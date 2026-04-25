"""Loss functions for training torchocr models."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models.detection import DBNetOutput


class DBLoss(nn.Module):
    """Differentiable Binarization training loss.

    Combines the three components of Liao et al. (AAAI 2020):

    1. **Probability-map BCE with online hard example mining (OHEM).**
       OHEM keeps every positive pixel plus the
       ``negative_ratio * n_pos`` hardest negatives, preventing the
       background class from drowning the loss.
    2. **Threshold-map L1 loss** restricted to the boundary band where
       the GT threshold is non-zero.
    3. **Approximate-binary Dice loss** computed on
       ``B_hat = sigmoid(k * (P - T))``. The factor ``k`` controls how
       sharply this differentiable form approximates the true
       Heaviside step ``P > T``: as ``k -> infinity`` ``B_hat`` becomes
       a step function, but its gradient also vanishes away from the
       decision boundary. The paper's ``k = 50`` is a tested
       compromise -- a tight transition (``~0.04`` units in ``P - T``)
       that still leaves backprop signal. Lower ``k`` flattens the
       step toward ``0.5``; higher ``k`` starves the gradients.

    Total loss: ``Lp + binary_weight * Lb + threshold_weight * Lt``.

    Args:
        k: Steepness of the differentiable binarization. Default 50.
        threshold_weight: Multiplier on the threshold-map L1 loss.
            Default 10, matching the paper's beta.
        binary_weight: Multiplier on the dice loss over the
            approximate binary map. Default 1, matching the paper's
            alpha. Set to 0 to disable the binary term entirely.
        negative_ratio: Negatives-to-positives ratio for the OHEM
            sampling in the probability BCE. Default 3.0.
        eps: Numerical-stability epsilon for BCE clamping and Dice.
    """

    def __init__(
        self,
        k: float = 50.0,
        threshold_weight: float = 10.0,
        binary_weight: float = 1.0,
        negative_ratio: float = 3.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive; got {k}.")
        if negative_ratio <= 0:
            raise ValueError(f"negative_ratio must be positive; got {negative_ratio}.")
        self.k = k
        self.threshold_weight = threshold_weight
        self.binary_weight = binary_weight
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(
        self,
        prediction: DBNetOutput,
        gt_probability: Tensor,
        gt_threshold: Tensor,
        training_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute DB loss components and their weighted sum.

        Args:
            prediction: Output of :class:`DBNet` for one batch.
            gt_probability: ``(B, 1, H, W)`` 0/1 (or soft) text mask.
            gt_threshold: ``(B, 1, H, W)`` float threshold-map target.
            training_mask: Optional ``(B, 1, H, W)`` boolean tensor of
                valid pixels. ``False`` entries are excluded from all
                three losses (use this to ignore "don't care" regions
                such as overlapping or illegible text).

        Returns:
            ``{"loss", "probability", "threshold", "binary"}``. Call
            ``.backward()`` on the ``"loss"`` entry; log the others.
        """
        prob = prediction.probability
        thresh = prediction.threshold
        if prob.shape != gt_probability.shape or thresh.shape != gt_threshold.shape:
            raise ValueError(
                f"Predictions and targets must share shape; got prob {tuple(prob.shape)} vs "
                f"{tuple(gt_probability.shape)}, threshold {tuple(thresh.shape)} vs "
                f"{tuple(gt_threshold.shape)}."
            )

        if training_mask is None:
            mask = torch.ones_like(gt_probability, dtype=torch.bool)
        else:
            mask = training_mask.to(torch.bool)

        prob_loss = self._bce_ohem(prob, gt_probability, mask)
        thresh_loss = self._threshold_l1(thresh, gt_threshold, mask)
        binary_loss = self._dice(prob, thresh, gt_probability, mask)

        total = (
            prob_loss
            + self.binary_weight * binary_loss
            + self.threshold_weight * thresh_loss
        )
        return {
            "loss": total,
            "probability": prob_loss,
            "threshold": thresh_loss,
            "binary": binary_loss,
        }

    def _bce_ohem(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        clamped = pred.clamp(min=self.eps, max=1.0 - self.eps)
        losses = F.binary_cross_entropy(clamped, target, reduction="none")

        positives = (target > 0.5) & mask
        negatives = (target <= 0.5) & mask
        n_pos = int(positives.sum().item())
        n_neg_cap = int(negatives.sum().item())
        n_neg = int(min(n_neg_cap, max(n_pos, 1) * self.negative_ratio))

        if n_pos == 0 and n_neg == 0:
            return (losses * mask).sum() / mask.sum().clamp(min=1)

        pos_loss = losses[positives].sum() if n_pos > 0 else losses.new_zeros(())
        if n_neg > 0:
            neg_losses = losses[negatives]
            if neg_losses.numel() > n_neg:
                neg_loss = neg_losses.topk(n_neg)[0].sum()
            else:
                neg_loss = neg_losses.sum()
        else:
            neg_loss = losses.new_zeros(())
        return (pos_loss + neg_loss) / max(n_pos + n_neg, 1)

    def _threshold_l1(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        active = mask & (target > 0)
        if not active.any():
            return pred.new_zeros(())
        return F.l1_loss(pred[active], target[active])

    def _dice(self, prob: Tensor, thresh: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        binary = torch.sigmoid(self.k * (prob - thresh))
        valid = mask.to(binary.dtype)
        binary = binary * valid
        target = target * valid
        intersection = (binary * target).sum()
        return 1.0 - (2.0 * intersection + self.eps) / (binary.sum() + target.sum() + self.eps)


class CRNNLoss(nn.Module):
    """Wrapper around :class:`torch.nn.CTCLoss` for CRNN-style logits.

    Applies ``log_softmax`` to the recognizer's logits before
    delegating to ``nn.CTCLoss``, which expects log probabilities.
    The blank index defaults to 0, matching :class:`CTCGreedyDecoder`.

    Args:
        blank_index: CTC blank class index. Default 0.
        reduction: Reduction passed to ``nn.CTCLoss`` -- 'mean', 'sum',
            or 'none'. Default 'mean'.
        zero_infinity: Whether to zero out infinite losses (caused by
            input lengths shorter than target lengths). Default True
            for training stability.
    """

    def __init__(
        self,
        blank_index: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = True,
    ) -> None:
        super().__init__()
        self.ctc = nn.CTCLoss(
            blank=blank_index,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """Compute the CTC loss.

        Args:
            logits: ``(T, B, num_classes)`` raw recognizer outputs.
            targets: Either ``(B, S_max)`` padded or
                ``(sum(target_lengths),)`` concatenated label tensor,
                per ``nn.CTCLoss`` convention.
            input_lengths: ``(B,)`` int tensor giving each sequence's
                effective length in time-steps.
            target_lengths: ``(B,)`` int tensor giving each target's
                length.
        """
        if logits.ndim != 3:
            raise ValueError(
                f"Expected logits of shape (T, B, num_classes); got {tuple(logits.shape)}."
            )
        log_probs = logits.log_softmax(dim=-1)
        return self.ctc(log_probs, targets, input_lengths, target_lengths)
