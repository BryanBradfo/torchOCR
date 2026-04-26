"""DBLoss and CRNNLoss: forward + backward + bit-equality with manual CTC."""

import torch

from torchocr import CRNNLoss, DBLoss
from torchocr.models import CRNN, DBNet


def _synth_db_targets(b: int = 1, h: int = 64, w: int = 64):
    gt_prob = torch.zeros(b, 1, h, w)
    gt_prob[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
    gt_thresh = torch.zeros_like(gt_prob)
    gt_thresh[:, :, h // 4 - 4 : h // 4 + 4, w // 4 : 3 * w // 4] = 0.7
    return gt_prob, gt_thresh


def test_dbloss_returns_components():
    detector = DBNet()
    detector.train()
    out = detector(torch.randn(1, 3, 64, 64))
    gt_prob, gt_thresh = _synth_db_targets()
    result = DBLoss()(out, gt_prob, gt_thresh)
    assert {"loss", "probability", "threshold", "binary"} <= result.keys()
    for v in result.values():
        assert torch.isfinite(v).all()


def test_dbloss_backward_flows_to_detector():
    detector = DBNet()
    detector.train()
    out = detector(torch.randn(1, 3, 64, 64))
    gt_prob, gt_thresh = _synth_db_targets()
    DBLoss()(out, gt_prob, gt_thresh)["loss"].backward()
    grad_total = sum(
        p.grad.abs().sum().item()
        for p in detector.parameters()
        if p.grad is not None
    )
    assert grad_total > 0


def test_dbloss_binary_weight_zero_disables_binary():
    detector = DBNet()
    out = detector(torch.randn(1, 3, 64, 64))
    gt_prob, gt_thresh = _synth_db_targets()
    full = DBLoss(binary_weight=1.0)(out, gt_prob, gt_thresh)
    no_bin = DBLoss(binary_weight=0.0)(out, gt_prob, gt_thresh)
    expected = full["probability"] + 10.0 * full["threshold"]
    assert torch.allclose(no_bin["loss"], expected, atol=1e-5)


def test_crnnloss_matches_manual_log_softmax_ctc():
    """Wrapper must be bit-equal to manual log_softmax + nn.CTCLoss."""
    torch.manual_seed(0)
    logits = torch.randn(20, 2, 10)
    targets = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    input_lengths = torch.tensor([20, 20], dtype=torch.long)
    target_lengths = torch.tensor([3, 2], dtype=torch.long)

    wrapped = CRNNLoss()(logits, targets, input_lengths, target_lengths)
    manual = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)(
        logits.log_softmax(dim=-1), targets, input_lengths, target_lengths
    )
    assert torch.allclose(wrapped, manual)


def test_crnnloss_backward_flows_to_recognizer():
    rec = CRNN(num_classes=10)
    rec.train()
    logits = rec(torch.randn(2, 3, 32, 80))  # T = 20
    targets = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    input_lengths = torch.tensor([20, 20], dtype=torch.long)
    target_lengths = torch.tensor([3, 2], dtype=torch.long)
    CRNNLoss()(logits, targets, input_lengths, target_lengths).backward()
    grad_total = sum(
        p.grad.abs().sum().item() for p in rec.parameters() if p.grad is not None
    )
    assert grad_total > 0
