"""Word crops stored in the deep-text-recognition-benchmark LMDB layout."""

from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image


class LMDBWords(Dataset[tuple[Tensor, str]]):
    """Cropped word images in the LMDB layout of clovaai's
    deep-text-recognition-benchmark.

    This is how the standard recognition corpora circulate (MJSynth,
    SynthText, IIIT5K, SVT, ICDAR 2013/2015 crops, CUTE80, MLT word crops):
    ``num-samples`` holds the count and samples ``i = 1..N`` are stored under
    ``image-%09d`` (encoded image bytes) and ``label-%09d`` (UTF-8 text).

    Requires the optional ``lmdb`` package (``pip install torchocr[lmdb]``).

    Args:
        root: Directory containing ``data.mdb``.
        label_filter: Optional predicate on the label; only matching samples
            are kept. Evaluated once on all labels at construction (images are
            not decoded), e.g. to drop words outside a model's charset.
        transform: Optional callable applied to the ``uint8`` image.

    Returns:
        ``(image, text)``: a ``(3, H, W)`` ``uint8`` RGB crop and its label.
    """

    def __init__(
        self,
        root: str | Path,
        label_filter: Callable[[str], bool] | None = None,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.root = Path(root)
        if not (self.root / "data.mdb").is_file():
            raise FileNotFoundError(f"No LMDB database (data.mdb) under {self.root}.")
        self.transform = transform
        self._env = None
        with self._open().begin() as txn:
            count = int(txn.get(b"num-samples"))
            labels = [txn.get(f"label-{i:09d}".encode()).decode("utf-8") for i in range(1, count + 1)]
        keep = [i for i, label in enumerate(labels, start=1) if label_filter is None or label_filter(label)]
        self.indices = keep
        self.labels = [labels[i - 1] for i in keep]
        # LMDB environments must not cross fork(): DataLoader workers reopen lazily.
        self._env.close()
        self._env = None

    def _open(self):
        try:
            import lmdb
        except ImportError as exc:
            raise ImportError("LMDBWords needs the lmdb package: pip install torchocr[lmdb]") from exc
        if self._env is None:
            self._env = lmdb.open(str(self.root), readonly=True, lock=False, readahead=False, meminit=False)
        return self._env

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        with self._open().begin() as txn:
            data = txn.get(f"image-{self.indices[index]:09d}".encode())
        image = decode_image(torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=ImageReadMode.RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]
