"""Shared pytest fixtures for the torchocr test suite."""

import pytest
import torch


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def ascii_charset() -> list[str]:
    """Blank + 95 ASCII printables = 96 classes."""
    return ["-"] + [chr(c) for c in range(32, 127)]
