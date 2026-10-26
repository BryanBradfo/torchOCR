"""Sphinx configuration for the torchocr documentation site.

Build with ``sphinx-build -b html docs docs/_build/html`` (``pip install -e .[docs]``).
Set ``TORCHOCR_DOCS_GALLERY=0`` to skip running the gallery examples, which
download pretrained weights.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torchocr  # noqa: E402

project = "torchocr"
author = "torchocr contributors"
copyright = "2026, torchocr contributors"
release = torchocr.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "weights_tables",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "torchvision": ("https://docs.pytorch.org/vision/stable", None),
}

sphinx_gallery_conf = {
    "examples_dirs": str(ROOT / "gallery"),
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "plot_gallery": os.environ.get("TORCHOCR_DOCS_GALLERY", "1") != "0",
    "download_all_examples": False,
    "remove_config_comments": True,
}

html_theme = "pydata_sphinx_theme"
html_title = f"torchocr {release}"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/BryanBradfo/torchOCR",
    "navbar_align": "left",
    "show_toc_level": 2,
}
templates_path = []
exclude_patterns = ["_build"]
