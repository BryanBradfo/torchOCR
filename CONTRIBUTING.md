# Contributing to torchocr

Thanks for considering a contribution. This document covers everything from environment setup to PR etiquette so you can ship changes quickly without round-trips on the basics.

## Code of conduct

Participation in this project is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). By contributing, you agree to uphold it. Report concerns to bryan.chen@polytechnique.edu.

## Ways to contribute

| Type | Where |
|---|---|
| Bug reports | GitHub Issues — include a minimal reproducer (10 lines of Python) and your `torch.__version__` |
| Feature proposals | GitHub Discussions or an Issue tagged `proposal:` — discuss before opening a PR |
| Code | Fork → branch → PR. Conventions below. |
| Docs / typos | Direct PR; no Issue needed |
| Pretrained weights | Email — there's a separate distribution process for checkpoint hosting |

## Development environment

torchocr requires Python ≥ 3.10 and a working PyTorch install. Editable install is the supported development path:

```bash
git clone https://github.com/BryanBradfo/torchOCR.git
cd torchOCR
pip install -e .
```

For GPU work, install a CUDA-enabled PyTorch build *before* the editable install:

```bash
# Replace cu128 with the wheel matching your driver — see https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Verify the install with the shape-contract smoke test:

```bash
python scripts/test_inference.py
# Expected: probability/threshold/logits shapes printed, exits 0
```

## Style invariants (match exactly)

These are *project conventions* established in v0.1.0. Match them in every PR — drift here forces reviewers to debate the same things twice.

- **Python ≥ 3.10.** Use union syntax: `Tensor | None`, never `Optional[Tensor]`.
- **No `from __future__ import annotations`** — modern syntax already works.
- **`from torch import Tensor, nn`** — never `torch.Tensor` in signatures.
- **Google-style docstrings.** Class docstring on the line below `class`. `Args:` / `Returns:` / `Note:` blocks.
- **Absolute imports within the package** (`from torchocr.core.structures import DocumentTensor` from outside; `from .structures import DocumentTensor` from inside `core/`).
- **Typed return containers are `@dataclass`es** — see [`DocumentTensor`](src/torchocr/core/structures.py) and [`DBNetOutput`](src/torchocr/models/detection.py). Not `dict[str, Tensor]`, not `NamedTuple`.
- **Input-shape guards** raise `ValueError` with the offending shape in the message:
  ```python
  if images.ndim != 4 or images.shape[1] != 3:
      raise ValueError(f"Expected (B, 3, H, W) images; got {tuple(images.shape)}.")
  ```
- **No back-compat shims.** v0.x is pre-stable; rename, replace, delete. Don't preserve aliases for the previous shape of an API.

## Architecture invariants

Don't break these without coordination — they're load-bearing for the pipeline:

| Component | Input | Output |
|---|---|---|
| `DBNet` | `(B, 3, H, W)`, H and W divisible by 32 | `DBNetOutput(probability, threshold)` both `(B, 1, H, W)` |
| `CRNN` | `(B, C, 32, W)`, height *exactly* 32 | `(T, B, num_classes)`, `T = W // 4` |
| `DBPostProcessor` | `DBNetOutput` | `(K, 5)` boxes formatted `[batch_idx, x1, y1, x2, y2]` |
| `CTCGreedyDecoder` | `(T, B, num_classes)` logits | `list[str]` length B |
| `OCRPipeline.__call__` | `(3, H, W)` image | populated `DocumentTensor` |

`OCRPipeline` is *not* an `nn.Module`. Move detector and recognizer to your device *before* constructing the pipeline.

CRNN forward returns *raw* logits — callers apply `log_softmax` for `nn.CTCLoss`.

## Running tests

There is no `tests/` directory yet (it's on the roadmap). Today's verification surface:

```bash
# Shape contracts
python scripts/test_inference.py

# Image demo (writes examples/sample_doc.jpg, examples/demo_output.jpg)
python examples/demo_inference.py

# PDF demo (writes examples/sample_doc.pdf)
python examples/demo_pdf.py

# CLI on the synthetic PDF
python scripts/infer.py --input examples/sample_doc.pdf --output-dir /tmp/torchocr_smoke
```

Every PR that touches `src/torchocr/` should keep all four green.

## Pull request process

### Branch naming

Use Conventional Commits prefixes as branch namespaces:

- `feat/<short-slug>` — new functionality
- `fix/<short-slug>` — bug fixes
- `docs/<short-slug>` — documentation only
- `chore/<short-slug>` — repo hygiene, deps, CI
- `refactor/<short-slug>` — code restructure without behavior change

### Commit messages

Conventional Commits with optional scope:

```
<type>(<scope>): <subject>

<body — wrapped at 72 cols, explains the why>
```

Examples from the v0.1.0 PR:

- `feat(models): implement DBNet and CRNN with shape-asserted forwards`
- `feat(losses): add DBLoss and CRNNLoss for training`
- `chore(gitignore): cover demo artifacts, weights, ML tracking dirs`

**One logical change per commit.** Don't bundle a feature commit with a docs fix and a CI tweak. Reviewers (and `git bisect`) thank you.

### PR description

Use the structure from the v0.1.0 PR:

- **Summary** — what this changes and why
- **What's included** — file-by-file inventory if substantial
- **Verification** — tests/demos you ran
- **What's intentionally not in this PR** — pre-empts reviewer "shouldn't this also do X?" comments
- **Test plan** — checklist of what reviewers should verify

### Review expectations

- Architecture changes (anything in `models/`, `pipelines.py`, `losses.py`) need at least one approving review.
- Docs / chore PRs can self-merge once CI is green (when CI exists).
- If a review comment is unclear, ask. Reviewers welcome push-back when warranted.

## Where to ask questions

- **Quick "how do I do X?"** — GitHub Discussions
- **"Is this a bug?"** — GitHub Issues, `bug:` tag
- **"I'm thinking of contributing X, is it in scope?"** — Discussions or `proposal:` Issue, *before* writing the code

Welcome aboard. The codebase is small and the design is opinionated; PRs that match existing patterns land fast.
