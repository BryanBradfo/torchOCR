# Credits

## PaddleOCR2Pytorch

[PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch) by Frotms is
licensed under Apache-2.0. torchocr's PaddleOCR-compatible weight conversion
path adapts the following ideas from PaddleOCR2Pytorch:

- **Architectural matches**: which PaddleOCR backbone, neck, and head
  combinations correspond to which checkpoints, and at what shapes
  (`pytorchocr/modeling/backbones/det_resnet_vd.py`,
  `pytorchocr/modeling/necks/db_fpn.py`,
  `pytorchocr/modeling/heads/det_db_head.py`).
- **Weight-loading recipe**: handling both Paddle 1.x dygraph and
  Paddle 2.x checkpoint formats
  (`pytorchocr/base_ocr_v20.py:94-103`).
- **Post-processing flow**: contour detection + polygon offset via
  `pyclipper` + minimum-area-rectangle scoring
  (`pytorchocr/postprocess/db_postprocess.py`).
- **Tensor-name remapping rules** for the converter:
  `running_mean`/`running_var` -> `_mean`/`_variance`,
  `stages.N` -> `stageN`, `head.binarize` / `head.thresh` nesting,
  Linear-weight transpose for the CTC head
  (`converter/det_converter.py`,
  `converter/ch_ppocr_v3_det_converter.py`,
  `converter/rec_converter.py`).
- **Recognition architecture details**: the `(2, 1)` recognizer-stride
  pattern, the `out_pool` final downsample, and the SequenceEncoder
  (`Im2Seq` + 2-layer BiLSTM) layout used by `ch_ppocr_v2.0_rec`
  (`pytorchocr/modeling/backbones/rec_resnet_vd.py`,
  `pytorchocr/modeling/necks/rnn.py`,
  `pytorchocr/modeling/heads/rec_ctc_head.py`).
- **Chinese charset file**: `src/torchocr/data/ppocr_keys_v1.txt`
  vendored verbatim from `pytorchocr/utils/ppocr_keys_v1.txt`.

torchocr re-implements these in the torchvision idiom (typed
dataclasses, modular composition, runtime shape contracts) rather
than mirroring PaddleOCR2Pytorch's class hierarchy or config-driven
architecture. PaddleOCR2Pytorch is required only at conversion time;
torchocr's inference path has no Paddle dependency.

## PaddleOCR

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) by Baidu is the
upstream source of the trained weights torchocr converts. Released
under Apache-2.0.

## Sample images in `examples/`

The Chinese, English, and Japanese demo images shipped under
`examples/` are vendored from PaddleOCR2Pytorch's `doc/imgs/` and
`doc/imgs_en/` directories (Apache-2.0):

| torchocr file | upstream path |
|---|---|
| `examples/chinese_receipt.jpg` | `PaddleOCR2Pytorch/doc/imgs/00018069.jpg` |
| `examples/chinese_typeset.jpg` | `PaddleOCR2Pytorch/doc/imgs/11.jpg` |
| `examples/english_doc.jpg` | `PaddleOCR2Pytorch/doc/imgs_en/img_10.jpg` |
| `examples/japanese.jpg` | `PaddleOCR2Pytorch/doc/imgs/japan_1.jpg` |

These are the canonical PaddleOCR demo images and exist purely to
exercise torchocr's pipeline against the same inputs upstream uses.
