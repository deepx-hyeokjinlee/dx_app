# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Two evaluation conventions some models were trained and scored with.

pad_location="edge": the image at the top-left of the letterbox, all padding
right/bottom (the compile JSONs' ``pad_location: edge``; automl EfficientDet and
GluonCV evaluate this way). COCO val2017: EfficientDet-d4 41.17 mAP top-left vs
40.62 centred on all 5 000 images; yolov3-gluon 416 41.94 vs 40.95 on 200.

box_coding="yolov3": the original YOLOv3 head GluonCV keeps -- centre
(sig(t) + grid) * stride, size exp(t) * anchor. yolov3-gluon decoded the
Ultralytics way scored 30.2 mAP on 200 images, the YOLOv3 way 40.9.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from common.processors import LetterboxPreprocessor, YOLOv5Postprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_edge_letterbox_puts_the_image_top_left():
    img = np.full((100, 200, 3), 200, np.uint8)  # wide: padding is vertical
    out, ctx = LetterboxPreprocessor(64, 64, pad_location="edge").process(img)
    assert (ctx.pad_x, ctx.pad_y) == (0, 0)
    # zero pads alone read as "stretched" in the postprocessors; the flag says otherwise
    assert ctx.pad_edge
    assert out.shape == (64, 64, 3)
    assert (out[:32] == 200).all() and (out[32:] == 114).all()


def test_centre_letterbox_is_unchanged():
    img = np.full((100, 200, 3), 200, np.uint8)
    out, ctx = LetterboxPreprocessor(64, 64).process(img)
    assert not ctx.pad_edge
    assert ctx.pad_y == 16 and (out[16:48] == 200).all() and (out[:16] == 114).all()


def test_unknown_pad_location_is_refused():
    with pytest.raises(ValueError):
        LetterboxPreprocessor(64, 64, pad_location="top")


def _one_cell(nhwc: bool) -> list:
    """One confident cell at stride 8 (10, 10), anchor 0 = 10x13 (YOLOv5 COCO)."""
    outs = []
    for g in (80, 40, 20):
        t = np.full((1, 3 * 85, g, g), -20.0, np.float32)
        if g == 80:
            t[0, 0:2, 10, 10] = np.log(3.0)       # sig = 0.75
            t[0, 2:4, 10, 10] = np.log(2.0)       # exp = 2
            t[0, 4:6, 10, 10] = 20.0
        outs.append(t.transpose(0, 2, 3, 1).copy() if nhwc else t)
    return outs


@pytest.mark.parametrize("nhwc", [False, True])
def test_yolov3_box_coding(nhwc):
    post = YOLOv5Postprocessor(640, 640, {"box_coding": "yolov3"})
    det = post._decode_multi_scale_outputs(_one_cell(nhwc))
    best = det[np.argmax(det[:, 4] * det[:, 5])]
    assert abs(best[0] - (0.75 + 10) * 8) < 1e-3   # centre x
    assert abs(best[2] - 2 * 10) < 1e-3              # width = exp(ln 2) * 10


def test_default_box_coding_is_ultralytics():
    post = YOLOv5Postprocessor(640, 640, {})
    det = post._decode_multi_scale_outputs(_one_cell(False))
    best = det[np.argmax(det[:, 4] * det[:, 5])]
    assert abs(best[0] - (2 * 0.75 - 0.5 + 10) * 8) < 1e-3
    assert abs(best[2] - (2 * 2 / 3) ** 2 * 10) < 1e-3


@pytest.mark.parametrize("variant,edge,coding", [
    ("efficientdet/efficientdet-d1_640x640", True, None),
    ("efficientdet/efficientdet-d4_1024x1024", True, None),
    ("yolov3/yolov3-gluon_416x416", True, "yolov3"),
    ("yolov3/yolov3-gluon_608x608", True, "yolov3"),
])
def test_variants_carry_their_convention(variant, edge, coding):
    for root in ("python_example", "cpp_example"):
        cfg = json.loads((PROJECT_ROOT / "src" / root / "object_detection" / variant / "config.json")
                         .read_text(encoding="utf-8"))
        assert (cfg["preprocessor"]["kwargs"].get("pad_location") == "edge") == edge
        assert cfg["postprocessor"].get("config_overrides", {}).get("box_coding") == coding



def test_efficientdet_maps_an_edge_letterbox_back_by_the_gain():
    """Zero pads used to read as a stretch, scaling x and y separately:
    EfficientDet-d1 fell to 3.2 mAP on 200 COCO images with the edge letterbox.
    A box drawn on the model input must land where the letterbox put it."""
    from common.processors.efficientdet_postprocessor import EfficientDetPostprocessor
    img = np.zeros((100, 200, 3), np.uint8)
    _, ctx = LetterboxPreprocessor(64, 64, pad_location="edge").process(img)  # gain 0.32
    post = EfficientDetPostprocessor.__new__(EfficientDetPostprocessor)
    post.input_width = post.input_height = 64
    box = post._scale_box(np.array([16.0, 8.0, 32.0, 24.0]), ctx, None, None)
    np.testing.assert_allclose(box, [50.0, 25.0, 100.0, 75.0], atol=1e-4)
