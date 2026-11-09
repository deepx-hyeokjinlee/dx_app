# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""A semantic-segmentation result is in image coordinates, in both trees.

Measured 2026-11-09 (sample_street.jpg 768x576): Python returned 576x768
for deeplab (pre-argmaxed outputs were always mapped back) but 256x256 for
unet (logits were mapped back only with resize_to_original, off by default),
while every C++ semantic postprocessor returned the network-size map and
left the un-letterboxing to its visualizer. Both trees now map every class
map back like a box: letterbox padding cut (scaled by map/input for a
smaller head), the rest resized NEAREST to the image size.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.base import PreprocessContext  # noqa: E402
from common.processors import SemanticSegmentationPostprocessor  # noqa: E402
from common.processors.segmentation_postprocessor import class_map_to_original  # noqa: E402


def _letterbox_ctx():
    # 200x100 image letterboxed into 64x64: scale 0.32, content 64x32, pad_y 16
    return PreprocessContext(pad_x=0, pad_y=16, scale=0.32, original_width=200,
                             original_height=100, input_width=64, input_height=64)


def test_a_letterboxed_map_loses_its_padding():
    cm = np.zeros((64, 64), np.int32)
    cm[16:48, :] = 3            # the content rows
    out = class_map_to_original(cm, _letterbox_ctx(), 64, 64)
    assert out.shape == (100, 200) and (out == 3).all()


def test_a_stride_4_map_scales_the_padding():
    cm = np.zeros((16, 16), np.int32)
    cm[4:12, :] = 5             # 16/64 of the input: pad 16 -> 4 rows
    out = class_map_to_original(cm, _letterbox_ctx(), 64, 64)
    assert out.shape == (100, 200) and (out == 5).all()


def test_the_logits_path_returns_the_image_size_by_default():
    pp = SemanticSegmentationPostprocessor(64, 64, {"num_classes": 3})
    logits = np.zeros((1, 3, 64, 64), np.float32)
    logits[0, 2, 16:48, :] = 1.0
    (res,) = pp.process([logits], _letterbox_ctx())
    assert (res.height, res.width) == (100, 200)
    assert set(np.unique(res.mask)) == {2}
