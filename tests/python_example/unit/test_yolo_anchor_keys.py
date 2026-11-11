# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""A config.json can only carry anchor strides as STRING keys.

YOLOv5Postprocessor looked the integer stride up in the override as given, so a
JSON-sourced table ({"8": ..., "16": ..., "32": ...}) matched no scale and every
level was skipped -- no detections, silently. PPUPostprocessor already converts
the keys; YOLOv5Postprocessor now does too. yolov7_640x640_nodecode is the
variant that needs it: without its own anchors it decoded with YOLOv5's and
scored 23.9 mAP on 200 COCO val2017 images against a 50.6 card.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.processors import YOLOv5Postprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V7 = {"8": [[12, 16], [19, 36], [40, 28]], "16": [[36, 75], [76, 55], [72, 146]],
      "32": [[142, 110], [192, 243], [459, 401]]}


def _one_box_outputs(size: int = 640) -> list:
    """Three raw heads, one confident cell: anchor 0 of stride 8 at grid (10, 10)."""
    outs = []
    for stride in (8, 16, 32):
        g = size // stride
        t = np.full((1, 3 * 85, g, g), -20.0, np.float32)
        if stride == 8:
            t[0, 0:4, 10, 10] = 0.0   # sig = 0.5: centre on the cell, w = anchor
            t[0, 4, 10, 10] = 20.0    # objectness
            t[0, 5, 10, 10] = 20.0    # class 0
        outs.append(t)
    return outs


def test_string_stride_keys_reach_the_decode():
    post = YOLOv5Postprocessor(640, 640, {"anchors": V7, "score_threshold": 0.25})
    assert set(post.ANCHORS) == {8, 16, 32}
    det = post._decode_multi_scale_outputs(_one_box_outputs())
    best = det[np.argmax(det[:, 4])]
    # YOLOv5 decode at sigmoid 0.5: w = (2 * 0.5)^2 * anchor_w = 12, h = 16
    assert abs(best[2] - 12.0) < 1e-3 and abs(best[3] - 16.0) < 1e-3


def test_yolov7_nodecode_variant_carries_the_yolov7_anchors():
    cfg = json.loads((PROJECT_ROOT / "src/python_example/object_detection/yolov7/"
                      "yolov7_640x640_nodecode/config.json").read_text(encoding="utf-8"))
    assert cfg["postprocessor"]["config_overrides"]["anchors"] == V7
