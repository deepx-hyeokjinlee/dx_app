# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Boxes and keypoints clip to [0, w] x [0, h]; detections carry class_name.

Measured 2026-11-09: the two trees clipped differently -- C++ scaleBox to
[0, w], 20 Python processors (and C++ centerpose/picodet/rtdetr and
scaleToOriginal) to [0, w - 1], C++ keypoints not at all -- so the same box
came out 1536.0 in C++ and 1535.0 in Python (yolov5-s on sample_face.jpg).
Both trees now clip like Ultralytics' clip_coords, [0, w]. And Python
detections left class_name empty where the C++ processor of the same model
fills it (resolve_class_name: config class_names, else COCO-80).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.base import PreprocessContext  # noqa: E402
from common.utility.preprocessing import scale_to_original  # noqa: E402

PROCESSORS = PROJECT_ROOT / "src" / "python_example" / "common" / "processors"


def test_no_python_processor_clips_coordinates_to_w_minus_1():
    offenders = []
    for p in sorted(PROCESSORS.glob("*.py")):
        if p.name == "sfa3d_bev_preprocessor.py":   # integer BEV pixel indices
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"original_(width|height) - 1|np\.clip\([^)]*\b(ow|oh) - 1\)", line):
                offenders.append(f"{p.name}:{i}")
    assert not offenders, offenders


def test_a_point_clips_to_the_image_size_not_one_less():
    ctx = PreprocessContext(original_width=100, original_height=50, scale=1.0)
    assert scale_to_original(150.0, 70.0, ctx) == (100.0, 50.0)
