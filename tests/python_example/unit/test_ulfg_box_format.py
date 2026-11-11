# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""ULFG box encoding is read off the STRUCTURE of the boxes tensor.

Decoded corners [x1, y1, x2, y2] are ordered (x2 > x1, y2 > y1) in every row --
100 % on ulfgfd rfb/slim-320 and rfb-640 -- while SSD anchor deltas are not
(3-21 %, the _nopp exports, which keep deltas through their tail). The three
implementations used to guess from the share of negative values instead; on
sample_crowd a DECODED rfb-320 tensor had 21 of its first 200 values below -0.01,
the C++ trees took it for deltas and decoded every box a second time (a 27x39
face came out 72x52). Edge priors legitimately carry corners slightly outside
the frame, so negatives say nothing about the encoding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.processors.ulfg_postprocessor import ULFGPostprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _decoded_with_negative_edges(n: int = 120) -> np.ndarray:
    boxes = np.tile(np.array([-0.03, -0.02, 0.04, 0.05], np.float32), (n, 1))
    boxes[0] = [0.25, 0.25, 0.75, 0.75]
    return boxes


def test_decoded_corners_with_negative_edge_values_stay_corners():
    post = ULFGPostprocessor(320, 240)
    boxes = _decoded_with_negative_edges()
    assert np.mean(boxes < 0) > 0.1  # the old Python rule would have said "deltas"
    x1, y1, x2, y2 = post._decode_boxes(boxes)
    assert abs((x2[0] - x1[0]) - 0.5 * 320) < 1e-3
    assert abs((y2[0] - y1[0]) - 0.5 * 240) < 1e-3


def test_flat_deltas_are_prior_decoded():
    post = ULFGPostprocessor(320, 240)
    n = 4420  # the 320x240 prior grid
    sign = np.where(np.arange(n) % 2, 1.0, -1.0).astype(np.float32)
    boxes = np.stack([0.3 * sign, -0.2 * sign, np.full(n, -0.5, np.float32), 0.4 * sign], axis=1)
    boxes[0] = 0.0  # prior 0 itself: 10x10 px at stride 8
    x1, y1, x2, y2 = post._decode_boxes(boxes)
    assert abs((x2[0] - x1[0]) - 10.0) < 0.5
    assert abs((y2[0] - y1[0]) - 10.0) < 0.5


def test_the_cpp_trees_use_the_same_structural_rule():
    lib = (PROJECT_ROOT / "src/postprocess/ulfgfd/ulfgfd_postprocess.cpp").read_text(encoding="utf-8")
    example = (PROJECT_ROOT / "src/cpp_example/common/processors/ulfg_face_postprocessor.hpp").read_text(
        encoding="utf-8")
    for source in (lib, example):
        assert "b[2] > b[0] && b[3] > b[1]" in source
        assert "< -0.01f" not in source
