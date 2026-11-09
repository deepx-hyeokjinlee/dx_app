# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""CLIPZeroShotPostprocessor must not normalise the engine's output in place.

Measured 2026-11-09: clip-img_vit-b32 --dump-tensors wrote an output tensor of
L2 norm 1.0 from Python and 12.9 from C++ for the same input -- the Python
postprocessor had divided the output buffer itself (a reshape view), so the
dump, and anything reading the outputs after it, saw the normalised vector.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.base import PreprocessContext  # noqa: E402
from common.processors import CLIPZeroShotPostprocessor  # noqa: E402


def test_the_output_buffer_is_left_as_the_engine_wrote_it(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"variant": "x", "checkpoint": "x", "embed_dim": 4, "template": "{}",
                                "labels": ["a", "b"], "prompts": ["a", "b"],
                                "embeddings": [[1, 0, 0, 0], [0, 1, 0, 0]]}), encoding="utf-8")
    pp = CLIPZeroShotPostprocessor(224, 224, {"prompt_bank": str(bank), "top_k": 2})
    out = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    before = out.copy()
    results = pp.process([out], PreprocessContext())
    np.testing.assert_array_equal(out, before)
    assert results and results[0].class_id == 1
