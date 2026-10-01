# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Two drifts found by running every task live from DX AI Studio (2026-10-01).

1. custom_ops.py was relocated into each variant folder, but the prompt bank stayed in
   the family folder where scripts/build_clip_prompt_bank.py writes it. The ViT-B/32
   zero-shot head then failed with FileNotFoundError on every input.
2. clip-img_vit-b32's C++ config said image_only false / stream inputs on, but the C++
   embedding runner has no video path at all ("Option 'v' does not exist"). The Python
   example does run video once it finds its bank (41.9 FPS on dogs.mp4), so only the C++
   config was wrong.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLIP = PROJECT_ROOT / "src/python_example/zero_shot_image_classification/clip"
B32 = "clip-img_vit-b32_256x256_datacomp-s34b-b86k"


def _custom_ops(variant):
    spec = importlib.util.spec_from_file_location(f"co_{variant}", CLIP / variant / "custom_ops.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_b32_zero_shot_head_finds_its_prompt_bank():
    mod = _custom_ops(B32)
    bank = mod.Vit_b_32_256_datacomp_s34b_b86kFactory._BANK
    assert Path(bank).is_file(), bank
    assert json.loads(Path(bank).read_text())["variant"] == B32


def test_every_cpp_embedding_example_is_image_only():
    """The C++ embedding runner has no -v/-c/-r: its configs must not offer stream input."""
    wrong = []
    for cfg in (PROJECT_ROOT / "src/cpp_example").glob("*/*/*/config.json"):
        d = cfg.parent
        cpp = next(d.glob("*_async.cpp"), None) or next(d.glob("*_sync.cpp"), None)
        if cpp is None or "embedding_runner.hpp" not in cpp.read_text(encoding="utf-8"):
            continue
        c = json.loads(cfg.read_text(encoding="utf-8"))
        if not c.get("image_only") or (c.get("cli") or {}).get("include_stream_inputs"):
            wrong.append(d.name)
    assert not wrong, wrong
