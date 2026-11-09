# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""--dump-tensors must dump or refuse, never accept and write nothing.

Measured 2026-11-09 (206): yolov5-n-face_640x640_sync -i <image> --dump-tensors
without --save wrote nothing -- 11 sync runners made their run directory only
for --save -- and casvit-m_224x224_async --dump-tensors --save wrote run_info
and the image but no tensor: the async runners registered the flag and never
read it. tests/cpp_example/test_dump_tensors.py always passes --save and only
picks three detection binaries, so neither showed. These read the runner
sources, so they hold on a machine with no NPU.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNERS = PROJECT_ROOT / "src" / "cpp_example" / "common" / "runner"
SYNC = sorted(p for p in RUNNERS.glob("sync_*runner.hpp") if "dump-tensors" in p.read_text(encoding="utf-8"))
ASYNC = sorted(p for p in RUNNERS.glob("async_*runner.hpp") if "dump-tensors" in p.read_text(encoding="utf-8"))


def test_the_runner_sets_are_found():
    assert len(SYNC) >= 13 and len(ASYNC) >= 14, (len(SYNC), len(ASYNC))


@pytest.mark.parametrize("path", SYNC, ids=lambda p: p.name)
def test_a_sync_runner_makes_its_run_directory_for_dump_tensors_alone(path):
    text = path.read_text(encoding="utf-8")
    # every condition that creates a run directory also admits --dump-tensors
    for m in re.finditer(r"if \(([^\n]*?)\) \{[^{}]*?(?:run_?[dD]ir) = makeRunDir", text, re.S):
        cond = m.group(1)
        if "saveMode" in cond:   # (a nested "run_dir.empty()" fallback is fine)
            assert "dumpTensors" in cond, f"{path.name}: run dir only for {cond!r}"
    assert "makeRunDir" in text


@pytest.mark.parametrize("path", SYNC, ids=lambda p: p.name)
def test_a_sync_runner_saves_the_image_only_for_save(path):
    text = path.read_text(encoding="utf-8")
    assert "saveMode || dumpEnabled" not in text, path.name


@pytest.mark.parametrize("path", ASYNC, ids=lambda p: p.name)
def test_an_async_runner_refuses_dump_tensors(path):
    text = path.read_text(encoding="utf-8")
    assert re.search(r"if \(args\.dumpTensors\) \{\s*dxapp::fatal_error\(", text), path.name
