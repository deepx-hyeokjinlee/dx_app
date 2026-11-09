# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""--dump-tensors must dump, in every runner that takes it.

Measured 2026-11-09 (206): yolov5-n-face_640x640_sync -i <image> --dump-tensors
without --save wrote nothing -- 11 sync runners made their run directory only
for --save -- and casvit-m_224x224_async --dump-tensors --save wrote run_info
and the image but no tensor: the async runners registered the flag and never
read it. tests/cpp_example/test_dump_tensors.py always passes --save and only
picks three detection binaries, so neither showed. These read the runner
sources, so they hold on a machine with no NPU. (A first fix refused the flag
in the async runners; they now dump through armAsyncDump/writeAsyncDump, and
the restoration runners write the model tensors -- per tile for tiled SR --
instead of the source frame alone.)
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
def test_an_async_runner_dumps(path):
    text = path.read_text(encoding="utf-8")
    assert "not supported by the async" not in text, path.name
    assert re.search(r"armAsyncDump\(", text) and re.search(r"writeAsyncDump\(", text), path.name
    for m in re.finditer(r"if \(([^\n]*?)\) \{[^{}]*?(?:run_?[dD]ir) = makeRunDir", text, re.S):
        if "saveMode" in m.group(1):
            assert "dumpTensors" in m.group(1), f"{path.name}: run dir only for {m.group(1)!r}"


def test_restoration_dumps_the_model_tensors():
    for name in ("sync_restoration_runner.hpp", "async_restoration_runner.hpp"):
        text = (RUNNERS / name).read_text(encoding="utf-8")
        assert "input_frame.bin" not in text, name
    sr = (RUNNERS.parent / "utility" / "sr_tiling.hpp").read_text(encoding="utf-8")
    assert "dumpTensorsToFiles(dump_dir" in sr
