"""The C++ [PROGRESS] contract (tests/progress/test_progress.cpp) builds and passes.

The runners include src/cpp_example/common/utility/progress.hpp; this compiles the contract
test against it with the host compiler, so a change to the header is checked without a full
dx_app build.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TEST_CPP = ROOT / "tests" / "progress" / "test_progress.cpp"
UTILITY = ROOT / "src" / "cpp_example" / "common" / "utility"


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not installed")
def test_progress_contract_builds_and_passes(tmp_path):
    exe = tmp_path / "test_progress"
    build = subprocess.run(["g++", "-std=c++17", "-O1", "-I", str(UTILITY), str(TEST_CPP),
                            "-o", str(exe), "-lpthread"], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "All Progress contract tests PASSED" in run.stdout
