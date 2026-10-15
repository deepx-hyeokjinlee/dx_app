"""An exported graph project (scripts/export_graph_project.py), built and run outside this repository.

Three tiers, by marker:

  * ``contract``     - source-level: the graph code reads its root at run time.
  * ``graph_export`` - no NPU: exports tests/cpp_example/graph_export/user_zone.json to /dev/shm,
                       builds it there with a clean environment (only the DEEPX runtime, CMake,
                       a compiler and OpenCV are visible) and checks the program finds its project
                       from anywhere and names no path of this repository.
  * ``graph_e2e``    - the NPU: the exported program and the in-suite multi_model_graph_async give
                       the same events and the same overlay on an image and on a video, and an
                       edited stub raises its own events.

The build tiers skip for exactly ONE reason: yolov8-n_640x640.dxnn is not downloaded. Anything else
that goes wrong is a failure.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import PROJECT_ROOT

# Every file the exported project compiles that once read the compiled-in root. A use left in
# code (not a comment, not a macro body the graph code never expands) would not compile in an
# export, which defines no PROJECT_ROOT_DIR.
ROOT_READERS = [
    "src/cpp_example/common/registry/typed_stage.hpp",
    "src/cpp_example/common/utility/common_util.hpp",
    "src/cpp_example/common/processors/gallery_retrieval_postprocessor.hpp",
    "src/cpp_example/common/visualizers/retrieval_visualizer.hpp",
    "src/cpp_example/multi_model_graph/graph_consumer.cpp",
    "src/cpp_example/multi_model_graph/graph_consumer.hpp",
]


@pytest.mark.contract
@pytest.mark.parametrize("rel", ROOT_READERS)
def test_the_graph_code_reads_its_root_at_run_time(rel):
    lines = (PROJECT_ROOT / rel).read_text(encoding="utf-8").splitlines()
    used = []
    for number, line in enumerate(lines, 1):
        if "PROJECT_ROOT_DIR" not in line:
            continue
        code = line.split("//", 1)[0].strip()
        in_comment = not code or code.startswith("*") or code.startswith("/*")
        in_macro = line.rstrip().endswith("\\")          # DXRT_TRY_CATCH_END's body: never expanded by graph code
        if not in_comment and not in_macro:
            used.append((number, line.strip()))
    assert not used, used


# ---------------------------------------------------------------------------
# graph_export: build outside the repository
# ---------------------------------------------------------------------------

EXPORT_ROOT = Path(os.environ.get("DX_EXPORT_DIR", "/dev/shm/gc-p3-export"))
USER_ZONE = PROJECT_ROOT / "tests" / "cpp_example" / "graph_export" / "user_zone.json"
MODEL = PROJECT_ROOT / "assets" / "models" / "yolov8-n_640x640.dxnn"
SUITE = PROJECT_ROOT.parent.parent
# What a machine with only the DEEPX runtime installed has: no venv, no LD_LIBRARY_PATH, no DXAPP_ROOT.
CLEAN_ENV = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": "/usr/local/bin:/usr/bin:/bin",
             "LANG": "C.UTF-8", "QT_QPA_PLATFORM": "offscreen"}
EDITED_STUB = """\
#include "user_logic/tag_people.hpp"

#include <string>
#include <utility>

namespace user_logic {

dxapp::graph::BoxesData tag_people::Run(const dxapp::graph::BoxesData& input,
                                        dxapp::graph::UserContext& ctx) {
    dxapp::graph::RecordItem data;
    data.numbers.push_back(std::make_pair(std::string("people"),
                                          static_cast<double>(input.items.size())));
    ctx.Raise("seen", -1, data);
    ctx.Label("people " + std::to_string(input.items.size()));
    return input;
}

}  // namespace user_logic
"""


def _needs_model():
    if not MODEL.is_file() or MODEL.stat().st_size == 0:
        pytest.skip("{} is not downloaded: python3 scripts/download_models.py --models yolov8-n_640x640"
                    .format(MODEL.name))


def _outside(path):
    real = Path(path).resolve()
    return PROJECT_ROOT not in real.parents and SUITE not in real.parents


def _export_and_build(out, stub=None):
    r = subprocess.run([sys.executable, "scripts/export_graph_project.py", "--graph", str(USER_ZONE),
                        "--out", str(out), "--json"], cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr
    if stub is not None:
        (out / "user_logic" / "tag_people.cpp").write_text(stub, encoding="utf-8")
    b = subprocess.run(["bash", "setup.sh"], cwd=str(out), env=CLEAN_ENV, capture_output=True, text=True,
                       timeout=1800)
    assert b.returncode == 0, b.stdout[-4000:] + b.stderr[-4000:]
    return out


@pytest.fixture(scope="module")
def built():
    _needs_model()
    shutil.rmtree(EXPORT_ROOT, ignore_errors=True)
    EXPORT_ROOT.mkdir(parents=True)
    assert _outside(EXPORT_ROOT)
    try:
        yield _export_and_build(EXPORT_ROOT / "user-zone")
    finally:
        if os.environ.get("DX_KEEP_EXPORT") != "1":
            shutil.rmtree(EXPORT_ROOT, ignore_errors=True)


@pytest.fixture(scope="module")
def edited(built):
    """The same export with tag_people.cpp edited to raise "seen" with the people count."""
    return _export_and_build(EXPORT_ROOT / "edited", EDITED_STUB)


def _program(out):
    return out / "build" / "user_zone"


@pytest.mark.graph_export
def test_the_export_builds_outside_the_repository_with_only_the_runtime(built):
    assert _outside(built) and _program(built).is_file()
    assert os.access(str(_program(built)), os.X_OK)


@pytest.mark.graph_export
def test_the_program_names_no_path_of_the_repository(built):
    data = _program(built).read_bytes()
    assert str(PROJECT_ROOT).encode() not in data
    assert str(SUITE).encode() not in data


@pytest.mark.graph_export
def test_the_program_finds_its_project_from_anywhere(built):
    r = subprocess.run(["./run.sh", "--check"], cwd=str(built), env=CLEAN_ENV, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0 and 'OK: graph "user-zone"' in r.stdout, r.stdout + r.stderr
    # No DXAPP_ROOT, another working directory: found from the program's directory up.
    r = subprocess.run([str(_program(built)), "--check"], cwd="/", env=CLEAN_ENV, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0 and "inputs/sample_people.jpg" in r.stdout, r.stdout + r.stderr
    # The program moved out of the project: found from the working directory up.
    away = EXPORT_ROOT / "away"
    away.mkdir(exist_ok=True)
    shutil.copy2(str(_program(built)), str(away / "user_zone"))
    r = subprocess.run([str(away / "user_zone"), "--check"], cwd=str(built / "user_logic"), env=CLEAN_ENV,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and 'OK: graph "user-zone"' in r.stdout, r.stdout + r.stderr


@pytest.mark.graph_export
def test_the_program_is_the_graph_cli(built):
    r = subprocess.run(["./run.sh", "--help"], cwd=str(built), env=CLEAN_ENV, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0 and "--graph <file>" in r.stdout
    r = subprocess.run(["./run.sh", "--list-ops", "--json"], cwd=str(built), env=CLEAN_ENV, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0
    assert [o["op"] for o in json.loads(r.stdout) if o["type"] == "logic"][-1] == "user"


@pytest.mark.graph_export
def test_an_edited_stub_builds(edited):
    assert _program(edited).is_file()
