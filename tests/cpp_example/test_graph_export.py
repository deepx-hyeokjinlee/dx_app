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
