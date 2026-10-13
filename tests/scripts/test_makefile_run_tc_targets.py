"""Makefile run_tc / run_tc_sonar: check the DX-RT service first, and let
run_tc_sonar write both coverage reports even when tests fail.

Without a reachable dxrtd every NPU test fails with "dxrt service is not
running", and run_tc_sonar only found out after an hour of runs. Both targets
now stop at once when `dxrt-cli -s` fails.

run_tc_sonar used `set -e`, so one failing test skipped gcovr and the Python
run and SonarQube got no coverage at all. Every step now runs; the target
still fails at the end if any step did.

The Makefile runs from a copy in a temporary tree, with recording stubs for
run_tc.sh, setup.sh, dxrt-cli, gcovr, coverage and pip. The Makefile is not
part of the public release (.github/release-excluded), so the tests skip
without it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = ROOT / "Makefile"

pytestmark = [
    pytest.mark.skipif(not MAKEFILE.exists(), reason="Makefile is internal-only"),
    pytest.mark.skipif(shutil.which("make") is None, reason="make not installed"),
]

# Appends "<name> <args>" to calls.log, then exits with the code in <name>.rc
# (default 0). run_tc.sh --python also writes the .coverage file pytest-cov
# leaves behind, unless no_python_coverage exists.
STUB = """#!/bin/bash
here="$(cd "$(dirname "$0")" && pwd)"
app="{app}"
echo "{name} $*" >> "$app/calls.log"
if [ "{name}" = "run_tc.sh" ] && [ "$1" = "--python" ] && [ ! -e "$app/no_python_coverage" ]; then
    mkdir -p "$app/tests/python_example" && : > "$app/tests/python_example/.coverage"
fi
case "{name} $1" in
    "run_tc.sh --cpp") rc_file="$app/cpp.rc" ;;
    "run_tc.sh --python") rc_file="$app/python.rc" ;;
    *) rc_file="$app/{name}.rc" ;;
esac
if [ -f "$rc_file" ]; then
    [ "{name}" = "dxrt-cli" ] && echo "[dxrt-exception] Service input & output exception {{dxrt service is not running}}"
    exit "$(cat "$rc_file")"
fi
exit 0
"""


def _stub(path: Path, app: Path) -> None:
    path.write_text(STUB.format(app=app, name=path.name))
    path.chmod(0o755)


@pytest.fixture
def app(tmp_path: Path) -> Path:
    app = tmp_path / "dx_app"
    app.mkdir()
    shutil.copy2(MAKEFILE, app / "Makefile")
    shutil.copy2(ROOT / "release.ver", app / "release.ver")
    for name in ("run_tc.sh", "setup.sh"):
        _stub(app / name, app)
    tools = app / "stub_bin"
    tools.mkdir()
    for name in ("dxrt-cli", "gcovr", "coverage", "pip"):
        _stub(tools / name, app)
    return app


def _make(app: Path, target: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PATH="{}:{}".format(app / "stub_bin", os.environ["PATH"]))
    return subprocess.run(["make", "--no-print-directory", target], cwd=app, env=env,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False, timeout=60)


def _calls(app: Path) -> list[str]:
    log = app / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


@pytest.mark.parametrize("target", ["run_tc", "run_tc_sonar"])
def test_an_unreachable_dxrt_service_stops_the_target_before_any_test(app, target):
    (app / "dxrt-cli.rc").write_text("1")
    result = _make(app, target)
    assert result.returncode != 0, result.stdout
    assert _calls(app) == ["dxrt-cli -s"], result.stdout
    assert "dxrt service is not running" in result.stdout
    assert "--pid=host" in result.stdout  # says how to fix a container


def test_sonar_writes_both_reports_when_the_cpp_tests_fail(app):
    (app / "cpp.rc").write_text("1")
    result = _make(app, "run_tc_sonar")
    calls = _calls(app)
    assert result.returncode != 0, result.stdout
    assert [c.split()[0] for c in calls] == [
        "dxrt-cli", "run_tc.sh", "gcovr", "run_tc.sh", "coverage"], calls
    assert (app / ".coverage").exists()
    assert "cpp-tests" in result.stdout


def test_sonar_fails_when_python_coverage_was_not_written(app):
    (app / "no_python_coverage").write_text("")
    result = _make(app, "run_tc_sonar")
    assert result.returncode != 0, result.stdout
    assert "coverage" not in [c.split()[0] for c in _calls(app)]
    assert "python-report" in result.stdout


def test_sonar_passes_when_every_step_passes(app):
    result = _make(app, "run_tc_sonar")
    calls = _calls(app)
    assert result.returncode == 0, result.stdout
    assert calls[0] == "dxrt-cli -s"
    assert "run_tc.sh --cpp --coverage" in calls
    assert "run_tc.sh --python --coverage" in calls
    assert calls[-1] == "coverage xml -o coverage.xml"
