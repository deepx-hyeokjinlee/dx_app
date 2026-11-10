"""Every variant's score_threshold reaches its Python postprocessor.

134 config.json files say score_threshold, which the C++ factories read; nine
Python postprocessors read only conf_threshold and fell back to their own
default (0.3), so the same model ran at different thresholds in the two
trees (yolov8-n: 0.25 in C++, 0.3 in Python). Built from the configs
themselves, so a new variant is covered the day it lands.
"""
import glob
import json
import os

import pytest

import common.processors as P

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "python_example")
ATTRS = ("conf_threshold", "score_threshold", "_score_threshold", "conf_thres")


def _variants():
    for path in sorted(glob.glob(os.path.join(ROOT, "*", "*", "*", "config.json"))):
        cfg = json.load(open(path))
        if "score_threshold" in (cfg.get("config") or {}):
            yield pytest.param(path, id=cfg["dxnn_file"])


@pytest.mark.parametrize("path", list(_variants()))
def test_score_threshold_reaches_the_postprocessor(path):
    cfg = json.load(open(path))
    spec = cfg["postprocessor"]
    tokens = {"input_width": int(cfg["input_width"]), "input_height": int(cfg["input_height"]),
              "config": dict(cfg["config"])}
    post = getattr(P, spec["class"])(*[tokens[a["token"]] for a in spec.get("args", [])],
                                     **(spec.get("kwargs") or {}))
    want = float(cfg["config"]["score_threshold"])
    have = [float(getattr(post, a)) for a in ATTRS if hasattr(post, a)]
    assert want in have, f"{spec['class']} holds {have}, config says {want}"
