# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""A C++ factory's constructor defaults are what its example runs with when
--config is not given; config.json is what the Python example always runs with.
They must agree, or the two languages report different detections for one
model (survey item F4: 191 factories disagreed, mostly score/nms thresholds,
fixed 2026-11-09). Every numeric config key that is also a constructor
parameter must have the same value."""
from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CPP = PROJECT_ROOT / "src" / "cpp_example"
_SIG = re.compile(r"\n    \w+Factory\((.*?)\)\s*(?::|\{)", re.S)
_PARAM = re.compile(r"(?:float|int|double)\s+(\w+)\s*=\s*([-\d.eE]+)f?")


def test_factory_constructor_defaults_match_config_json():
    bad = []
    for cfg_path in sorted(CPP.glob("*/*/*/config.json")):
        factory = sorted((cfg_path.parent / "factory").glob("*_factory.hpp"))
        if not factory:
            continue
        sig = _SIG.search(factory[0].read_text(encoding="utf-8"))
        if not sig:
            continue
        defaults = {k: float(v) for k, v in _PARAM.findall(sig.group(1))}
        config = json.loads(cfg_path.read_text(encoding="utf-8")).get("config") or {}
        for key, value in config.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if key in defaults and abs(defaults[key] - float(value)) > 1e-6:
                bad.append(f"{cfg_path.parent.name}: {key} default {defaults[key]} != config {value}")
    assert not bad, bad
