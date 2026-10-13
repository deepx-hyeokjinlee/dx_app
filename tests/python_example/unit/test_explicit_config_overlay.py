"""config.json is read only when given with --config, as the C++ runners do.

Every example ships a per-variant config.json whose settings sit in a nested
``"config"`` object (``{"config": {"score_threshold": 0.25, ...}}``). The C++
ModelConfig overlays that object onto the top level, so ``--config`` changes
thresholds and ``sr_tile_halo``. The Python runners passed the raw JSON on and
only read top-level keys, so the same file changed nothing: a halo of 1 ran at
4, a score threshold of 0.99 still drew every box.

Without --config the Python runners also searched for a config.json next to
the model or the script and loaded it. The C++ runners never do: a run without
--config uses the defaults the factory was built with.
"""
import argparse
import json

from common.config import overlay_nested_config
from common.runner.sr_tiling import resolve_halo_with_source
from common.runner.sync_runner import _load_run_config, _resolve_config_path


def test_the_nested_config_overlays_the_top_level():
    data = {"task": "super_resolution", "score_threshold": 0.1,
            "config": {"score_threshold": 0.9, "sr_tile_halo": 1}}
    merged = overlay_nested_config(data)
    assert merged["score_threshold"] == 0.9   # the nested value wins, as in C++
    assert merged["sr_tile_halo"] == 1
    assert merged["task"] == "super_resolution"
    assert data["score_threshold"] == 0.1      # the input is not modified


def test_a_config_that_is_not_an_object_is_left_alone():
    data = {"config": "not-an-object", "score_threshold": 0.1}
    assert overlay_nested_config(data) == data


def _variant_config(tmp_path, nested):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"variant": "espcn-x4_17x17", "config": nested}))
    return str(path)


def test_a_given_config_applies_its_nested_settings(tmp_path):
    config = _load_run_config(_variant_config(tmp_path, {"sr_tile_halo": 1}))
    assert resolve_halo_with_source(config=config, tile_h=17, tile_w=17) == (1, "config.json")


def test_without_config_no_config_json_is_searched(tmp_path):
    model = tmp_path / "espcn-x4_17x17.dxnn"
    model.write_bytes(b"")
    _variant_config(tmp_path, {"sr_tile_halo": 1})  # right next to the model
    args = argparse.Namespace(config=None, model=str(model))
    assert _resolve_config_path(args) is None


def test_no_config_path_gives_no_config():
    assert _load_run_config(None) == {}
