"""MODEL_IMAGE_OVERRIDE is keyed by the names the harness looks it up with.

resolve_image_for_model() is called with an example's variant name (the
binary or script stem, e.g. "espcn-x4_17x17"). The keys were still the
pre-restructure model names ("espcn_x4"), so no override ever applied and
every model fell back to its task's default image: Real-ESRGAN got the 275x150
ESPCN crop instead of its 165x90 one, the pose models no people image, ...
"""
import json

from test_helpers.constants import MODEL_IMAGE_OVERRIDE, PROJECT_ROOT
from test_helpers.utils import resolve_image_for_model

REGISTRY = PROJECT_ROOT / "config" / "model_registry.json"


def _variants():
    return {entry["variant"] for entry in json.loads(REGISTRY.read_text())
            if entry.get("variant")}


def test_every_key_is_a_registered_variant():
    unknown = sorted(set(MODEL_IMAGE_OVERRIDE) - _variants())
    assert not unknown, "not a variant name in model_registry.json: {}".format(unknown)


def test_every_override_image_exists():
    missing = sorted(img for img in MODEL_IMAGE_OVERRIDE.values()
                     if not (PROJECT_ROOT / img).is_file())
    assert not missing


def test_super_resolution_models_get_their_own_crop():
    assert resolve_image_for_model("espcn-x4_17x17", "super_resolution") \
        == "sample/img/sample_lowres275x150.png"
    assert resolve_image_for_model("realesrgan-x8_192x192", "super_resolution") \
        == "sample/img/sample_lowres165x90.png"
