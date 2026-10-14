#!/usr/bin/env python3
"""
Generate launch.json configurations for all examples

Usage:
    Generate launch.json file:
       python3 generate_launch_configs.py > launch.json
    
    Note: This will overwrite your existing launch.json file.
          Make sure to backup your launch.json if needed.
"""

import json

# Example configurations: (name, binary, model, video, image, rtsp_stream)
# Example configurations are DERIVED from config/model_registry.json rather than
# hard-coded. The hard-coded list had drifted badly: 252 of its 276 .dxnn references
# named files from an older naming era that are no longer on disk. Under the
# dx-modelzoo family layout the C++ executable basename IS the family, and the sample
# media comes from each variant's own config, so all of it is derivable.
#
# One representative variant per family keeps the config count manageable; every other
# variant is reachable from the same binary by swapping -m, because both entry points
# derive the variant from the model path.
import json as _json
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
_REG = _json.loads((_ROOT / "config" / "model_registry.json").read_text(encoding="utf-8"))


def _display(family):
    return family.replace("_", " ").title()


def _variant_media(entry):
    cfg_path = (_ROOT / "src" / "python_example" / entry["task"] / entry["family"]
                / entry["variant"] / "config.json")
    if not cfg_path.is_file():
        return None, None
    cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
    img = cfg.get("default_image") or "sample/img/sample_street.jpg"
    vid = cfg.get("default_video") or "assets/videos/snowboard.mp4"
    return img, _Path(vid).name


_seen = {}
for _e in sorted(_REG, key=lambda x: (x["task"], x["family"], x["variant"])):
    if _e["family"] in _seen or _e.get("alias_of") is not None:
        continue
    _img, _vid = _variant_media(_e)
    if _img is None:
        continue
    _seen[_e["family"]] = (
        _display(_e["family"]), _e["family"], _e["dxnn_file"], _vid, _img, "stream6")

# (name, binary, model, video, image, rtsp_stream) -- the shape create_config expects.
examples = list(_seen.values())

# Multi-model example configurations: (name, binary, model1, model2, video, image, rtsp)
# Empty: the yolov7_x_deeplabv3 entry that used to live here names a binary that exists
# in neither example tree (CMake's *_x_* multi-model glob matches nothing), and its two
# .dxnn names are from the pre-rename era. It generated 12 dead launch configs.
multi_model_examples = [
]

def create_config(name, binary, mode, input_type, model, video, image, rtsp_stream):
    """Create a single launch configuration"""
    
    # Determine arguments based on input type
    if input_type == "Image":
        args = ["-m", f"assets/models/{model}", "-i", image, "-l", "30"]
    elif input_type == "Image Dir":
        args = ["-m", f"assets/models/{model}", "-i", "sample/img", "-l", "30"]
    elif input_type == "Video":
        args = ["-m", f"assets/models/{model}", "-v", f"assets/videos/{video}"]
    elif input_type == "Video + Save":
        args = ["-m", f"assets/models/{model}", "-v", f"assets/videos/{video}", "-s"]
    elif input_type == "RTSP":
        args = ["-m", f"assets/models/{model}", "-r", f"rtsp://192.168.30.100:8554/{rtsp_stream}"]
    elif input_type == "Camera":
        args = ["-m", f"assets/models/{model}", "-c", "0"]
    else:
        args = []
    
    config = {
        "type": "cppdbg",
        "request": "launch",
        "name": f"Demo: {name} {mode} ({input_type})",
        "program": "${workspaceFolder}/bin/" + binary + "_" + mode.lower(),
        "args": args,
        "cwd": "${workspaceFolder}",
        "stopAtEntry": False,
        "environment": [],
        "externalConsole": False,
        "MIMode": "gdb",
        "setupCommands": [
            {
                "description": "Enable pretty-printing for gdb",
                "text": "-enable-pretty-printing",
                "ignoreFailures": True
            }
        ],
        "preLaunchTask": f"build: {binary}_{mode.lower()}"
    }
    
    return config

def create_multi_model_config(name, binary, mode, input_type, model1, model2, video, image, rtsp_stream):
    """Create a single launch configuration for multi-model examples"""
    
    # Determine arguments based on input type
    if input_type == "Image":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-i", image, "-l", "30"]
    elif input_type == "Image Dir":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-i", "sample/img", "-l", "30"]
    elif input_type == "Video":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-v", f"assets/videos/{video}"]
    elif input_type == "Video + Save":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-v", f"assets/videos/{video}", "-s"]
    elif input_type == "RTSP":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-r", f"rtsp://192.168.30.100:8554/{rtsp_stream}"]
    elif input_type == "Camera":
        args = ["-y", f"assets/models/{model1}", "-d", f"assets/models/{model2}", "-c", "0"]
    else:
        args = []
    
    config = {
        "type": "cppdbg",
        "request": "launch",
        "name": f"Demo: {name} {mode} ({input_type})",
        "program": "${workspaceFolder}/bin/" + binary + "_" + mode.lower(),
        "args": args,
        "cwd": "${workspaceFolder}",
        "stopAtEntry": False,
        "environment": [],
        "externalConsole": False,
        "MIMode": "gdb",
        "setupCommands": [
            {
                "description": "Enable pretty-printing for gdb",
                "text": "-enable-pretty-printing",
                "ignoreFailures": True
            }
        ],
        "preLaunchTask": f"build: {binary}_{mode.lower()}"
    }
    
    return config

# Generate all configurations
all_configs = []

for name, binary, model, video, image, rtsp_stream in examples:
    for mode in ["Sync", "Async"]:
        for input_type in ["Image", "Image Dir", "Video", "Video + Save", "RTSP", "Camera"]:
            config = create_config(name, binary, mode, input_type, model, video, image, rtsp_stream)
            all_configs.append(config)

# Generate multi-model configurations
for name, binary, model1, model2, video, image, rtsp_stream in multi_model_examples:
    for mode in ["Sync", "Async"]:
        for input_type in ["Image", "Image Dir", "Video", "Video + Save", "RTSP", "Camera"]:
            config = create_multi_model_config(name, binary, mode, input_type, model1, model2, video, image, rtsp_stream)
            all_configs.append(config)

# Output JSON in launch.json format (pretty-printed)
launch_json = {
    "version": "0.2.0",
    "configurations": all_configs
}
print(json.dumps(launch_json, indent=4))
