#!/usr/bin/env python3
"""Classifiers preprocess the way their compile JSON says.

The model zoo publishes, next to each .dxnn, the DX-COM compile JSON it was
built with (https://sdk.deepx.ai/modelzoo/q-lite-json/2_5_0/<stem>.json). Its
`default_loader.preprocessings` is the model's evaluation transform -- for
most classifiers torchvision's Resize(shorter side S) + CenterCrop. The
examples stretched every classifier to its input instead. On ImageNetV2
(10 000 labelled images) the compile JSON's transform scores 2.3-4.7 top-1
above the stretch (resnet18 52.18 -> 56.91, mobilenetv2 55.01 -> 59.53).

  config/modelzoo_compile_preprocessing.json
      the snapshot: each .dxnn's default_loader.preprocessings, as published.
      Rebuilt with --snapshot DIR (a directory of the downloaded JSONs).

  --apply   writes, for every image_classification variant whose JSON resizes
            and centre-crops:
              python_example/.../config.json  preprocessor.kwargs.crop_resize
              cpp_example/.../config.json     the same
              cpp_example/.../factory/*.hpp   setCenterCrop(...) in
                                              createPreprocessor()
  --check   exits 1 when any of those disagrees with the snapshot (default).

Mapping (resize step -> crop_resize):
  {mode: torchvision, size: S, interpolation: I}  {"short": S, "interpolation": i,
                                                    "resampler": "pillow"}
  {mode: torchvision, size: [H, W]}               {"width": W, "height": H,
                                                    "resampler": "pillow"}
  {mode: pycls, size: S}                          {"short": S, "resampler": "opencv"}
  {width: W, height: H} (W, H != the crop)        {"width": W, "height": H,
                                                    "resampler": "opencv"}
and the crop must equal the model input (it is checked, not assumed).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SNAPSHOT = os.path.join(ROOT, "config", "modelzoo_compile_preprocessing.json")
FACTORY_LINE = "return std::make_unique<SimpleResizePreprocessor>(input_width, input_height);"


def build_snapshot(json_dir: str) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        data = json.load(open(path))
        out[os.path.basename(path)[:-5] + ".dxnn"] = (data.get("default_loader") or {}).get("preprocessings") or []
    return out


def crop_resize_for(steps: list, input_w: int, input_h: int):
    """(crop_resize dict, description) or (None, reason)."""
    resize = next((s["resize"] for s in steps if "resize" in s), None)
    crop = next((s["centercrop"] for s in steps if "centercrop" in s), None)
    if resize is None or crop is None:
        return None, "no resize + centercrop"
    if (crop["width"], crop["height"]) != (input_w, input_h):
        return None, f"crop {crop['width']}x{crop['height']} is not the input {input_w}x{input_h}"
    mode = resize.get("mode")
    interp = str(resize.get("interpolation", "BILINEAR")).upper()
    if interp not in ("BILINEAR", "BICUBIC", "LINEAR"):
        return None, f"interpolation {interp} not supported"
    interp = "bicubic" if interp == "BICUBIC" else "bilinear"
    if mode == "torchvision" and isinstance(resize["size"], list):
        # torchvision Resize((h, w)): exactly that size, the aspect not kept
        h, w = (int(v) for v in resize["size"])
        return ({"width": w, "height": h, "interpolation": interp, "resampler": "pillow"},
                f"torchvision Resize(({h}, {w}), {interp}) + CenterCrop({input_w})")
    if mode == "torchvision":
        return ({"short": int(resize["size"]), "interpolation": interp, "resampler": "pillow"},
                f"torchvision Resize({resize['size']}, {interp}) + CenterCrop({input_w})")
    if mode == "pycls":
        return ({"short": int(resize["size"]), "interpolation": interp, "resampler": "opencv"},
                f"pycls scale({resize['size']}) + CenterCrop({input_w})")
    if mode is None and "width" in resize and "height" in resize:
        if (resize["width"], resize["height"]) == (input_w, input_h):
            return None, "resize already equals the crop: a plain stretch"
        return ({"width": int(resize["width"]), "height": int(resize["height"]),
                 "interpolation": interp, "resampler": "opencv"},
                f"resize {resize['width']}x{resize['height']} + CenterCrop({input_w})")
    return None, f"resize mode {mode!r} not handled"


def factory_call(cr: dict) -> str:
    interp = "cv::INTER_CUBIC" if cr["interpolation"] == "bicubic" else "cv::INTER_LINEAR"
    return (f"pre->setCenterCrop({cr.get('short', 0)}, {cr.get('width', 0)}, {cr.get('height', 0)}, "
            f"{interp}, /*pillow=*/{'true' if cr['resampler'] == 'pillow' else 'false'});")


def variants():
    for py in sorted(glob.glob(os.path.join(ROOT, "src/python_example/image_classification/*/*/config.json"))):
        rel = os.path.relpath(os.path.dirname(py), os.path.join(ROOT, "src/python_example"))
        cpp_dir = os.path.join(ROOT, "src/cpp_example", rel)
        yield py, os.path.join(cpp_dir, "config.json"), glob.glob(os.path.join(cpp_dir, "factory", "*.hpp"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", metavar="JSON_DIR", help="rebuild the snapshot from downloaded compile JSONs")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if args.snapshot:
        snap = build_snapshot(args.snapshot)
        with open(SNAPSHOT, "w") as f:
            json.dump(snap, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"snapshot: {len(snap)} compile JSONs -> {os.path.relpath(SNAPSHOT, ROOT)}")
    snap = json.load(open(SNAPSHOT))

    wrong, applied, skipped, no_json = [], 0, {}, []
    for py_cfg, cpp_cfg, factories in variants():
        cfg = json.load(open(py_cfg))
        dxnn = cfg["dxnn_file"]
        if dxnn not in snap:
            no_json.append(dxnn)
            continue
        cr, why = crop_resize_for(snap[dxnn], int(cfg["input_width"]), int(cfg["input_height"]))
        if cr is None:
            skipped[dxnn] = why
            continue
        for path in (py_cfg, cpp_cfg):
            c = json.load(open(path))
            kw = c.setdefault("preprocessor", {}).setdefault("kwargs", {})
            if kw.get("crop_resize") != cr:
                if args.apply:
                    kw["crop_resize"] = cr
                    with open(path, "w") as f:
                        json.dump(c, f, indent=2, ensure_ascii=False)
                        f.write("\n")
                else:
                    wrong.append(f"{os.path.relpath(path, ROOT)}: crop_resize {kw.get('crop_resize')} != {cr}")
        if len(factories) != 1:
            wrong.append(f"{dxnn}: expected one C++ factory, found {len(factories)}")
            continue
        src = open(factories[0]).read()
        call = factory_call(cr)
        if call not in src:
            if args.apply:
                block = re.search(r"\n([ \t]*)(?://[^\n]*\n[ \t]*)?" + re.escape(FACTORY_LINE) +
                                  r"|\n([ \t]*)auto pre = std::make_unique<SimpleResizePreprocessor>\(input_width, input_height\);\n"
                                  r"[ \t]*pre->setCenterCrop\([^\n]*\n[ \t]*return pre;", src)
                if block is None:
                    wrong.append(f"{os.path.relpath(factories[0], ROOT)}: createPreprocessor not in the expected form")
                    continue
                ind = block.group(1) or block.group(2)
                new = (f"\n{ind}// The compile JSON's evaluation transform (q-lite-json/2_5_0/{dxnn[:-5]}.json):\n"
                       f"{ind}// {crop_resize_for(snap[dxnn], int(cfg['input_width']), int(cfg['input_height']))[1]}.\n"
                       f"{ind}auto pre = std::make_unique<SimpleResizePreprocessor>(input_width, input_height);\n"
                       f"{ind}{call}\n{ind}return pre;")
                src = src[:block.start()] + new + src[block.end():]
                with open(factories[0], "w") as f:
                    f.write(src)
            else:
                wrong.append(f"{os.path.relpath(factories[0], ROOT)}: missing `{call}`")
                continue
        applied += 1

    print(f"classifiers following their compile JSON's resize + crop: {applied}")
    print(f"no compile JSON published: {len(no_json)}; JSON without resize+crop: {len(skipped)}")
    for dxnn, why in sorted(skipped.items()):
        print(f"  skip {dxnn}: {why}")
    if wrong:
        print(f"FAIL -- {len(wrong)} disagreement(s); run with --apply:")
        for w in wrong:
            print(f"  {w}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
