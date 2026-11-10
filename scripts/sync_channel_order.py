#!/usr/bin/env python3
"""The C++ examples feed the channel order the Python config says.

`preprocessor.kwargs.bgr: true` in python_example/<variant>/config.json is the
source: the model was trained on cv2's BGR (YOLOX, NanoDet, RetinaFace, ...)
and the frame goes in without the BGR->RGB swap. Measured on labelled data
(COCO val2017, ImageNetV2, COCO person keypoints as a face proxy), feeding
those models RGB cost 3.1-4.7 mAP, 13.8 top-1 and 13.8 face AP.

For every variant this keeps three places in agreement:
  python_example/.../config.json   preprocessor.kwargs.bgr
  cpp_example/.../config.json      the same
  cpp_example/.../factory/*.hpp    createPreprocessor() passes colour code -1
                                   (DetectionPreprocessor / SimpleResizePreprocessor)

  sync_channel_order.py                    # --check (default): exit 1 on disagreement
  sync_channel_order.py --apply            # make the C++ side follow Python
  sync_channel_order.py --set-bgr V [V..]  # set bgr: true on these variants, then apply
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = os.path.join(ROOT, "src", "python_example")
CPP = os.path.join(ROOT, "src", "cpp_example")
CTOR = re.compile(r"std::make_unique<(DetectionPreprocessor|SimpleResizePreprocessor)>\(\s*"
                  r"input_width,\s*input_height(\s*,\s*(?:/\*color_conversion=\*/)?\s*-1)?\s*\)")


def load(path):
    with open(path) as f:
        return json.load(f)


def dump(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def set_kw(path, bgr: bool) -> bool:
    if not os.path.exists(path):
        return False
    c = load(path)
    kw = c.setdefault("preprocessor", {}).setdefault("kwargs", {})
    if bool(kw.get("bgr", False)) == bgr:
        return False
    if bgr:
        kw["bgr"] = True
    else:
        kw.pop("bgr", None)
    dump(path, c)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--set-bgr", nargs="+", metavar="VARIANT", default=[])
    args = ap.parse_args()

    for v in args.set_bgr:
        path = os.path.join(PY, v, "config.json")
        if not os.path.exists(path):
            print(f"ERROR: no such variant {v}", file=sys.stderr)
            return 2
        set_kw(path, True)
    apply = args.apply or bool(args.set_bgr)

    wrong, bgr_variants = [], 0
    for py_cfg in sorted(glob.glob(os.path.join(PY, "*", "*", "*", "config.json"))):
        rel = os.path.relpath(os.path.dirname(py_cfg), PY)
        c = load(py_cfg)
        pre = c.get("preprocessor") or {}
        bgr = bool((pre.get("kwargs") or {}).get("bgr", False))
        bgr_variants += bgr
        cpp_dir = os.path.join(CPP, rel)
        cpp_cfg = os.path.join(cpp_dir, "config.json")
        if os.path.exists(cpp_cfg):
            have = bool(((load(cpp_cfg).get("preprocessor") or {}).get("kwargs") or {}).get("bgr", False))
            if have != bgr:
                if apply:
                    set_kw(cpp_cfg, bgr)
                else:
                    wrong.append(f"{os.path.relpath(cpp_cfg, ROOT)}: bgr {have} != python {bgr}")
        for fac in glob.glob(os.path.join(cpp_dir, "factory", "*.hpp")):
            src = open(fac).read()
            m = CTOR.search(src)
            if m is None:
                continue  # a preprocessor with its own colour handling (mean/float, crops, ...)
            has = m.group(2) is not None
            if has == bgr:
                continue
            if not apply:
                wrong.append(f"{os.path.relpath(fac, ROOT)}: feeds {'BGR' if has else 'RGB'}, "
                             f"python config says {'BGR' if bgr else 'RGB'}")
                continue
            new = (f"std::make_unique<{m.group(1)}>(input_width, input_height, /*color_conversion=*/-1)"
                   if bgr else f"std::make_unique<{m.group(1)}>(input_width, input_height)")
            src = src[:m.start()] + new + src[m.end():]
            with open(fac, "w") as f:
                f.write(src)

    print(f"variants fed BGR (preprocessor.kwargs.bgr): {bgr_variants}")
    if wrong:
        print(f"FAIL -- {len(wrong)} disagreement(s); run with --apply:")
        for w in wrong:
            print(f"  {w}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
