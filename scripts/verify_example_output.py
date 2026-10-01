#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Does a PASS actually draw anything?

scripts/sweep_npu_inference.py answers "did it run": it requires the runner's inference
line, so a script exiting 0 without touching the NPU is NO_INFERENCE rather than PASS.
It does NOT answer "was there a result" -- a postprocessor returning an empty list on
every frame passes that bar cleanly, and so does a visualizer that draws nothing.

This closes that gap the cheapest honest way: run one model per task with --save and
compare the rendered output against the input image. Identical pixels mean nothing was
drawn. A different size means the app produced a new image entirely (a depth map, a
super-resolved frame, a side-by-side), which also counts as having produced something.

One representative per task, chosen as the alphabetically first PASS in that task, so
the sample is not cherry-picked.

The three embedding tasks (face_recognition, the eigenplaces models dx-modelzoo files
under super_resolution, and zero_shot_image_classification) are reported SKIP here:
their input is a directory of image pairs and their output is a vector, not a picture.
Verify those with the embedding check in the same session -- a finite, unit-norm,
image-dependent vector -- rather than by looking for pixels.

Usage:
    python3 scripts/verify_example_output.py
"""
import subprocess, sys, json, tempfile, shutil
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]   # was a developer's home path
PY_BIN = str(ROOT.parent / "venv-dx-runtime" / "bin" / "python")


def main():
    reg = {e["variant"]: e for e in json.loads((ROOT / "config/model_registry.json").read_text())}
    reports = sorted((ROOT / "artifacts/npu_sweep").glob("sweep-python_example-sync-*.json"))
    if not reports:
        sys.exit("no artifacts/npu_sweep/sweep-python_example-sync-*.json -- run the NPU sweep first")
    report = reports[-1]
    results = {r["variant"]: r for r in json.loads(report.read_text())["results"]}

    # One representative per task: the first PASS in each, alphabetically -- no cherry-picking.
    per_task = {}
    for variant in sorted(results):
        if results[variant]["status"] != "PASS":
            continue
        entry = reg.get(variant)
        if entry is None:
            # The report predates a rename (efficientnet-lite0_256x256 -> _224x224), so
            # the registry no longer knows this variant. A stale row is not a failure.
            continue
        per_task.setdefault(entry["task"], variant)

    rows, drew, blank, failed = [], 0, 0, 0
    for task, variant in sorted(per_task.items()):
        r = results[variant]
        entry = ROOT / r["entry"]
        src_image = ROOT / r["input"] if r.get("input") else None
        if src_image is None or not src_image.is_file():
            rows.append((task, variant, "SKIP", "input is not a single image")); continue
        out = Path(tempfile.mkdtemp(prefix="dx_render_"))
        try:
            p = subprocess.run(
                [PY_BIN, str(entry), "-m", str(ROOT / "assets/models" / reg[variant]["dxnn_file"]),
                 "--image", str(src_image), "--no-display", "--save", "--save-dir", str(out)],
                cwd=entry.parent, capture_output=True, text=True, timeout=300)
            images = sorted(out.rglob("*.jpg")) + sorted(out.rglob("*.png"))
            if p.returncode != 0 or not images:
                failed += 1
                rows.append((task, variant, "NO-OUTPUT", (p.stderr or p.stdout).strip().splitlines()[-1][:70] if (p.stderr or p.stdout).strip() else f"rc={p.returncode}"))
                continue
            rendered = cv2.imread(str(images[0]))
            original = cv2.imread(str(src_image))
            if rendered is None:
                failed += 1; rows.append((task, variant, "UNREADABLE", images[0].name)); continue
            if original is not None and rendered.shape == original.shape:
                diff = cv2.absdiff(rendered, original)
                changed = int((diff.max(axis=2) > 8).sum())
                pct = 100.0 * changed / (rendered.shape[0] * rendered.shape[1])
                if changed == 0:
                    blank += 1; rows.append((task, variant, "UNCHANGED", "identical to the input"))
                else:
                    drew += 1; rows.append((task, variant, "DREW", f"{pct:.1f}% of pixels changed"))
            else:
                # A different size means the app produced a new image (depth map, SR, matte).
                drew += 1
                rows.append((task, variant, "DREW", f"output {rendered.shape[1]}x{rendered.shape[0]} vs input {original.shape[1]}x{original.shape[0]}" if original is not None else "new image"))
        except subprocess.TimeoutExpired:
            failed += 1; rows.append((task, variant, "TIMEOUT", ""))
        finally:
            shutil.rmtree(out, ignore_errors=True)

    print(f"{'task':<32} {'variant':<42} {'verdict':<11} detail")
    for task, variant, verdict, detail in rows:
        print(f"{task:<32} {variant:<42} {verdict:<11} {detail}")
    print(f"\nDREW {drew}   UNCHANGED {blank}   NO-OUTPUT/failed {failed}   of {len(rows)} tasks")
    sys.exit(1 if (blank or failed) else 0)


if __name__ == "__main__":
    main()
