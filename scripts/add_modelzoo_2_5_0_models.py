#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Emit the 147 DX Model Zoo 2_5_0 additions into every file that describes them.

Outputs, all generated and none of them a source:

    config/model_registry.json          +147 entries, +'published' on all of them
    tests/data/processor_specs.json      +147 entries
    tests/data/modelzoo_cv_tree.json    'provisional' section (147)
    scripts/modelzoo_manifest.json      +147 rows

The single source is scripts/data/modelzoo_2_5_0.py. Change a family name there and
re-run this; never hand-edit the four files above.

Idempotent by construction: every row this script owns is REMOVED and re-inserted
rather than patched in place. That matters because the predecessor was bitten for real
-- renaming a registry field to 'alias_of' while the emitter only stripped the NEW
field names left 'duplicate_of' alive in all 353 entries, so two spellings coexisted.
Deleting first makes a rename impossible to half-apply.

Existing entries keep their exact position: the registry's 353 rows are NOT sorted (nor
sorted case-insensitively), so re-ordering them would produce a huge, meaningless diff
and bury the real change. New rows are appended, sorted by variant.

Usage:
    python3 scripts/add_modelzoo_2_5_0_models.py                 # write
    python3 scripts/add_modelzoo_2_5_0_models.py --snapshot-only # provisional section only
    python3 scripts/add_modelzoo_2_5_0_models.py --check         # exit 1 if a write would change anything
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data.modelzoo_2_5_0 import expand, self_check  # noqa: E402

REGISTRY = ROOT / "config" / "model_registry.json"
SPECS = ROOT / "tests" / "data" / "processor_specs.json"
SNAPSHOT = ROOT / "tests" / "data" / "modelzoo_cv_tree.json"
MANIFEST = ROOT / "scripts" / "modelzoo_manifest.json"

# The legacy short code the add_model tooling keys off, keyed by the LEGACY task name
# so it matches what the existing 353 entries carry.
CSV_TASK = {
    "object_detection": "OD",
    "instance_segmentation": "ISEG",
    "pose_estimation": "POSE",
    "classification": "IC",
    "reid": "REID",
    "semantic_segmentation": "SEG",
    # Every one of the 10 existing embedding entries -- clip, arcface, eigenplaces --
    # carries FREC, so a new one does too rather than inventing a code.
    "embedding": "FREC",
    "super_resolution": "SR",
    "anomaly_detection": "AD",          # no precedent; this task is new
}

# Registry key order, exactly as the existing entries spell it, plus 'published'.
REGISTRY_KEYS = (
    "model_name", "dxnn_file", "original_name", "csv_task", "add_model_task",
    "postprocessor", "input_width", "input_height", "config", "source", "supported",
    "variant", "family", "task", "task_legacy", "image_only", "zoo_canonical",
    "alias_of", "published",
)

# Tasks the legacy runner treats as image-only; a variant in one of them gets
# image_only=True without a per-variant flag, which is how the existing 17 work.
IMAGE_ONLY_LEGACY_TASKS = {
    "embedding", "reid", "attribute_recognition", "object_pose_estimation",
    "3d_object_detection",
}


def registry_entry(row: dict) -> dict:
    entry = {
        "model_name": row["legacy_model_name"],
        "dxnn_file": f"{row['variant']}.dxnn",
        "original_name": row["display_name"],
        "csv_task": CSV_TASK[row["task_legacy"]],
        # Mirrors task_legacy, as it does in all 353 existing entries.
        "add_model_task": row["task_legacy"],
        "postprocessor": row["postprocessor"]["class"],
        "input_width": row["width"],
        "input_height": row["height"],
        "config": row["config"],
        "source": row["reference"],
        "supported": True,
        "variant": row["variant"],
        "family": row["family"],
        "task": row["task"],
        "task_legacy": row["task_legacy"],
        "image_only": row["task_legacy"] in IMAGE_ONLY_LEGACY_TASKS,
        # False because it is what the flag means: resolved from the upstream
        # dx-modelzoo snapshot. These 147 are not in it (they are in the
        # provisional table), and they become canonical when upstream publishes
        # them and harvest_modelzoo_tree.py moves them across.
        "zoo_canonical": False,
        "alias_of": None,
        "published": row["published"],
    }
    assert tuple(entry) == REGISTRY_KEYS, tuple(entry)
    return entry


def spec_entry(row: dict) -> dict:
    """The processor-spec entry for one new variant.

    'factory_class' is informational for a new family -- the generator derives the
    class name from the family itself -- but 'factory_bases' and the matching import
    line are load-bearing: family_base() picks them off the first member it finds.
    """
    family = row["family"]
    pascal = "".join(p[:1].upper() + p[1:] for p in family.split("_") if p)
    if not pascal[:1].isalpha():
        pascal = "N" + pascal
    return {
        "cli": {
            "include_kitti_paths": False,
            "include_output": False,
            # Image-only variants drop the stream flags, matching the 12 existing ones.
            "include_stream_inputs": row["task_legacy"] not in IMAGE_ONLY_LEGACY_TASKS,
        },
        "dir_config": row["config"],
        # IPoseFactory declares get_num_keypoints abstract, so a pose family that
        # does not carry it cannot be instantiated at all -- which is how all six
        # pose variants failed the first full sweep. The generator emits whatever is
        # here verbatim, exactly as it does for the pre-existing pose families.
        "extra_methods": row["extra_methods"],
        "factory_bases": row["factory_bases"],
        "factory_class": f"{pascal}Factory",
        "family": family,
        "imports": row["imports"],
        "model_name": row["legacy_model_name"],
        "postprocessor": row["postprocessor"],
        "preprocessor": row["preprocessor"],
        "registry_config": row["config"],
        # legacy_media() matches _MODEL_SAMPLE_IMAGE_OVERRIDE against this as a
        # lowercase-alphanumeric prefix, so it must be the family, not the variant.
        "source_dir": family,
        "source_task": row["task"],
        "task": row["task"],
        "variant": row["variant"],
        "visualizer": row["visualizer"],
    }


def manifest_entry(row: dict) -> dict:
    """Download URLs.

    DX Model Zoo published 2_5_0 on 2026-09-30 and moved its whole q-lite tier there,
    so q-lite/2_5_0 is now the ordinary case: 492 of the 496 files the page offers live
    there, and the remaining 4 keep their measured q-master/2_4_0 URLs.

    The tier comes from the row, not from `published`. A q-lite URL can still be
    unpublished when its variant is in UNPUBLISHED; that set is empty, so
    vit-l-p16_512x512_swag is published like the other q-lite rows.
    """
    stem = row["variant"]
    base, version = row["manifest_tier"]
    return {
        # 'name' is the legacy model_name in all 352 existing rows -- it is what
        # `download_models.py --model <name>` whitelists against -- while the .dxnn
        # filename comes from the URL basename. Keeping the convention means --model
        # works the same way across all 499.
        "name": row["legacy_model_name"],
        "category": row["manifest_category"],
        "dxnn_url": f"https://sdk.deepx.ai/modelzoo/{base}-dxnn/{version}/{stem}.dxnn",
        # Zoo sidecar JSON is not used by the examples. Keep the key so every
        # row has the same shape, but leave it empty so download_models.py
        # does not fetch it.
        "json_url": None,
        # Declared, so download_models.py can tell an expected 403 from a real one.
        **({"pending": True} if not row["published"] else {}),
    }


def _manifest_stem(entry: dict) -> str | None:
    """The .dxnn stem a manifest row points at -- its identity."""
    url = entry.get("dxnn_url")
    if not url:
        return None
    name = url.rsplit("/", 1)[-1]
    return name[: -len(".dxnn")] if name.endswith(".dxnn") else name


def _dump(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def build_registry(rows: list[dict]) -> str:
    owned = {r["variant"] for r in rows}
    existing = json.loads(REGISTRY.read_text(encoding="utf-8"))
    kept = []
    for entry in existing:
        if entry.get("variant") in owned:
            continue                      # ours: dropped, re-added below
        entry.setdefault("published", True)
        kept.append({k: entry[k] for k in REGISTRY_KEYS if k in entry})
    kept.extend(registry_entry(r) for r in sorted(rows, key=lambda r: r["variant"]))
    return _dump(kept)


def build_specs(rows: list[dict]) -> str:
    specs = json.loads(SPECS.read_text(encoding="utf-8"))
    for row in rows:
        specs.pop(row["variant"], None)
        specs[row["variant"]] = spec_entry(row)
    return _dump(dict(sorted(specs.items())))


def build_snapshot(rows: list[dict]) -> str:
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    variants = {r["variant"]: {"task": r["task"], "family": r["family"]}
                for r in sorted(rows, key=lambda r: r["variant"])}
    # Keep the original date while the assignments themselves are unchanged. Stamping
    # today unconditionally made --check fail every day after the first, for a file
    # whose content had not moved -- and an idempotence check that cries wolf daily
    # stops being read.
    previous = snap.get("provisional") or {}
    assigned_at = (previous.get("assigned_at")
                   if previous.get("variants") == variants
                   else datetime.now().strftime("%Y-%m-%d"))
    snap["provisional"] = {
        "note": (
            "Unpublished in dx-modelzoo as of assigned_at. The (task, family) values "
            "here are OUR assignments, following dx-modelzoo's postprocessing-chain "
            "rule, because the authoritative upstream table does not exist for these "
            "stems yet. When upstream publishes one it must move into 'variants'; "
            "test_provisional_never_overlaps_the_harvested_table fails until it does, "
            "and harvest_modelzoo_tree.py reports any stem upstream assigns "
            "differently."
        ),
        "assigned_at": assigned_at or datetime.now().strftime("%Y-%m-%d"),
        "variant_count": len(rows),
        "variants": variants,
    }
    return _dump(snap)


def build_manifest(rows: list[dict]) -> str:
    # Drop by the .dxnn stem, never by 'name': the stem is the stable identity, so
    # re-running after a change to the 'name' convention still removes the old row
    # instead of leaving a duplicate behind (which is exactly what happened once).
    owned_stems = {r["variant"] for r in rows}
    owned_names = {r["legacy_model_name"] for r in rows}
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    kept = [
        e for e in existing
        if _manifest_stem(e) not in owned_stems and e.get("name") not in owned_names
    ]
    kept.extend(manifest_entry(r) for r in sorted(rows, key=lambda r: r["variant"]))
    return _dump(kept)


BUILDERS = {
    "registry": (REGISTRY, build_registry),
    "specs": (SPECS, build_specs),
    "snapshot": (SNAPSHOT, build_snapshot),
    "manifest": (MANIFEST, build_manifest),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if writing would change any file")
    ap.add_argument("--snapshot-only", action="store_true",
                    help="write only the snapshot's provisional section")
    args = ap.parse_args()

    self_check()
    rows = expand()
    targets = ["snapshot"] if args.snapshot_only else list(BUILDERS)

    drift = []
    for name in targets:
        path, build = BUILDERS[name]
        before = path.read_text(encoding="utf-8")
        after = build(rows)
        if before == after:
            print(f"[DXAPP] [INFO] {name}: unchanged")
            continue
        if args.check:
            drift.append(name)
            print(f"[DXAPP] [ERROR] {name}: would change ({path})", file=sys.stderr)
            continue
        path.write_text(after, encoding="utf-8")
        print(f"[DXAPP] [INFO] {name}: written ({path})")

    if drift:
        print(f"[DXAPP] [ERROR] {len(drift)} file(s) out of date: {', '.join(drift)}. "
              "Run scripts/add_modelzoo_2_5_0_models.py.", file=sys.stderr)
        return 1

    if not args.check and not args.snapshot_only:
        reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
        specs = json.loads(SPECS.read_text(encoding="utf-8"))
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        published = sum(1 for e in reg if e["published"])
        print(f"[DXAPP] [INFO] registry {len(reg)} entries ({published} published), "
              f"specs {len(specs)}, manifest {len(man)}, provisional {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
