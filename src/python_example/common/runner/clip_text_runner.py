# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Runner for the CLIP text tower: prompts in, joint-space embeddings out.

The image runners cannot drive this model: its input is text. Prompts come from
``-t / --text`` (repeatable) and ``--text-file`` (one per line); with neither,
three built-in prompts are used. Each prompt is tokenized and embedded on the CPU
(ClipTextPreprocessor), run through the NPU transformer, and projected into the
1024-d CLIP RN50 space (CLIPTextPostprocessor). With two or more distinct
prompts the runner prints their cosine-similarity matrix.

The CPU-side weights are read from ``--text-assets``, else the config key
``text_assets``, else the directory holding the .dxnn
(``scripts/export_clip_text_assets.py`` writes them).

``--save`` writes ``embeddings.json`` (prompt, token count, truncation flag,
vector) to the run directory; ``DXAPP_VERIFY=1`` dumps each result like every
other runner. The async runner submits every prompt with ``run_async`` before
waiting for the first. ``cpp_postprocessor`` (the ``*_cpp_postprocess``
scripts) swaps in dx_postprocess.CLIPTextPostProcess.

This is the C++ ``common/runner/clip_text_runner.hpp`` in Python.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import load_config
from ..processors.clip_text_encoder import resolve_assets_dir
from .interrupts import interrupt_scope
from .run_dir import create_run_dir, write_run_info
from .verify_serialize import dump_verify_json, is_verify_enabled

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = ["a photo of a dog", "a photo of a cat", "a photo of a car"]


def collect_prompts(args) -> List[str]:
    """``--text`` values, then ``--text-file`` lines; the built-in set if both are empty."""
    prompts = list(getattr(args, "text", None) or [])
    text_file = getattr(args, "text_file", None)
    if text_file:
        try:
            lines = Path(text_file).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.error(f"cannot read --text-file {text_file}: {exc}")
            sys.exit(1)
        prompts += [ln.rstrip("\r") for ln in lines if ln.strip()]
    if not prompts:
        logger.info("No prompt given (-t / --text-file). Using the built-in prompts.")
        prompts = list(DEFAULT_PROMPTS)
    return prompts


def cosine_matrix(embeddings: List[np.ndarray]) -> np.ndarray:
    m = np.stack([np.asarray(e, dtype=np.float64) for e in embeddings])
    return m @ m.T


class ClipTextRunner:
    """One prompt at a time: encode, run, project."""

    async_mode = False

    def __init__(self, factory, on_engine_init=None):
        self.factory = factory
        self._on_engine_init = on_engine_init
        self._cpp_postprocessor = None
        self._cpp_convert_fn = None
        self.text_assets_dir: Optional[str] = None

    def run(self, args) -> None:
        with interrupt_scope():
            try:
                self._run(args)
            except KeyboardInterrupt:
                logger.info("\nInterrupted by user.")

    def _run(self, args) -> None:
        from .sync_runner import (_check_dxrt_version, _check_model_version,
                                  _resolve_config_path, _validate_model)

        if getattr(args, "image", None):
            logger.error(
                f"{self.factory.get_model_name()} takes text, not an image: give prompts with "
                "-t \"a photo of a dog\" (repeatable) or --text-file <file, one prompt per line>.")
            sys.exit(2)
        _check_dxrt_version()
        prompts = collect_prompts(args)
        _validate_model(args)

        config_path = _resolve_config_path(args)
        if config_path:
            config = load_config(config_path, verbose=getattr(args, "show_log", False))
            if config:
                self.factory.load_config(config)
        cfg = dict(self.factory.config)
        if getattr(args, "text_assets", None):
            cfg["text_assets"] = args.text_assets
        self.text_assets_dir = resolve_assets_dir(cfg, args.model)
        self.factory.config["text_assets"] = self.text_assets_dir

        from dx_engine import InferenceEngine  # type: ignore[import-not-found]

        ie = InferenceEngine(args.model)
        _check_model_version(ie)
        pre = self.factory.create_preprocessor(0, 0)
        post = self.factory.create_postprocessor(0, 0)
        shape = list(ie.get_input_tensors_info()[0]["shape"])
        expected = [1, pre.get_input_height(), pre.get_input_width()]
        if shape != expected:
            logger.error(f"model input {shape} does not match the CLIP text weights {expected}")
            sys.exit(1)
        if self._on_engine_init is not None:
            self._on_engine_init(self)

        logger.info(f"Task: {self.factory.get_task_type()}")
        logger.info(f"Model loaded: {args.model}")
        logger.info(f"Text assets: {self.text_assets_dir}")
        logger.info(f"Prompts: {len(prompts)} ({'async' if self.async_mode else 'sync'})")

        loop = getattr(args, "loop", 1)
        total = loop if loop and loop > 1 else len(prompts)
        t_pre = t_inf = t_post = 0.0
        encoded = []
        start = time.perf_counter()
        for i in range(total):
            text = prompts[i % len(prompts)]
            t0 = time.perf_counter()
            tensor, ctx = pre.process(text)
            t_pre += time.perf_counter() - t0
            job = ie.run_async([tensor]) if self.async_mode else None
            encoded.append((text, tensor, ctx, pre.last_token_count, pre.last_truncated, job))

        last = {}
        for i, (text, tensor, ctx, tokens, cut, job) in enumerate(encoded):
            t1 = time.perf_counter()
            outputs = ie.wait(job) if self.async_mode else ie.run([tensor])
            t2 = time.perf_counter()
            if self._cpp_postprocessor is not None:
                results = self._cpp_convert_fn(
                    self._cpp_postprocessor.postprocess(list(outputs), ctx.eot_position), ctx)
            else:
                results = post.process(outputs, ctx)
            t3 = time.perf_counter()
            t_inf += t2 - t1
            t_post += t3 - t2
            if not results:
                continue
            if is_verify_enabled():
                dump_verify_json(results, f"text:{text}", args.model, "embedding",
                                 (ctx.input_height, ctx.input_width),
                                 verbose=getattr(args, "show_log", False))
            e = np.asarray(results[0].embedding, dtype=np.float32)
            last[text] = (e, tokens, cut)
            print(f"[{i}] \"{text}\"  tokens={tokens}{' (cut at 77)' if cut else ''}  "
                  f"eot={ctx.eot_position}  dim={e.size}  |e|={np.linalg.norm(e):.4f}")
        wall = time.perf_counter() - start

        names = [p for p in dict.fromkeys(prompts) if p in last]
        if len(names) >= 2:
            sim = cosine_matrix([last[n][0] for n in names])
            print("\nCosine similarity:")
            for n, row in zip(names, sim):
                label = n if len(n) <= 48 else n[:45] + "..."
                print("  " + "".join(f"{v:8.4f}" for v in row) + f"  {label}")

        if getattr(args, "save", False):
            run_dir = create_run_dir("text", f"{len(names)}-prompts",
                                     getattr(args, "save_dir", None))
            write_run_info(run_dir, args.model, "text")
            out = {"model": Path(args.model).name,
                   "prompts": [{"text": n, "tokens": last[n][1], "truncated": last[n][2],
                                "embedding": [float(v) for v in last[n][0]]} for n in names]}
            (run_dir / "embeddings.json").write_text(json.dumps(out, indent=1) + "\n",
                                                     encoding="utf-8")
            logger.info(f"Saved {run_dir / 'embeddings.json'}")

        print(f"\n{total} prompt(s): preprocess {1000 * t_pre / total:.2f} ms, inference "
              f"{1000 * t_inf / total:.2f} ms, postprocess {1000 * t_post / total:.2f} ms (avg); "
              f"{1000 * wall:.2f} ms total")


class AsyncClipTextRunner(ClipTextRunner):
    """Every prompt submitted with run_async, then waited for in order."""

    async_mode = True
