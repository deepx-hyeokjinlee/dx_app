#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""Write the CPU half of OpenAI CLIP RN50's text tower for clip-text_resnet50_77x512_openai.

The .dxnn is the text transformer only: [1,77,512] token+position embedding in,
[1,77,512] hidden states out. The tokenizer, the embedding lookup, ``ln_final`` and
``text_projection`` run on the CPU (clip_text_encoder.hpp / clip_text_encoder.py) and
need two files, written here into --out (default: next to the .dxnn, where the
examples look first):

  clip_rn50_text.nxclip   "NXCLIPT1" u32 vocab u32 context u32 width u32 embed_dim,
                          then token_embedding, positional_embedding,
                          ln_final.weight, ln_final.bias, text_projection (float32,
                          ~103 MB -- too large to ship in the repository)
  clip_bpe_merges.txt     the BPE merges in rank order (OpenAI's
                          bpe_simple_vocab_16e6.txt.gz, header line dropped)

The inputs are OpenAI's own files (MIT licence, github.com/openai/CLIP): the RN50
TorchScript checkpoint and the BPE vocabulary. Pass local copies with --checkpoint /
--bpe, or let the script download them (TLS verified) into --cache. Both are checked
against the sha256 OpenAI publishes (the checkpoint's is its download URL's path
segment) before anything is read; a mismatch is an error.

This is a BUILD-TIME tool: it needs torch to read the TorchScript checkpoint. The
examples themselves need only numpy (Python) or nothing (C++).

  python scripts/export_clip_text_assets.py --out assets/models
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import struct
import sys
import urllib.request
from pathlib import Path

# openai/CLIP clip/clip.py _MODELS["RN50"]: the hash is the URL's path segment.
RN50_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
RN50_URL = f"https://openaipublic.azureedge.net/clip/models/{RN50_SHA256}/RN50.pt"
# openai/CLIP clip/bpe_simple_vocab_16e6.txt.gz as shipped with the repository.
BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
BPE_URL = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> Path:
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, help="OpenAI RN50.pt (TorchScript); downloaded if omitted")
    ap.add_argument("--bpe", type=Path, help="bpe_simple_vocab_16e6.txt.gz; downloaded if omitted")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "dx_app" / "clip_rn50",
                    help="where downloads go (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "models",
                    help="output directory: the one holding the .dxnn (default: %(default)s)")
    args = ap.parse_args()

    checkpoint = args.checkpoint or fetch(RN50_URL, args.cache / "RN50.pt")
    bpe = args.bpe or fetch(BPE_URL, args.cache / "bpe_simple_vocab_16e6.txt.gz")
    for path, want in ((checkpoint, RN50_SHA256), (bpe, BPE_SHA256)):
        got = sha256(path)
        if got != want:
            print(f"ERROR: {path} sha256 {got} is not OpenAI's {want}", file=sys.stderr)
            return 1

    try:
        import torch  # build-time only
    except ImportError as exc:  # pragma: no cover - build-time
        print(f"ERROR: {exc}. This build-time tool needs torch to read the TorchScript "
              "checkpoint; the examples themselves do not.", file=sys.stderr)
        return 1

    sd = torch.jit.load(str(checkpoint), map_location="cpu").eval().state_dict()
    tok = sd["token_embedding.weight"].float().numpy()
    pos = sd["positional_embedding"].float().numpy()
    ln_w = sd["ln_final.weight"].float().numpy()
    ln_b = sd["ln_final.bias"].float().numpy()
    proj = sd["text_projection"].float().numpy()
    vocab, width = tok.shape
    context = pos.shape[0]
    embed_dim = proj.shape[1]
    if pos.shape != (context, width) or proj.shape[0] != width:
        print(f"ERROR: unexpected CLIP text shapes {tok.shape} {pos.shape} {proj.shape}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    weights = args.out / "clip_rn50_text.nxclip"
    with open(weights, "wb") as f:
        f.write(b"NXCLIPT1")
        f.write(struct.pack("<4I", vocab, context, width, embed_dim))
        for array in (tok, pos, ln_w, ln_b, proj):
            f.write(array.astype("<f4").tobytes(order="C"))
    merges = gzip.open(bpe).read().decode("utf-8").split("\n")[1:]
    (args.out / "clip_bpe_merges.txt").write_text("\n".join(merges), encoding="utf-8")
    print(f"wrote {weights} (vocab {vocab}, context {context}, width {width}, embed {embed_dim}) "
          f"and {args.out / 'clip_bpe_merges.txt'} ({len(merges)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
