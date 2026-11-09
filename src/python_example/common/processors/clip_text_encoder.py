# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""The CPU half of the CLIP RN50 (OpenAI) text tower around the NPU transformer.

``clip-text_resnet50_77x512_openai.dxnn`` takes the [1,77,512] token+position
embedding and returns the transformer's [1,77,512] hidden states; the tokenizer,
the embedding lookup, ``ln_final`` and ``text_projection`` are outside the
compiled graph. This module is that outside::

    text -> ClipTokenizer (OpenAI simple_tokenizer.py, MIT) -> 77 ids
         -> ClipTextWeights.embed   (token_embedding[id] + positional)
         -> NPU transformer
         -> ClipTextWeights.project (ln_final at the EOT position, @ text_projection,
            L2-normalised) -> 1024-d, the space clip-img_resnet50_224x224_openai lives in

The weights are OpenAI's RN50 checkpoint (MIT). ``scripts/export_clip_text_assets.py``
checks its sha256 and writes the two files read here, ``clip_rn50_text.nxclip`` and
``clip_bpe_merges.txt``, next to the .dxnn. ``cpp_example/common/processors/
clip_text_encoder.hpp`` is the same algorithm line for line, so the two trees
produce the same ids and the same input tensor.

Tokenizer differences from the Python original, both disclosed: ftfy's mojibake
repair is not done, and the regex classes \\p{L} / \\p{N} are approximated -- ASCII
letters and every non-ASCII code point that is not whitespace count as letters,
ASCII digits as numbers. Plain-English prompts tokenize identically.
"""
from __future__ import annotations

import functools
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

CONTEXT = 77
WEIGHTS_FILE = "clip_rn50_text.nxclip"
MERGES_FILE = "clip_bpe_merges.txt"

_SPECIAL = ("<|startoftext|>", "<|endoftext|>")
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
_ENTITIES = (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"))
_SPACE, _LETTER, _DIGIT, _OTHER = range(4)


def bytes_to_unicode() -> List[Tuple[int, str]]:
    """GPT-2's reversible byte -> printable-unicode map, in dict insertion order."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [(b, chr(c)) for b, c in zip(bs, cs)]


def clean(text: str) -> str:
    """basic_clean + whitespace_clean + lower(): the five HTML entities unescaped
    twice, whitespace runs to one space, trimmed, ASCII lower-cased."""
    s = text
    for _ in range(2):
        for entity, char in _ENTITIES:
            s = s.replace(entity, char)
    data = s.encode("utf-8")
    out = bytearray()
    space = False
    for c in data:
        if c in b" \t\n\r\f\v":
            space = True
            continue
        if space and out:
            out += b" "
        space = False
        out.append(c - ord("A") + ord("a") if ord("A") <= c <= ord("Z") else c)
    return out.decode("utf-8", errors="replace")


def _kind(c: str) -> int:
    o = ord(c)
    if c == " ":
        return _SPACE
    if o >= 0x80 or ("a" <= c <= "z") or ("A" <= c <= "Z"):
        return _LETTER
    if "0" <= c <= "9":
        return _DIGIT
    return _OTHER


def pre_tokenize(s: str) -> List[str]:
    """The simple_tokenizer.py pattern, alternatives in its order:
    <|startoftext|> | <|endoftext|> | 's|'t|'re|'ve|'m|'ll|'d | \\p{L}+ | \\p{N} | [^\\s\\p{L}\\p{N}]+"""
    out: List[str] = []
    i = 0
    while i < len(s):
        hit = next((p for p in _SPECIAL + _CONTRACTIONS if s.startswith(p, i)), None)
        if hit is not None:
            out.append(hit)
            i += len(hit)
            continue
        k = _kind(s[i])
        if k == _SPACE:
            i += 1
            continue
        if k == _DIGIT:
            out.append(s[i])
            i += 1
            continue
        j = i
        while j < len(s) and _kind(s[j]) == k:
            j += 1
        out.append(s[i:j])
        i = j
    return out


class ClipTokenizer:
    """OpenAI CLIP's byte-level BPE (simple_tokenizer.py), 77-token context."""

    def __init__(self, merges_path: str | Path):
        lines = [ln.rstrip("\r") for ln in Path(merges_path).read_text(encoding="utf-8").split("\n")]
        merges = [tuple(ln.split(" ", 1)) for ln in lines if ln and " " in ln]
        wanted = 49152 - 256 - 2
        if len(merges) < wanted:
            raise ValueError(f"BPE merges file has {len(merges)} merges, expected {wanted}: {merges_path}")
        merges = merges[:wanted]
        table = bytes_to_unicode()
        self._byte_encoder = {b: u for b, u in table}
        vocab = [u for _, u in table]
        vocab += [u + "</w>" for _, u in table]
        vocab += ["".join(m) for m in merges]
        vocab += list(_SPECIAL)
        self.encoder = {tok: i for i, tok in enumerate(vocab)}
        self._ranks = {m: i for i, m in enumerate(merges)}
        self.sot = self.encoder["<|startoftext|>"]
        self.eot = self.encoder["<|endoftext|>"]

    def _bpe(self, token: str) -> List[str]:
        word = list(token)
        if not word:
            return []
        word[-1] += "</w>"
        while len(word) > 1:
            best, at = None, 0
            for i in range(len(word) - 1):
                r = self._ranks.get((word[i], word[i + 1]))
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if best is None:
                break
            first, second = word[at], word[at + 1]
            merged, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == first and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        return word

    def encode(self, text: str) -> List[int]:
        """BPE ids of ``text``, without SOT/EOT."""
        ids: List[int] = []
        for piece in pre_tokenize(clean(text)):
            mapped = "".join(self._byte_encoder[b] for b in piece.encode("utf-8"))
            ids.extend(self.encoder[sub] for sub in self._bpe(mapped) if sub in self.encoder)
        return ids

    def tokenize(self, text: str) -> Tuple[List[int], bool]:
        """clip.tokenize(text, truncate=True): [SOT, ids..., EOT] zero-padded to 77;
        too long is cut and the last slot made EOT. Returns (ids, truncated)."""
        ids = [self.sot] + self.encode(text) + [self.eot]
        cut = len(ids) > CONTEXT
        if cut:
            ids = ids[:CONTEXT]
            ids[-1] = self.eot
        return ids + [0] * (CONTEXT - len(ids)), cut

    def __len__(self) -> int:
        return len(self.encoder)


class ClipTextWeights:
    """The text tower's CPU-side weights (scripts/export_clip_text_assets.py)::

        "NXCLIPT1"  u32 vocab  u32 context  u32 width  u32 embed_dim  (little-endian)
        token_embedding [vocab*width]  positional [context*width]
        ln_final weight [width]  ln_final bias [width]  text_projection [width*embed_dim]

    all float32.
    """

    def __init__(self, path: str | Path):
        data = Path(path).read_bytes()
        if data[:8] != b"NXCLIPT1":
            raise ValueError(f"not an NXCLIPT1 file: {path}")
        self.vocab, self.context, self.width, self.embed_dim = struct.unpack_from("<4I", data, 8)
        sizes = (self.vocab * self.width, self.context * self.width, self.width, self.width,
                 self.width * self.embed_dim)
        if len(data) < 24 + 4 * sum(sizes):
            raise ValueError(f"truncated CLIP text weights: {path}")
        arrays, at = [], 24
        for n in sizes:
            arrays.append(np.frombuffer(data, dtype="<f4", count=n, offset=at))
            at += 4 * n
        self.token = arrays[0].reshape(self.vocab, self.width)
        self.position = arrays[1].reshape(self.context, self.width)
        self.ln_w, self.ln_b = arrays[2], arrays[3]
        self.projection = arrays[4].reshape(self.width, self.embed_dim)

    def embed(self, ids: List[int]) -> np.ndarray:
        """token_embedding[id] + positional_embedding, [context, width] float32."""
        idx = np.clip(np.asarray(ids[: self.context], dtype=np.int64), 0, self.vocab - 1)
        out = np.zeros((self.context, self.width), dtype=np.float32)
        out[: len(idx)] = self.token[idx] + self.position[: len(idx)]
        return out

    def project(self, hidden: np.ndarray, eot_position: int) -> np.ndarray:
        """ln_final (eps 1e-5) of the hidden state at ``eot_position``,
        @ text_projection, L2-normalised: the joint-space text embedding."""
        x = np.asarray(hidden, dtype=np.float32).reshape(-1, self.width)[eot_position].astype(np.float64)
        normed = ((x - x.mean()) / np.sqrt(x.var() + 1e-5)).astype(np.float32) * self.ln_w + self.ln_b
        out = normed.astype(np.float32) @ self.projection
        n = float(np.linalg.norm(out.astype(np.float64)))
        return (out / n if n > 1e-12 else out).astype(np.float32)


def eot_position(ids: List[int]) -> int:
    """The EOT position: the largest id (EOT is the last vocab entry), as CLIP's
    text.argmax(dim=-1)."""
    return int(np.argmax(np.asarray(ids)))


class ClipTextAssets:
    """The tokenizer and the weights, loaded once and shared by the pre- and
    postprocessor."""

    def __init__(self, tokenizer: ClipTokenizer, weights: ClipTextWeights):
        self.tokenizer = tokenizer
        self.weights = weights


@functools.lru_cache(maxsize=4)
def load_clip_text_assets(directory: str) -> ClipTextAssets:
    """Load ``clip_rn50_text.nxclip`` and ``clip_bpe_merges.txt`` from ``directory``.

    Raises FileNotFoundError naming the export command when either is missing.
    """
    d = Path(directory)
    weights_path, merges_path = d / WEIGHTS_FILE, d / MERGES_FILE
    for p in (weights_path, merges_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"[DXAPP] [ERROR] CLIP text assets: {p} not found. Write them once with: "
                f"python scripts/export_clip_text_assets.py --out {d}")
    tokenizer = ClipTokenizer(merges_path)
    weights = ClipTextWeights(weights_path)
    if weights.context != CONTEXT or weights.vocab != len(tokenizer):
        raise ValueError(
            f"[DXAPP] [ERROR] CLIP text assets disagree: weights context {weights.context} / "
            f"vocab {weights.vocab}, tokenizer context {CONTEXT} / vocab {len(tokenizer)}")
    return ClipTextAssets(tokenizer, weights)


def resolve_assets_dir(config: Optional[dict], model_path: Optional[str]) -> str:
    """``config["text_assets"]``, else the directory holding the .dxnn."""
    explicit = (config or {}).get("text_assets")
    if explicit:
        return str(explicit)
    return str(Path(model_path).resolve().parent) if model_path else "."
