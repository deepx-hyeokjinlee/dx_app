# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""The CPU half of the CLIP RN50 text tower (common/processors/clip_text_encoder.py).

clip-text_resnet50_77x512_openai.dxnn is the transformer only, so the example
tokenizes, embeds, and projects on the CPU. Before this, the example fed the model
an image resize (a uint8 tensor its float32 input refused), took hidden-state row
76 as the end-of-text token (all 77 rows are non-zero, so "the last non-zero row"
is always 76; measured: min row norm 8.6), and returned that 512-d row without
ln_final / text_projection.

The synthetic tests pin the mechanics with tiny files; the real-asset test pins
token ids against OpenAI's SimpleTokenizer and runs when the exported assets are
present (DXAPP_CLIP_TEXT_ASSETS, else assets/models).
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "python_example"))

from common.base import PreprocessContext  # noqa: E402
from common.processors import ClipTextPreprocessor, CLIPTextPostprocessor  # noqa: E402
from common.processors.clip_text_encoder import (  # noqa: E402
    MERGES_FILE, WEIGHTS_FILE, ClipTextWeights, ClipTokenizer, bytes_to_unicode, clean,
    eot_position, load_clip_text_assets, pre_tokenize)

N_MERGES = 49152 - 256 - 2
VOCAB = 512 + N_MERGES + 2          # bytes, bytes</w>, merges, SOT, EOT


def write_merges(path: Path) -> None:
    """Rank 0 merges 'h' + 'e</w>'; the rest are pairs no text can produce."""
    lines = ["h e</w>"] + [f"Ā{i} ā" for i in range(N_MERGES - 1)]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_weights(path: Path, vocab: int, context: int, width: int, embed: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    arrays = [rng.standard_normal(n).astype("<f4") for n in
              (vocab * width, context * width, width, width, width * embed)]
    with open(path, "wb") as f:
        f.write(b"NXCLIPT1" + struct.pack("<4I", vocab, context, width, embed))
        for a in arrays:
            f.write(a.tobytes())
    return arrays


def test_byte_map_is_a_bijection_onto_printable_characters():
    table = bytes_to_unicode()
    assert sorted(b for b, _ in table) == list(range(256))
    assert len({u for _, u in table}) == 256
    assert dict(table)[ord("!")] == "!" and dict(table)[0] == chr(256)


def test_clean_unescapes_collapses_whitespace_and_lowercases_ascii():
    assert clean("  A &amp;amp; B\t\nC  ") == "a & b c"
    assert clean("Naïve CAFÉ") == "naïve cafÉ"   # only ASCII is lower-cased


def test_pre_tokenize_follows_the_simple_tokenizer_pattern():
    assert pre_tokenize("the dog's ball isn't red, it's blue.") == [
        "the", "dog", "'s", "ball", "isn", "'t", "red", ",", "it", "'s", "blue", "."]
    assert pre_tokenize("10pm <|endoftext|>") == ["1", "0", "pm", "<|endoftext|>"]


def test_tokenize_applies_the_merges_and_pads_to_77(tmp_path):
    write_merges(tmp_path / "m.txt")
    tok = ClipTokenizer(tmp_path / "m.txt")
    assert len(tok) == VOCAB and (tok.sot, tok.eot) == (VOCAB - 2, VOCAB - 1)
    ids, cut = tok.tokenize("He")
    assert ids[:3] == [tok.sot, 512, tok.eot] and ids[3:] == [0] * 74 and not cut
    assert eot_position(ids) == 2


def test_tokenize_cuts_at_77_and_ends_with_eot(tmp_path):
    write_merges(tmp_path / "m.txt")
    tok = ClipTokenizer(tmp_path / "m.txt")
    ids, cut = tok.tokenize("he " * 100)
    assert cut and len(ids) == 77 and ids[-1] == tok.eot and eot_position(ids) == 76


def test_too_few_merges_is_an_error(tmp_path):
    (tmp_path / "m.txt").write_text("h e\n", encoding="utf-8")
    with pytest.raises(ValueError, match="merges"):
        ClipTokenizer(tmp_path / "m.txt")


def test_embed_and_project_match_the_formulas(tmp_path):
    tok, pos, ln_w, ln_b, proj = write_weights(tmp_path / "w.nxclip", 7, 3, 4, 2)
    w = ClipTextWeights(tmp_path / "w.nxclip")
    x = w.embed([5, 2, 0])
    np.testing.assert_array_equal(x, tok.reshape(7, 4)[[5, 2, 0]] + pos.reshape(3, 4))
    hidden = np.arange(12, dtype=np.float32).reshape(3, 4) ** 1.5
    h = hidden[1].astype(np.float64)
    want = ((h - h.mean()) / np.sqrt(h.var() + 1e-5)) * ln_w + ln_b
    want = want @ proj.reshape(4, 2)
    np.testing.assert_allclose(w.project(hidden, 1), want / np.linalg.norm(want), rtol=1e-5)


def test_truncated_weights_are_an_error(tmp_path):
    write_weights(tmp_path / "w.nxclip", 7, 3, 4, 2)
    data = (tmp_path / "w.nxclip").read_bytes()
    (tmp_path / "short.nxclip").write_bytes(data[:-4])
    with pytest.raises(ValueError, match="truncated"):
        ClipTextWeights(tmp_path / "short.nxclip")


@pytest.fixture
def synthetic_assets(tmp_path):
    write_merges(tmp_path / MERGES_FILE)
    write_weights(tmp_path / WEIGHTS_FILE, VOCAB, 77, 4, 3)
    load_clip_text_assets.cache_clear()
    return str(tmp_path)


def test_preprocessor_and_postprocessor_carry_the_eot_row(synthetic_assets):
    cfg = {"text_assets": synthetic_assets}
    pre, post = ClipTextPreprocessor(4, 77, cfg), CLIPTextPostprocessor(4, 77, cfg)
    tensor, ctx = pre.process("He")
    assert tensor.shape == (1, 77, 4) and tensor.dtype == np.float32
    assert ctx.eot_position == 2 and pre.last_token_count == 3
    hidden = np.random.default_rng(1).standard_normal((1, 77, 4)).astype(np.float32)
    (result,) = post.process([hidden], ctx)
    np.testing.assert_allclose(result.embedding, pre.assets.weights.project(hidden[0], 2))
    assert result.embedding.shape == (3,) and abs(np.linalg.norm(result.embedding) - 1) < 1e-6


def test_postprocessor_refuses_a_context_without_an_eot_row(synthetic_assets):
    post = CLIPTextPostprocessor(4, 77, {"text_assets": synthetic_assets})
    with pytest.raises(ValueError, match="end-of-text"):
        post.process([np.zeros((1, 77, 4), np.float32)], PreprocessContext())


def test_preprocessor_refuses_an_image(synthetic_assets):
    pre = ClipTextPreprocessor(4, 77, {"text_assets": synthetic_assets})
    with pytest.raises(TypeError, match="takes text"):
        pre.process(np.zeros((224, 224, 3), np.uint8))


def test_missing_assets_name_the_export_command(tmp_path):
    load_clip_text_assets.cache_clear()
    with pytest.raises(FileNotFoundError, match="export_clip_text_assets.py"):
        load_clip_text_assets(str(tmp_path))


def test_text_options_exist_only_for_text_models(monkeypatch):
    from common.runner.args import parse_common_args
    monkeypatch.setattr(sys, "argv", ["x", "-t", "a", "-t", "b", "--text-assets", "/d"])
    args = parse_common_args(include_text_inputs=True, include_stream_inputs=False)
    assert args.text == ["a", "b"] and args.text_assets == "/d"
    with pytest.raises(SystemExit):
        parse_common_args(include_stream_inputs=False)


REAL = Path(os.environ.get("DXAPP_CLIP_TEXT_ASSETS", PROJECT_ROOT / "assets" / "models"))


@pytest.mark.skipif(not (REAL / MERGES_FILE).is_file(),
                    reason="real CLIP text assets not exported (scripts/export_clip_text_assets.py)")
def test_real_ids_equal_openai_simple_tokenizer():
    """Pinned from OpenAI's SimpleTokenizer (bpe_simple_vocab_16e6)."""
    tok = ClipTokenizer(REAL / MERGES_FILE)
    assert tok.encode("a photo of a dog") == [320, 1125, 539, 320, 1929]
    assert tok.encode("the dog's ball isn't red, it's blue.") == [
        518, 1929, 568, 1069, 2923, 713, 736, 267, 585, 568, 1746, 269]
    assert tok.encode("naïve café") == [1097, 35689, 563, 15304]
