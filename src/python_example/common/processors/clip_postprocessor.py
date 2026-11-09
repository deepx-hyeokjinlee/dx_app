"""
CLIP (Contrastive Language-Image Pre-Training) Postprocessor

Feature embedding extraction for both image and text encoders.

Image Encoder:
  Input:  [1, 3, H, W]  — preprocessed image
  Output: [1, D]         — image feature vector (D=512, 768, 1024 depending on backbone)
  → Uses GenericEmbeddingPostprocessor (same algorithm as ArcFace)

Text Encoder (clip-text_resnet50_77x512_openai: the transformer only):
  Input:  [1, 77, 512] — token + position embedding (ClipTextPreprocessor)
  Output: [1, 77, 512] — hidden states; ln_final at the end-of-text token,
          @ text_projection -> [1024] (CLIPTextPostprocessor)
  The CPU half is clip_text_encoder.py, the same algorithm as the C++ tree.

The output embedding is L2-normalized for cosine similarity computation.
Image-text similarity = dot(image_embedding, text_embedding)
"""

import numpy as np
from typing import List, Tuple

from ..base import IPostprocessor, IPreprocessor, PreprocessContext, EmbeddingResult
from .clip_text_encoder import eot_position, load_clip_text_assets
from .embedding_postprocessor import GenericEmbeddingPostprocessor


def _assets_from(config: dict):
    directory = (config or {}).get("text_assets")
    if not directory:
        raise ValueError(
            "[DXAPP] [ERROR] CLIP text: config has no 'text_assets' directory (the runner "
            "sets it from --text-assets, the config, or the .dxnn directory)")
    return load_clip_text_assets(str(directory))


class CLIPImagePostprocessor(GenericEmbeddingPostprocessor):
    """
    Postprocessor for CLIP image encoder.

    Thin wrapper around GenericEmbeddingPostprocessor with
    CLIP-specific defaults (model_type="image_encoder").
    """

    def __init__(self, input_width: int = 224, input_height: int = 224, config: dict = None):
        super().__init__(
            input_width=input_width,
            input_height=input_height,
            config=config,
            model_type="image_encoder",
            model_name="clip_image",
        )


class ClipTextPreprocessor(IPreprocessor):
    """Text -> 77 token ids -> the [1, 77, 512] float32 model input.

    ``process`` takes the prompt string (the text runner passes it where the
    image runners pass an image) and records the end-of-text position in the
    context; the postprocessor projects the hidden state at that row.
    """

    def __init__(self, input_width: int = 512, input_height: int = 77, config: dict = None):
        self.config = config or {}
        self.assets = _assets_from(self.config)
        self.last_token_count = 0
        self.last_truncated = False

    def process(self, text) -> Tuple[np.ndarray, PreprocessContext]:
        if not isinstance(text, str):
            raise TypeError(
                "[DXAPP] [ERROR] clip-text takes text, not an image: give prompts with "
                "-t / --text or --text-file")
        ids, self.last_truncated = self.assets.tokenizer.tokenize(text)
        self.last_token_count = sum(1 for i in ids if i != 0)
        tensor = self.assets.weights.embed(ids)[np.newaxis]
        ctx = PreprocessContext(input_width=self.assets.weights.width,
                                input_height=self.assets.weights.context)
        ctx.eot_position = eot_position(ids)
        return tensor, ctx

    def get_input_width(self) -> int:
        return self.assets.weights.width

    def get_input_height(self) -> int:
        return self.assets.weights.context


class CLIPTextPostprocessor(IPostprocessor):
    """Hidden states -> the L2-normalised joint-space text embedding.

    The end-of-text row is the arg-max of the token ids (CLIP's
    ``text.argmax(dim=-1)``), carried in ``ctx.eot_position``. It cannot be
    recovered from the hidden states: all 77 rows are non-zero. A [1, D] output
    (a graph that already ends in the projection) is L2-normalised as it is.
    """

    def __init__(self, input_width: int = 512, input_height: int = 77, config: dict = None):
        self.config = config or {}
        self.assets = _assets_from(self.config)

    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext) -> List[EmbeddingResult]:
        output = np.asarray(outputs[0], dtype=np.float32)
        w = self.assets.weights
        if output.ndim == 3:
            if output.shape[1:] != (w.context, w.width):
                raise ValueError(
                    f"[DXAPP] [ERROR] CLIPTextPostprocessor: hidden states {output.shape} do "
                    f"not match the text weights (1, {w.context}, {w.width})")
            eot = getattr(ctx, "eot_position", -1)
            if not 0 <= eot < w.context:
                raise ValueError(
                    "[DXAPP] [ERROR] CLIPTextPostprocessor: no end-of-text position in the "
                    "context (the input must come from ClipTextPreprocessor)")
            embedding = w.project(output[0], eot)
        else:
            embedding = output.flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        return [EmbeddingResult(embedding=embedding.astype(np.float32), model_type="text_encoder")]

    def get_model_name(self) -> str:
        return "clip_text"
