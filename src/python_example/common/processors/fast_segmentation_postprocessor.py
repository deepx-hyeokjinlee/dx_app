"""
Generic fast semantic segmentation postprocessor.

This postprocessor is optimized for reduced-resolution class logits. It computes
argmax at the model output resolution first, then resizes the single-channel
class map to the original image size. This avoids upsampling every class logit
channel to full resolution.
"""

from typing import List, Optional

import cv2
import numpy as np

from ..base import IPostprocessor, PreprocessContext, SegmentationResult
from .segmentation_postprocessor import class_map_to_original


class FastSegmentationPostprocessor(IPostprocessor):
    """Fast semantic segmentation postprocessor for low-resolution logits."""

    def __init__(
        self,
        input_width: int = 2048,
        input_height: int = 1024,
        config: Optional[dict] = None,
        *,
        num_classes: Optional[int] = None,
    ):
        self.input_width = input_width
        self.input_height = input_height
        self.config = config or {}
        self.num_classes = num_classes or self.config.get("num_classes", 19)

    def process(self, outputs: List[np.ndarray], ctx: PreprocessContext) -> List[SegmentationResult]:
        if not outputs:
            return []

        class_map = self._to_class_map(outputs[0])
        if class_map is None:
            return []

        class_map = self._resize_to_original(class_map, ctx)
        return [self._make_result(class_map)]

    def _to_class_map(self, output: np.ndarray) -> Optional[np.ndarray]:
        """Convert NCHW/NHWC logits or pre-argmaxed outputs to a class map."""
        if output.ndim == 4:
            logits = output[0]
            if logits.shape[0] == 1 or np.issubdtype(logits.dtype, np.integer):
                return logits.squeeze().astype(np.int32)
            if (logits.ndim == 3 and logits.shape[2] < logits.shape[0]
                    and logits.shape[2] < logits.shape[1]):
                logits = np.transpose(logits, (2, 0, 1))
            return np.argmax(logits, axis=0).astype(np.int32)

        if output.ndim == 3:
            if output.shape[0] == 1 or np.issubdtype(output.dtype, np.integer):
                return output.squeeze().astype(np.int32)
            if output.shape[2] < output.shape[0] and output.shape[2] < output.shape[1]:
                output = np.transpose(output, (2, 0, 1))
            return np.argmax(output, axis=0).astype(np.int32)

        if output.ndim == 2:
            return output.astype(np.int32)

        return None

    def _resize_to_original(self, class_map: np.ndarray, ctx: PreprocessContext) -> np.ndarray:
        """Resize the single-channel class map to original image size."""
        return class_map_to_original(class_map, ctx, self.input_width, self.input_height)
    def _make_result(class_map: np.ndarray) -> SegmentationResult:
        height, width = class_map.shape
        unique_classes = np.unique(class_map).tolist()
        return SegmentationResult(
            mask=class_map,
            width=width,
            height=height,
            class_ids=unique_classes,
            class_names=[],
        )

    def get_model_name(self) -> str:
        return "fast_segmentation"
