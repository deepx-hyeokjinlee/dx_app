"""LetterboxPreprocessor(bgr=True) feeds the frame without the BGR->RGB swap.

YOLOX, NanoDet and RetinaFace-mobilenet0.25 were trained on cv2 (BGR)
images; fed RGB they lost 3.1-4.7 COCO mAP (YOLOX/NanoDet) and 10.7 face AP
(RetinaFace, COCO-keypoint proxy).
"""
import numpy as np

from common.processors.letterbox_preprocessor import LetterboxPreprocessor


def _frame():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[..., 0] = 10   # B
    img[..., 2] = 200  # R
    return img


def test_default_swaps_to_rgb():
    out, _ = LetterboxPreprocessor(64, 64).process(_frame())
    assert tuple(out[32, 32]) == (200, 0, 10)


def test_bgr_keeps_the_frame_order():
    out, _ = LetterboxPreprocessor(64, 64, bgr=True).process(_frame())
    assert tuple(out[32, 32]) == (10, 0, 200)
