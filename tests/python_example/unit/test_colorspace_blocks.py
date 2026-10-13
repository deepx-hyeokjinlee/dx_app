"""ycrcb_limited_to_bgr converts in row blocks.

On a super-resolved frame (7680x4320 at x4 from 1080p) converting the whole
image at once allocated ~2 GB of float32 temporaries, the largest part of the
Python ESPCN memory peak. The arithmetic per pixel is unchanged, so the result
must be identical whether or not a block boundary falls inside the image.
"""
import tracemalloc

import numpy as np
import pytest

from common.utility import colorspace


def _whole_image(ycrcb):
    """The previous implementation, verbatim: the reference for the result."""
    y = ycrcb[:, :, 0].astype(np.float32) - colorspace.Y_LIMITED_MIN
    cr = ycrcb[:, :, 1].astype(np.float32) - 128.0
    cb = ycrcb[:, :, 2].astype(np.float32) - 128.0
    yy = colorspace._IY * y
    r = yy + colorspace._IR_CR * cr
    g = yy + colorspace._IG_CB * cb + colorspace._IG_CR * cr
    b = yy + colorspace._IB_CB * cb
    out = np.stack([b, g, r], axis=2)
    return np.clip(out, 0.0, 255.0).round().astype(np.uint8)


@pytest.mark.parametrize("rows", [1, 255, 256, 257, 700])
def test_the_result_is_unchanged_across_block_boundaries(rows):
    ycrcb = np.random.default_rng(rows).integers(0, 256, (rows, 37, 3), dtype=np.uint8)
    out = colorspace.ycrcb_limited_to_bgr(ycrcb)
    assert out.dtype == np.uint8 and out.shape == ycrcb.shape
    np.testing.assert_array_equal(out, _whole_image(ycrcb))


def _peak_bytes(fn, arg):
    tracemalloc.start()
    try:
        fn(arg)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_scratch_memory_does_not_grow_with_the_image():
    """Beyond the uint8 result, the conversion needs one block of rows of
    scratch however tall the image is. Whole-image it needed ~20x the result."""
    small = np.zeros((1024, 1024, 3), np.uint8)
    large = np.zeros((4096, 1024, 3), np.uint8)
    scratch_small = _peak_bytes(colorspace.ycrcb_limited_to_bgr, small) - small.nbytes
    scratch_large = _peak_bytes(colorspace.ycrcb_limited_to_bgr, large) - large.nbytes
    assert scratch_large <= scratch_small + 1024 * 1024, (scratch_small, scratch_large)
    assert _peak_bytes(_whole_image, large) > 10 * large.nbytes  # the cost removed
