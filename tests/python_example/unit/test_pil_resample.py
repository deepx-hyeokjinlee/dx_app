"""pil_resample: Pillow's antialiased resize, byte for byte, without Pillow."""
import numpy as np
import pytest

from common.utility import pil_resample

BL = [99,132,137,134,151,134,116,120,140,111,125,131,140,124,103,108,127,139,114,141,125,132,137,137,112,128,142,120,116,122,132,138,134,112,127,142,117,143,122,129,135,140,115,128,146,151,130,114,123,129,143,114,134,138,120,139,120,117,130,155]
BC = [19,72,127,96,159,223,192,179,110,65,49,101,76,141,188,166,225,45,87,154,183,188,200,86,117,88,112,75,133,187,170,140,97,157,77,79,161,145,111,109,124,84,90,118,189,153,174,148,118,83,82,105,89,165,129,95,76,49,100,159,148,190,166,160,128,97,57,88,149,117,149,158,44,107,168,130,175,167,187,176,49,80,79,147,113,172,191,169,115,62,138,197,205,165,104,68,33,76,139,115,178,221,186,150,89,49,57,116,192,139,85,63,83,141,107,176,174,205,187,67,87,77,124,86,147,210,96,105,137,105,166,197,157,127,94,88,108,106,95,145,205,178,159,125,89,165,149,172,171,136,106,79,101,65,133,164,170,187,145,143,106,93,175,177,97,157,114,84,67,120,179,142,174,137,168,148,66,71,95,167,209,29,88,66,113,178,153,205,189,144,74,62,31,90,158,127,182,235]


def _pattern():
    w, h = 13, 9
    return np.array([[[(x * 37 + y * 91 + c * 53) % 256 for c in range(3)] for x in range(w)]
                     for y in range(h)], dtype=np.uint8)


def test_matches_pillow_shrinking_bilinear():
    # PIL 13.0.0 Image.fromarray(src).resize((5, 4), BILINEAR)
    assert pil_resample.resize(_pattern(), 5, 4).reshape(-1).tolist() == BL


def test_matches_pillow_upscaling_bicubic():
    assert pil_resample.resize(_pattern(), 6, 11, bicubic=True).reshape(-1).tolist() == BC


def test_short_side_and_centre_crop_follow_torchvision():
    src = _pattern()                                   # 13 x 9: the short side is the height
    full = pil_resample.resize(src, 8, 6)              # short 6 -> 8 x 6 (int(6 * 13 / 9))
    crop = pil_resample.resize_short_center_crop(src, 6, 4, 4)
    assert np.array_equal(crop, full[1:5, 2:6])        # round((6-4)/2), round((8-4)/2)


def test_against_installed_pillow_when_there_is_one():
    pil = pytest.importorskip("PIL.Image")
    rng = np.random.RandomState(0)
    for _ in range(6):
        h, w = rng.randint(5, 90, size=2)
        img = rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
        ow, oh = rng.randint(3, 120, size=2)
        for bicubic, filt in ((False, pil.BILINEAR), (True, pil.BICUBIC)):
            ref = np.asarray(pil.fromarray(img).resize((int(ow), int(oh)), filt))
            assert np.array_equal(pil_resample.resize(img, int(ow), int(oh), bicubic), ref)
