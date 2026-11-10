"""NanoDet-Plus: four levels (ceil grid) and priors at the cell corner.

nanodet-plus 416 emits 52+26+13+7 = 3598 rows; its head places priors at
x * stride (upstream nanodet_plus_head), so config center_offset is 0.0.
Decoding at the centre (0.5) moved every box half a stride: COCO val2017
(150 images) 17.8 -> 29.5 mAP.
"""
import numpy as np

from common.processors.nanodet_postprocessor import NanoDetPostprocessor

ROW = 1 + 4 * 2  # one class, reg_max 1


def _plus_output(row_index):
    x = np.full((3598, ROW), -10.0, dtype=np.float32)
    x[:, 1:] = 0.0              # every side 0.5 stride from the prior
    x[row_index, 0] = 10.0      # the one confident row
    return [x[None]]


def _post(**cfg):
    return NanoDetPostprocessor(416, 416, dict({"num_classes": 1, "reg_max": 1,
                                                "score_threshold": 0.3}, **cfg))


def test_plus_stride64_prior_at_the_corner(ctx):
    # row 3549 + 8: the 7x7 stride-64 level's cell (1, 1) -> prior (64, 64)
    res = _post(center_offset=0.0).process(_plus_output(3549 + 8), ctx)
    assert len(res) == 1
    assert np.allclose(res[0].box, [32, 32, 96, 96], atol=1e-3)


def test_classic_default_is_the_cell_centre(ctx):
    res = _post().process(_plus_output(3549 + 8), ctx)
    assert np.allclose(res[0].box, [64, 64, 128, 128], atol=1e-3)
