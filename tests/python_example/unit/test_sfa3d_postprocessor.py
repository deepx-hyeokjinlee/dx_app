"""SFA3D postprocessor unit tests."""

import math

import numpy as np
import pytest

from common.base import DetectionResult
from common.processors import Detection3DResult, SFA3DPostprocessor
from common.utility import convert_cpp_sfa3d


def _make_outputs():
    return [
        np.zeros((1, 3, 152, 152), dtype=np.float32),
        np.zeros((1, 2, 152, 152), dtype=np.float32),
        np.zeros((1, 2, 152, 152), dtype=np.float32),
        np.zeros((1, 1, 152, 152), dtype=np.float32),
        np.zeros((1, 3, 152, 152), dtype=np.float32),
    ]


class TestSFA3DPostprocessor:
    def test_returns_empty_when_no_peak_above_threshold(self, ctx):
        postprocessor = SFA3DPostprocessor(608, 608)
        outputs = _make_outputs()
        outputs[0].fill(-10.0)  # every heatmap logit well below the score threshold

        results = postprocessor.process(outputs, ctx)

        assert results == []

    def test_decodes_single_peak_to_one_3d_box(self, ctx):
        postprocessor = SFA3DPostprocessor(608, 608)
        outputs = _make_outputs()
        outputs[0].fill(-10.0)
        y, x = 10, 20
        outputs[0][0, 1, y, x] = 10.0   # class 1 (Car) heatmap peak
        outputs[2][0, 0, y, x] = 1.0    # direction: yaw_im
        outputs[2][0, 1, y, x] = 0.0    # direction: yaw_re
        outputs[3][0, 0, y, x] = 2.5    # z_coord (network regresses z - minZ)
        outputs[4][0, 0, y, x] = 1.5    # dim_h
        outputs[4][0, 1, y, x] = 8.0    # dim_w
        outputs[4][0, 2, y, x] = 16.0   # dim_l

        results = postprocessor.process(outputs, ctx)

        assert len(results) == 1
        det = results[0]
        assert isinstance(det, Detection3DResult)
        assert det.class_id == 1
        assert det.class_name == "Car"
        assert det.confidence > 0.99
        # BEV centre: (col/row) * (input / grid) with zero offset → 20*4, 10*4
        assert det.bev_x == pytest.approx(80.0)
        assert det.bev_y == pytest.approx(40.0)
        # dims are regressed directly in metres (no exp)
        assert (det.dim_h, det.dim_w, det.dim_l) == pytest.approx((1.5, 8.0, 16.0))
        # z regresses (z - minZ); minZ == -2.73
        assert det.z3d == pytest.approx(2.5 - 2.73)
        # SFA3D's convert_det_to_real_values negates it: -atan2(im=1, re=0) == -pi/2
        assert det.yaw == pytest.approx(-math.pi / 2.0)
        # the row axis is x, forward DOWN the raster: x = row * (50 / 152)
        assert det.x3d == pytest.approx(10 * 50.0 / 152)
        assert det.y3d == pytest.approx(-25.0 + 20 * 50.0 / 152)

    def test_offsets_move_x_with_channel_1_and_y_with_channel_0(self, ctx):
        postprocessor = SFA3DPostprocessor(608, 608)
        outputs = _make_outputs()
        outputs[0].fill(-10.0)
        outputs[0][0, 1, 10, 20] = 10.0
        outputs[1][0, 0, 10, 20] = 0.25   # cen_offset x (column)
        outputs[1][0, 1, 10, 20] = 0.5    # cen_offset y (row)

        det = postprocessor.process(outputs, ctx)[0]

        assert det.x3d == pytest.approx(10.5 * 50.0 / 152)
        assert det.y3d == pytest.approx(-25.0 + 20.25 * 50.0 / 152)

    def test_a_neighbour_of_a_stronger_peak_is_suppressed(self, ctx):
        postprocessor = SFA3DPostprocessor(608, 608)
        outputs = _make_outputs()
        outputs[0].fill(-10.0)
        outputs[0][0, 1, 10, 20] = 10.0
        outputs[0][0, 1, 10, 21] = 9.0    # above threshold, but not a 3x3 maximum
        outputs[0][0, 1, 40, 40] = 8.0    # a separate peak

        results = postprocessor.process(outputs, ctx)

        assert [(round(d.x3d, 3), round(d.y3d, 3)) for d in results] == [
            (round(10 * 50 / 152, 3), round(-25 + 20 * 50 / 152, 3)),
            (round(40 * 50 / 152, 3), round(-25 + 40 * 50 / 152, 3)),
        ]

    def test_top_k_is_taken_over_all_classes_together(self, ctx):
        postprocessor = SFA3DPostprocessor(608, 608, {"max_detections": 2})
        outputs = _make_outputs()
        outputs[0].fill(-10.0)
        outputs[0][0, 0, 10, 10] = 9.0
        outputs[0][0, 1, 50, 50] = 10.0
        outputs[0][0, 2, 90, 90] = 8.0

        results = postprocessor.process(outputs, ctx)

        assert [d.class_id for d in results] == [1, 0]

    def test_decodes_negative_logit_peak_above_threshold(self, ctx):
        postprocessor = SFA3DPostprocessor(608, 608)
        outputs = _make_outputs()
        outputs[0].fill(-10.0)
        y, x = 10, 20
        outputs[0][0, 0, y, x] = -0.5   # class 0 (Pedestrian): sigmoid(-0.5) > 0.3 threshold
        outputs[4][0, 1, y, x] = 8.0
        outputs[4][0, 2, y, x] = 16.0

        results = postprocessor.process(outputs, ctx)

        assert len(results) == 1
        det = results[0]
        assert isinstance(det, Detection3DResult)
        assert det.class_id == 0
        assert det.class_name == "Pedestrian"
        assert det.confidence == pytest.approx(1.0 / (1.0 + math.exp(0.5)), rel=1e-6)
        assert det.bev_x == pytest.approx(80.0)
        assert det.bev_y == pytest.approx(40.0)


class TestConvertCppSFA3D:
    def test_convert_cpp_sfa3d_maps_rows_to_detection_results(self):
        detections = np.array(
            [
                [80.0, 40.0, 2.5, 1.5, 8.0, 16.0, math.pi / 2.0, 0.95, 2.0],
            ],
            dtype=np.float32,
        )

        results = convert_cpp_sfa3d(detections)

        assert len(results) == 1
        det = results[0]
        assert isinstance(det, DetectionResult)
        assert det.class_id == 2
        assert det.confidence == pytest.approx(0.95)
        np.testing.assert_allclose(det.box, [72.0, 36.0, 88.0, 44.0], atol=1e-6)

    def test_convert_cpp_sfa3d_keeps_model_space_boxes_with_ctx(self, ctx):
        detections = np.array(
            [
                [80.0, 40.0, 0.0, 1.5, 8.0, 16.0, 0.0, 0.95, 1.0],
            ],
            dtype=np.float32,
        )

        results = convert_cpp_sfa3d(detections, ctx)

        assert len(results) == 1
        np.testing.assert_allclose(results[0].box, [72.0, 36.0, 88.0, 44.0], atol=1e-6)


class TestPointcloudToBev:
    """maudzung/SFA3D makeBEVMap, the encoding the model was trained on."""

    def test_highest_point_rules_its_cell(self):
        from common.processors.sfa3d_bev_preprocessor import pointcloud_to_bev
        pts = np.array([
            [10.0, 0.0, 0.0, 0.5],     # top of a 2-point cell
            [10.0, 0.0, -1.0, 0.9],    # lower: its intensity must not win
            [5.0, -25.0, -2.7, 0.2],   # y == y_min -> column 0
            [20.0, 3.0, 1.0, 0.3],     # tie on z with the next: the first stays
            [20.0, 3.0, 1.0, 0.7],
            [50.0, 0.0, 0.0, 1.0],     # x == x_max: row 608, cropped as upstream
            [60.0, 0.0, 0.0, 1.0],     # outside the window
        ], dtype=np.float32)
        bev = pointcloud_to_bev(pts)
        res = np.float32(50.0 / 608)
        assert bev.shape == (608, 608, 3) and bev.dtype == np.uint8
        row = int(np.floor(np.float32(10.0) / res))
        assert row == 121   # forward runs DOWN the raster
        assert tuple(bev[row, 304]) == (128, 174, 67)
        # log(2) / log(64) * 255 = 42.50000000000001 in float64 -> 43
        assert tuple(bev[int(np.floor(np.float32(5.0) / res)), 0]) == (51, 2, 43)
        c = bev[int(np.floor(np.float32(20.0) / res)), int(np.floor(np.float32(3.0) / res)) + 304]
        assert c[0] == 77
        assert np.count_nonzero(bev.any(axis=2)) == 3
