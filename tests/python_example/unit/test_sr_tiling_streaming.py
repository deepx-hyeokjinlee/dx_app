"""run_and_assemble_tiles stitches each tile as soon as its output arrives.

The collect-then-assemble path (run_tiles_pipelined + assemble_tiles) holds
every tile output of a frame until the last one is done: 25,560 outputs,
~0.5 GB, for a 1920x1080 frame through ESPCN x4. Streaming keeps only the
in-flight window alive, and the image must not change by a single pixel.

Mirrors the C++ cases in src/cpp_example/common/test/common_unit_test.cpp.
"""
import weakref

import numpy as np
import pytest

from common.runner.sr_tiling import (
    assemble_tiles, plan_tiles, run_and_assemble_tiles, run_tiles_pipelined)

TILE = 17
SCALE = 2


class _Output(list):
    """wait()'s result: a list subclass, so the engine can hold a weak reference."""


class UpscalingEngine:
    """A nearest-neighbour x2 'model': a 17x17 uint8 tile -> 34x34 floats in [0, 1].

    Records, at every wait(), how many outputs it handed out earlier are still
    referenced by the caller. fail_wait lists job ids whose wait() raises.
    """

    def __init__(self, fail_wait=()):
        self.jobs = {}
        self.next_id = 0
        self.handed_out = []
        self.max_alive = 0
        self.fail_wait = set(fail_wait)

    def run_async(self, tensors):
        job = self.next_id
        self.next_id += 1
        self.jobs[job] = np.array(tensors[0][:, :, 0], copy=True)
        return job

    def wait(self, job_id):
        tile = self.jobs.pop(job_id)
        alive = sum(1 for ref in self.handed_out if ref() is not None)
        self.max_alive = max(self.max_alive, alive)
        if job_id in self.fail_wait:
            raise RuntimeError("wait failed")
        up = np.kron(tile.astype(np.float32) / 255.0, np.ones((SCALE, SCALE), np.float32))
        out = _Output([up[np.newaxis, np.newaxis]])
        self.handed_out.append(weakref.ref(out))
        return out


def _frame(h=40, w=61, halo=4):
    padded_h, padded_w, plans = plan_tiles(h, w, TILE, TILE, halo)
    plane = np.random.default_rng(0).integers(0, 256, (padded_h, padded_w), dtype=np.uint8)
    return padded_h, padded_w, plans, plane


def _collected(engine, padded_h, padded_w, plans, plane):
    outputs = run_tiles_pipelined(engine, lambda t: t, plane, plans, TILE, TILE)
    return assemble_tiles(plans, outputs, padded_h * SCALE, padded_w * SCALE, SCALE, SCALE)


def _streamed(engine, padded_h, padded_w, plans, plane, inflight=16):
    return run_and_assemble_tiles(engine, lambda t: t, plane, plans, TILE, TILE,
                                  padded_h * SCALE, padded_w * SCALE, SCALE, SCALE,
                                  inflight=inflight)


def test_streaming_gives_the_same_image_as_collect_then_assemble():
    padded_h, padded_w, plans, plane = _frame()
    expected, expected_done = _collected(UpscalingEngine(), padded_h, padded_w, plans, plane)
    got, got_done = _streamed(UpscalingEngine(), padded_h, padded_w, plans, plane)
    assert got_done == expected_done == len(plans)
    np.testing.assert_array_equal(got, expected)


def test_streaming_keeps_no_finished_tile_output_alive():
    padded_h, padded_w, plans, plane = _frame()
    streaming = UpscalingEngine()
    _streamed(streaming, padded_h, padded_w, plans, plane, inflight=4)
    assert streaming.max_alive == 0
    # The collecting path holds every earlier output: the cost streaming removes.
    collecting = UpscalingEngine()
    _collected(collecting, padded_h, padded_w, plans, plane)
    assert collecting.max_alive == len(plans) - 1


def test_a_tile_whose_wait_fails_stays_black_and_is_not_counted():
    padded_h, padded_w, plans, plane = _frame()
    expected, expected_done = _collected(UpscalingEngine(fail_wait={3}), padded_h, padded_w,
                                         plans, plane)
    got, got_done = _streamed(UpscalingEngine(fail_wait={3}), padded_h, padded_w, plans, plane)
    assert got_done == expected_done == len(plans) - 1
    np.testing.assert_array_equal(got, expected)


class FailingSubmitEngine(UpscalingEngine):
    def __init__(self, fail_at):
        super().__init__()
        self.fail_at = fail_at
        self.waited = []

    def run_async(self, tensors):
        if self.next_id == self.fail_at:
            raise RuntimeError("submit failed")
        return super().run_async(tensors)

    def wait(self, job_id):
        self.waited.append(job_id)
        return super().wait(job_id)


def test_a_failed_submit_waits_for_the_jobs_already_in_flight():
    padded_h, padded_w, plans, plane = _frame()
    engine = FailingSubmitEngine(fail_at=2)
    with pytest.raises(RuntimeError, match="submit failed"):
        _streamed(engine, padded_h, padded_w, plans, plane)
    assert engine.waited == [0, 1]
