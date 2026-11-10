"""A classifier head with a leading background row (TF-slim: 1001 outputs)."""
import numpy as np

from common.processors.classification_postprocessor import ClassificationPostprocessor


def _logits():
    x = np.zeros(1001, dtype=np.float32)
    x[6] = 8.0      # ImageNet 5
    x[1000] = 7.0   # ImageNet 999
    return [x[None]]


def test_background_row_is_dropped_and_ids_are_imagenet(ctx):
    post = ClassificationPostprocessor(224, 224, {"top_k": 2, "background_class": True})
    res = post.process(_logits(), ctx)
    assert [r.class_id for r in res] == [5, 999]
    z = np.exp(8.0) + np.exp(7.0) + 999.0      # softmax over all 1001 rows
    assert abs(res[0].confidence - np.exp(8.0) / z) < 1e-6
    assert res[0].class_name                   # an ImageNet name, not empty


def test_without_the_flag_nothing_moves(ctx):
    post = ClassificationPostprocessor(224, 224, {"top_k": 1})
    assert post.process(_logits(), ctx)[0].class_id == 6
