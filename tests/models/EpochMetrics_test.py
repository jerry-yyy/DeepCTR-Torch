# -*- coding: utf-8 -*-
"""Training metrics must be computed on the whole epoch, not averaged over batches.

The reference is always scikit-learn / numpy applied to every prediction the model made during the
epoch. Predictions are captured with a forward hook and matched to their labels through a unique
input feature, so the reference does not depend on the code under test or on the batch order.
"""
import warnings

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score

from deepctr_torch.inputs import DenseFeat, SparseFeat
from deepctr_torch.models import DeepFM, MMOE

SAMPLE_SIZE = 10
# batch sizes: 1 (every batch single-class), incomplete last batches (3, 4, 7), one full batch, larger than the data
BATCH_SIZES = [1, 3, 4, 7, 10, 32]
# with batch_size=4 and shuffle=False the second batch contains a single class
BINARY_LABELS = np.array([0, 1, 0, 1, 0, 0, 0, 0, 1, 0], dtype="float64")


def _features():
    return [SparseFeat("user", 10, embedding_dim=4), SparseFeat("item", 12, embedding_dim=4), DenseFeat("row", 1)]


def _inputs(sample_size=SAMPLE_SIZE, seed=0):
    rng = np.random.RandomState(seed)
    # "row" is unique per sample and exactly representable in float32: it identifies the sample inside a batch
    return {"user": rng.randint(0, 10, sample_size), "item": rng.randint(0, 12, sample_size),
            "row": np.arange(sample_size) / 64.0}


def _fit_and_capture(model, x, y, **fit_kwargs):
    """Fit for one epoch and return (history, labels, predictions) in the order the model saw the samples."""
    row_column = model.feature_index["row"][0]
    rows, predictions = [], []

    def hook(module, inputs, output):
        rows.append(np.rint(inputs[0][:, row_column].detach().cpu().numpy() * 64.0).astype(int))
        predictions.append(output.detach().cpu().numpy().astype("float64"))

    handle = model.register_forward_hook(hook)
    try:
        history = model.fit(x, y, epochs=1, **fit_kwargs).history
    finally:
        handle.remove()
    rows = np.concatenate(rows)
    assert sorted(rows.tolist()) == list(range(len(y)))  # every sample was seen exactly once
    return history, np.asarray(y)[rows], np.concatenate(predictions)


def _reference_logloss(y_true, y_pred, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def _mmoe(task_types, task_names):
    return MMOE(_features(), task_types=list(task_types), task_names=list(task_names), device="cpu")


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("shuffle", [False, True])
def test_single_task_train_metrics_equal_whole_epoch_reference(batch_size, shuffle):
    model = DeepFM(_features(), _features(), device="cpu")
    model.compile("adam", "binary_crossentropy", metrics=["auc", "binary_crossentropy", "acc", "mse"])

    history, y_seen, pred = _fit_and_capture(model, _inputs(), BINARY_LABELS, batch_size=batch_size, verbose=2,
                                             shuffle=shuffle)

    assert history["auc"][0] == roc_auc_score(y_seen, pred[:, 0])  # exactly the whole-epoch AUC
    assert history["binary_crossentropy"][0] == pytest.approx(_reference_logloss(y_seen, pred[:, 0]), abs=1e-9)
    assert history["acc"][0] == pytest.approx(accuracy_score(y_seen, pred[:, 0] > 0.5), abs=1e-12)
    assert history["mse"][0] == pytest.approx(mean_squared_error(y_seen, pred[:, 0]), abs=1e-12)


def test_single_class_batches_do_not_turn_epoch_auc_into_nan():
    model = DeepFM(_features(), _features(), device="cpu")
    model.compile("adam", "binary_crossentropy", metrics=["auc"])

    history, y_seen, pred = _fit_and_capture(model, _inputs(), BINARY_LABELS, batch_size=4, verbose=2, shuffle=False)

    assert len(np.unique(BINARY_LABELS[4:8])) == 1  # the scenario really contains a single-class batch
    assert np.isfinite(history["auc"][0])
    assert history["auc"][0] == roc_auc_score(y_seen, pred[:, 0])


def test_regression_metric_is_not_biased_by_incomplete_last_batch():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 5, 5], dtype="float64")  # the short last batch holds the large errors
    model = DeepFM(_features(), _features(), task="regression", device="cpu")
    model.compile("adam", "mse", metrics=["mse"])

    history, y_seen, pred = _fit_and_capture(model, _inputs(), y, batch_size=4, verbose=2, shuffle=False)

    batch_mean = np.mean([mean_squared_error(y_seen[i:i + 4], pred[i:i + 4, 0]) for i in range(0, SAMPLE_SIZE, 4)])
    reference = mean_squared_error(y_seen, pred[:, 0])
    assert abs(batch_mean - reference) > 1.0  # averaging batch values would be clearly wrong here
    assert history["mse"][0] == pytest.approx(reference, abs=1e-12)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_multi_task_list_metrics_are_pooled_over_the_whole_epoch(batch_size):
    y = np.stack([BINARY_LABELS, BINARY_LABELS[::-1]], axis=1)
    model = _mmoe(("binary", "binary"), ("ctr", "ctcvr"))
    model.compile("adam", ["binary_crossentropy"] * 2, metrics=["auc", "mse"])

    history, y_seen, pred = _fit_and_capture(model, _inputs(), y, batch_size=batch_size, verbose=2, shuffle=True)

    assert history["auc"][0] == roc_auc_score(y_seen.reshape(-1), pred.reshape(-1))
    assert history["mse"][0] == pytest.approx(mean_squared_error(y_seen.reshape(-1), pred.reshape(-1)), abs=1e-12)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("shuffle", [False, True])
def test_per_task_train_metrics_equal_whole_epoch_reference_for_mixed_tasks(batch_size, shuffle):
    rating = np.array([0, 0, 0, 0, 0, 0, 0, 0, 5, 5], dtype="float64")
    y = np.stack([BINARY_LABELS, rating, BINARY_LABELS[::-1]], axis=1)
    model = _mmoe(("binary", "regression", "binary"), ("like", "rating", "share"))
    model.compile("adam", ["binary_crossentropy", "mse", "binary_crossentropy"],
                  metrics={"like": ["auc", "logloss", "acc"], "rating": ["mse", "mae"], "share": ["auc"]})

    history, y_seen, pred = _fit_and_capture(model, _inputs(), y, batch_size=batch_size, verbose=0, shuffle=shuffle)

    assert history["like_auc"][0] == roc_auc_score(y_seen[:, 0], pred[:, 0])
    assert history["share_auc"][0] == roc_auc_score(y_seen[:, 2], pred[:, 2])
    assert history["like_logloss"][0] == pytest.approx(_reference_logloss(y_seen[:, 0], pred[:, 0]), abs=1e-9)
    assert history["like_acc"][0] == pytest.approx(accuracy_score(y_seen[:, 0], pred[:, 0] > 0.5), abs=1e-12)
    assert history["rating_mse"][0] == pytest.approx(mean_squared_error(y_seen[:, 1], pred[:, 1]), abs=1e-12)
    assert history["rating_mae"][0] == pytest.approx(mean_absolute_error(y_seen[:, 1], pred[:, 1]), abs=1e-12)


def test_only_one_task_has_metrics_and_other_columns_are_not_needed():
    y = np.stack([BINARY_LABELS, np.linspace(0.0, 5.0, SAMPLE_SIZE), BINARY_LABELS[::-1]], axis=1)
    model = _mmoe(("binary", "regression", "binary"), ("like", "rating", "share"))
    model.compile("adam", ["binary_crossentropy", "mse", "binary_crossentropy"], metrics={"share": ["auc", "logloss"]})

    history, y_seen, pred = _fit_and_capture(model, _inputs(), y, batch_size=4, verbose=0, shuffle=True)

    assert set(history) == {"loss", "share_auc", "share_logloss"}
    assert history["share_auc"][0] == roc_auc_score(y_seen[:, 2], pred[:, 2])
    assert history["share_logloss"][0] == pytest.approx(_reference_logloss(y_seen[:, 2], pred[:, 2]), abs=1e-9)


def test_task_with_single_class_epoch_is_nan_and_does_not_affect_other_tasks():
    y = np.stack([np.zeros(SAMPLE_SIZE), BINARY_LABELS], axis=1)  # "ctr" never has a positive label
    model = _mmoe(("binary", "binary"), ("ctr", "ctcvr"))
    model.compile("adam", ["binary_crossentropy"] * 2, metrics={"ctr": ["auc", "logloss"], "ctcvr": ["auc"]})

    with pytest.warns(UserWarning, match="ctr") as record:
        history, y_seen, pred = _fit_and_capture(model, _inputs(), y, batch_size=4, verbose=0, shuffle=False)

    assert np.isnan(history["ctr_auc"][0])
    assert history["ctcvr_auc"][0] == roc_auc_score(y_seen[:, 1], pred[:, 1])
    assert history["ctr_logloss"][0] == pytest.approx(_reference_logloss(y_seen[:, 0], pred[:, 0]), abs=1e-9)
    assert np.isfinite(history["loss"][0])
    assert not any("ctcvr" in str(warning.message) for warning in record)


def test_single_task_single_class_epoch_auc_is_nan_with_warning():
    model = DeepFM(_features(), _features(), device="cpu")
    model.compile("adam", "binary_crossentropy", metrics=["auc", "mse"])

    with pytest.warns(UserWarning, match="only one class"):
        history, y_seen, pred = _fit_and_capture(model, _inputs(), np.ones(SAMPLE_SIZE), batch_size=4, verbose=2,
                                                 shuffle=False)

    assert np.isnan(history["auc"][0])
    assert history["mse"][0] == pytest.approx(mean_squared_error(y_seen, pred[:, 0]), abs=1e-12)
    assert np.isfinite(history["loss"][0])


def test_single_class_evaluation_auc_is_nan_with_warning_for_list_metrics():
    model = DeepFM(_features(), _features(), device="cpu")
    model.compile("adam", "binary_crossentropy", metrics=["auc"])

    with pytest.warns(UserWarning, match="only one class"):
        result = model.evaluate(_inputs(), np.zeros(SAMPLE_SIZE))

    assert np.isnan(result["auc"])


def test_list_metrics_are_still_skipped_during_training_when_verbose_is_zero():
    model = DeepFM(_features(), _features(), device="cpu")
    model.compile("adam", "binary_crossentropy", metrics=["auc"])

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing is computed, so nothing can warn
        history = model.fit(_inputs(), np.ones(SAMPLE_SIZE), batch_size=4, epochs=1, verbose=0).history

    assert set(history) == {"loss"}


def test_train_metrics_are_reset_between_epochs():
    model = DeepFM(_features(), _features(), task="regression", device="cpu")
    model.compile("adam", "mse", metrics=["mse"])
    x, y = _inputs(), np.linspace(0.0, 5.0, SAMPLE_SIZE)
    per_epoch = []
    handle = model.register_forward_hook(
        lambda module, inputs, output: per_epoch.append(output.detach().cpu().numpy().astype("float64")))

    history = model.fit(x, y, batch_size=SAMPLE_SIZE, epochs=3, verbose=2, shuffle=False).history
    handle.remove()

    assert len(per_epoch) == 3
    for epoch in range(3):
        assert history["mse"][epoch] == pytest.approx(mean_squared_error(y, per_epoch[epoch][:, 0]), abs=1e-12)
