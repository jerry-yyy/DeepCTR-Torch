# -*- coding: utf-8 -*-
"""Tests for per-task metrics (``compile(metrics={task_name: [...]})``).

Reference values are computed with scikit-learn / numpy on each label column
separately, never with the helpers under test.
"""
import os
import warnings

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score

from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from deepctr_torch.inputs import DenseFeat, SparseFeat
from deepctr_torch.models import DeepFM, ESMM, MMOE, PLE, SharedBottom


def _data(sample_size=45, task_types=("binary", "regression"), seed=0):
    rng = np.random.RandomState(seed)
    feature_columns = [SparseFeat("user", 10, embedding_dim=4), SparseFeat("item", 12, embedding_dim=4),
                       DenseFeat("price", 1)]
    x = {"user": rng.randint(0, 10, sample_size), "item": rng.randint(0, 12, sample_size),
         "price": rng.random_sample(sample_size)}
    columns = []
    for task_type in task_types:
        if task_type == "binary":
            label = rng.randint(0, 2, sample_size).astype("float64")
            label[:2] = [0.0, 1.0][:sample_size]  # both classes whenever sample_size >= 2
            columns.append(label)
        else:
            columns.append(rng.random_sample(sample_size) * 5.0)
    return x, np.stack(columns, axis=1), feature_columns


def _losses(task_types):
    return ["binary_crossentropy" if task_type == "binary" else "mse" for task_type in task_types]


def _mmoe(feature_columns, task_types=("binary", "regression"), task_names=("like", "watch"), device="cpu"):
    return MMOE(feature_columns, task_types=list(task_types), task_names=list(task_names), device=device)


def _reference_logloss(y_true, y_pred, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def _fixed_predictions(model, monkeypatch, predictions):
    monkeypatch.setattr(model, "predict", lambda x, batch_size=256: np.asarray(predictions, dtype="float64"))


def test_mixed_tasks_evaluate_matches_per_column_sklearn():
    x, y, feature_columns = _data()
    model = _mmoe(feature_columns)
    model.compile("adam", loss=["binary_crossentropy", "mse"],
                  metrics={"like": ["auc", "logloss", "accuracy"], "watch": ["mse", "mae"]})

    result = model.evaluate(x, y, batch_size=32)
    pred = model.predict(x, 32)

    assert set(result) == {"like_auc", "like_logloss", "like_accuracy", "watch_mse", "watch_mae"}
    assert result["like_auc"] == pytest.approx(roc_auc_score(y[:, 0], pred[:, 0]), abs=1e-12)
    assert result["like_logloss"] == pytest.approx(_reference_logloss(y[:, 0], pred[:, 0]), abs=1e-9)
    assert result["like_accuracy"] == pytest.approx(accuracy_score(y[:, 0], pred[:, 0] > 0.5), abs=1e-12)
    assert result["watch_mse"] == pytest.approx(mean_squared_error(y[:, 1], pred[:, 1]), abs=1e-12)
    assert result["watch_mae"] == pytest.approx(mean_absolute_error(y[:, 1], pred[:, 1]), abs=1e-12)


def test_two_binary_tasks_report_per_task_auc_not_pooled_auc(monkeypatch):
    y = np.array([[0, 0], [1, 1]], dtype="float64")
    pred = np.array([[0.01, 0.80], [0.02, 0.90]], dtype="float64")
    assert roc_auc_score(y.reshape(-1), pred.reshape(-1)) == pytest.approx(0.75)  # pooled value, a different quantity

    x, _, feature_columns = _data(sample_size=2, task_types=("binary", "binary"))
    model = _mmoe(feature_columns, ("binary", "binary"), ("ctr", "ctcvr"))
    model.compile("adam", loss=["binary_crossentropy"] * 2, metrics={"ctr": ["auc"], "ctcvr": ["auc"]})
    _fixed_predictions(model, monkeypatch, pred)

    assert model.evaluate(x, y) == {"ctr_auc": 1.0, "ctcvr_auc": 1.0}


def test_mixed_fixed_predictions_from_issue_example(monkeypatch):
    y = np.array([[0, 4.2], [1, 8.5], [0, 2.3], [1, 7.6]])
    pred = np.array([[0.1, 4.0], [0.8, 8.0], [0.2, 2.0], [0.9, 7.8]])
    x, _, feature_columns = _data(sample_size=4)
    model = _mmoe(feature_columns)
    model.compile("adam", loss=["binary_crossentropy", "mse"], metrics={"like": ["auc"], "watch": ["mse"]})
    _fixed_predictions(model, monkeypatch, pred)

    result = model.evaluate(x, y)

    assert result["like_auc"] == pytest.approx(1.0)
    assert result["watch_mse"] == pytest.approx(0.105)


@pytest.mark.parametrize("batch_size", [1, 7, 32, 256])
def test_evaluate_is_independent_of_batch_size(batch_size):
    x, y, feature_columns = _data(sample_size=45)  # 45 is not a multiple of 7 or 32
    model = _mmoe(feature_columns)
    model.compile("adam", loss=["binary_crossentropy", "mse"], metrics={"like": ["auc", "logloss"], "watch": ["mse"]})
    reference_pred = model.predict(x, 45)

    result = model.evaluate(x, y, batch_size=batch_size)

    assert result["like_auc"] == pytest.approx(roc_auc_score(y[:, 0], reference_pred[:, 0]), abs=1e-12)
    assert result["like_logloss"] == pytest.approx(_reference_logloss(y[:, 0], reference_pred[:, 0]), abs=1e-6)
    assert result["watch_mse"] == pytest.approx(mean_squared_error(y[:, 1], reference_pred[:, 1]), abs=1e-6)


def test_metric_columns_follow_task_names_not_dict_order():
    x, y, feature_columns = _data()
    model = _mmoe(feature_columns)
    model.compile("adam", loss=["binary_crossentropy", "mse"], metrics={"watch": ["mse"], "like": ["auc"]})

    result = model.evaluate(x, y)
    pred = model.predict(x)

    assert result["like_auc"] == pytest.approx(roc_auc_score(y[:, 0], pred[:, 0]), abs=1e-12)
    assert result["watch_mse"] == pytest.approx(mean_squared_error(y[:, 1], pred[:, 1]), abs=1e-12)


def test_three_tasks_with_one_task_left_without_metrics():
    task_types = ("binary", "regression", "binary")
    x, y, feature_columns = _data(task_types=task_types)
    model = _mmoe(feature_columns, task_types, ("click", "watch", "share"))
    model.compile("adam", loss=_losses(task_types), metrics={"share": ["auc"], "watch": ["mae"]})

    result = model.evaluate(x, y)
    pred = model.predict(x)

    assert set(result) == {"watch_mae", "share_auc"}
    assert result["share_auc"] == pytest.approx(roc_auc_score(y[:, 2], pred[:, 2]), abs=1e-12)
    assert result["watch_mae"] == pytest.approx(mean_absolute_error(y[:, 1], pred[:, 1]), abs=1e-12)


@pytest.mark.parametrize("metrics, message", [
    ({"like": ["auc"], "wacth": ["mse"]}, "wacth"),
    ({"like": ["f1"]}, "f1"),
    ({"watch": ["auc"]}, "regression"),
    ({"watch": ["logloss"]}, "regression"),
    ({"watch": ["accuracy"]}, "regression"),
    ({"like": ["auc", "auc"]}, "like_auc"),
    ({"like": [roc_auc_score]}, "string"),
    ({"like": "auc"}, "list"),
])
def test_invalid_task_metric_configuration_is_rejected(metrics, message):
    _, _, feature_columns = _data()
    model = _mmoe(feature_columns)

    with pytest.raises(ValueError, match=message):
        model.compile("adam", loss=["binary_crossentropy", "mse"], metrics=metrics)


def test_duplicate_task_names_are_rejected():
    _, _, feature_columns = _data()
    model = _mmoe(feature_columns, ("binary", "binary"), ("ctr", "ctr"))

    with pytest.raises(ValueError, match="unique"):
        model.compile("adam", loss=["binary_crossentropy"] * 2, metrics={"ctr": ["auc"]})


def test_dict_metrics_on_single_task_model_are_rejected():
    _, _, feature_columns = _data()
    model = DeepFM(feature_columns, feature_columns, device="cpu")

    with pytest.raises(ValueError, match="task_names"):
        model.compile("adam", "binary_crossentropy", metrics={"ctr": ["auc"]})


def test_evaluate_rejects_labels_with_wrong_number_of_columns():
    x, y, feature_columns = _data()
    model = _mmoe(feature_columns)
    model.compile("adam", loss=["binary_crossentropy", "mse"], metrics={"like": ["auc"]})

    with pytest.raises(ValueError, match="2 columns"):
        model.evaluate(x, y[:, 0])
    with pytest.raises(ValueError, match="2 columns"):
        model.evaluate(x, np.concatenate([y, y[:, :1]], axis=1))


def test_history_has_train_and_validation_task_metrics_with_verbose_zero():
    x, y, feature_columns = _data()
    model = _mmoe(feature_columns)
    model.compile("adam", loss=["binary_crossentropy", "mse"], metrics={"like": ["auc"], "watch": ["mse", "mae"]})

    history = model.fit(x, y, batch_size=16, epochs=2, verbose=0, validation_data=(x, y)).history

    expected = {"loss", "like_auc", "watch_mse", "watch_mae", "val_like_auc", "val_watch_mse", "val_watch_mae"}
    assert set(history) == expected
    assert all(len(values) == 2 for values in history.values())


def test_train_task_metrics_use_all_epoch_predictions_not_batch_average():
    x, y, feature_columns = _data(sample_size=45)
    model = _mmoe(feature_columns)
    model.compile("adam", loss=["binary_crossentropy", "mse"], metrics={"like": ["auc"], "watch": ["mse"]})
    seen = []
    handle = model.register_forward_hook(lambda module, inputs, output: seen.append(output.detach().cpu().numpy()))

    history = model.fit(x, y, batch_size=7, epochs=1, verbose=0, shuffle=False).history
    handle.remove()
    epoch_pred = np.concatenate(seen).astype("float64")

    assert epoch_pred.shape == y.shape
    assert history["like_auc"][0] == pytest.approx(roc_auc_score(y[:, 0], epoch_pred[:, 0]), abs=1e-12)
    assert history["watch_mse"][0] == pytest.approx(mean_squared_error(y[:, 1], epoch_pred[:, 1]), abs=1e-9)


def test_single_class_batch_does_not_abort_training():
    x, y, feature_columns = _data(sample_size=8, task_types=("binary", "binary"))
    y[:, :] = 0.0
    y[0, :] = 1.0  # both classes in the epoch, single-class in three of four batches
    model = _mmoe(feature_columns, ("binary", "binary"), ("ctr", "ctcvr"))
    model.compile("adam", loss=["binary_crossentropy"] * 2, metrics={"ctr": ["auc"], "ctcvr": ["auc", "logloss"]})

    history = model.fit(x, y, batch_size=2, epochs=1, verbose=2, shuffle=False).history

    assert np.isfinite(history["ctr_auc"][0])
    assert np.isfinite(history["ctcvr_logloss"][0])


def test_auc_on_single_class_evaluation_set_is_nan_with_task_warning():
    x, y, feature_columns = _data(task_types=("binary", "binary"))
    y[:, 1] = 0.0
    model = _mmoe(feature_columns, ("binary", "binary"), ("ctr", "ctcvr"))
    model.compile("adam", loss=["binary_crossentropy"] * 2, metrics={"ctr": ["auc"], "ctcvr": ["auc", "logloss"]})
    pred = model.predict(x)

    with pytest.warns(UserWarning, match="ctcvr"):
        result = model.evaluate(x, y)

    assert np.isnan(result["ctcvr_auc"])
    assert result["ctr_auc"] == pytest.approx(roc_auc_score(y[:, 0], pred[:, 0]), abs=1e-12)
    assert result["ctcvr_logloss"] == pytest.approx(_reference_logloss(y[:, 1], pred[:, 1]), abs=1e-9)


def test_callbacks_never_treat_nan_metric_as_best(tmp_path):
    x, y, feature_columns = _data(task_types=("binary", "binary"))
    val_y = y.copy()
    val_y[:, 0] = 1.0  # validation AUC of task "ctr" is undefined
    model = _mmoe(feature_columns, ("binary", "binary"), ("ctr", "ctcvr"))
    model.compile("adam", loss=["binary_crossentropy"] * 2, metrics={"ctr": ["auc"]})
    early_stopping = EarlyStopping(monitor="val_ctr_auc", mode="max", patience=2)
    filepath = str(tmp_path / "best.ckpt")
    checkpoint = ModelCheckpoint(filepath, monitor="val_ctr_auc", mode="max", save_best_only=True,
                                 save_weights_only=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        history = model.fit(x, y, batch_size=16, epochs=5, verbose=0, validation_data=(x, val_y),
                            callbacks=[early_stopping, checkpoint]).history

    assert np.isnan(history["val_ctr_auc"]).all()
    assert len(history["val_ctr_auc"]) == 2  # stopped by patience, no epoch counted as an improvement
    assert early_stopping.best == -np.inf
    assert not os.path.exists(filepath)


def test_early_stopping_and_checkpoint_monitor_task_metric(tmp_path):
    x, y, feature_columns = _data()
    model = _mmoe(feature_columns)
    model.compile("adam", loss=["binary_crossentropy", "mse"], metrics={"like": ["auc"], "watch": ["mse"]})
    early_stopping = EarlyStopping(monitor="val_like_auc", patience=1)  # mode="auto" must maximise *_auc
    filepath = str(tmp_path / "best.ckpt")
    # the whole model (including the compiled metric functions) must stay picklable
    checkpoint = ModelCheckpoint(filepath, monitor="val_watch_mse", save_best_only=True, save_weights_only=False)

    history = model.fit(x, y, batch_size=16, epochs=3, verbose=0, validation_data=(x, y),
                        callbacks=[early_stopping, checkpoint]).history

    assert early_stopping.monitor_op == np.greater
    assert early_stopping.best == max(history["val_like_auc"])
    assert checkpoint.best == min(history["val_watch_mse"])
    assert os.path.exists(filepath)


def _build(model_class, feature_columns, task_types, task_names):
    return model_class(feature_columns, task_types=list(task_types), task_names=list(task_names), device="cpu")


@pytest.mark.parametrize("model_class, task_types", [
    (SharedBottom, ("binary", "binary")),
    (SharedBottom, ("binary", "regression")),
    (MMOE, ("binary", "binary")),
    (MMOE, ("binary", "regression")),
    (PLE, ("binary", "binary")),
    (PLE, ("binary", "regression")),
    (ESMM, ("binary", "binary")),
])
def test_task_metrics_across_multitask_models(model_class, task_types):
    x, y, feature_columns = _data(sample_size=33, task_types=task_types)  # batch_size=16 leaves a last batch of 1
    task_names = ("first", "second")
    model = _build(model_class, feature_columns, task_types, task_names)
    second_metrics = ["auc"] if task_types[1] == "binary" else ["mse", "mae"]
    model.compile("adam", loss=_losses(task_types), metrics={"first": ["auc", "logloss"], "second": second_metrics})

    history = model.fit(x, y, batch_size=16, epochs=1, verbose=0, validation_data=(x, y)).history
    result = model.evaluate(x, y, batch_size=16)
    pred = model.predict(x, 16)

    assert set(history) == {"loss"} | set(result) | {"val_" + name for name in result}
    assert result["first_auc"] == pytest.approx(roc_auc_score(y[:, 0], pred[:, 0]), abs=1e-12)
    assert result["first_logloss"] == pytest.approx(_reference_logloss(y[:, 0], pred[:, 0]), abs=1e-9)
    if task_types[1] == "binary":
        assert result["second_auc"] == pytest.approx(roc_auc_score(y[:, 1], pred[:, 1]), abs=1e-12)
    else:
        assert result["second_mse"] == pytest.approx(mean_squared_error(y[:, 1], pred[:, 1]), abs=1e-12)
        assert result["second_mae"] == pytest.approx(mean_absolute_error(y[:, 1], pred[:, 1]), abs=1e-12)
    for name, value in result.items():
        assert history["val_" + name][0] == pytest.approx(value, abs=1e-12)


@pytest.mark.parametrize("sample_size, batch_size", [(1, 1), (5, 1)])
def test_task_metrics_with_tiny_inputs(sample_size, batch_size):
    x, y, feature_columns = _data(sample_size=sample_size)
    model = _mmoe(feature_columns)
    model.compile("adam", loss=["binary_crossentropy", "mse"], metrics={"like": ["logloss"], "watch": ["mse"]})

    history = model.fit(x, y, batch_size=batch_size, epochs=1, verbose=0).history
    result = model.evaluate(x, y, batch_size=batch_size)
    pred = model.predict(x, batch_size)

    assert np.isfinite(history["like_logloss"][0]) and np.isfinite(history["watch_mse"][0])
    assert result["like_logloss"] == pytest.approx(_reference_logloss(y[:, 0], pred[:, 0]), abs=1e-9)
    assert result["watch_mse"] == pytest.approx(mean_squared_error(y[:, 1], pred[:, 1]), abs=1e-12)


def test_legacy_list_metrics_keep_pooled_values_and_history_keys():
    x, y, feature_columns = _data(task_types=("binary", "binary"))
    model = _mmoe(feature_columns, ("binary", "binary"), ("ctr", "ctcvr"))
    model.compile("adam", loss=["binary_crossentropy"] * 2, metrics=["auc", "mse"])

    silent = model.fit(x, y, batch_size=16, epochs=1, verbose=0, validation_data=(x, y)).history
    verbose = model.fit(x, y, batch_size=16, epochs=1, verbose=2, validation_data=(x, y)).history
    result = model.evaluate(x, y)
    pred = model.predict(x)

    assert set(silent) == {"loss", "val_auc", "val_mse"}
    assert set(verbose) == {"loss", "auc", "mse", "val_auc", "val_mse"}
    assert set(result) == {"auc", "mse"}
    assert result["auc"] == pytest.approx(roc_auc_score(y.reshape(-1), pred.reshape(-1)), abs=1e-12)
    assert result["mse"] == pytest.approx(mean_squared_error(y.reshape(-1), pred.reshape(-1)), abs=1e-12)


def test_legacy_single_task_api_is_unchanged():
    x, y, feature_columns = _data(task_types=("binary",))
    model = DeepFM(feature_columns, feature_columns, device="cpu")
    model.compile("adam", "binary_crossentropy", metrics=["auc", "binary_crossentropy", "acc"])

    history = model.fit(x, y, batch_size=16, epochs=1, verbose=2, validation_data=(x, y)).history
    result = model.evaluate(x, y)
    pred = model.predict(x)

    assert set(history) == {"loss", "auc", "binary_crossentropy", "acc",
                            "val_auc", "val_binary_crossentropy", "val_acc"}
    assert result["auc"] == pytest.approx(roc_auc_score(y[:, 0], pred[:, 0]), abs=1e-12)
    assert result["binary_crossentropy"] == pytest.approx(_reference_logloss(y[:, 0], pred[:, 0]), abs=1e-9)
    assert result["acc"] == pytest.approx(accuracy_score(y[:, 0], pred[:, 0] > 0.5), abs=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_task_metrics_on_cuda_match_sklearn():
    x, y, feature_columns = _data()
    model = _mmoe(feature_columns, device="cuda:0")
    model.compile("adam", loss=["binary_crossentropy", "mse"], metrics={"like": ["auc"], "watch": ["mse"]})

    history = model.fit(x, y, batch_size=16, epochs=1, verbose=0, validation_data=(x, y)).history
    result = model.evaluate(x, y)
    pred = model.predict(x)

    assert set(history) == {"loss", "like_auc", "watch_mse", "val_like_auc", "val_watch_mse"}
    assert result["like_auc"] == pytest.approx(roc_auc_score(y[:, 0], pred[:, 0]), abs=1e-12)
    assert result["watch_mse"] == pytest.approx(mean_squared_error(y[:, 1], pred[:, 1]), abs=1e-12)
