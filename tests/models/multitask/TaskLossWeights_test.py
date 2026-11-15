# -*- coding: utf-8 -*-
"""Tests for explicit per-task loss weights (``compile(loss_weights=...)``).

Loss values and gradients are checked against a hand-written PyTorch reference
that never calls ``fit``.
"""
import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from deepctr_torch.inputs import DenseFeat, SparseFeat
from deepctr_torch.models import DeepFM, ESMM, MMOE, PLE, SharedBottom

TASK_NAMES = ("like", "watch")


def _data(sample_size=24, task_types=("binary", "regression"), seed=0):
    rng = np.random.RandomState(seed)
    feature_columns = [SparseFeat("user", 10, embedding_dim=4), SparseFeat("item", 12, embedding_dim=4),
                       DenseFeat("price", 1)]
    x = {"user": rng.randint(0, 10, sample_size), "item": rng.randint(0, 12, sample_size),
         "price": rng.random_sample(sample_size)}
    columns = []
    for task_type in task_types:
        if task_type == "binary":
            columns.append(rng.randint(0, 2, sample_size).astype("float64"))
        else:
            columns.append(rng.random_sample(sample_size) * 5.0)
    return x, np.stack(columns, axis=1), feature_columns


def _mmoe(feature_columns, task_types=("binary", "regression"), task_names=TASK_NAMES, device="cpu", l2=0.0):
    # Regularization is off by default here so that data-loss gradients can be compared exactly.
    return MMOE(feature_columns, task_types=list(task_types), task_names=list(task_names), device=device,
                l2_reg_linear=0, l2_reg_embedding=l2, l2_reg_dnn=0)


def _sgd(model):
    return torch.optim.SGD(model.parameters(), lr=0.1)  # no momentum, no weight decay


def _input_tensor(model, x, device="cpu"):
    columns = [np.asarray(x[name]).reshape(-1, 1) for name in model.feature_index]
    return torch.from_numpy(np.concatenate(columns, axis=-1)).to(device).float()


def _reference_task_losses(model, x, y, reduction="mean", device="cpu"):
    """Per-task data losses written directly in PyTorch for a binary + regression model."""
    model.train()
    pred = model(_input_tensor(model, x, device))
    target = torch.from_numpy(y).to(device).float()
    return [F.binary_cross_entropy(pred[:, 0], target[:, 0], reduction=reduction),
            F.mse_loss(pred[:, 1], target[:, 1], reduction=reduction)]


def _reference_gradients(model, x, y, weights, device="cpu"):
    model.zero_grad()
    losses = _reference_task_losses(model, x, y, device=device)
    total = sum(weight * loss for weight, loss in zip(weights, losses))
    total.backward()
    gradients = {name: (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
                 for name, p in model.named_parameters()}
    return gradients, float(total.item())


def _fit_one_full_batch(model, x, y):
    return model.fit(x, y, batch_size=len(y), epochs=1, verbose=0, shuffle=False).history


def _assert_gradients_match(model, reference_gradients, atol=1e-7):
    checked = 0
    for name, parameter in model.named_parameters():
        gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
        assert torch.allclose(gradient, reference_gradients[name], rtol=1e-5, atol=atol), name
        checked += 1
    assert checked == len(reference_gradients)


@pytest.mark.parametrize("loss_weights, as_list", [
    ([1.0, 0.2], [1.0, 0.2]),
    ((0.3, 2.5), [0.3, 2.5]),
    ({"like": 1.0, "watch": 0.2}, [1.0, 0.2]),
    ({"watch": 0.2, "like": 1.0}, [1.0, 0.2]),  # dict order must not matter
])
def test_weighted_loss_and_gradients_match_manual_pytorch(loss_weights, as_list):
    x, y, feature_columns = _data()
    model = _mmoe(feature_columns)
    reference = copy.deepcopy(model)
    reference_gradients, reference_loss = _reference_gradients(reference, x, y, as_list)

    model.compile(_sgd(model), loss=["binary_crossentropy", "mse"], loss_weights=loss_weights)
    history = _fit_one_full_batch(model, x, y)

    assert history["loss"][0] == pytest.approx(reference_loss, rel=1e-6)
    _assert_gradients_match(model, reference_gradients)
    names = [name for name, _ in model.named_parameters()]
    assert any(name.startswith("embedding_dict") for name in names)  # shared parameters were compared
    assert any(name.startswith("tower_dnn_final_layer.1") for name in names)  # task-specific ones too
    assert reference_gradients["tower_dnn_final_layer.1.weight"].abs().sum() > 0


def test_default_weights_equal_unweighted_sum_reference():
    x, y, feature_columns = _data()
    model = _mmoe(feature_columns)
    reference_gradients, reference_loss = _reference_gradients(copy.deepcopy(model), x, y, [1.0, 1.0])

    model.compile(_sgd(model), loss=["binary_crossentropy", "mse"])
    history = _fit_one_full_batch(model, x, y)

    assert history["loss"][0] == pytest.approx(reference_loss, rel=1e-6)
    _assert_gradients_match(model, reference_gradients)


@pytest.mark.parametrize("loss_weights", [[1.0, 1.0], {"like": 1, "watch": 1}])
def test_none_and_all_ones_train_identically(loss_weights):
    x, y, feature_columns = _data(sample_size=50)

    def train(weights):
        model = _mmoe(feature_columns)
        if weights is None:
            model.compile("adam", loss=["binary_crossentropy", "mse"])
        else:
            model.compile("adam", loss=["binary_crossentropy", "mse"], loss_weights=weights)
        torch.manual_seed(7)
        history = model.fit(x, y, batch_size=16, epochs=2, verbose=0).history
        return model.state_dict(), history

    default_state, default_history = train(None)
    ones_state, ones_history = train(loss_weights)

    assert default_history["loss"] == ones_history["loss"]
    for name, value in default_state.items():
        assert torch.equal(value, ones_state[name]), name


def test_zero_weight_removes_that_task_data_loss():
    x, y, feature_columns = _data()
    model = _mmoe(feature_columns)
    reference = copy.deepcopy(model)
    reference.zero_grad()
    like_loss = _reference_task_losses(reference, x, y)[0]  # only the "like" loss, "watch" never enters
    like_loss.backward()
    reference_gradients = {name: (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
                           for name, p in reference.named_parameters()}

    model.compile(_sgd(model), loss=["binary_crossentropy", "mse"], loss_weights={"like": 1.0, "watch": 0.0})
    history = _fit_one_full_batch(model, x, y)

    assert history["loss"][0] == pytest.approx(float(like_loss.item()), rel=1e-6)
    _assert_gradients_match(model, reference_gradients)
    watch_tower = model.tower_dnn_final_layer[1].weight
    assert watch_tower.grad is None or torch.count_nonzero(watch_tower.grad) == 0


def test_regularization_is_added_once_and_not_scaled_by_task_weights():
    x, y, feature_columns = _data()
    l2 = 0.01

    def logged_loss(weights):
        model = _mmoe(feature_columns, l2=l2)
        reference = copy.deepcopy(model)
        task_losses = [float(loss.item()) for loss in _reference_task_losses(reference, x, y)]
        reg = float(sum(l2 * torch.sum(torch.square(p)) for p in reference.embedding_dict.parameters()).item())
        model.compile(_sgd(model), loss=["binary_crossentropy", "mse"], loss_weights=weights)
        return _fit_one_full_batch(model, x, y)["loss"][0], task_losses, reg

    loss_small, task_losses, reg = logged_loss([1.0, 0.5])
    loss_large, _, _ = logged_loss([10.0, 5.0])

    assert reg > 0
    assert loss_small == pytest.approx(1.0 * task_losses[0] + 0.5 * task_losses[1] + reg, rel=1e-6)
    assert loss_large == pytest.approx(10.0 * task_losses[0] + 5.0 * task_losses[1] + reg, rel=1e-6)


def test_auxiliary_loss_is_added_once_and_not_scaled_by_task_weights():
    x, y, feature_columns = _data()

    def logged_loss(weights):
        model = _mmoe(feature_columns)
        task_losses = [float(loss.item()) for loss in _reference_task_losses(copy.deepcopy(model), x, y)]
        model.add_auxiliary_loss(torch.tensor([0.74]), 0.5)  # constant auxiliary loss of 0.37
        model.compile(_sgd(model), loss=["binary_crossentropy", "mse"], loss_weights=weights)
        return _fit_one_full_batch(model, x, y)["loss"][0], task_losses

    loss_default, task_losses = logged_loss(None)
    loss_weighted, _ = logged_loss([10.0, 5.0])

    assert loss_default == pytest.approx(task_losses[0] + task_losses[1] + 0.37, rel=1e-6)
    assert loss_weighted == pytest.approx(10.0 * task_losses[0] + 5.0 * task_losses[1] + 0.37, rel=1e-6)


def test_loss_weights_respect_sum_reduction():
    x, y, feature_columns = _data()
    model = _mmoe(feature_columns)
    reference_losses = _reference_task_losses(copy.deepcopy(model), x, y, reduction="sum")
    expected = (0.3 * reference_losses[0].item() + 2.0 * reference_losses[1].item()) / len(y)

    model.compile(_sgd(model), loss=["binary_crossentropy", "mse"], loss_weights=[0.3, 2.0])
    model.loss_reduction = "sum"
    history = _fit_one_full_batch(model, x, y)

    assert history["loss"][0] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("loss_weights, message", [
    ([1.0], "2"),
    ([1.0, 1.0, 1.0], "2"),
    ({"like": 1.0}, "watch"),
    ({"like": 1.0, "watch": 1.0, "share": 1.0}, "share"),
    ({"like": 1.0, "wacth": 1.0}, "wacth"),
    ([1.0, -0.1], "non-negative"),
    ([1.0, float("nan")], "finite"),
    ([float("inf"), 1.0], "finite"),
    ([0.0, 0.0], "zero"),
    (["a", 1.0], "number"),
    (0.5, "list"),
])
def test_invalid_loss_weights_are_rejected(loss_weights, message):
    _, _, feature_columns = _data()
    model = _mmoe(feature_columns)

    with pytest.raises(ValueError, match=message):
        model.compile("adam", loss=["binary_crossentropy", "mse"], loss_weights=loss_weights)


def test_loss_weights_need_one_loss_per_task():
    _, _, feature_columns = _data()
    model = _mmoe(feature_columns)
    with pytest.raises(ValueError, match="per-task"):
        model.compile("adam", loss="mse", loss_weights=[1.0, 1.0])

    single_task = DeepFM(feature_columns, feature_columns, device="cpu")
    with pytest.raises(ValueError, match="per-task"):
        single_task.compile("adam", "binary_crossentropy", loss_weights=[1.0])


def test_duplicate_task_names_reject_dict_loss_weights():
    _, _, feature_columns = _data()
    model = _mmoe(feature_columns, task_names=("like", "like"))

    with pytest.raises(ValueError, match="unique"):
        model.compile("adam", loss=["binary_crossentropy", "mse"], loss_weights={"like": 1.0})


@pytest.mark.parametrize("model_class, task_types, loss_weights", [
    (SharedBottom, ("binary", "regression"), [1.0, 0.2]),
    (SharedBottom, ("binary", "binary"), {"first": 0.5, "second": 2.0}),
    (MMOE, ("binary", "binary"), {"first": 0.5, "second": 2.0}),
    (PLE, ("binary", "regression"), [1.0, 0.2]),
    (PLE, ("binary", "binary"), {"first": 0.0, "second": 1.0}),
    (ESMM, ("binary", "binary"), {"first": 1.0, "second": 0.5}),
    (ESMM, ("binary", "binary"), [1.0, 0.5]),
])
def test_loss_weights_across_multitask_models(model_class, task_types, loss_weights):
    x, y, feature_columns = _data(sample_size=33, task_types=task_types)  # batch_size=16 leaves a last batch of 1
    model = model_class(feature_columns, task_types=list(task_types), task_names=["first", "second"], device="cpu")
    losses = ["binary_crossentropy" if task_type == "binary" else "mse" for task_type in task_types]
    model.compile("adam", loss=losses, loss_weights=loss_weights)

    history = model.fit(x, y, batch_size=16, epochs=2, verbose=0, validation_data=(x, y)).history

    assert len(history["loss"]) == 2 and np.isfinite(history["loss"]).all()
    assert model.predict(x, 16).shape == (33, 2)


def test_three_tasks_weighted_by_name():
    task_types = ("binary", "regression", "binary")
    x, y, feature_columns = _data(task_types=task_types)
    model = _mmoe(feature_columns, task_types, ("click", "watch", "share"))
    reference = copy.deepcopy(model)
    reference.train()
    pred = reference(_input_tensor(reference, x))
    target = torch.from_numpy(y).float()
    expected = (2.0 * F.binary_cross_entropy(pred[:, 0], target[:, 0])
                + 0.5 * F.mse_loss(pred[:, 1], target[:, 1])
                + 0.0 * F.binary_cross_entropy(pred[:, 2], target[:, 2])).item()

    model.compile(_sgd(model), loss=["binary_crossentropy", "mse", "binary_crossentropy"],
                  loss_weights={"share": 0.0, "click": 2.0, "watch": 0.5})
    history = _fit_one_full_batch(model, x, y)

    assert history["loss"][0] == pytest.approx(expected, rel=1e-6)


def test_legacy_positional_compile_still_works():
    x, y, feature_columns = _data(task_types=("binary",))
    model = DeepFM(feature_columns, feature_columns, device="cpu")
    model.compile("adam", "binary_crossentropy", ["binary_crossentropy"])

    history = model.fit(x, y, batch_size=16, epochs=1, verbose=0, validation_data=(x, y)).history

    assert set(history) == {"loss", "val_binary_crossentropy"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_weighted_gradients_match_manual_pytorch_on_cuda():
    x, y, feature_columns = _data()
    model = _mmoe(feature_columns, device="cuda:0")
    reference_gradients, reference_loss = _reference_gradients(copy.deepcopy(model), x, y, [1.0, 0.2], device="cuda:0")

    model.compile(_sgd(model), loss=["binary_crossentropy", "mse"], loss_weights={"like": 1.0, "watch": 0.2})
    history = _fit_one_full_batch(model, x, y)

    assert history["loss"][0] == pytest.approx(reference_loss, rel=1e-5)
    _assert_gradients_match(model, reference_gradients, atol=1e-6)
