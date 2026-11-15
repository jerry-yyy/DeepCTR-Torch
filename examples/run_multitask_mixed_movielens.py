# -*- coding: utf-8 -*-
import os

import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from deepctr_torch.callbacks import EarlyStopping
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import MMOE

if __name__ == "__main__":
    data = pd.read_csv("./movielens_sample.txt")
    sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]

    # Two targets of different types. Both are derived from the same rating, so they are
    # strongly correlated: this example only demonstrates the API, it is not a benchmark.
    data["like"] = (data["rating"] >= 4).astype("float32")  # binary classification
    data["rating"] = data["rating"].astype("float32")  # regression
    target = ["like", "rating"]

    # 1.Label Encoding for sparse features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # 2.count #unique features for each sparse field
    dnn_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4) for feat in sparse_features]
    feature_names = get_feature_names(dnn_feature_columns)

    # 3.generate input data for model
    split_boundary = int(data.shape[0] * 0.8)
    train, test = data[:split_boundary], data[split_boundary:]
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = MMOE(dnn_feature_columns, task_types=['binary', 'regression'], task_names=target, device=device)
    # `metrics` maps each task name to its own metrics. Columns of `y` follow `task_names`.
    model.compile("adam", loss=["binary_crossentropy", "mse"],
                  metrics={"like": ["auc", "logloss"], "rating": ["mse", "mae"]})

    epochs = int(os.getenv("DEEPCTR_EXAMPLE_EPOCHS", "10"))
    # Per-task metrics are logged as `<task>_<metric>` / `val_<task>_<metric>` and can be monitored by callbacks.
    early_stopping = EarlyStopping(monitor="val_rating_mse", patience=2, mode="min")
    history = model.fit(train_model_input, train[target].values, batch_size=32, epochs=epochs, verbose=2,
                        validation_split=0.2, callbacks=[early_stopping])

    print("")
    print("evaluate:", model.evaluate(test_model_input, test[target].values, batch_size=256))
    pred_ans = model.predict(test_model_input, 256)
    print("like test AUC", round(roc_auc_score(test["like"].values, pred_ans[:, 0]), 4))
    print("rating test MSE", round(mean_squared_error(test["rating"].values, pred_ans[:, 1]), 4))
    print("rating test MAE", round(mean_absolute_error(test["rating"].values, pred_ans[:, 1]), 4))
