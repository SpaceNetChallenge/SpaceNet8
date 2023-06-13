import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import fire
import numpy as np
import pandas as pd
import pytorch_pfn_extras as ppe
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

import somen
from somen.pytorch_utility import models
from somen.pytorch_utility.configs import TrainingConfig
from somen.pytorch_utility.metrics import LossMetric
from somen.types import PathLike


@dataclass
class Config:
    fold_index: int
    training: TrainingConfig
    model: Union[models.MLPConfig, models.Tabular1DCNNConfig]


_logger = logging.getLogger(__name__)


def build_model(model_config: Union[models.MLPConfig, models.Tabular1DCNNConfig]) -> nn.Module:
    if isinstance(model_config, models.MLPConfig):
        return models.MLP(model_config)
    elif isinstance(model_config, models.Tabular1DCNNConfig):
        return models.Tabular1DCNN(model_config)
    raise ValueError


def preprocess_input(X: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> np.ndarray:
    _logger.info("Start preprocessing...")

    null_check_cols = [
        "book.log_return1.realized_volatility",
        "book_150.log_return1.realized_volatility",
        "book_300.log_return1.realized_volatility",
        "book_450.log_return1.realized_volatility",
        "trade.log_return.realized_volatility",
        "trade_150.log_return.realized_volatility",
        "trade_300.log_return.realized_volatility",
        "trade_450.log_return.realized_volatility",
    ]

    for c in null_check_cols:
        if c in X.columns:
            X[f"{c}_isnull"] = X[c].isnull().astype(int)

    cat_cols = ["stock_id"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_num = X[num_cols].values.astype(np.float32)
    X_cat = np.nan_to_num(X[cat_cols].values.astype(np.int32))

    if scaler is None:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
    else:
        X_num = scaler.transform(X_num)

    X_num = np.nan_to_num(X_num, posinf=0, neginf=0)

    _logger.info("Finish preprocessing...")
    return X_num, X_cat, scaler


def train(
    config_path: str,
    *overrides: Sequence[str],
    input_dir: PathLike = "/kaggle/input/1st-place-public-2nd-place-solution/",
) -> None:
    input_dir = Path(input_dir)

    config = somen.file_io.load_yaml_as_dataclass(Config, config_path, overrides)
    working_dir = Path(f"working/{Path(config_path).stem}/{config.fold_index}/")
    working_dir.mkdir(parents=True, exist_ok=True)
    somen.logger.configure_default_loggers(working_dir / "log.txt")

    X = pd.read_feather(input_dir / "X.f")
    y = pd.read_feather(input_dir / "y.f")["target"].to_numpy().astype(np.float32)
    folds = somen.file_io.load_pickle(input_dir / "folds.pkl")

    X_num, X_cat, scaler = preprocess_input(X)
    somen.file_io.save_pickle(scaler, working_dir / "scaler.pkl")

    train_indices, valid_indices = folds[config.fold_index]
    train_set = torch.utils.data.TensorDataset(
        *map(torch.from_numpy, (X_num[train_indices], X_cat[train_indices], y[train_indices]))
    )
    valid_set = torch.utils.data.TensorDataset(
        *map(torch.from_numpy, (X_num[valid_indices], X_cat[valid_indices], y[valid_indices]))
    )
    collate_fn = ppe.dataloaders.utils.CollateAsDict(names=["x_num", "x_cat", "target"])

    config.model.num_dim = X_num.shape[1]
    model = build_model(config.model)

    config.training.macro_metrics = [LossMetric(["rmspe"])]
    somen.pytorch_utility.train(config.training, model, train_set, valid_set, working_dir, collate_fn)

    torch.save(model.state_dict(), working_dir / "final.pt")


def predict_valid(
    config_path: str,
    *overrides: Sequence[str],
    input_dir: PathLike = "/kaggle/input/1st-place-public-2nd-place-solution/",
) -> None:
    input_dir = Path(input_dir)

    config = somen.file_io.load_yaml_as_dataclass(Config, config_path, overrides)
    working_dir = Path(f"working/{Path(config_path).stem}/{config.fold_index}/")
    somen.logger.configure_default_loggers(working_dir / "log.txt")
    _logger.info("Config:\n" + str(config))

    folds = somen.file_io.load_pickle(input_dir / "folds.pkl")
    X_valid = pd.read_feather(input_dir / "X.f").iloc[folds[config.fold_index][1]]

    scaler = somen.file_io.load_pickle(working_dir / "scaler.pkl")
    X_num, X_cat, _ = preprocess_input(X_valid, scaler)
    valid_set = torch.utils.data.TensorDataset(*map(torch.from_numpy, (X_num, X_cat)))
    collate_fn = ppe.dataloaders.utils.CollateAsDict(names=["x_num", "x_cat"])

    config.model.num_dim = X_num.shape[1]
    model = build_model(config.model)
    model.load_state_dict(torch.load(working_dir / "bests" / "val_loss_best.pt", map_location="cpu"))
    ppe.to(model, config.training.device)

    (pred,) = somen.pytorch_utility.predict(
        model,
        valid_set,
        config.training.batch_size * 2,
        config.training.num_workers,
        config.training.device,
        collate_fn,
        config.training.pin_memory,
        config.training.progress_bar,
    )
    somen.file_io.save_array(pred, working_dir / "pred_valid.h5")


if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "predict_valid": predict_valid,
        }
    )
