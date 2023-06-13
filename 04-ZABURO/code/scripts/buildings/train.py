import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import fire
import pandas as pd
import somen
import torch
from somen.pytorch_utility.configs import TrainingConfig
from somen.types import PathLike

from sn8.datasets.building_segmentation_dataset import SN2_AOI, BuildingSegmentationDataset, build_building_dataset
from sn8.inference import predict
from sn8.loss_functions import BuildingLossFn
from sn8.models.configs import UnetConfig


@dataclass
class Config:
    fold_index: int
    training: TrainingConfig
    model: UnetConfig


_logger = logging.getLogger(__name__)


def download(config_path: PathLike) -> None:
    config = somen.file_io.load_yaml_as_dataclass(Config, config_path)
    config.model.build(classes=2)


def train(
    config_path: PathLike,
    *overrides: Sequence[str],
    working_base: PathLike = "/wdata/working",
    wdata_base: PathLike = "/wdata",
    image_list_csv: PathLike = "/wdata/building_image_list.csv",
) -> None:
    config = somen.file_io.load_yaml_as_dataclass(Config, config_path, overrides)
    working_dir = Path(f"{working_base}/buildings/{Path(config_path).stem}/{config.fold_index}/").resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    somen.logger.configure_default_loggers(working_dir / "log.txt")
    _logger.info("Config:\n" + str(config))

    train_set, train_sampler, valid_sets = build_building_dataset(wdata_base, image_list_csv, config.fold_index)
    config.training.train_sampler = train_sampler

    model = config.model.build(classes=2)
    config.training.objective = BuildingLossFn()
    somen.pytorch_utility.train(
        config=config.training, model=model, train_set=train_set, valid_sets=valid_sets, working_dir=working_dir
    )

    torch.save(model.state_dict(), working_dir / "final.pt")


def predict_valid(
    config_path: PathLike,
    *overrides: Sequence[str],
    working_base: PathLike = "/wdata/working",
    wdata_base: PathLike = "/wdata",
    image_list_csv: PathLike = "/wdata/building_image_list.csv",
) -> None:
    config = somen.file_io.load_yaml_as_dataclass(Config, config_path, overrides)
    working_dir = Path(f"{working_base}/buildings/{Path(config_path).stem}/{config.fold_index}/").resolve()
    somen.logger.configure_default_loggers(working_dir / "log.txt")
    _logger.info("Config:\n" + str(config))

    model = config.model.build(classes=2)
    model.load_state_dict(torch.load(working_dir / "final.pt", map_location="cpu"))

    out_dir = working_dir / "pred_valid"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(image_list_csv)
    for aoi, sub in df[df["fold_index"] == config.fold_index].groupby("AOI"):
        pad = 27 if aoi in SN2_AOI else 22
        valid_set = BuildingSegmentationDataset(
            sub["image"].map(lambda x: Path(wdata_base) / x).to_list(), None, pad=pad
        )

        predict(
            model,
            out_dir,
            valid_set,
            config.training.batch_size_valid,
            config.training.device,
            config.training.num_workers,
            config.training.pin_memory,
            config.training.progress_bar,
        )


def predict_test(
    config_path: PathLike,
    image_dir: PathLike,
    *overrides: Sequence[str],
    working_base: PathLike = "/wdata/working",
    d4_tta: bool = False,
    snapshot_name: str = "final.pt",
) -> None:
    config = somen.file_io.load_yaml_as_dataclass(Config, config_path, overrides)
    working_dir = Path(f"{working_base}/buildings/{Path(config_path).stem}/{config.fold_index}/").resolve()
    somen.logger.configure_default_loggers(working_dir / "log.txt")
    _logger.info("Config:\n" + str(config))

    image_paths = sorted(Path(image_dir).glob("*.tif"))
    valid_set = BuildingSegmentationDataset(image_paths, None, pad=22)

    model = config.model.build(classes=2)
    _logger.info(f"Loading state_dict from: {working_dir / snapshot_name}")
    model.load_state_dict(torch.load(working_dir / snapshot_name, map_location="cpu"))

    out_dir = working_dir / "pred_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    predict(
        model,
        out_dir,
        valid_set,
        valid_set.image_paths,
        d4_tta,
        config.training.batch_size_valid,
        config.training.device,
        config.training.num_workers,
        config.training.pin_memory,
        config.training.progress_bar,
    )


if __name__ == "__main__":
    fire.Fire({"train": train, "predict_valid": predict_valid, "predict_test": predict_test, "download": download})
