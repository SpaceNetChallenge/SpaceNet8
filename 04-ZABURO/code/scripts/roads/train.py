import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import fire
import pandas as pd
import somen
import torch
from somen.pytorch_utility.configs import TrainingConfig
from somen.types import PathLike

from sn8.datasets.sn5_multi_class_segmentation_dataset import (
    OutputType,
    SN5MultiClassSegmentationDataset,
    build_road_dataset,
    output_type_to_num_classes,
)
from sn8.inference import predict, sliding_predict
from sn8.loss_functions import RoadLossFn
from sn8.models.configs import UnetConfig


@dataclass
class Config:
    fold_index: int
    training: TrainingConfig
    model: UnetConfig
    focal_label_smoothing: float = 0.0
    unused_aoi: Optional[Sequence[str]] = None
    output_type: OutputType = "speed"
    crop_size_train: Optional[int] = 512
    crop_size_inference: Optional[int] = None
    junction_loss_weight: float = 1.0
    bce_weight: float = 0.0
    focal_weight: float = 0.75
    dice_weight: float = 0.25


_logger = logging.getLogger(__name__)


def download(config_path: PathLike) -> None:
    config = somen.file_io.load_yaml_as_dataclass(Config, config_path)
    config.model.build(output_type_to_num_classes[config.output_type])


def train(
    config_path: PathLike,
    *overrides: Sequence[str],
    working_base: PathLike = "/wdata/working",
    wdata_base: PathLike = "/wdata",
    image_list_csv: PathLike = "/wdata/road_image_list.csv",
) -> None:
    config = somen.file_io.load_yaml_as_dataclass(Config, config_path, overrides)
    working_dir = Path(f"{working_base}/roads/{Path(config_path).stem}/{config.fold_index}/").resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    somen.logger.configure_default_loggers(working_dir / "log.txt")
    _logger.info("Config:\n" + str(config))

    train_set, valid_sets = build_road_dataset(
        wdata_base,
        image_list_csv,
        config.fold_index,
        config.output_type,
        config.crop_size_train,
        config.crop_size_inference,
    )

    model = config.model.build(output_type_to_num_classes[config.output_type])
    config.training.objective = RoadLossFn(
        config.output_type,
        config.focal_label_smoothing,
        config.junction_loss_weight,
        config.bce_weight,
        config.focal_weight,
        config.dice_weight,
    )
    somen.pytorch_utility.train(
        config=config.training, model=model, train_set=train_set, valid_sets=valid_sets, working_dir=working_dir
    )

    torch.save(model.state_dict(), working_dir / "final.pt")


def predict_valid(
    config_path: PathLike,
    *overrides: Sequence[str],
    working_base: PathLike = "/wdata/working",
    wdata_base: PathLike = "/wdata",
    image_list_csv: PathLike = "/wdata/road_image_list.csv",
    d4_tta: bool = False,
    snapshot_name: str = "final.pt",
) -> None:
    config = somen.file_io.load_yaml_as_dataclass(Config, config_path, overrides)
    working_dir = Path(f"{working_base}/roads/{Path(config_path).stem}/{config.fold_index}/").resolve()
    somen.logger.configure_default_loggers(working_dir / "log.txt")
    _logger.info("Config:\n" + str(config))

    df = pd.read_csv(image_list_csv)
    image_paths = df.loc[df["fold_index"] == config.fold_index, "image"].map(lambda x: Path(wdata_base) / x).to_list()
    valid_set = SN5MultiClassSegmentationDataset(
        image_paths,
        None,
        config.output_type,
        pad=22 if config.crop_size_inference is None else None,
    )

    model = config.model.build(output_type_to_num_classes[config.output_type])
    _logger.info(f"Loading state_dict from: {working_dir / snapshot_name}")
    model.load_state_dict(torch.load(working_dir / snapshot_name, map_location="cpu"))

    out_dir = working_dir / "pred_valid"
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.crop_size_inference is None:
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
    else:
        sliding_predict(
            model,
            out_dir,
            valid_set,
            valid_set.image_paths,
            config.crop_size_inference,
            d4_tta,
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
) -> None:
    config = somen.file_io.load_yaml_as_dataclass(Config, config_path, overrides)
    working_dir = Path(f"{working_base}/roads/{Path(config_path).stem}/{config.fold_index}/").resolve()
    somen.logger.configure_default_loggers(working_dir / "log.txt")
    _logger.info("Config:\n" + str(config))

    image_paths = sorted(Path(image_dir).glob("*.tif"))
    valid_set = SN5MultiClassSegmentationDataset(
        image_paths, None, config.output_type, pad=22 if config.crop_size_inference is None else None
    )

    model = config.model.build(output_type_to_num_classes[config.output_type])
    model.load_state_dict(torch.load(working_dir / "final.pt", map_location="cpu"))

    out_dir = working_dir / "pred_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.crop_size_inference is None:
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
    else:
        sliding_predict(
            model,
            out_dir,
            valid_set,
            valid_set.image_paths,
            config.crop_size_inference,
            d4_tta,
            config.training.batch_size_valid,
            config.training.device,
            config.training.num_workers,
            config.training.pin_memory,
            config.training.progress_bar,
        )


if __name__ == "__main__":
    fire.Fire({"train": train, "predict_valid": predict_valid, "predict_test": predict_test, "download": download})
