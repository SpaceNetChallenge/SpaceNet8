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

from sn8.datasets.flood_segmentation_dataset import FloodSegmentationDataset, build_flood_dataset
from sn8.inference import predict, sliding_predict
from sn8.loss_functions import FloodLossFn
from sn8.models.configs import UnetSiameseConfig


@dataclass
class Config:
    fold_index: int
    training: TrainingConfig
    model: UnetSiameseConfig
    bce_weight: float = 1.0
    focal_weight: float = 0.0
    dice_weight: float = 0.0
    ratio_to_flooded: float = 1.0
    image_level_bce_weight: float = 0.0
    crop_size_inference: Optional[int] = None
    use_xview2: bool = False


_logger = logging.getLogger(__name__)


def download(config_path: PathLike) -> None:
    config = somen.file_io.load_yaml_as_dataclass(Config, config_path)
    config.model.build(classes=1)


def train(
    config_path: PathLike,
    *overrides: Sequence[str],
    working_base: PathLike = "/wdata/working",
    wdata_base: PathLike = "/wdata",
    image_list_csv: PathLike = "/wdata/flood_image_list.csv",
    xview2_base: Optional[PathLike] = "/wdata/geotiffs",
    xview2_image_list_csv: Optional[PathLike] = "/wdata/xview2_image_list.csv",
) -> None:
    config = somen.file_io.load_yaml_as_dataclass(Config, config_path, overrides)
    working_dir = Path(f"{working_base}/floods/{Path(config_path).stem}/{config.fold_index}/").resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    somen.logger.configure_default_loggers(working_dir / "log.txt")
    _logger.info("Config:\n" + str(config))

    if not config.use_xview2:
        xview2_base = None
        xview2_image_list_csv = None

    train_set, train_sampler, valid_sets = build_flood_dataset(
        wdata_base,
        image_list_csv,
        config.fold_index,
        config.image_level_bce_weight > 0,
        config.ratio_to_flooded,
        xview2_base,
        xview2_image_list_csv,
    )
    config.training.train_sampler = train_sampler

    model = config.model.build(classes=1)
    config.training.objective = FloodLossFn(
        config.bce_weight, config.focal_weight, config.dice_weight, config.image_level_bce_weight
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
    image_list_csv: PathLike = "/wdata/flood_image_list.csv",
    d4_tta: bool = False,
    snapshot_name: str = "final.pt",
) -> None:
    config = somen.file_io.load_yaml_as_dataclass(Config, config_path, overrides)
    working_dir = Path(f"{working_base}/floods/{Path(config_path).stem}/{config.fold_index}/").resolve()
    somen.logger.configure_default_loggers(working_dir / "log.txt")
    _logger.info("Config:\n" + str(config))

    def as_wdata_path(rel_path: str) -> Path:
        return Path(wdata_base) / rel_path

    df = pd.read_csv(image_list_csv)
    sub = df[df["fold_index"] == config.fold_index]
    valid_set = FloodSegmentationDataset(
        sub["pre"].map(as_wdata_path).to_list(),
        sub["post1"].map(as_wdata_path).to_list(),
        None,
        None,
        pad=22 if config.crop_size_inference is None else None,
    )

    model = config.model.build(classes=1)
    _logger.info(f"Loading state_dict from: {working_dir / snapshot_name}")
    model.load_state_dict(torch.load(working_dir / snapshot_name, map_location="cpu"))

    out_dir = working_dir / "pred_valid"
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.crop_size_inference is None:
        predict(
            model,
            out_dir,
            valid_set,
            valid_set.pre_paths,
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
            valid_set.pre_paths,
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
    snapshot_name: str = "final.pt",
) -> None:
    config = somen.file_io.load_yaml_as_dataclass(Config, config_path, overrides)
    working_dir = Path(f"{working_base}/floods/{Path(config_path).stem}/{config.fold_index}/").resolve()
    somen.logger.configure_default_loggers(working_dir / "log.txt")
    _logger.info("Config:\n" + str(config))

    image_dir = Path(image_dir)
    label_image_mapping_csvs = list(image_dir.glob("*label_image_mapping.csv"))
    if len(label_image_mapping_csvs) != 1:
        raise RuntimeError

    print(label_image_mapping_csvs[0])
    df = pd.read_csv(label_image_mapping_csvs[0])

    test_set = FloodSegmentationDataset(
        df["pre-event image"].map(lambda x: image_dir / "PRE-event" / x),
        df["post-event image 1"].map(lambda x: image_dir / "POST-event" / x),
        None,
        None,
        pad=22 if config.crop_size_inference is None else None,
    )

    model = config.model.build(classes=1)
    _logger.info(f"Loading state_dict from: {working_dir / snapshot_name}")
    model.load_state_dict(torch.load(working_dir / snapshot_name, map_location="cpu"))

    out_dir = working_dir / "pred_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.crop_size_inference is None:
        predict(
            model,
            out_dir,
            test_set,
            test_set.pre_paths,
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
            test_set,
            test_set.pre_paths,
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
