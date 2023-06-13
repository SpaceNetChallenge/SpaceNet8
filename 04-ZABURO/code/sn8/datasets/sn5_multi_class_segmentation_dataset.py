from pathlib import Path
from typing import Dict, Literal, Mapping, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import skimage.io
import torch
from somen.types import PathLike
from torch import Tensor

OutputType = Literal["speed", "binary", "speed-and-junction"]
output_type_to_num_classes: Mapping[OutputType, int] = {"speed": 8, "binary": 1, "speed-and-junction": 9}


class SN5MultiClassSegmentationDataset:
    def __init__(
        self,
        image_paths: Sequence[Path],
        mask_paths: Optional[Sequence[Path]],
        output_type: OutputType,
        augs: Optional[A.Compose] = None,
        pad: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mask_paths is not None:
            assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.output_type = output_type
        self.augs = augs
        self.pad = pad

        if output_type == "speed-and-junction" and mask_paths is not None:
            self.junction_mask_paths = [
                path.parent.parent / "train_mask_binned_junction" / path.name for path in mask_paths
            ]
            assert all(p.exists() for p in self.junction_mask_paths)
            if self.augs is not None:
                self.augs.add_targets({"junction": "mask"})
        else:
            self.junction_mask_paths = None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        inputs = {}
        inputs["image"] = skimage.io.imread(self.image_paths[index])
        if self.mask_paths is not None:
            inputs["mask"] = skimage.io.imread(self.mask_paths[index])

            if self.output_type == "binary" and inputs["mask"].shape[-1] > 1:
                inputs["mask"] = inputs["mask"][..., [-1]]

            if self.output_type == "speed-and-junction":
                assert self.junction_mask_paths is not None
                inputs["junction"] = skimage.io.imread(self.junction_mask_paths[index])[..., np.newaxis]
                assert inputs["junction"].ndim == 3

        if self.pad is not None:
            for key, img in inputs.items():
                img = cv2.copyMakeBorder(img, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_REFLECT)
                # img.shape[-1] == 1 の時 squeeze されてしまうので元に戻す
                if img.ndim == 2:
                    img = img[..., np.newaxis]
                inputs[key] = img

        if self.augs is not None:
            inputs = self.augs(**inputs)

        ret = {"x": torch.from_numpy(np.moveaxis(inputs["image"] / 255.0, 2, 0).astype(np.float32))}
        if "mask" in inputs:
            ret["target"] = torch.from_numpy(np.moveaxis(inputs["mask"] / 255.0, 2, 0).astype(np.float32))

            if "junction" in inputs:
                ret["target"] = torch.cat(
                    [ret["target"], torch.from_numpy(np.moveaxis(inputs["junction"] / 255.0, 2, 0).astype(np.float32))]
                )
        return ret


def build_road_dataset(
    wdata_base: PathLike,
    image_list_csv: PathLike,
    fold_index: int,
    output_type: OutputType,
    crop_size_train: Optional[int],
    crop_size_inference: Optional[int] = None,
) -> Tuple[SN5MultiClassSegmentationDataset, Mapping[str, SN5MultiClassSegmentationDataset]]:
    def as_wdata_path(rel_path: str) -> Path:
        return Path(wdata_base) / rel_path

    df = pd.read_csv(image_list_csv)

    train_index_mask = df["fold_index"] != fold_index
    train_indices = df.index[train_index_mask].to_numpy()

    crop = [] if crop_size_train is None else [A.RandomCrop(height=crop_size_train, width=crop_size_train, p=1.0)]
    train_augs = A.Compose(
        crop
        + [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=30, p=0.75),
            A.ColorJitter(p=0.5),
        ]
    )
    train_set = SN5MultiClassSegmentationDataset(
        df.loc[train_indices, "image"].map(as_wdata_path).to_list(),
        df.loc[train_indices, "mask"].map(as_wdata_path).to_list(),
        output_type,
        train_augs,
        pad=22 if crop_size_train is None else None,  # crop するときは padding しない
    )

    if crop_size_inference is not None:
        valid_augs = A.Compose([A.CenterCrop(height=crop_size_inference, width=crop_size_inference, p=1.0)])
    else:
        valid_augs = None

    valid_sets = {}
    for aoi, sub in df[df["fold_index"] == fold_index].groupby("AOI"):
        valid_sets[aoi] = SN5MultiClassSegmentationDataset(
            sub["image"].map(lambda x: Path(wdata_base) / x).to_list(),
            sub["mask"].map(lambda x: Path(wdata_base) / x).to_list(),
            output_type,
            augs=valid_augs,
            pad=22 if crop_size_inference is None else None,  # crop するときは padding しない
        )

    return train_set, valid_sets
