from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import skimage.io
import torch
from torch import Tensor

from somen.types import PathLike

SN2_AOI = {"Vegas", "Paris", "Shanghai", "Khartoum"}


class BuildingSegmentationDataset:
    def __init__(
        self,
        image_paths: Sequence[Path],
        mask_paths: Optional[Sequence[Path]],
        augs: Optional[A.Compose] = None,
        pad: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mask_paths is not None:
            assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augs = augs
        self.pad = pad

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        inputs = {}
        inputs["image"] = skimage.io.imread(self.image_paths[index])
        if self.mask_paths is not None:
            inputs["mask"] = skimage.io.imread(self.mask_paths[index])

        if self.pad is not None:
            inputs = {
                key: cv2.copyMakeBorder(img, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_REFLECT)
                for key, img in inputs.items()
            }

        if self.augs is not None:
            inputs = self.augs(**inputs)

        ret = {"x": torch.from_numpy(np.moveaxis(inputs["image"] / 255.0, -1, 0).astype(np.float32))}
        if "mask" in inputs:
            ret["target"] = torch.from_numpy(np.moveaxis(inputs["mask"] / 255.0, -1, 0).astype(np.float32))
        return ret


class BalancedSampler(torch.utils.data.sampler.Sampler[int]):

    """SpaceNet 2 の画像を 1/4 に Down Sample する"""

    def __init__(self, data_source: BuildingSegmentationDataset, aoi_list: Sequence[str]) -> None:
        self.data_source = data_source
        self.aoi_list = aoi_list

        indices_by_aoi = defaultdict(list)
        for i, aoi in enumerate(aoi_list):
            if aoi in SN2_AOI:
                indices_by_aoi[aoi].append(i)
            else:
                indices_by_aoi["Others"].append(i)

        self.indices_by_aoi = {key: np.asarray(value) for key, value in indices_by_aoi.items()}
        self.num_samples = sum(
            [len(value) if key == "Others" else len(value) // 4 for key, value in self.indices_by_aoi.items()]
        )

    def __iter__(self) -> Iterator[int]:
        indices = []
        for aoi, sub_indices in self.indices_by_aoi.items():
            if aoi == "Others":
                indices += list(sub_indices)
            else:
                indices += list(np.random.choice(sub_indices, size=len(sub_indices) // 4, replace=False))

        yield from (indices[i] for i in np.random.permutation(len(indices)))

    def __len__(self) -> int:
        return self.num_samples


def build_building_dataset(
    wdata_base: PathLike, image_list_csv: PathLike, fold_index: int
) -> Tuple[BuildingSegmentationDataset, BalancedSampler, Mapping[str, BuildingSegmentationDataset]]:
    df = pd.read_csv(image_list_csv)

    train_indices = df.index[df["fold_index"] != fold_index].to_numpy()
    train_augs = A.Compose(
        [
            A.RandomCrop(height=512, width=512, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=30, p=0.75),
            A.ColorJitter(p=0.5),
        ]
    )

    train_set = BuildingSegmentationDataset(
        df.loc[train_indices, "image"].map(lambda x: Path(wdata_base) / x).to_list(),
        df.loc[train_indices, "mask"].map(lambda x: Path(wdata_base) / x).to_list(),
        train_augs,
    )
    train_sampler = BalancedSampler(train_set, df.loc[train_indices, "AOI"].to_list())

    valid_sets = {}
    for aoi, sub in df[df["fold_index"] == fold_index].groupby("AOI"):
        pad = 27 if aoi in SN2_AOI else 22
        valid_sets[aoi] = BuildingSegmentationDataset(
            sub["image"].map(lambda x: Path(wdata_base) / x).to_list(),
            sub["mask"].map(lambda x: Path(wdata_base) / x).to_list(),
            pad=pad,
        )

    return train_set, train_sampler, valid_sets
