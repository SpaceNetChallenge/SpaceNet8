from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import skimage.io
import torch
from somen.types import PathLike
from torch import Tensor


class FloodSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pre_paths: Sequence[Path],
        post_paths: Sequence[Path],
        mask_paths: Optional[Sequence[Path]],
        flooded: Optional[Sequence[bool]],
        shared_augs: Optional[A.Compose] = None,
        individual_augs: Optional[A.Compose] = None,
        pad: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert len(pre_paths) == len(post_paths)
        if mask_paths is not None:
            assert len(pre_paths) == len(mask_paths)
        if flooded is not None:
            assert len(pre_paths) == len(flooded)
        self.pre_paths = pre_paths
        self.post_paths = post_paths
        self.mask_paths = mask_paths
        self.flooded = flooded
        self.shared_augs = shared_augs
        self.individual_augs = individual_augs
        self.pad = pad

    def __len__(self) -> int:
        return len(self.pre_paths)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        inputs = {}
        inputs["image"] = skimage.io.imread(self.pre_paths[index]).astype(np.uint8)
        inputs["image-post"] = cv2.resize(
            skimage.io.imread(self.post_paths[index]).astype(np.uint8), inputs["image"].shape[:2]
        )

        if self.mask_paths is not None:
            mask = skimage.io.imread(self.mask_paths[index])
            if mask.ndim == 3:
                # SpaceNet8
                mask = np.stack(
                    [
                        (mask[..., :] > 0).any(axis=-1),
                        (mask[..., [1, 3]] > 0).any(axis=-1),  # flooded
                    ],
                    axis=-1,
                ).astype(np.uint8)
            else:
                # xview2
                mask = np.stack([2 <= mask, 3 <= mask], axis=-1).astype(np.uint8)
            inputs["mask"] = mask

        if self.pad is not None:
            inputs = {
                key: cv2.copyMakeBorder(img, self.pad, self.pad, self.pad, self.pad, cv2.BORDER_REFLECT)
                for key, img in inputs.items()
            }

        if self.shared_augs is not None:
            inputs = self.shared_augs(**inputs)

        if self.individual_augs is not None:
            inputs["image"] = self.individual_augs(image=inputs["image"])["image"]
            inputs["image-post"] = self.individual_augs(image=inputs["image-post"])["image"]

        ret = {
            "x1": torch.from_numpy(np.moveaxis(inputs["image"] / 255.0, -1, 0).astype(np.float32)),
            "x2": torch.from_numpy(np.moveaxis(inputs["image-post"] / 255.0, -1, 0).astype(np.float32)),
        }
        if "mask" in inputs:
            ret["target"] = torch.from_numpy(np.moveaxis(inputs["mask"], 2, 0).astype(np.float32))

        if self.flooded is not None:
            ret["flood_exists"] = self.flooded[index]

        return ret


class NonFloodedDownSampler(torch.utils.data.sampler.Sampler[int]):

    """Non-flooded な画像を Down-sample する"""

    def __init__(self, data_source: FloodSegmentationDataset, flooded: Sequence[bool], ratio_to_flooded: float) -> None:
        self.data_source = data_source
        self.flooded = flooded
        self.ratio_to_flooded = ratio_to_flooded

        self.flooded_indices = np.where(flooded)[0]
        self.non_flooded_indices = np.where(~np.asarray(flooded))[0]
        self.num_non_flooded_samples = min(
            len(self.non_flooded_indices), int(len(self.flooded_indices) * ratio_to_flooded)
        )
        self.num_samples = len(self.flooded_indices) + self.num_non_flooded_samples

    def __iter__(self) -> Iterator[int]:
        indices = list(self.flooded_indices) + list(
            np.random.choice(self.non_flooded_indices, size=self.num_non_flooded_samples, replace=False)
        )
        assert len(indices) == self.num_samples
        yield from (indices[i] for i in np.random.permutation(len(indices)))

    def __len__(self) -> int:
        return self.num_samples


def build_flood_dataset(
    wdata_base: PathLike,
    image_list_csv: PathLike,
    fold_index: int,
    use_flooded_flag: bool,
    ratio_to_flooded: float,
    xview2_base: Optional[PathLike],
    xview2_image_list_csv: Optional[PathLike],
) -> Tuple[FloodSegmentationDataset, NonFloodedDownSampler, Mapping[str, FloodSegmentationDataset]]:
    df = pd.read_csv(image_list_csv)

    df["pre"] = df["pre"].map(lambda x: Path(wdata_base) / x)
    df["post1"] = df["post1"].map(lambda x: Path(wdata_base) / x)
    df["mask"] = df["mask"].map(lambda x: Path(wdata_base) / x)

    if xview2_base is not None:
        assert xview2_image_list_csv is not None
        xview2_df = pd.read_csv(xview2_image_list_csv)
        xview2_df["pre"] = xview2_df["pre"].map(lambda x: Path(xview2_base) / x)
        xview2_df["post1"] = xview2_df["post1"].map(lambda x: Path(xview2_base) / x)
        xview2_df["mask"] = xview2_df["mask"].map(lambda x: Path(xview2_base) / x)
        df = pd.concat([df, xview2_df], axis=0).reset_index(drop=True)

    train_indices = df.index[df["fold_index"] != fold_index].to_numpy()
    shared_augs = A.Compose(
        [
            A.RandomCrop(height=512, width=512, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=30, p=0.75),
        ],
        additional_targets={"image-post": "image"},
    )
    individual_augs = A.Compose([A.ColorJitter(p=0.5)])

    train_set = FloodSegmentationDataset(
        df.loc[train_indices, "pre"].to_list(),
        df.loc[train_indices, "post1"].to_list(),
        df.loc[train_indices, "mask"].to_list(),
        df.loc[train_indices, "flooded"].to_list() if use_flooded_flag else None,
        shared_augs,
        individual_augs,
    )
    train_sampler = NonFloodedDownSampler(train_set, df.loc[train_indices, "flooded"].to_list(), ratio_to_flooded)

    valid_sets = {}
    for aoi, sub in df[df["fold_index"] == fold_index].groupby("AOI"):
        valid_sets[aoi] = FloodSegmentationDataset(
            sub["pre"].to_list(),
            sub["post1"].to_list(),
            sub["mask"].to_list(),
            sub["flooded"].to_list() if use_flooded_flag else None,
            pad=22,
        )

    return train_set, train_sampler, valid_sets
