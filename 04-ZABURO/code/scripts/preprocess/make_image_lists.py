from pathlib import Path

import fire
import numpy as np
import pandas as pd
from osgeo import gdal


def make_road_image_list(wdata_base: Path, out_path: Path) -> None:
    train_dir = wdata_base / "road_train"
    image_paths = [p for p in train_dir.glob("images_8bit_base/PS-RGB/*.tif")]
    mask_paths = [train_dir / "masks_base" / "train_mask_binned_mc" / p.name for p in image_paths]
    assert all([p.exists() for p in mask_paths])

    df = pd.DataFrame(
        {
            "image": [str(path.relative_to(wdata_base)) for path in image_paths],
            "mask": [str(path.relative_to(wdata_base)) for path in mask_paths],
        }
    )

    geo_transforms = [gdal.Open(str(path)).GetGeoTransform() for path in image_paths]
    longitude, _, _, latitude, _, _ = zip(*geo_transforms)
    df["longitude"] = longitude
    df["latitude"] = latitude

    filename_to_aoi = {}
    for p in (wdata_base / "Germany_Training_Public" / "PRE-event").glob("*.tif"):
        filename_to_aoi[p.name] = "Germany"
    for p in (wdata_base / "Louisiana-East_Training_Public" / "PRE-event").glob("*.tif"):
        filename_to_aoi[p.name] = "Louisiana-East"

    def get_aoi(x: str) -> str:
        if "AOI" in x:
            return x.split("/")[-1].split("_")[5]
        else:
            return filename_to_aoi[x.split("/")[-1]]

    df["AOI"] = df["image"].map(get_aoi)
    df = df.sort_values(by=["AOI", "longitude", "latitude"]).reset_index(drop=True)

    n_folds = 5
    df["fold_index"] = -1
    for _, sub in df.index.to_series().groupby(df["AOI"]):
        sections = np.linspace(0, len(sub), n_folds + 1).astype(int)
        for i, (start, end) in enumerate(zip(sections, sections[1:])):
            df.loc[sub.iloc[start:end], "fold_index"] = i
    assert (df["fold_index"] != -1).all()

    print(df.groupby("AOI")["fold_index"].value_counts())
    df.to_csv(out_path, index=False)


def make_building_image_list(wdata_base: Path, out_path: Path) -> None:
    train_dir = wdata_base / "building_train"
    image_paths = [p for p in train_dir.glob("images_8bit_base/*.tif")]
    mask_paths = [train_dir / "masks_base" / p.name for p in image_paths]
    assert all([p.exists() for p in mask_paths])

    df = pd.DataFrame(
        {
            "image": [str(path.relative_to(wdata_base)) for path in image_paths],
            "mask": [str(path.relative_to(wdata_base)) for path in mask_paths],
        }
    )

    geo_transforms = [gdal.Open(str(path)).GetGeoTransform() for path in image_paths]
    longitude, _, _, latitude, _, _ = zip(*geo_transforms)
    df["longitude"] = longitude
    df["latitude"] = latitude

    filename_to_aoi = {}
    for p in (wdata_base / "Germany_Training_Public" / "PRE-event").glob("*.tif"):
        filename_to_aoi[p.name] = "Germany"
    for p in (wdata_base / "Louisiana-East_Training_Public" / "PRE-event").glob("*.tif"):
        filename_to_aoi[p.name] = "Louisiana-East"

    def get_aoi(x: str) -> str:
        if "AOI" in x:
            return x.split("/")[-1].split("_")[3]
        else:
            return filename_to_aoi[x.split("/")[-1]]

    df["AOI"] = df["image"].map(get_aoi)
    df = df.sort_values(by=["AOI", "longitude", "latitude"]).reset_index(drop=True)

    n_folds = 5
    df["fold_index"] = -1
    for _, sub in df.index.to_series().groupby(df["AOI"]):
        sections = np.linspace(0, len(sub), n_folds + 1).astype(int)
        for i, (start, end) in enumerate(zip(sections, sections[1:])):
            df.loc[sub.iloc[start:end], "fold_index"] = i
    assert (df["fold_index"] != -1).all()

    print(df.groupby("AOI")["fold_index"].value_counts())
    df.to_csv(out_path, index=False)


def make_flood_image_list(wdata_base: Path, out_path: Path) -> None:
    label_image_mapping = pd.concat(
        [
            pd.read_csv(wdata_base / "Germany_Training_Public/Germany_Training_Public_label_image_mapping.csv"),
            pd.read_csv(
                wdata_base / "Louisiana-East_Training_Public/Louisiana-East_Training_Public_label_image_mapping.csv"
            ),
        ],
        axis=0,
    ).set_index("pre-event image")

    rows = []
    flood_paths = sorted((wdata_base / "Germany_Training_Public/annotations/masks/").glob("flood*.tif")) + sorted(
        (wdata_base / "Louisiana-East_Training_Public/annotations/masks/").glob("flood*.tif")
    )
    for path in flood_paths:
        ds = gdal.Open(str(path))
        geotransform = ds.GetGeoTransform()
        img = ds.ReadAsArray()
        dataset_root = path.parent.parent.parent
        pre_path = dataset_root / "PRE-event" / path.name[len("flood_") :]
        post1_path = dataset_root / "POST-event" / label_image_mapping.loc[pre_path.name, "post-event image 1"]
        rows.append(
            {
                "AOI": dataset_root.name.split("_")[0],
                "pre": pre_path.relative_to(wdata_base),
                "post1": post1_path.relative_to(wdata_base),
                "mask": path.relative_to(wdata_base),
                "flooded": (img[[1, 3]] > 0).any(),
                "longitude": geotransform[0],
                "latitude": geotransform[3],
            }
        )

    df = pd.DataFrame(rows)
    del rows

    df = df.sort_values(by=["AOI", "flooded", "longitude", "latitude"]).reset_index(drop=True)
    n_folds = 5
    df["fold_index"] = -1
    for _, sub in df.index.to_series().groupby([df["AOI"], df["flooded"]]):
        sections = np.linspace(0, len(sub), n_folds + 1).astype(int)
        for i, (start, end) in enumerate(zip(sections, sections[1:])):
            df.loc[sub.iloc[start:end], "fold_index"] = i
    assert (df["fold_index"] != -1).all()

    print(df.groupby(["AOI", "flooded"])["fold_index"].value_counts())
    df.to_csv(out_path, index=False)


def main(wdata_base: str = "/wdata") -> None:
    path = Path(wdata_base)
    make_road_image_list(path, path / "road_image_list.csv")
    make_building_image_list(path, path / "building_image_list.csv")
    make_flood_image_list(path, path / "flood_image_list.csv")


if __name__ == "__main__":
    fire.Fire(main)
