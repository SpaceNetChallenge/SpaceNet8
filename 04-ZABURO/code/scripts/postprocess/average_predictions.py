from pathlib import Path
from typing import Sequence

import fire
import numpy as np
import skimage.io
from osgeo import gdal, osr
from tqdm import tqdm


def main(out_dir: str, *in_dirs: Sequence[str]) -> None:
    print(f"out_dir: {out_dir}")
    print(f"in_dirs: {in_dirs}")
    in_dir_paths = [Path(in_dir) for in_dir in in_dirs]

    image_lists = [sorted(in_dir.glob("*.tif")) for in_dir in in_dir_paths]
    image_names = [p.name for p in image_lists[0]]
    for image_list in image_lists[1:]:
        assert image_names == [p.name for p in image_list]

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(exist_ok=True, parents=True)

    for image_name in tqdm(image_names):
        in_ds = gdal.Open(str(in_dir_paths[0] / image_name))
        avg_img = in_ds.ReadAsArray().astype(np.float32)
        if avg_img.ndim == 2:
            avg_img = avg_img[np.newaxis]

        for in_dir_path in in_dir_paths[1:]:
            img = skimage.io.imread(in_dir_path / image_name)
            if img.ndim == 2:
                img = img[np.newaxis]
            else:
                img = np.moveaxis(img, 2, 0)
            assert img.shape == avg_img.shape
            avg_img = avg_img + img.astype(np.float32)
        avg_img = (avg_img / len(in_dir_paths)).clip(0, 255).astype(np.uint8)

        out_ds = gdal.GetDriverByName("GTiff").Create(
            str(out_dir_path / image_name),
            avg_img.shape[2],
            avg_img.shape[1],
            avg_img.shape[0],
            gdal.GDT_Byte,
        )
        out_ds.SetGeoTransform(in_ds.GetGeoTransform())

        sr = osr.SpatialReference()
        sr.ImportFromWkt(in_ds.GetProjectionRef())
        out_ds.SetProjection(sr.ExportToWkt())

        out_ds.WriteArray(avg_img)
        del in_ds, out_ds


if __name__ == "__main__":
    fire.Fire(main)
