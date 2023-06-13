import json
import multiprocessing as mp
from pathlib import Path
from typing import Sequence, Tuple

import fire
import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr
from somen.types import PathLike
from tqdm import tqdm


def write_flood_mask(image_path: str, polygon_wkts: Sequence[Tuple[int, str]], out_path: str) -> None:
    gdata = gdal.Open(image_path)

    outdriver = ogr.GetDriverByName("MEMORY")
    outDataSource = outdriver.CreateDataSource("memData")
    _ = outdriver.Open("memData", 1)
    # outLayer = outDataSource.CreateLayer("states_extent", raster_srs, geom_type=ogr.wkbMultiPolygon)
    outLayer = outDataSource.CreateLayer("states_extent", osr.SpatialReference(), geom_type=ogr.wkbMultiPolygon)

    burnField = "burn"
    idField = ogr.FieldDefn(burnField, ogr.OFTInteger)
    outLayer.CreateField(idField)
    featureDefn = outLayer.GetLayerDefn()

    for burn_value, wkt in polygon_wkts:
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
        outFeature.SetField(burnField, burn_value)
        outLayer.CreateFeature(outFeature)
        del outFeature

    target_ds = gdal.GetDriverByName("GTiff").Create(out_path, gdata.RasterXSize, gdata.RasterYSize, 1, gdal.GDT_Byte)
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    gdal.RasterizeLayer(target_ds, [1], outLayer, options=["ATTRIBUTE=%s" % burnField])

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(gdata.GetProjectionRef())

    target_ds.SetGeoTransform(gdata.GetGeoTransform())
    target_ds.SetProjection(raster_srs.ExportToWkt())


subtypes = ["un-classified", "no-damage", "minor-damage", "major-damage", "destroyed"]
subtype_to_index = {subtype: i for i, subtype in enumerate(subtypes, 1)}


def _task(post_json_path: Path) -> dict:
    with post_json_path.open() as fp:
        data = json.load(fp)

    flooded = False
    polygon_wkts = []
    for entry in data["features"]["xy"]:
        burn_value = subtype_to_index[entry["properties"]["subtype"]]
        if 3 <= burn_value:
            flooded = True
        polygon_wkts.append((burn_value, entry["wkt"]))

    post_image_path = post_json_path.parent.parent / "images" / (post_json_path.stem + ".tif")
    pre_image_path = (
        post_json_path.parent.parent
        / "images"
        / (post_json_path.stem.replace("post_disaster", "pre_disaster") + ".tif")
    )
    assert post_image_path.exists()
    assert pre_image_path.exists()

    out_path = post_json_path.parent.parent / "masks" / (post_json_path.stem + ".tif")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    write_flood_mask(str(post_image_path), polygon_wkts, str(out_path))
    assert out_path.exists()

    geotransform = gdal.Open(str(post_image_path)).GetGeoTransform()

    return {
        "pre": pre_image_path,
        "post1": post_image_path,
        "mask": out_path,
        "flooded": flooded,
        "longitude": geotransform[0],
        "latitude": geotransform[3],
    }


def main(geotiffs_dir: PathLike, output_df: str, n_jobs: int) -> None:
    geotiffs_dir = Path(geotiffs_dir)

    target_disasters = [
        "hurricane-florence",
        "hurricane-harvey",
        "midwest-flooding",
        "nepal-flooding",
    ]

    rows = []
    for target_disaster in target_disasters:
        args = sorted(geotiffs_dir.glob(f"**/*{target_disaster}*post_disaster.json"))
        with mp.Pool(processes=n_jobs) as pool:
            for row in tqdm(pool.imap(_task, args), total=len(args)):
                row["disaster"] = target_disaster
                row["pre"] = row["pre"].relative_to(geotiffs_dir)
                row["post1"] = row["post1"].relative_to(geotiffs_dir)
                row["mask"] = row["mask"].relative_to(geotiffs_dir)
                rows.append(row)

    df = pd.DataFrame(rows)
    n_folds = 5
    df["fold_index"] = -1
    for _, sub in df.index.to_series().groupby([df["disaster"], df["flooded"]]):
        sections = np.linspace(0, len(sub), n_folds + 1).astype(int)
        for i, (start, end) in enumerate(zip(sections, sections[1:])):
            df.loc[sub.iloc[start:end], "fold_index"] = i
    assert (df["fold_index"] != -1).all()

    df.to_csv(output_df, index=False)


if __name__ == "__main__":
    fire.Fire(main)
