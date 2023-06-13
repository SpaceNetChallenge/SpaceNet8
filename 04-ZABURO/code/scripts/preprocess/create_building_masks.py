from pathlib import Path
from typing import Sequence

import fire
import geopandas as gpd
import numpy as np
from osgeo import gdal, ogr, osr
from shapely.geometry import Polygon
from skimage.morphology import erosion, square
from tqdm import tqdm


def write_building_mask(image_path: str, polygons: Sequence[Polygon], out_path: str) -> None:
    gdata = gdal.Open(image_path)

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(gdata.GetProjectionRef())

    outdriver = ogr.GetDriverByName("MEMORY")
    outDataSource = outdriver.CreateDataSource("memData")
    tmp = outdriver.Open("memData", 1)
    outLayer = outDataSource.CreateLayer("states_extent", raster_srs, geom_type=ogr.wkbMultiPolygon)

    burnField = "burn"
    idField = ogr.FieldDefn(burnField, ogr.OFTInteger)
    outLayer.CreateField(idField)
    featureDefn = outLayer.GetLayerDefn()

    for j, geomShape in enumerate(polygons):
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(ogr.CreateGeometryFromWkt(geomShape.wkt))
        outFeature.SetField(burnField, j + 1)
        outLayer.CreateFeature(outFeature)
        del outFeature

    target_ds = gdal.GetDriverByName("GTiff").Create(out_path, gdata.RasterXSize, gdata.RasterYSize, 2, gdal.GDT_Byte)
    target_ds.SetGeoTransform(gdata.GetGeoTransform())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    gdal.RasterizeLayer(target_ds, [1], outLayer, options=["ATTRIBUTE=%s" % burnField])
    band.FlushCache()

    labels = band.ReadAsArray()
    edge_width = 1
    max_label_index = labels.max()

    footprint = labels > 0

    border = np.zeros_like(labels, dtype="bool")
    for label_index in range(1, max_label_index + 1):
        tmp = labels == label_index
        kernel = square(2 * edge_width + 1)
        tmp = erosion(tmp, kernel) ^ tmp
        border = border | tmp

    band = target_ds.GetRasterBand(1)
    band.WriteArray((255 * footprint).astype(np.uint8))
    band = target_ds.GetRasterBand(2)
    band.WriteArray((255 * border).astype(np.uint8))


def sn2(image_dir: str, geojson_dir: str, output_dir: str) -> None:
    image_paths = sorted(Path(image_dir).glob("*.tif"))
    geojson_paths = {p.stem.split("_")[-1]: p for p in Path(geojson_dir).glob("*.geojson")}

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    for image_path in tqdm(image_paths):
        image_id = image_path.stem.split("_")[-1]
        geojson_path = geojson_paths[image_id]
        output_path = output_dir_path / image_path.name

        gdf = gpd.read_file(geojson_path)
        write_building_mask(str(image_path), gdf["geometry"].to_list(), str(output_path))


def sn8(image_dir: str, geojson_dir: str, output_dir: str) -> None:
    image_paths = sorted(Path(image_dir).glob("*.tif"))
    geojson_paths = {p.stem: p for p in Path(geojson_dir).glob("*.geojson")}

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    for image_path in tqdm(image_paths):
        image_id = image_path.stem[image_path.stem.find("_") + 1 :]
        geojson_path = geojson_paths[image_id]
        output_path = output_dir_path / image_path.name

        gdf = gpd.read_file(geojson_path)
        write_building_mask(str(image_path), gdf.loc[gdf["building"] == "yes", "geometry"].to_list(), str(output_path))


if __name__ == "__main__":
    fire.Fire({"sn2": sn2, "sn8": sn8})
