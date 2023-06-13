from pathlib import Path
from typing import Sequence, Union

import fiona
import fire
import numpy as np
import pandas as pd
import shapely.wkt
import skimage.measure
from fiona.crs import from_epsg
from osgeo import gdal, ogr, osr
from shapely.geometry import Polygon, mapping
from skimage.morphology import opening, square


def make_wgs84_utm_srs(longitude, latitude):
    """create a Spatial Reference object that is a WGS84 UTM projected coord system
    from longitude and latitude values."""
    north = int(latitude > 0)
    approx_zone = int((longitude + 180) / 6)
    srs = osr.SpatialReference()
    srs.SetUTM(approx_zone, north)  # zone, north=1,
    srs.SetWellKnownGeogCS("WGS84")
    return srs


def geo_coords_to_image_coords(image_geotran, in_wkt):
    """translates WKT geometry in geographic coordinates wgs84 (latitude, longitude)
    to WKT geometry in image coordinates (col, row)"""
    xmin = image_geotran[0]
    xres = image_geotran[1]
    ymax = image_geotran[3]
    yres = image_geotran[5]

    shapely_poly = shapely.wkt.loads(in_wkt)
    x, y = shapely_poly.exterior.coords.xy

    outcoords = []  # [(x. y),(x, y), ...]
    for coord in range(len(x)):
        outcoords.append((int((x[coord] - xmin) / xres), int((y[coord] - ymax) / yres)))
    out_wkt = Polygon(outcoords).wkt
    return out_wkt


def polygonize_pred_mask(target_ds):
    """ polygonize the input raster band in target_ds """
    band = target_ds.GetRasterBand(1)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(target_ds.GetProjectionRef())

    out_driver = ogr.GetDriverByName("MEMORY")
    out_ds = out_driver.CreateDataSource("memData")
    _ = out_driver.Open("memData", 1)
    out_layer = out_ds.CreateLayer("polygonize", raster_srs, geom_type=ogr.wkbMultiPolygon)
    field = ogr.FieldDefn("maskid", ogr.OFTInteger)
    out_layer.CreateField(field)

    gdal.Polygonize(band, None, out_layer, 0, [], callback=None)
    return out_ds


def remove_small_polygons_and_simplify(in_ds, area_threshold=5, simplify=True, simplify_tolerance=5):
    """ removes polygon features from the input dataset that are below the area threshold and simplifies polygons. """
    in_layer = in_ds.GetLayer("polygonize")
    extent = in_layer.GetExtent()
    longitude_origin = extent[0]
    latitude_origin = extent[2]
    target_srs = make_wgs84_utm_srs(longitude_origin, latitude_origin)
    source = osr.SpatialReference()  # the input dataset is in wgs84
    source.ImportFromEPSG(4326)
    source.SetAxisMappingStrategy(
        osr.OAMS_TRADITIONAL_GIS_ORDER
    )  # this is required for gdal>=3.0. The axis ordering is different between 3.x (y, x) and 2.x (x, y). see: https://github.com/OSGeo/gdal/issues/1546
    transform_to_utm = osr.CoordinateTransformation(source, target_srs)  # transform from wgs84 gcs to utm
    transform_to_wgs = osr.CoordinateTransformation(target_srs, source)  # transform from utm to wgs84 gcs

    fids_to_remove = []
    out_features = []  # features to save to shapefile in geojson format
    for feature in in_layer:
        mask_val = feature.GetField("maskid")
        if mask_val != 0:  # background is zero. only simplify non-zero polygons
            geom = feature.GetGeometryRef()
            geom.Transform(transform_to_utm)
            area = geom.GetArea()
            if area < area_threshold:
                fids_to_remove.append(feature.GetFID())
            else:
                # feature is above area threshold, so do the simplification
                if simplify:
                    geom = geom.SimplifyPreserveTopology(simplify_tolerance)
                geom.Transform(transform_to_wgs)
                wktgeom = geom.ExportToWkt()
                out_features.append(
                    {"geometry": wktgeom, "properties": {"fid": feature.GetFID(), "mask_val": mask_val}}
                )
        if mask_val == 0:  # always remove the background polygon
            fids_to_remove.append(feature.GetFID())
    for i in fids_to_remove:
        in_layer.DeleteFeature(i)
    # print(f"  removed {len(fids_to_remove)-1} building detection polygon features")
    return out_features


def get_flood_attributed_building_mask(barr, farr, perc_positive=0.25):
    """barr is building array, farr is flood array.
    perc_positive means any blob with number of flood pixels above this percentage, will be classified as fully flooded. otherwise it is not flooded"""
    barr = np.where(barr == 255, 1, 0)
    intersect = np.zeros(farr.shape)
    intersect = np.where((farr > 0) & (barr > 0), farr, barr)
    out_arr = np.zeros(farr.shape)

    binary_arr = np.where(intersect > 0, 1, 0)
    labeled_binary = skimage.measure.label(binary_arr)
    props = skimage.measure.regionprops(labeled_binary)

    for i in props:
        row_idxs = i["coords"][:, 0]
        col_idxs = i["coords"][:, 1]

        out_rowidxs = i["coords"][:, 0]
        out_colidxs = i["coords"][:, 1]

        # out_arr[out_rowidxs, out_colidxs] = np.argmax(np.bincount(intersect[row_idxs,col_idxs]))

        binned = np.bincount(intersect[row_idxs, col_idxs])
        if len(binned) > 2:
            if (
                binned[2] / np.sum(binned)
            ) > perc_positive:  # if flooded building pixels account for more than perc_positive of all pixels.
                out_arr[out_rowidxs, out_colidxs] = 2
            else:
                out_arr[out_rowidxs, out_colidxs] = 1
        else:
            out_arr[out_rowidxs, out_colidxs] = 1
    return out_arr


def main(
    mask_path: str,
    out_dir: str,
    flood_mask: str,
    square_size: int = 5,
    min_area: int = 5,
    simplify_tolerance: float = 0.75,
    perc_positive: float = 0.5,
    building_th: float = 0.5,
    flood_th: float = 0.5,
) -> None:
    Path(out_dir).mkdir(exist_ok=True)
    out_prefix = str(Path(out_dir) / (Path(mask_path).stem))

    ds = gdal.Open(mask_path)
    nrows = ds.RasterYSize
    ncols = ds.RasterXSize
    geotran = ds.GetGeoTransform()
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(ds.GetProjectionRef())
    in_arr = ds.ReadAsArray()
    in_arr = in_arr / 255
    in_arr = ((in_arr[0] * (1 - in_arr[1]) >= building_th) * 255).astype(np.uint8)
    ds = None
    # print("morph opening...")
    building_arr = opening(in_arr, square(square_size))

    flood_ds = gdal.Open(flood_mask)
    flood_arr = flood_ds.ReadAsArray()
    assert flood_arr.ndim == 2
    flood_arr = ((flood_arr / 255 >= flood_th) * 2).astype(np.uint8)  # in {0, 2}^n

    # print("flood attribution...")
    out_arr = get_flood_attributed_building_mask(building_arr, flood_arr, perc_positive=perc_positive)

    driver = gdal.GetDriverByName("MEM")
    outopen_ds = driver.Create("", ncols, nrows, 1, gdal.GDT_Byte)
    outopen_ds.SetGeoTransform(geotran)
    band = outopen_ds.GetRasterBand(1)
    outopen_ds.SetProjection(raster_srs.ExportToWkt())
    band.WriteArray(out_arr)
    band.FlushCache()
    # print("polygonize...")
    out_ds = polygonize_pred_mask(outopen_ds)
    # print("remove small polygons and simplify...")
    feats = remove_small_polygons_and_simplify(
        out_ds, area_threshold=min_area, simplify=True, simplify_tolerance=simplify_tolerance
    )

    columns = ["Object", "Flooded", "WKT_Pix", "WKT_Geo"]

    if len(feats) == 0:
        df = pd.DataFrame([["Building", False, "POLYGON EMPTY", "POLYGON EMPTY"]], columns=columns)
    else:
        rows = []
        for f in feats:
            wkt_image_coords = geo_coords_to_image_coords(geotran, f["geometry"])
            is_flooded = f["properties"]["mask_val"] == 2
            rows.append(["Building", is_flooded, wkt_image_coords, f["geometry"]])
        df = pd.DataFrame(rows, columns=columns)

    df.to_csv(f"{out_prefix}_wkt_df.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
