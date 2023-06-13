from typing import Sequence

import geopandas as gpd
import numpy as np
import osmnx
import shapely
from osgeo import gdal, ogr, osr

from sn8.data_prep.dsu import DSU


def _merge_upto_2_degree(roads: Sequence[shapely.geometry.LineString]):
    # 道路の重なりを交差点として検出する方法において、次数2の点を交差点として検出するのを避けるため、事前にマージしておく
    # マージは LineString の端点同士の距離が十分小さいものから貪欲に行う

    # 端点同士の距離を計算
    pos = np.concatenate(
        [
            np.stack(list(map(lambda x: x.coords[0], roads))),
            np.stack(list(map(lambda x: x.coords[-1], roads))),
        ]
    )
    dists = np.sqrt(((pos[:, np.newaxis, :] - pos[np.newaxis, :, :]) ** 2).sum(axis=-1))

    # 貪欲にマッチング
    dsu = DSU(len(pos))
    for i in range(dists.shape[0] - 1):
        if dsu.size(i) >= 2:
            continue
        for j in np.argsort(dists[i, i + 1 :]) + i + 1:
            if dists[i, j] >= 1e-3:
                break
            if dsu.size(j) >= 2:
                continue
            dsu.merge(i, j)
            break

    # 端点の index を道路の index に変換
    dsu2 = DSU(len(roads))
    for group in dsu.groups():
        for i, j in zip(group, group[1:]):
            dsu2.merge(i % len(roads), j % len(roads))

    return [shapely.ops.unary_union([roads[i] for i in group]) for group in dsu2.groups()]


def _rasterize(image_path: str, mask_path_out: str, roads: gpd.GeoDataFrame) -> None:
    gdata = gdal.Open(image_path)
    target_ds = gdal.GetDriverByName("GTiff").Create(
        mask_path_out, gdata.RasterXSize, gdata.RasterYSize, 1, gdal.GDT_Byte
    )

    target_ds.SetGeoTransform(gdata.GetGeoTransform())

    # set raster info
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(gdata.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    outdriver = ogr.GetDriverByName("MEMORY")
    outDataSource = outdriver.CreateDataSource("memData")
    _ = outdriver.Open("memData", 1)
    outLayer = outDataSource.CreateLayer("states_extent", raster_srs, geom_type=ogr.wkbMultiPolygon)

    # burn
    burnField = "burn"
    idField = ogr.FieldDefn(burnField, ogr.OFTInteger)
    outLayer.CreateField(idField)
    featureDefn = outLayer.GetLayerDefn()

    for geomShape in roads:
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(ogr.CreateGeometryFromWkt(geomShape.wkt))
        outFeature.SetField(burnField, 255)
        outLayer.CreateFeature(outFeature)
        del outFeature

    gdal.RasterizeLayer(target_ds, [1], outLayer, options=["ATTRIBUTE=%s" % burnField])


def _make_empty_mask(image_path: str, mask_path_out: str) -> None:
    gdata = gdal.Open(image_path)
    target_ds = gdal.GetDriverByName("GTiff").Create(
        mask_path_out, gdata.RasterXSize, gdata.RasterYSize, 1, gdal.GDT_Byte
    )

    target_ds.SetGeoTransform(gdata.GetGeoTransform())

    # set raster info
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(gdata.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)


def junction_mask(
    geojson_path: str,
    image_path: str,
    mask_path_out: str,
    road_buffer_meters: float = 2.0,
    junction_buffer_meters: float = 8.0,
) -> None:

    try:
        gdf = gpd.read_file(geojson_path)
    except Exception:
        print("Can't read geojson:", geojson_path)
        print("Making empty mask:", mask_path_out)
        _make_empty_mask(image_path, mask_path_out)
        return

    if len(gdf) == 0:
        print("Empty geojson is loaded. Making empty mask:", mask_path_out)
        _make_empty_mask(image_path, mask_path_out)
        return

    gdf_utm = osmnx.project_gdf(gdf)

    roads = []
    for line in gdf_utm["geometry"]:
        if isinstance(line, shapely.geometry.MultiLineString):
            roads.extend(line.geoms)
        else:
            roads.append(line)
    roads = _merge_upto_2_degree(roads)
    roads = [road.buffer(road_buffer_meters) for road in roads]

    junctions = []
    for i in range(len(roads)):
        p_i = roads[i]
        for j in range(i + 1, len(roads)):
            p_j = roads[j]
            intersection = p_i.intersection(p_j)
            if intersection.area == 0:
                continue
            if isinstance(intersection, shapely.geometry.MultiPolygon):
                for geom in intersection.geoms:
                    junctions.append(geom.centroid.buffer(junction_buffer_meters))
            else:
                junctions.append(intersection.centroid.buffer(junction_buffer_meters))
    junctions = [shapely.ops.unary_union(junctions)]
    gdf_junction = osmnx.project_gdf(gpd.GeoDataFrame(geometry=gpd.GeoSeries(junctions, crs=gdf_utm.crs)), gdf.crs)

    _rasterize(image_path, mask_path_out, gdf_junction["geometry"])
