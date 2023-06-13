import os
import math
import glob
import argparse
from typing import List, Tuple

from osgeo import gdal, ogr, osr
from shapely.geometry import Polygon
import numpy as np

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",
                        required=True,
                        type=str,
                        help="path to the directory holding the AOIs directories")
    parser.add_argument("--aoi_dirs",
                        required=True,
                        nargs="+",
                        type=str,
                        help="directory names of the AOIs (e.g. Germany_Training_Public, Louisiana-East_Training_Public)")
    args = parser.parse_args()
    return args

def get_utm_epsg(lat: float, lon: float) -> int:
    """
    determines the utm zone given the latitude and longitude and then returns the zone's EPSG code
    """
    zone = (math.floor((lon + 180)/6) % 60) + 1
    north = True if lat >= 0 else False
    epsg_code = 32600
    epsg_code += zone
    if not north:
        epsg_code += 100
    return epsg_code

def write_geotiff(output_tif: str,
                  ncols: int,
                  nrows: int,
                  xmin: float,
                  xres: float,
                  ymax: float,
                  yres: float,
                 raster_srs: osr.SpatialReference,
                 label_arr: np.ndarray) -> None:
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif, ncols, nrows, len(label_arr), gdal.GDT_Byte)
    out_ds.SetGeoTransform((xmin, xres, 0, ymax, 0, yres))
    out_ds.SetProjection(raster_srs.ExportToWkt())
    for i in range(len(label_arr)):
        outband = out_ds.GetRasterBand(i+1)
        outband.WriteArray(label_arr[i])
        outband.SetNoDataValue(0)
        outband.FlushCache()
    out_ds = None

def buffer_roads(roads_shapefile_filename,
                 in_layer,
                 target_srs,
                 buffer_distance=2.0,
                 add_speed=True):
    """ buffers roads by some distance in meters. adds speed attribute to the line segments. """
    
    # reproject shp geometries to UTM to do buffer in meters
    # add speed attribute to the shapefile.
    source = osr.SpatialReference()
    if int(gdal.VersionInfo()[0]) == 3: # axis ordering in gdal 3 is different. see migration: https://github.com/OSGeo/gdal/blob/release/3.0/gdal/MIGRATION_GUIDE.TXT#L12
        source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    source.ImportFromEPSG(4326)

    target = osr.SpatialReference()
    target.ImportFromEPSG(target_srs)
    transform_to_utm = osr.CoordinateTransformation(source, target) # transform from wgs84 gcs to utm
    transform_to_gcs = osr.CoordinateTransformation(target, source) # transform from utm to wgs84 gcs
    
    mem_driver = ogr.GetDriverByName('ESRI Shapefile')
    buffer_ds = mem_driver.CreateDataSource(f"{roads_shapefile_filename[:-4]}_buffered.shp")
    buffer_lyr = buffer_ds.CreateLayer(os.path.basename(roads_shapefile_filename[:-4])+"_buffered.shp", source, geom_type=ogr.wkbPolygon)
    
    mask_id = ogr.FieldDefn("mask_id", ogr.OFTInteger)
    buffer_lyr.CreateField(mask_id)
    if add_speed:
        speed_id = ogr.FieldDefn("speed_id", ogr.OFTInteger)
        buffer_lyr.CreateField(speed_id)
    
    feature_def = buffer_lyr.GetLayerDefn()
    
    for feature in in_layer:
        in_geom = feature.GetGeometryRef()
        in_geom.Transform(transform_to_utm) # reproject to appropriate UTM zone for the buffer to take place in meters
        buffered_geom = in_geom.Buffer(buffer_distance) # buffer the geometry in meters
        buffered_geom.Transform(transform_to_gcs) # reproject back to WGS84
        
        out_feature = ogr.Feature(feature_def)
        out_feature.SetGeometry(buffered_geom)
        out_feature.SetField('mask_id', feature.GetField("mask_id")) # add the mask_id field into the MEM ds
        
        if add_speed:
            speed_mph = feature.GetField("speed_mph")
            burn_val = speed_to_burn_func(speed_mph) # get the binned speed value. 1 - 7
            out_feature.SetField('speed_id', burn_val)
        
        buffer_lyr.CreateFeature(out_feature)
        out_feature = None
    buffer_ds = None
    
def speed_to_burn_func(speed_mph, bin_size_mph=10.0, channel_value_mult=1):
    '''from cresi repo. 
    Speeds are binned into 7 bins. 
    bin every 10 mph or so
    Convert speed estimate to appropriate channel
    bin = 0 if speed = 0'''
    return int( int(math.ceil(speed_mph / bin_size_mph)) * channel_value_mult)
    
def create_mask_for_tile(pre_image,
                         buildings_shapefile,
                         roads_shapefile,
                         output_image_directory):
    #create_tiled_bldg_geojsons = True # MULTIPOLYGONs
    #create_tiled_road_geojsons = True # these are non-buffered. LINESTRINGs
    
    mask_types = ["binary_road", "binary_building", "flood", "road_speed"]
    road_buffer_distance = 3.0 # in meters
    
    n_speed_channels = 7 
    
    # 0 background
    # 1 non-floded building
    # 2 flooded building
    # 3 non-flooded road
    # 4 flooded road
    
    driver = ogr.GetDriverByName('ESRI Shapefile')
    bldgs_ds = driver.Open(buildings_shapefile, 0)
    bldgs_layer = bldgs_ds.GetLayer()
    bldgs_extent = bldgs_layer.GetExtent()
    
    driver2 = ogr.GetDriverByName('ESRI Shapefile')
    roads_ds = driver.Open(roads_shapefile, 0)
    roads_layer = roads_ds.GetLayer()
    roads_extent = roads_layer.GetExtent()
    
    lat = roads_extent[2]
    lon = roads_extent[0]
    target_crs = get_utm_epsg(lat, lon)

    # buffer the roads
    buffer_roads(roads_shapefile, roads_layer, target_crs, buffer_distance=road_buffer_distance, add_speed=True)
    driver3 = ogr.GetDriverByName('ESRI Shapefile')
    buffered_roads_ds = driver.Open(f"{roads_shapefile[:-4]}_buffered.shp", 0)
    buffered_roads_layer = buffered_roads_ds.GetLayer()
    
    preim_ds = gdal.Open(pre_image)
    preim_nrows = preim_ds.RasterYSize
    preim_ncols = preim_ds.RasterXSize
    assert(preim_nrows == preim_ncols)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(preim_ds.GetProjectionRef()) # preims are in wgs84

    band_count = preim_ds.RasterCount
    preim_geotran = preim_ds.GetGeoTransform()
    xmin = preim_geotran[0]
    xres = preim_geotran[1]
    ymax = preim_geotran[3]
    yres = preim_geotran[5]
    
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', preim_ncols, preim_nrows,  1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((xmin, xres, 0, ymax, 0, yres))
    band = target_ds.GetRasterBand(1)
    target_ds.SetProjection(raster_srs.ExportToWkt())

    # Rasterize flooded buildings
    gdal.RasterizeLayer(target_ds, [1], bldgs_layer, options=['ATTRIBUTE=mask_id'])
    building_flood_arr = target_ds.ReadAsArray()
    # get binary building array (0 no-building, 1 building)
    building_binary_arr = np.zeros(building_flood_arr.shape)
    building_binary_arr[building_flood_arr>0.5] = 1
    target_ds = None

    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', preim_ncols, preim_nrows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((xmin, xres, 0, ymax, 0, yres))
    band = target_ds.GetRasterBand(1)
    target_ds.SetProjection(raster_srs.ExportToWkt())

    # Rasterize flooded roads
    gdal.RasterizeLayer(target_ds, [1], buffered_roads_layer, options=['ATTRIBUTE=mask_id'])
    # get binary road array (0 no-road, 1 road)
    road_flood_arr = target_ds.ReadAsArray()
    road_binary_arr = np.zeros(road_flood_arr.shape)
    road_binary_arr[road_flood_arr > 0.5] = 1
    road_arr_sum = np.sum(road_binary_arr)

    if "road_speed" in mask_types:
        # rasterize with speed_id.
        # break single channel into multichannel
        driver = gdal.GetDriverByName('MEM')
        target_ds2 = driver.Create('', preim_ncols, preim_nrows, 1, gdal.GDT_Byte)
        target_ds2.SetGeoTransform((xmin, xres, 0, ymax, 0, yres))
        band = target_ds2.GetRasterBand(1)
        target_ds2.SetProjection(raster_srs.ExportToWkt())

        # Rasterize flooded roads
        gdal.RasterizeLayer(target_ds2, [1], buffered_roads_layer, options=['ATTRIBUTE=speed_id'])
        # get binary road array (0 no-road, 1 road)
        road_speed_arr = target_ds2.ReadAsArray()
        road_arr_shape = road_speed_arr.shape
        # now break road_speed_arr into multichannel format
        channeled_road_speed_arr = np.zeros((n_speed_channels+1,road_arr_shape[0], road_arr_shape[1])) # +1 to add a last channel with all roads in it.
        for b in range(n_speed_channels):
            channeled_road_speed_arr[b][road_speed_arr==b+1] = 1 # +1 because speed_id starts at 1, not zero.
        channeled_road_speed_arr[-1][road_speed_arr>0.5] = 1 # last channel includes all roads in it, regardless of speed.

    if "binary_building" in mask_types:
        out_imagename = f"building_{os.path.basename(pre_image).split('.')[0]}.tif"
        output_tif = os.path.join(output_image_directory, out_imagename)
        write_geotiff(output_tif, preim_ncols, preim_nrows,
                      xmin, xres, ymax, yres,
                     raster_srs, [building_binary_arr])

    if "binary_road" in mask_types:
        # road mask tile
        out_imagename = f"road_{os.path.basename(pre_image).split('.')[0]}.tif"
        output_tif = os.path.join(output_image_directory, out_imagename)
        write_geotiff(output_tif, preim_ncols, preim_nrows,
                      xmin, xres, ymax, yres,
                     raster_srs, [road_binary_arr])

    if "flood" in mask_types:
        # now combine the building flood and road flood masks into a single flood mask
        # channel 1 has non-flooded buildings
        # channel 2 has flooded buildings
        # channel 3 has non-flooded roads
        # channel 4 has flooded roads
        flood_arr = np.zeros((4, preim_ncols, preim_nrows))
        flood_arr[0][building_flood_arr==1] = 1
        flood_arr[1][building_flood_arr==2] = 1
        flood_arr[2][road_flood_arr==1] = 1
        flood_arr[3][road_flood_arr==2] = 1

        # flood mask tile
        out_imagename = f"flood_{os.path.basename(pre_image).split('.')[0]}.tif"
        output_tif = os.path.join(output_image_directory, out_imagename)
        write_geotiff(output_tif, preim_ncols, preim_nrows,
                      xmin, xres, ymax, yres,
                     raster_srs, flood_arr)

    if "road_speed" in mask_types:
        # road speed mask tile
        out_imagename = f"roadspeed_{os.path.basename(pre_image).split('.')[0]}.tif"
        output_tif = os.path.join(output_image_directory, out_imagename)
        write_geotiff(output_tif, preim_ncols, preim_nrows,
                      xmin, xres, ymax, yres,
                     raster_srs, channeled_road_speed_arr)
    preim_ds = None
    roads_ds = None
    bldgs_ds = None
    
def match_im_label(anno, bldgs, roads, pre_images):
    out_pre = []
    out_anno = []
    out_bu = []
    out_ro = []
    for i in anno:
        tileid = os.path.basename(i).split('.')[0]
        pre_im = [j for j in pre_images if f"_{tileid}.tif" in j][0]
        build = [j for j in bldgs if f"buildings_{tileid}.shp" in j][0]
        road = [j for j in roads if f"roads_{tileid}.shp" in j][0]
        
        out_anno.append(i)
        out_pre.append(pre_im)
        out_bu.append(build)
        out_ro.append(road)
        
    return out_anno, out_bu, out_ro, out_pre
    
if __name__ == "__main__":
    args = parse_args()
    root_dir = args.root_dir
    aois = args.aoi_dirs

    geojsons = []
    pre_images = []
    post_images = []
    build_labels = []
    road_labels = []
    for i in aois:
        anno = glob.glob(os.path.join(root_dir, i, "annotations", "*.geojson"))
        bldgs = glob.glob(os.path.join(root_dir, i, "annotations", "prepped_cleaned", "buildings*.shp"))
        roads = glob.glob(os.path.join(root_dir, i, "annotations", "prepped_cleaned", "roads*.shp"))
        pre = glob.glob(os.path.join(root_dir, i, "PRE-event", "*.tif"))

        an, bu, ro, preims = match_im_label(anno, bldgs, roads, pre)
        
        geojsons.extend(an)
        build_labels.extend(bu)
        road_labels.extend(ro)
        pre_images.extend(preims)

    print("creating masks...")
    for i in range(len(pre_images)):
        pre_image = pre_images[i]
        buildings_shapefile = build_labels[i]
        roads_shapefile = road_labels[i]
        out_dir_root = os.path.dirname(geojsons[i])
        output_image_directory = os.path.join(out_dir_root, "masks")
        if not os.path.exists(output_image_directory):
            os.mkdir(output_image_directory)
            os.chmod(output_image_directory, 0o777)
            
        create_mask_for_tile(pre_image,
                            buildings_shapefile,
                            roads_shapefile,
                            output_image_directory)
        print(f"{np.round((i+1)/len(pre_images)*100, 2)}%   ", end="\r")
    print()