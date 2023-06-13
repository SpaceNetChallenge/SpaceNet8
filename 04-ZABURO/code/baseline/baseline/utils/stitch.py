"""
utility for stitching prediction tiles together
"""

import glob
import os

import numpy as np
from osgeo import gdal
from osgeo import osr

def write_geotiff(output_tif, ncols, nrows,
                  xmin, xres,ymax, yres,
                 raster_srs, label_arr):
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

def stitch_tiles(tile_filenames, out_dir, out_filedesc):
    assert(len(tile_filenames) > 0)
    tile_filenames.sort()
    origin_rows = []
    origin_cols = []

    for i in tile_filenames:
        split_filename = os.path.basename(i).split("_")
        origin_rows.append(int(split_filename[-3]))
        origin_cols.append(int(split_filename[-2]))
    unique_rows = np.unique(origin_rows)
    unique_cols = np.unique(origin_cols)
    print("shape: ", unique_rows, unique_cols)
    originrow = np.min(unique_rows)
    origincol = np.min(unique_cols)

    ds = gdal.Open(tile_filenames[0])
    tilex_size = ds.RasterXSize
    tiley_size = ds.RasterYSize
    nbands = ds.RasterCount
    ds = None

    outnrows = len(unique_rows)*(tiley_size) + tiley_size
    outncols = len(unique_cols)*(tilex_size) + tilex_size
    out_arr = np.zeros((nbands, outnrows, outncols), dtype=np.uint8)
    print(out_arr.shape)

    origin_row_filename = None # need the origin filename for geotransform info
    origin_col_filename = None
    for i in tile_filenames:
        if str(originrow) in os.path.basename(i):
            origin_row_filename = i
        if str(origincol) in os.path.basename(i):
            origin_col_filename = i
    print("origin row file", os.path.basename(origin_row_filename))
    print("origin col file:", os.path.basename(origin_col_filename))
    # now make the geotransform
    ds = gdal.Open(origin_row_filename)
    geotran = ds.GetGeoTransform()
    ymax = geotran[3]
    yres = geotran[5]
    ds = None
    ds = gdal.Open(origin_col_filename)
    geotran = ds.GetGeoTransform()
    xmin = geotran[0]
    xres = geotran[1]
    out_geotran = (xmin, xres, 0, ymax, 0, yres)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(ds.GetProjectionRef())
    ds = None
    
    count = 0
    for i in tile_filenames:
        count+=1
        ds = gdal.Open(i)
        geotran = ds.GetGeoTransform()
        row = abs(round((geotran[3] - out_geotran[3]) / yres))
        col = abs(round((geotran[0] - out_geotran[0]) / xres))
        arr = ds.ReadAsArray()
        #print(count, i, arr.shape)
        #print(col, col+tilex_size)
        #print(out_arr[:,row:row+tiley_size,col:col+tilex_size].shape)
        out_arr[:,row:row+tiley_size,col:col+tilex_size] = arr.astype(np.uint8)
        ds = None

    output_tif = os.path.join(out_dir, out_filedesc)
    write_geotiff(output_tif, outncols, outnrows,
                  xmin, xres,ymax, yres,
                  raster_srs, out_arr)
    

if __name__ == "__main__":
    in_dir = "" # path to prediction dir for flood predictions or foundation feature predictions
    out_dir = "" # output directory to save the stitched prediction .tif
    pred_type = "" # use either: floodpred, buildingpred, roadspeedpred

    preds = glob.glob(in_dir + f"/*{pred_type}.tif")

    print(f"stitching {len(preds)} prediction tiles")
    stitch_tiles(preds, out_dir, out_filedesc="all_stitched_{pred_type}.tif")