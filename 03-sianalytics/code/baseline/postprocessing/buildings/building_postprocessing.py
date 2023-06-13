import os
import glob
import argparse

import numpy as np
import fiona
from fiona.crs import from_epsg
from osgeo import gdal, ogr, osr
import shapely.wkt
from shapely.geometry import mapping, shape, Polygon
import skimage.measure
from skimage.morphology import square, opening
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation_pred_dir",
                        type=str,
                        required=True)
    parser.add_argument("--flood_pred_dir",
                        type=str,
                        required=True)
    parser.add_argument("--out_submission_csv",
                        type=str,
                        required=True)
    parser.add_argument("--out_shapefile_dir",
                        type=str,
                        required=False,
                        default=None)
    parser.add_argument("--square_size",
                        type=int,
                        required=False,
                        default=5)
    parser.add_argument("--min_area",
                        type=float,
                        required=False,
                        default=5)
    parser.add_argument("--simplify_tolerance",
                        type=float,
                        required=False,
                        default=0.75)
    parser.add_argument("--percent_positive",
                        type=float,
                        required=False,
                        default=0.5)    
    args = parser.parse_args()
    return args

def make_wgs84_utm_srs(longitude, latitude):
    """ create a Spatial Reference object that is a WGS84 UTM projected coord system
    from longitude and latitude values."""
    north = int(latitude > 0)
    approx_zone = int((longitude + 180) / 6)
    srs = osr.SpatialReference()
    srs.SetUTM(approx_zone, north) # zone, north=1,
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
    
    outcoords = [] # [(x. y),(x, y), ...]
    for coord in range(len(x)):\
        outcoords.append((int((x[coord]-xmin)/xres), int((y[coord]-ymax)/yres)))
    out_wkt = Polygon(outcoords).wkt
    return out_wkt
    
def morph_opening_on_prediction(in_file, square_size=5):
    """ do morphological opening operation on a binary building prediction mask. """
    ds = gdal.Open(in_file)
    nrows = ds.RasterYSize
    ncols = ds.RasterXSize
    geotran = ds.GetGeoTransform()
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(ds.GetProjectionRef())
    in_arr = ds.ReadAsArray()
    ds = None
    
    out = opening(in_arr, square(square_size))
    
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', ncols, nrows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geotran)
    band = target_ds.GetRasterBand(1)
    target_ds.SetProjection(raster_srs.ExportToWkt())
    band.WriteArray(out)
    band.FlushCache()
    return out, target_ds
    
def polygonize_pred_mask(target_ds):
    """ polygonize the input raster band in target_ds """
    band = target_ds.GetRasterBand(1)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(target_ds.GetProjectionRef())
    
    out_driver = ogr.GetDriverByName('MEMORY')
    out_ds = out_driver.CreateDataSource('memData')
    tmp = out_driver.Open('memData', 1)
    out_layer = out_ds.CreateLayer("polygonize", raster_srs,
                                   geom_type=ogr.wkbMultiPolygon)
    field = ogr.FieldDefn('maskid', ogr.OFTInteger)
    out_layer.CreateField(field)
    
    gdal.Polygonize(band, None, out_layer, 0, [], callback=None)
    return out_ds
    
def remove_small_polygons_and_simplify(in_ds, area_threshold=5, simplify=True, simplify_tolerance=5):
    """ removes polygon features from the input dataset that are below the area threshold and simplifies polygons. """
    in_layer = in_ds.GetLayer('polygonize')
    extent = in_layer.GetExtent()
    longitude_origin = extent[0]
    latitude_origin = extent[2]
    target_srs = make_wgs84_utm_srs(longitude_origin, latitude_origin)
    source = osr.SpatialReference() # the input dataset is in wgs84
    source.ImportFromEPSG(4326)
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) # this is required for gdal>=3.0. The axis ordering is different between 3.x (y, x) and 2.x (x, y). see: https://github.com/OSGeo/gdal/issues/1546
    transform_to_utm = osr.CoordinateTransformation(source, target_srs) # transform from wgs84 gcs to utm
    transform_to_wgs = osr.CoordinateTransformation(target_srs, source) # transform from utm to wgs84 gcs
    
    fids_to_remove = []
    out_features = [] # features to save to shapefile in geojson format
    for feature in in_layer:
        mask_val = feature.GetField('maskid')
        if mask_val != 0: # background is zero. only simplify non-zero polygons
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
                out_features.append({'geometry': wktgeom,
                                     'properties': {'fid':feature.GetFID(),
                                                     'mask_val':mask_val}})
        if mask_val == 0: # always remove the background polygon
            fids_to_remove.append(feature.GetFID())
    for i in fids_to_remove:
        in_layer.DeleteFeature(i)
    #print(f"  removed {len(fids_to_remove)-1} building detection polygon features")
    return out_features

def save_to_shapefile(in_features, outfile):
    """ saves features to a shapefile. in_features should be in wgs84 coord system and be in geojson type format. feature geometry should be WKT """
    dest_crs = from_epsg(4326)
    schema = {'geometry': 'Polygon',
            'properties': {'fid': 'int',
                           'mask_val': 'int'}}
    with fiona.open(outfile, 'w', driver='ESRI Shapefile', crs=dest_crs, schema=schema) as c:
        for f in in_features:
            write_rec = {'geometry':mapping(shapely.wkt.loads(f['geometry'])),
                         'properties':f['properties']}
            c.write(write_rec)
            
def get_flood_attributed_building_mask(barr, farr, perc_positive=0.25):
    """ barr is building array, farr is flood array.
    perc_positive means any blob with number of flood pixels above this percentage, will be classified as fully flooded. otherwise it is not flooded """
    barr = np.where(barr == 255, 1, 0)
    intersect = np.zeros(farr.shape)
    intersect = np.where((farr>0) & (barr > 0), farr, barr)
    out_arr = np.zeros(farr.shape)

    binary_arr = np.where(intersect > 0, 1, 0)
    labeled_binary = skimage.measure.label(binary_arr)
    props = skimage.measure.regionprops(labeled_binary)

    for i in props:
        row_idxs = i["coords"][:,0]
        col_idxs = i["coords"][:,1]

        out_rowidxs = i["coords"][:,0]
        out_colidxs = i["coords"][:,1]

        #out_arr[out_rowidxs, out_colidxs] = np.argmax(np.bincount(intersect[row_idxs,col_idxs]))
        
        summed = np.sum(intersect[row_idxs,col_idxs])
        binned = np.bincount(intersect[row_idxs,col_idxs])
        if len(binned) > 2:
            if (binned[2] / np.sum(binned)) > perc_positive: # if flooded building pixels account for more than perc_positive of all pixels.
                out_arr[out_rowidxs, out_colidxs] = 2
            else:
                out_arr[out_rowidxs, out_colidxs] = 1
        else:
            out_arr[out_rowidxs, out_colidxs] = 1
    return out_arr

def main_w_flood(root_dir,
                flood_dir,
                out_submission_csv,
                out_shapefile_dir=None,
                square_size=5,
                min_area=5,
                simplify_tolerance=0.75,
                perc_positive=0.5):
    """ function that conflates the flood predictions with the buildings in the foundation features predictions. 
    
    Parameters:
    --------------
    root_dir (str): filepath to directory containing foundation feature predictions.
    flood_dir (str): filepath to directory containing flood predictions
    out_submission_csv (csv): absolute filepath for the output submission filename
    out_shapefile_dir (str): filepath to a directory to save output shapefile. If None, then no shapefiles are written
    square_size (int): the size of the structuring element used in morphological opening
    min_area (int): in sq. meters, the minimum area of a building feature for it to be kept. Only buildings with are above this threshold are kept
    simplify_tolerance (float): distance tolerance in polygon simplification (geom.Simplify())
    perc_positive (float): the percentage of flooded pixels within a building detection in order to consider the building instance entirely "flooded"

    Returns:
    --------
    None

    """
    bld_preds = glob.glob(root_dir + "/*buildingpred.tif")
    print(f"postprocessing {len(bld_preds)} tiles")

    cols = ['ImageId','Object','Flooded','Wkt_Pix','Wkt_Geo']
    record_list = []
    count = 0
    for in_file in bld_preds:
        name_root = os.path.basename(in_file).replace("_buildingpred.tif", '')
        #print(in_file)
        
        # morphological opening
        ds = gdal.Open(in_file)
        nrows = ds.RasterYSize
        ncols = ds.RasterXSize
        geotran = ds.GetGeoTransform()
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(ds.GetProjectionRef())
        in_arr = ds.ReadAsArray()
        ds = None
        #print("morph opening...")
        building_arr = opening(in_arr, square(square_size))

        flood_ds = gdal.Open(os.path.join(flood_dir, os.path.basename(in_file).replace("buildingpred.tif", "floodpred.tif")))
        flood_arr = flood_ds.ReadAsArray()

        #print("flood attribution...")
        out_arr = get_flood_attributed_building_mask(building_arr, flood_arr, perc_positive=perc_positive)

        driver = gdal.GetDriverByName('MEM')
        outopen_ds = driver.Create('', ncols, nrows, 1, gdal.GDT_Byte)
        outopen_ds.SetGeoTransform(geotran)
        band = outopen_ds.GetRasterBand(1)
        outopen_ds.SetProjection(raster_srs.ExportToWkt())
        band.WriteArray(out_arr)
        band.FlushCache()
        #print("polygonize...")
        out_ds = polygonize_pred_mask(outopen_ds)
        #print("remove small polygons and simplify...")
        feats = remove_small_polygons_and_simplify(out_ds,
                                                   area_threshold=min_area,
                                                   simplify=True,
                                                   simplify_tolerance=simplify_tolerance)

        #print("save shapefile...")
        if out_shapefile_dir is not None:
            out_shp_filename = os.path.join(out_shapefile_dir, os.path.basename(in_file[:-4])+".shp")
            save_to_shapefile(feats, out_shp_filename)

        ds = gdal.Open(in_file)
        image_geotran = ds.GetGeoTransform()
        ds = None
        if len(feats) == 0: # no buildings detecting in the tile, write no prediction to submission
            record_list.append([name_root, 'Building', 'Null', 'POLYGON EMPTY', 'POLYGON EMPTY'])
        else:
            for f in feats:
                wkt_image_coords = geo_coords_to_image_coords(image_geotran, f['geometry'])
                flood_val = 'True' if f['properties']['mask_val'] == 2 else 'False'
                record_list.append([name_root, 'Building', flood_val, wkt_image_coords, f['geometry']])
        count+=1
        print(f"{np.round((count/len(bld_preds))*100, 2)}%  ", end="\r")
    print()
    df = pd.DataFrame(record_list, columns=cols)
    df.to_csv(out_submission_csv, index=False)

if __name__ == "__main__":
    args = parse_args()
    
    root_dir = args.foundation_pred_dir
    flood_dir = args.flood_pred_dir
    out_submission_csv = args.out_submission_csv
    out_shapefile_dir = args.out_shapefile_dir
    square_size = args.square_size
    min_area = args.min_area
    simplify_tolerance = args.simplify_tolerance
    perc_positive = args.percent_positive

    main_w_flood(root_dir,
                flood_dir,
                out_submission_csv,
                out_shapefile_dir,
                square_size,
                min_area,
                simplify_tolerance,
                perc_positive)