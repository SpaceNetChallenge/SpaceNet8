import fiona
import numpy as np
import pandas as pd
import shapely.wkt
import skimage.measure
from fiona.crs import from_epsg
from osgeo import gdal, ogr, osr
from shapely.geometry import Polygon, mapping
from skimage.morphology import opening, square


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
