import os
import glob
import numpy as np
from osgeo import gdal, ogr, osr
import json
from shapely.geometry import Polygon
from skimage import io
import cv2
from tqdm import tqdm
from itertools import chain
np.random.seed(13)


def get_transforms(input_raster='', targetsr='', geom_transform=''):
    '''
      Convert latitude, longitude coords to pixexl coords.
      From spacenet geotools
    '''
    sourcesr = osr.SpatialReference()
    sourcesr.ImportFromEPSG(4326)

    if targetsr == '':
        src_raster = gdal.Open(input_raster)
        targetsr = osr.SpatialReference()
        targetsr.ImportFromWkt(src_raster.GetProjectionRef())
    coord_trans = osr.CoordinateTransformation(sourcesr, targetsr)
    if geom_transform == '':
        src_raster = gdal.Open(input_raster)
        transform = src_raster.GetGeoTransform()
    else:
        transform = geom_transform
    x_origin = transform[0]
    y_origin = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]

    rasterx, rastery = src_raster.RasterXSize, src_raster.RasterYSize
    return x_origin, y_origin, pixel_width, pixel_height, coord_trans,  rasterx, rastery


def latlon2pixel(latlon, x_origin, y_origin, pixel_width, pixel_height, coord_trans):
    '''
    Convert latitude, longitude coords to pixexl coords.
    From spacenet geotools
    '''
    geom = ogr.Geometry(ogr.wkbPoint)
    pix_latlon = []
    for item in latlon:
        lat, lon = item
        geom.AddPoint(lon, lat)
        geom.Transform(coord_trans)
        x_pix = (geom.GetPoint()[0] - x_origin) / pixel_width
        y_pix = (geom.GetPoint()[1] - y_origin) / pixel_height
        pix_latlon.append([x_pix, y_pix])
    return pix_latlon


def mask_for_polygons(polygons, im_size=(1024, 1024)):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 255)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def make_masks(img_subdirs, continue_on_exist=True):
    for subdir in img_subdirs:
        sub_outdir = os.path.join(subdir,  'masks')
        if not os.path.isdir(sub_outdir):
            os.makedirs(sub_outdir, exist_ok=True)
        images_paths = glob.glob(subdir+'/PS-RGB-u8/*.tif')
        for image_path in tqdm(images_paths):
            out_mask_path = os.path.join(sub_outdir, os.path.basename(image_path).replace('.tif', '.png'))
            if os.path.isfile(out_mask_path) and continue_on_exist:
                continue
            x_origin, y_origin, pixel_width, pixel_height, coord_trans, rasterx, rastery = get_transforms(image_path)
            ann_path = os.path.join(subdir, 'geojson_buildings', os.path.basename(image_path).replace('.tif', '.geojson').replace('PS-RGB', 'geojson_buildings'))
            anns = json.load(open(ann_path))
            pix_poly = []
            for buildings in anns['features']:
                if buildings['geometry'] is None:
                    continue
                type = buildings['geometry']['type']
                building_coords = buildings['geometry']['coordinates']
                #if type != 'Polygon' or type != 'MultiPolygon':
                #    print('Warn: is a ->', type)
                for b in building_coords:
                    if type == 'MultiPolygon':
                        for bi in b:
                            geo_points = [[item[1], item[0]] for item in bi]
                            pix_points = latlon2pixel(geo_points, x_origin, y_origin, pixel_width, pixel_height, coord_trans)
                            pix_poly.append(Polygon(pix_points))
                    elif type=='Polygon':
                        geo_points = [[item[1], item[0]] for item in b]
                        pix_points = latlon2pixel(geo_points, x_origin, y_origin, pixel_width, pixel_height,coord_trans)
                        pix_poly.append(Polygon(pix_points))
            mask = mask_for_polygons(pix_poly, im_size=(rastery, rasterx))
            #plt.subplot(121)
            #plt.imshow(tifffile.imread(image_path))
            #plt.subplot(122)
            #plt.imshow(mask)
            #plt.show()
            io.imsave(out_mask_path, mask, check_contrast=False)
            

if __name__ == '__main__':
    import sys
    IM_SUB_DIRS = sys.argv[1]
    if not isinstance(IM_SUB_DIRS, list):
        IM_SUB_DIRS = [IM_SUB_DIRS]
    make_masks(IM_SUB_DIRS, continue_on_exist=False)
