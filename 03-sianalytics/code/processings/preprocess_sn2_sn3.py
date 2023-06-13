# Copyright SI-Anlaytics. All Rights Reserved.
# Written by Doyoung Jeong (github.com/tmits37)

import os
import rasterio
import glob
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from PIL import Image
import math

from rasterio.plot import show
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool


ROOT_DIR = '/data/SpaceNet2/'
ROOT_DIR = '/data/SpaceNet3/'
AOIS = ['AOI_2_Vegas', 'AOI_3_Paris', 'AOI_4_Shanghai', 'AOI_5_Khartoum']

def min_max_normalize(image, percentile):
    image = image.astype('float32')
    mask = np.mean(image, axis=2) != 0

    percent_min = np.percentile(image, percentile, axis=(0, 1))
    percent_max = np.percentile(image, 100-percentile, axis=(0, 1))

    if image.shape[1] * image.shape[0] - np.sum(mask) > 0:
        mdata = np.ma.masked_equal(image, 0, copy=False)
        mdata = np.ma.filled(mdata, np.nan)
        percent_min = np.nanpercentile(mdata, percentile, axis=(0, 1))

    norm = (image-percent_min) / (percent_max - percent_min)
    norm[norm < 0] = 0
    norm[norm > 1] = 1
    norm = (norm * 255).astype('uint8') * mask[:, :, np.newaxis]

    return norm


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


def make_input_args(imglist, geojsondir, img_outdir, ann_outdir, number):
    num_list = len(imglist)
    numbers = range(number, number+num_list, 1)

    input_args = []
    for idx, imgname in enumerate(imglist):
        args = [imgname, geojsondir, img_outdir, ann_outdir, numbers[idx]]
        input_args.append(dict(filename=imgname,
                              geojsondir=geojsondir,
                              img_outdir=img_outdir,
                              ann_outdir=ann_outdir,
                              number=numbers[idx]))

    number = number + num_list

    return input_args, number


def mapping(filename, geojsondir, img_outdir, ann_outdir, number):
    basename = os.path.basename(filename).replace('_PS-RGB_', '_geojson_roads_')
    basename = basename.replace('.tif', '.geojson')

    src = rasterio.open(filename)
    try:
        gdf = gpd.read_file(os.path.join(geojsondir, basename))

        epsg = get_utm_epsg(gdf.total_bounds[1], gdf.total_bounds[0])

        tmp_gdf = gdf.copy()
        tmp_gdf = gdf.to_crs(f'EPSG:{epsg}')

        tmp_geoms = []
        for i, feat in tmp_gdf.iterrows():
            geom = feat['geometry']
            tmp_geom = geom.buffer(3)
            tmp_geoms.append(tmp_geom)

        tmp_gdf['geometry'] = tmp_geoms
        tmp_gdf = tmp_gdf.to_crs('EPSG:4326')

        # the value you want in the output raster where a shape exists
        burnval = 1
        shapes = ((features['geometry'], burnval) for features in tmp_gdf)

        shapes = []
        for i, feat in tmp_gdf.iterrows():
            kt = [feat['geometry'], 1]
            shapes.append(kt)

        try:
            k = rasterize(shapes, out_shape=(src.height, src.width), transform=src.transform, all_touched=True)
        except:
            k = np.zeros((src.height, src.width)).astype('uint8')
    except:
        k = np.zeros((src.height, src.width)).astype('uint8')

    label = Image.fromarray(k).convert('P')
    label.putpalette(np.array(
        [[0,0,0],
        [255, 255, 255],
        ], dtype=np.uint8))

    img = src.read().transpose(1,2,0) # [H, W, C]
    img_norm = min_max_normalize(img, percentile=2)
    image = Image.fromarray(img_norm)

    basename = str(number).zfill(5) + '.png'

    image.save(os.path.join(img_outdir, basename))
    label.save(os.path.join(ann_outdir, basename))


def map_function(data):
    mapping(**data)


if __name__ == '__main__':
    outdir = os.path.join(ROOT_DIR, 'mmstyle')
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'img_dir'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'ann_dir'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'img_dir', 'train'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'ann_dir', 'train'), exist_ok=True)

    img_outdir = os.path.join(outdir, 'img_dir', 'train')
    ann_outdir = os.path.join(outdir, 'ann_dir', 'train')

    number = 0

    for aoi in AOIS:
        rootdir = os.path.join(ROOT_DIR, aoi)
        banddir = os.path.join(rootdir, 'PS-RGB')
        geojsondir = os.path.join(rootdir, 'geojson_roads')
        imglist = glob.glob(banddir + '/*.tif')

        print(aoi)
        print(len(imglist))
        input_args, number = make_input_args(imglist, geojsondir, img_outdir, ann_outdir, number)

        pool = Pool(20)
        for _ in tqdm(pool.imap_unordered(map_function, input_args), total=len(input_args)):
            pass
        pool.close()
        pool.join()

        print(number)