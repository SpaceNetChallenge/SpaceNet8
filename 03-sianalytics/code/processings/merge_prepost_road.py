import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

import cv2
from osgeo import gdal


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', type=str, required=True)
    parser.add_argument('--image-path', type=str,
                        default='/data/SpaceNet8/Testing/Louisiana-West_Test_Public')
    parser.add_argument('--threshold', type=int, default=127)
    parser.add_argument('--preonly', action='store_true')
    parser.add_argument('--premask', action='store_false')

    args = parser.parse_args()

    return args


def merge_prepostroad(img_name, rootdir, predir, postdir, postdir_2, datadir, threshold=127, preonly=False, premask=False):
    img_id = os.path.basename(img_name)
    src = gdal.Open(os.path.join(predir, img_id))
    pre = src.ReadAsArray()[1]

    if preonly:
        union_val = pre>=threshold
        kernel = np.ones((7,7), np.uint8)
        iteration=5
        img_erode = union_val.astype('uint8')
        for i in range(iteration):
            img_dil = cv2.dilate(img_erode.astype('uint8'), kernel)
            img_erode = cv2.erode(img_dil.astype('uint8'), kernel)

        union_val = img_erode

    else:
        post = gdal.Open(os.path.join(postdir, img_id)).ReadAsArray()[1]
        # post_2 = gdal.Open(os.path.join(postdir_2, img_id)).ReadAsArray()[1]
        # union_val = np.logical_or(np.logical_or(pre>=threshold, post>=threshold), post_2>=threshold)
        pre = pre.astype('float32')
        post = post.astype('float32')
        union = pre + post
        union_val = union >= 2*threshold
        kernel = np.ones((7,7), np.uint8)
        iteration=5
        img_erode = union_val.astype('uint8')
        for i in range(iteration):
            img_dil = cv2.dilate(img_erode.astype('uint8'), kernel)
            img_erode = cv2.erode(img_dil.astype('uint8'), kernel)

        union_val = img_erode

        if premask:
            prename = img_id.replace('_floodpred.tif', '.tif')
            data = gdal.Open(os.path.join(datadir, prename)).ReadAsArray().astype('float32')
            mask = np.sum(data, axis=0)==0
            union_val[mask] = 0
    
    union_val = union_val.astype('uint8')
    
    [rows, cols] = pre.shape
    dstpath = os.path.join(rootdir, img_id)
    driver = gdal.GetDriverByName('GTiff')
    dst = driver.Create(dstpath, cols, rows, 1, gdal.GDT_Byte)
    dst.SetGeoTransform(src.GetGeoTransform())
    dst.SetProjection(src.GetProjection())
    dst.GetRasterBand(1).WriteArray(union_val)
    dst.FlushCache()
    
    src = None
    dst = None
    

def map_function(data):
    merge_prepostroad(**data)


if __name__ == '__main__':
    args = parse_args()

    rootdir = args.rootdir
    threshold = args.threshold
    preonly = args.preonly
    premask = args.premask

    # prefix
    predir = os.path.join(rootdir, 'pre')
    postdir = os.path.join(rootdir, 'post1')
    postdir_2 = os.path.join(rootdir, 'post2')
    datadir = os.path.join(args.image_path, 'PRE-event')

    imglist = glob.glob(predir + '/*.tif')
    imglist = sorted(imglist)

    input_args = []
    for img_name in imglist:
        arg = dict()
        arg['img_name'] = img_name
        arg['rootdir'] = rootdir
        arg['predir'] = predir
        arg['postdir'] = postdir
        arg['postdir_2'] = postdir_2
        arg['datadir'] = datadir
        arg['threshold'] = threshold
        arg['preonly'] = preonly
        arg['premask'] = premask
        input_args.append(arg)

    pool = Pool(20)
    for _ in tqdm(pool.imap_unordered(map_function, input_args), total=len(input_args)):
        pass
    pool.close()
    pool.join()
    print("logit merging finished")

