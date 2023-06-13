import argparse
import os
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import pandas as pd
from osgeo import gdal
from skimage import io
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/data/train')
    parser.add_argument('--artifact_dir', default='/wdata')
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


def get_mapping_csv(args, aoi):
    mapping_csv_path = glob(os.path.join(args.root_dir, aoi, '*_mapping.csv'))
    assert len(mapping_csv_path) == 1, mapping_csv_path
    return mapping_csv_path[0]


def warp_image(images, args, aoi, out_dir):
    # get width and height of pre image
    pre, post1, post2 = images  # pre, post-1, post-2 image file names
    pre_path = os.path.join(args.root_dir, aoi, 'PRE-event', pre)
    assert os.path.exists(pre_path), pre_path
    pre_image = io.imread(pre_path)
    h, w = pre_image.shape[:2]

    # warp post-1 image
    post1_path = os.path.join(args.root_dir, aoi, 'POST-event', post1)
    assert os.path.exists(post1_path), post1_path
    ds1 = gdal.Warp(
        os.path.join(out_dir, post1),
        post1_path,
        width=w,
        height=h,
        resampleAlg=gdal.GRIORA_Bilinear,
        outputType=gdal.GDT_Byte
    )
    ds1 = None

    # warp post-2 image
    if isinstance(post2, str):
        post2_path = os.path.join(args.root_dir, aoi, 'POST-event', post2)
        assert os.path.exists(post2_path), post2_path
        ds2 = gdal.Warp(
            os.path.join(out_dir, post2),
            post2_path,
            width=w,
            height=h,
            resampleAlg=gdal.GRIORA_Bilinear,
            outputType=gdal.GDT_Byte
        )
        ds2 = None


def warp_images(args, aoi):
    out_dir = 'warped_posts_test' if args.test else 'warped_posts_train'
    out_dir = os.path.join(args.artifact_dir, out_dir, aoi)
    os.makedirs(out_dir, exist_ok=True)

    mapping_csv_path = get_mapping_csv(args, aoi)
    df = pd.read_csv(mapping_csv_path)
    images_list = df[['pre-event image', 'post-event image 1', 'post-event image 2']].values
    with Pool(os.cpu_count()) as pool:
        with tqdm(total=len(images_list)) as pbar:
            for _ in pool.imap_unordered(partial(warp_image, args=args, aoi=aoi, out_dir=out_dir), images_list):
                pbar.update()


def main():
    args = parse_args()
    aois = [d for d in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, d))]
    for aoi in aois:
        print(f'warping post images of {aoi} AOI')
        warp_images(args, aoi)

if __name__ == '__main__':
    main()
