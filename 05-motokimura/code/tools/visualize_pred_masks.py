import argparse
import os
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foundation', type=str, default=None)
    parser.add_argument('--flood', type=str, default=None)
    parser.add_argument('--out_dir', default='/wdata/vis_preds/exp_xxxx')
    parser.add_argument('--root_dir', default='/data/test')
    parser.add_argument('--artifact_dir', default='/wdata')
    return parser.parse_args()


def get_mapping_csv(args, aoi):
    mapping_csv_path = glob(os.path.join(args.root_dir, aoi, '*_mapping.csv'))
    assert len(mapping_csv_path) == 1, mapping_csv_path
    return mapping_csv_path[0]


def load_flood_mask(images, args, aoi):
    pre, _, _ = images
    path = os.path.join(args.flood, aoi, pre)
    pred = io.imread(path).astype(float)

    h, w = pred.shape[:2]
    vis = np.zeros((h, w, 3), dtype=float)

    # TODO: map channels from meta.json
    vis[:, :, 0] = np.clip(pred[:, :, 0] + pred[:, :, 2], a_max=255, a_min=0)
    vis[:, :, 1] = np.clip(pred[:, :, 1] + pred[:, :, 3], a_max=255, a_min=0)

    return vis.astype(np.uint8)

def load_foundation_mask(images, args, aoi):
    pre, _, _ = images
    path = os.path.join(args.foundation, aoi, pre)
    pred = io.imread(path).astype(float)

    h, w = pred.shape[:2]
    vis = np.zeros((h, w, 3), dtype=float)

    # TODO: map channels from meta.json
    vis[:, :, 0] = pred[:, :, 1]
    vis[:, :, 1] = np.clip(pred[:, :, 0] + pred[:, :, 3], a_max=255, a_min=0)
    vis[:, :, 2] = np.clip(pred[:, :, 2] + pred[:, :, 4], a_max=255, a_min=0)

    return vis.astype(np.uint8)


def load_images(images, args, aoi):
    # load pre image
    pre, post1, post2 = images  # pre, post-1, post-2 image file names
    pre_path = os.path.join(args.root_dir, aoi, 'PRE-event', pre)
    assert os.path.exists(pre_path), pre_path
    pre_image = io.imread(pre_path)
    h, w = pre_image.shape[:2]

    # load post-1 image
    warped_dir = 'warped_posts_test'
    post1_path = os.path.join(args.artifact_dir, warped_dir, aoi, post1)
    assert os.path.exists(post1_path), post1_path
    post1_image = io.imread(post1_path)

    # load post-2 image if exists
    if isinstance(post2, str):
        post2_path = os.path.join(args.artifact_dir, warped_dir, aoi, post2)
        assert os.path.exists(post2_path), post2_path
        post2_image = io.imread(post2_path)
    else:
        post2_image = np.zeros((h, w, 3), dtype=np.uint8)

    return pre_image, post1_image, post2_image


def visualize_image(images, args, aoi, out_dir):
    pre_image, post1_image, post2_image = load_images(images, args, aoi)

    h, w = pre_image.shape[:2]
    canvas_images = np.zeros((h, 3 * w, 3), dtype=np.uint8)
    canvas_images[:, :w] = pre_image
    canvas_images[:, w:2*w] = post1_image
    canvas_images[:, 2*w:] = post2_image

    pre, _, _ = images
    image_id, _ = os.path.splitext(pre)
    out_path = os.path.join(out_dir, f'{image_id}.jpg')

    canvas = np.zeros((2 * h, 3 * w, 3), dtype=np.uint8)
    # 1st row
    canvas[:h, :] = canvas_images

    # 2nd row
    if args.foundation is not None:
        foundation_mask = load_foundation_mask(images, args, aoi)
        canvas[h:, :w] = foundation_mask
    if args.flood is not None:
        flood_mask = load_flood_mask(images, args, aoi)
        canvas[h:, w:2*w] = flood_mask

    io.imsave(out_path, canvas, check_contrast=False)


def visualize_aoi(args, aoi):
    out_dir = os.path.join(args.out_dir, aoi)
    os.makedirs(out_dir, exist_ok=True)

    mapping_csv_path = get_mapping_csv(args, aoi)
    df = pd.read_csv(mapping_csv_path)
    images_list = df[['pre-event image', 'post-event image 1', 'post-event image 2']].values
    with Pool(os.cpu_count()) as pool:
        with tqdm(total=len(images_list)) as pbar:
            for _ in pool.imap_unordered(partial(visualize_image, args=args, aoi=aoi, out_dir=out_dir), images_list):
                pbar.update()


def main():
    args = parse_args()
    aois = [d for d in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, d))]
    for aoi in aois:
        print(f'visualizing {aoi} AOI')
        visualize_aoi(args, aoi)


if __name__ == '__main__':
    main()
