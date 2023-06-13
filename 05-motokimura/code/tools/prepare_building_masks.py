import argparse
import os
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import cv2
import numpy as np
import pandas as pd
from shapely.wkt import loads
from skimage import io
from skimage.morphology import dilation, erosion, square
from skimage.segmentation import watershed
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='/data/train')
    parser.add_argument('--edge_width', type=int, default=3)
    parser.add_argument('--contact_width', type=int, default=9)
    parser.add_argument('--artifact_dir', default='/wdata')
    return parser.parse_args()


def polygon_to_mask(poly, image_size):
    mask = np.zeros(image_size, np.uint8)
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords)]
    interiors = [int_coords(pi.coords) for pi in poly.interiors]
    cv2.fillPoly(mask, exteriors, 1)
    cv2.fillPoly(mask, interiors, 0)
    return mask


def get_gt_csv(args, aoi):
    gt_csv_path = glob(os.path.join(args.train_dir, aoi, '*_reference.csv'))
    assert len(gt_csv_path) == 1, gt_csv_path
    return gt_csv_path[0]


def prepare_3channel_mask(image_id, args, aoi, out_dir):
    gt_csv_path = get_gt_csv(args, aoi)
    df = pd.read_csv(gt_csv_path)
    df = df[(df.ImageId == image_id) & (df.Object == 'Building')]
    wkt_polys = df.Wkt_Pix

    image_path = os.path.join(args.train_dir, aoi, 'PRE-event', f'{image_id}.tif')
    image = io.imread(image_path)
    h, w = image.shape[:2]

    labels = np.zeros((h, w), dtype='uint16')
    current_label = 0  # building instance id
    for wkt_poly in wkt_polys:
        poly = loads(wkt_poly)
        if not poly.is_empty:
            current_label += 1
            msk = polygon_to_mask(poly, (h, w))
            labels[msk > 0] = current_label

    if labels.max() == 0:  # no building
        mask = np.zeros((h, w, 3), dtype='uint8')
        io.imsave(os.path.join(out_dir, f'{image_id}.png'), mask, check_contrast=False)
        return

    # generate 3-channel mask from labels
    # channel-1: building footprint
    footprint_mask = labels > 0
    # channel-2: building border
    border_mask = np.zeros_like(labels, dtype='bool')
    for l in range(1, labels.max() + 1):
        tmp_lbl = labels == l
        _k = square(args.edge_width)
        tmp = erosion(tmp_lbl, _k)
        tmp = tmp ^ tmp_lbl
        border_mask = border_mask | tmp
    # channel-3: building contact
    tmp = dilation(labels > 0, square(args.contact_width))
    tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = tmp | border_mask
    tmp = dilation(tmp, square(args.contact_width))
    contact_mask = np.zeros_like(labels, dtype='bool')
    for y0 in range(labels.shape[0]):
        for x0 in range(labels.shape[1]):
            if not tmp[y0, x0]:
                continue
            if labels[y0, x0] == 0:
                sz = 3
            else:
                sz = 1
            uniq = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1), max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
            if len(uniq[uniq > 0]) > 1:
                contact_mask[y0, x0] = True
    mask = np.stack((255 * footprint_mask, 255 * border_mask, 255 * contact_mask)).astype('uint8')
    mask = np.rollaxis(mask, 0, 3)
    io.imsave(os.path.join(out_dir, f'{image_id}.png'), mask, check_contrast=False)


def prepare_3channel_masks(args, aoi):
    out_3channels = os.path.join(args.artifact_dir, 'masks_building_3channel', aoi)
    os.makedirs(out_3channels, exist_ok=True)

    gt_csv_path = get_gt_csv(args, aoi)
    df = pd.read_csv(gt_csv_path)
    image_ids = df.ImageId.unique()
    with Pool(os.cpu_count()) as pool:
        with tqdm(total=len(image_ids)) as pbar:
            for _ in pool.imap_unordered(partial(prepare_3channel_mask, args=args, aoi=aoi, out_dir=out_3channels), image_ids):
                pbar.update()


def prepare_flood_mask(image_id, args, aoi, out_dir):
    gt_csv_path = get_gt_csv(args, aoi)
    df = pd.read_csv(gt_csv_path)
    df = df[(df.ImageId == image_id) & (df.Object == 'Building')]
    wkt_polys_flooded = df[df.Flooded == 'True'].Wkt_Pix
    wkt_polys_not_flooded = df[df.Flooded != 'True'].Wkt_Pix

    image_path = os.path.join(args.train_dir, aoi, 'PRE-event', f'{image_id}.tif')
    image = io.imread(image_path)
    h, w = image.shape[:2]

    labels = np.zeros((h, w), dtype='uint16')
    current_label = 0  # building instance id
    for wkt_poly in wkt_polys_flooded:
        poly = loads(wkt_poly)
        if not poly.is_empty:
            current_label += 1
            msk = polygon_to_mask(poly, (h, w))
            labels[msk > 0] = current_label
    flooded_max_label = current_label
    for wkt_poly in wkt_polys_not_flooded:
        poly = loads(wkt_poly)
        if not poly.is_empty:
            current_label += 1
            msk = polygon_to_mask(poly, (h, w))
            labels[msk > 0] = current_label

    # generate 2-channel mask from labels
    # channel-1: flooded building footprint
    flooded_mask = (labels > 0) & (labels <= flooded_max_label)
    # channel-2: flooded building footprint
    not_flooded_mask = (labels > 0) & (labels > flooded_max_label)
    # channel-3: no-data
    zeros = np.zeros((h, w), dtype='uint8')

    mask = np.stack((255 * flooded_mask, 255 * not_flooded_mask, zeros)).astype('uint8')
    mask = np.rollaxis(mask, 0, 3)
    io.imsave(os.path.join(out_dir, f'{image_id}.png'), mask, check_contrast=False)


def prepare_flood_masks(args, aoi):
    out_flood = os.path.join(args.artifact_dir, 'masks_building_flood', aoi)
    os.makedirs(out_flood, exist_ok=True)

    gt_csv_path = get_gt_csv(args, aoi)
    df = pd.read_csv(gt_csv_path)
    image_ids = df.ImageId.unique()
    with Pool(os.cpu_count()) as pool:
        with tqdm(total=len(image_ids)) as pbar:
            for _ in pool.imap_unordered(partial(prepare_flood_mask, args=args, aoi=aoi, out_dir=out_flood), image_ids):
                pbar.update()


def main():
    args = parse_args()
    aois = [d for d in os.listdir(args.train_dir) if os.path.isdir(os.path.join(args.train_dir, d))]
    for aoi in aois:
        print(f'preparing 3-channel building masks of {aoi} AOI')
        prepare_3channel_masks(args, aoi)
    for aoi in aois:
        print(f'preparing flooded building masks of {aoi} AOI')
        prepare_flood_masks(args, aoi)


if __name__ == '__main__':
    main()
