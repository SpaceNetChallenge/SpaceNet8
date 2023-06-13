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
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='/data/train')
    parser.add_argument('--road_width', type=int, default=12)
    parser.add_argument('--junction_radius', type=int, default=16)
    parser.add_argument('--artifact_dir', default='/wdata')
    return parser.parse_args()


def get_gt_csv(args, aoi):
    gt_csv_path = glob(os.path.join(args.train_dir, aoi, '*_reference.csv'))
    assert len(gt_csv_path) == 1, gt_csv_path
    return gt_csv_path[0]


def prepare_road_mask(image_id, args, aoi, out_dir):
    gt_csv_path = get_gt_csv(args, aoi)
    df = pd.read_csv(gt_csv_path)
    df = df[(df.ImageId == image_id) & (df.Object == 'Road')]
    wkt_lines_flooded = df[df.Flooded == 'True'].Wkt_Pix
    wkt_lines_not_flooded = df[df.Flooded != 'True'].Wkt_Pix

    image_path = os.path.join(args.train_dir, aoi, 'PRE-event', f'{image_id}.tif')
    image = io.imread(image_path)
    h, w = image.shape[:2]

    # generate flooded road mask (chennel-1) and not flooded road mask (chennel-2)
    flooded_mask = np.zeros((h, w), dtype='uint8')  # channel-1: flooded road mask
    not_flooded_mask = np.zeros((h, w), dtype='uint8')  # channel-2: not flooded road mask

    end_point_counts = {}  # used later to make junction_mask
    a, b = 10, 100000

    for wkt_lines, mask in [(wkt_lines_flooded, flooded_mask), (wkt_lines_not_flooded, not_flooded_mask)]:
        for wkt_line in wkt_lines:
            line = loads(wkt_line)
            if len(line.coords) == 0:  # no road
                continue

            xs, ys = line.coords.xy

            x_int = int(round(xs[0] * a))
            y_int = int(round(ys[0] * a))
            k = x_int * b + y_int
            if k not in end_point_counts:
                end_point_counts[k] = 0
            end_point_counts[k] += 1

            for i in range(len(xs) - 1):
                x_int = int(round(xs[i + 1] * a))
                y_int = int(round(ys[i + 1] * a))
                k = x_int * b + y_int
                if k not in end_point_counts:
                    end_point_counts[k] = 0
                if i == len(xs) - 2:
                    end_point_counts[k] += 1
                else:
                    end_point_counts[k] += 2

                cv2.line(mask, (int(xs[i]), int(ys[i])), (int(xs[i + 1]), int(ys[i + 1])), 255, args.road_width)

    # generate road junction mask (chennel-3)
    junction_mask = np.zeros((h, w), dtype='uint8')  # channel-2: road junction mask
    for k, count in end_point_counts.items():
        if count < 3:
            continue
        x_int = int(k / b)
        y_int = k - x_int * b
        x_int = int(x_int / a)
        y_int = int(y_int / a)
        cv2.circle(junction_mask, (x_int, y_int), args.junction_radius, 255, -1)
    junction_mask = (flooded_mask + not_flooded_mask > 0) * junction_mask

    mask = np.stack([flooded_mask, not_flooded_mask, junction_mask]).transpose(1, 2, 0)
    io.imsave(os.path.join(out_dir, f'{image_id}.png'), mask, check_contrast=False)


def prepare_road_masks(args, aoi):
    out_dir = os.path.join(args.artifact_dir, 'masks_road', aoi)
    os.makedirs(out_dir, exist_ok=True)

    gt_csv_path = get_gt_csv(args, aoi)
    df = pd.read_csv(gt_csv_path)
    image_ids = df.ImageId.unique()
    with Pool(os.cpu_count()) as pool:
        with tqdm(total=len(image_ids)) as pbar:
            for _ in pool.imap_unordered(partial(prepare_road_mask, args=args, aoi=aoi, out_dir=out_dir), image_ids):
                pbar.update()


def main():
    args = parse_args()
    aois = [d for d in os.listdir(args.train_dir) if os.path.isdir(os.path.join(args.train_dir, d))]
    for aoi in aois:
        print(f'preparing road masks of {aoi} AOI')
        prepare_road_masks(args, aoi)


if __name__ == '__main__':
    main()
