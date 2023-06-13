import argparse
import os
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import numpy as np
from skimage import io
from tqdm import tqdm

# isort: off
from spacenet8_model.utils.misc import save_array_as_geotiff
# isort: on


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flood', required=True)
    parser.add_argument('--building', required=True)
    parser.add_argument('--weight', type=float, default=0.5)
    parser.add_argument('--artifact_dir', default='/wdata')
    parser.add_argument('--val', action='store_true')
    return parser.parse_args()


def refine(flood_path, args):
    flood = io.imread(flood_path)

    aoi = os.path.basename(os.path.dirname(flood_path))
    fn = os.path.basename(flood_path)
    building_path = os.path.join(args.building, aoi, fn)
    building = io.imread(building_path)[:, :, 0]  # 0: flood_building_channel, TODO
    assert len(building.shape) == 2, building.shape
    assert building.shape[0] == flood.shape[0], (building.shape, flood.shape)
    assert building.shape[1] == flood.shape[1], (building.shape, flood.shape)

    building_channel = 0  # 0: flood_building_channel, TODO
    w = args.weight
    flood[:, :, building_channel] = w * building.astype(float) + (1 - w) * flood[:, :, building_channel].astype(float)

    assert flood.min() >= 0
    assert flood.max() <= 255
    flood = flood.astype(np.uint8)
    flood = flood.transpose(2, 0, 1)  # HWC to CHW

    exp_dir = os.path.basename(os.path.dirname(os.path.dirname(flood_path)))

    if args.val:
        out_dir = os.path.join(args.artifact_dir, '_val/refined_preds_3', exp_dir, aoi)
    else:
        out_dir = os.path.join(args.artifact_dir, 'refined_preds_3', exp_dir, aoi)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fn)

    save_array_as_geotiff(flood, flood_path, out_path)


def main():
    args = parse_args()
    print(f'flood_building mask weight = {args.weight}')
    aois = [d for d in os.listdir(args.flood) if os.path.isdir(os.path.join(args.flood, d))]

    flood_paths = []
    for aoi in aois:
        paths = glob(os.path.join(args.flood, aoi, '*.tif'))
        paths.sort()
        flood_paths.extend(paths)

    with Pool(os.cpu_count()) as pool:
        with tqdm(total=len(flood_paths)) as pbar:
            for _ in pool.imap_unordered(partial(refine, args=args), flood_paths):
                pbar.update()


if __name__ == '__main__':
    main()
