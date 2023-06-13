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
    parser.add_argument('--foundation', required=True)  # /path/to/refined_preds/exp_* directory
    parser.add_argument('--building', required=True)
    parser.add_argument('--weight', type=float, default=0.5)
    parser.add_argument('--artifact_dir', default='/wdata')
    parser.add_argument('--val', action='store_true')
    return parser.parse_args()


def refine(foundation_path, args):
    foundation = io.imread(foundation_path)

    aoi = os.path.basename(os.path.dirname(foundation_path))
    fn = os.path.basename(foundation_path)
    building_path = os.path.join(args.building, aoi, fn)
    building = io.imread(building_path)
    assert len(building.shape) == 2, building.shape
    assert building.shape[0] == foundation.shape[0], (building.shape, foundation.shape)
    assert building.shape[1] == foundation.shape[1], (building.shape, foundation.shape)

    building_channel = 0  # TODO
    w = args.weight
    foundation[:, :, building_channel] = w * building.astype(float) + (1 - w) * foundation[:, :, building_channel].astype(float)

    assert foundation.min() >= 0
    assert foundation.max() <= 255
    foundation = foundation.astype(np.uint8)
    foundation = foundation.transpose(2, 0, 1)  # HWC to CHW

    exp_dir = os.path.basename(os.path.dirname(os.path.dirname(foundation_path)))

    if args.val:
        out_dir = os.path.join(args.artifact_dir, '_val/refined_preds_2', exp_dir, aoi)
    else:
        out_dir = os.path.join(args.artifact_dir, 'refined_preds_2', exp_dir, aoi)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fn)

    save_array_as_geotiff(foundation, foundation_path, out_path)


def main():
    args = parse_args()
    print(f'building mask weight = {args.weight}')
    aois = [d for d in os.listdir(args.foundation) if os.path.isdir(os.path.join(args.foundation, d))]

    foundation_paths = []
    for aoi in aois:
        paths = glob(os.path.join(args.foundation, aoi, '*.tif'))
        paths.sort()
        foundation_paths.extend(paths)

    with Pool(os.cpu_count()) as pool:
        with tqdm(total=len(foundation_paths)) as pbar:
            for _ in pool.imap_unordered(partial(refine, args=args), foundation_paths):
                pbar.update()


if __name__ == '__main__':
    main()
