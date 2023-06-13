import argparse
import os
from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm

# isort: off
from spacenet8_model.utils.misc import save_array_as_geotiff
# isort: on


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', nargs='+', type=int, required=True)
    parser.add_argument('--root_dir', default='/data/test')
    parser.add_argument('--artifact_dir', default='/wdata')
    return parser.parse_args()


def ensemble(pre_image_info, args):
    pre_fn = pre_image_info['file_name']
    aoi = pre_image_info['aoi']

    for i, exp_id in enumerate(args.exp_id):
        pred_path = os.path.join(args.artifact_dir, 'preds', f'exp_{exp_id:05d}', aoi, pre_fn)
        pred = io.imread(pred_path).astype(float)

        if len(pred.shape) == 2:
            pred = pred[:, :, np.newaxis]

        if i == 0:
            h, w, c = pred.shape
            ensembled = np.zeros(shape=[h, w, c], dtype=float)

        ensembled += pred

    ensembled /= len(args.exp_id)
    assert ensembled.min() >= 0
    assert ensembled.max() <= 255
    ensembled = ensembled.astype(np.uint8)
    ensembled = ensembled.transpose(2, 0, 1)  # HWC to CHW

    pre_path = os.path.join(args.root_dir, aoi, 'PRE-event', pre_fn)
    assert os.path.exists(pre_path), pre_path

    exp_dir = 'exp_'
    for exp_id in args.exp_id:
        exp_dir += f'{exp_id:05d}-'
    exp_dir = exp_dir[:-1]  # remove '-'
    out_dir = os.path.join(args.artifact_dir, 'ensembled_preds', exp_dir, aoi)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, pre_fn)
    save_array_as_geotiff(ensembled, pre_path, out_path)


def main():
    args = parse_args()

    df = pd.read_csv(os.path.join(args.artifact_dir, 'test.csv'))
    pre_image_info = []
    for i, row in df.iterrows():
        pre_image_info.append({
            'file_name': row['pre-event image'],
            'aoi': row['aoi']
        })

    with Pool(os.cpu_count()) as pool:
        with tqdm(total=len(pre_image_info)) as pbar:
            for _ in pool.imap_unordered(partial(ensemble, args=args), pre_image_info):
                pbar.update()


if __name__ == '__main__':
    main()
