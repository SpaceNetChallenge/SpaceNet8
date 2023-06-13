import argparse
import os

import numpy as np
import pandas as pd
from skimage import io
from tqdm.contrib import tzip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='/data/train')
    parser.add_argument('--test_dir', default='/data/test')
    parser.add_argument('--artifact_dir', default='/wdata')
    parser.add_argument('--fold_ver', default='folds_v3')
    parser.add_argument('--train_only', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    return parser.parse_args()


def compute_mse(im1, im2, mask):
    return np.mean((im1[mask].astype(float) - im2[mask].astype(float))**2)


def measure_image_similarities(pre_paths, post1_paths, post2_paths):
    mse1, mse2 = [], []

    for pre_path, post1_path, post2_path in tzip(pre_paths, post1_paths, post2_paths):
        pre = io.imread(pre_path)
        mask = pre.sum(axis=2) > 0  # remove black pixels. shape: [1300, 1300]

        post1 = io.imread(post1_path)
        mse1.append(compute_mse(pre, post1, mask))

        if isinstance(post2_path, str):
            post2 = io.imread(post2_path)
            mse2.append(compute_mse(pre, post2, mask))
        else:
            mse2.append(None)

    return mse1, mse2


def process_train_val_test_core(df, data_dir, warped_posts_dir):
    pre_paths = df['pre-event image'].values
    post1_paths = df['post-event image 1'].values
    post2_paths = df['post-event image 2'].values
    aois = df['aoi'].values
    pre_paths = [os.path.join(data_dir, aoi, 'PRE-event', fn) for fn, aoi in zip(pre_paths, aois)]
    post1_paths = [os.path.join(warped_posts_dir,  aoi, fn) for fn, aoi in zip(post1_paths, aois)]
    post2_paths = [os.path.join(warped_posts_dir,  aoi, fn) if isinstance(fn, str) else None
        for fn, aoi in zip(post2_paths, aois)]
    mse1, mse2 = measure_image_similarities(pre_paths, post1_paths, post2_paths)
    df['mse1'] = mse1
    df['mse2'] = mse2
    return df


def process_test(args):
    print(f'processing test..')
    csv_path = os.path.join(args.artifact_dir, 'test.csv')
    df = pd.read_csv(csv_path)
    df = process_train_val_test_core(df, data_dir=args.test_dir, warped_posts_dir=os.path.join(args.artifact_dir, 'warped_posts_test'))
    df.to_csv(csv_path, index=False)


def process_train_val(args):
    n_folds = 5  # XXX: fixed
    warped_posts_dir = os.path.join(args.artifact_dir, 'warped_posts_train')

    for i in range(n_folds):
        # train-set
        print(f'processing train-{i}..')
        csv_path = os.path.join(args.artifact_dir, args.fold_ver, f'train_{i}.csv')
        df = pd.read_csv(csv_path)
        df = process_train_val_test_core(df, data_dir=args.train_dir, warped_posts_dir=warped_posts_dir)
        df.to_csv(csv_path, index=False)

        # val-set
        print(f'processing val-{i}..')
        csv_path = os.path.join(args.artifact_dir, args.fold_ver, f'val_{i}.csv')
        df = pd.read_csv(csv_path)
        df = process_train_val_test_core(df, data_dir=args.train_dir, warped_posts_dir=warped_posts_dir)
        df.to_csv(csv_path, index=False)


def process_mosaic(args):
    n_folds = 5  # XXX: fixed
    for i in range(n_folds):
        print(f'processing mosaic train-{i}..')
        csv_path = os.path.join(args.artifact_dir, 'mosaics', f'train_{i}', 'mosaics.csv')
        df = pd.read_csv(csv_path)
        pre_paths = df['pre'].values
        post1_paths = df['post1'].values
        post2_paths = df['post2'].values
        mse1, mse2 = measure_image_similarities(pre_paths, post1_paths, post2_paths)
        df['mse1'] = mse1
        df['mse2'] = mse2
        df.to_csv(csv_path, index=False)


def main():
    args = parse_args()

    assert (not args.train_only) or (not args.test_only), (args.train_only, args.test_only)

    if not args.train_only:
        process_test(args)

    if not args.test_only:
        process_train_val(args)
        process_mosaic(args)


if __name__ == '__main__':
    main()
