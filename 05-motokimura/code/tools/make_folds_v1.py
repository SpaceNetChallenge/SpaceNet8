import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='/data/train')
    parser.add_argument('--artifact_dir', default='/wdata')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=777)
    return parser.parse_args()


def get_mapping_csv(args, aoi):
    mapping_csv_path = glob(os.path.join(args.train_dir, aoi, '*_mapping.csv'))
    assert len(mapping_csv_path) == 1, mapping_csv_path
    return mapping_csv_path[0]


def main():
    args = parse_args()

    out_dir = os.path.join(args.artifact_dir, 'folds_v1')
    os.makedirs(out_dir, exist_ok=True)

    columns= ['label', 'pre-event image', 'post-event image 1', 'post-event image 2', 'aoi']
    rows_all_folds = [[] for _ in range(args.n_folds)]

    aois = [d for d in os.listdir(args.train_dir) if os.path.isdir(os.path.join(args.train_dir, d))]
    aois.sort()
    for aoi in aois:
        mapping_csv_path = get_mapping_csv(args, aoi)
        df = pd.read_csv(mapping_csv_path)
        df = shuffle(df, random_state=args.seed)
        df = df.reset_index()
        df['aoi'] = aoi

        rows = df[columns].values
        rows_split = np.array_split(rows, args.n_folds)
        for i in range(args.n_folds):
            rows_all_folds[i].extend(rows_split[i])
    
    for i in range(args.n_folds):
        val_rows = rows_all_folds[i]
        val_rows = shuffle(val_rows, random_state=args.seed)
        val_df = pd.DataFrame(val_rows, columns=columns)
        val_df.to_csv(os.path.join(out_dir, f'val_{i}.csv'), index=False)

        train_mask = np.ones(args.n_folds, dtype=bool)
        train_mask[i] = False
        train_rows = np.array(rows_all_folds, dtype=object)[train_mask]
        train_rows = np.concatenate(train_rows, axis=0)
        train_rows = shuffle(train_rows, random_state=args.seed)
        train_df = pd.DataFrame(train_rows, columns=columns)
        train_df.to_csv(os.path.join(out_dir, f'train_{i}.csv'), index=False)


if __name__ == '__main__':
    main()
