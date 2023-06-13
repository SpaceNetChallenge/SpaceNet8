import argparse
import os
from glob import glob

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default='/data/test')
    parser.add_argument('--artifact_dir', default='/wdata')
    return parser.parse_args()


def get_mapping_csv(args, aoi):
    mapping_csv_path = glob(os.path.join(args.test_dir, aoi, '*_mapping.csv'))
    assert len(mapping_csv_path) == 1, mapping_csv_path
    return mapping_csv_path[0]


def main():
    args = parse_args()

    out_path = os.path.join(args.artifact_dir, 'test.csv')

    columns= ['label', 'pre-event image', 'post-event image 1', 'post-event image 2', 'aoi']
    rows_all_aois = []

    aois = [d for d in os.listdir(args.test_dir) if os.path.isdir(os.path.join(args.test_dir, d))]
    aois.sort()
    for aoi in aois:
        mapping_csv_path = get_mapping_csv(args, aoi)
        df = pd.read_csv(mapping_csv_path)
        df['aoi'] = aoi

        rows = df[columns].values
        rows_all_aois.extend(rows)

    train_df = pd.DataFrame(rows_all_aois, columns=columns)
    train_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()
