import argparse
import os
from glob import glob

import pandas as pd
from sklearn.utils import shuffle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='/data/train')
    parser.add_argument('--artifact_dir', default='/wdata')
    parser.add_argument('--seed', type=int, default=777)
    return parser.parse_args()


def get_mapping_csv(args, aoi):
    mapping_csv_path = glob(os.path.join(args.train_dir, aoi, '*_mapping.csv'))
    assert len(mapping_csv_path) == 1, mapping_csv_path
    return mapping_csv_path[0]


def main():
    args = parse_args()

    out_dir = os.path.join(args.artifact_dir, 'folds_v2')
    os.makedirs(out_dir, exist_ok=True)

    columns= ['label', 'pre-event image', 'post-event image 1', 'post-event image 2', 'aoi']

    fold_tiles_mapping = {
        0: {
            'aoi': 'Louisiana-East_Training_Public',
            'tile_starts_with': '105001001A0FFC00_0'
        },
        1: {
            'aoi': 'Louisiana-East_Training_Public',
            'tile_starts_with': '10400100684A4B00_1'
        },
        2: {
            'aoi': 'Louisiana-East_Training_Public',
            'tile_starts_with': '10300100AF395C00_2'
        },
        3: {
            'aoi': 'Germany_Training_Public',
            'tile_starts_with': '10500500C4DD7000_0'
        }
    }

    aois = [d for d in os.listdir(args.train_dir) if os.path.isdir(os.path.join(args.train_dir, d))]
    aois.sort()
    df_all_aois = []
    for aoi in aois:
        mapping_csv_path = get_mapping_csv(args, aoi)
        df = pd.read_csv(mapping_csv_path)
        df['aoi'] = aoi
        assert (df.columns == columns).all()
        df_all_aois.append(df)
    df_all_aois = pd.concat(df_all_aois)

    for fold_id, v in fold_tiles_mapping.items():
        aoi = v['aoi']
        tile_starts_with = v['tile_starts_with']

        val_df, train_df = [], []
        for i, row in df_all_aois.iterrows():
            if (row['aoi'] == aoi) and (row['pre-event image'].startswith(tile_starts_with)):
                val_df.append(row)
            else:
                train_df.append(row)

        val_df = pd.DataFrame(val_df)
        train_df = pd.DataFrame(train_df)
        assert len(val_df) + len(train_df) == len(df_all_aois)

        train_df = shuffle(train_df, random_state=args.seed)
        val_df = shuffle(val_df, random_state=args.seed)

        val_df.to_csv(os.path.join(out_dir, f'val_{fold_id}.csv'), index=False)
        train_df.to_csv(os.path.join(out_dir, f'train_{fold_id}.csv'), index=False)

        print(f'fold={fold_id}, n_val: {len(val_df)}, n_train: {len(train_df)}')


if __name__ == '__main__':
    main()
