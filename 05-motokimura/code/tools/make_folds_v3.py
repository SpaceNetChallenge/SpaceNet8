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

    out_dir = os.path.join(args.artifact_dir, 'folds_v3')
    os.makedirs(out_dir, exist_ok=True)

    fold_tiles_mapping = {
        0: [
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '105001001A0FFC00_0',
                'y_min': 1,
                'y_max': 5
            },
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '10400100684A4B00_1',
                'y_min': 77,
                'y_max': 81
            },
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '10300100AF395C00_2',
                'y_min': 43,
                'y_max': 46
            },
            {
                'aoi': 'Germany_Training_Public',
                'tile_group': '10500500C4DD7000_0',
                'x_min': 28,
                'x_max': 31
            },
        ],
        1: [
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '105001001A0FFC00_0',
                'y_min': 6,
                'y_max': 11
            },
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '10400100684A4B00_1',
                'y_min': 82,
                'y_max': 87
            },
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '10300100AF395C00_2',
                'y_min': 32,
                'y_max': 36
            },
            {
                'aoi': 'Germany_Training_Public',
                'tile_group': '10500500C4DD7000_0',
                'x_min': 39,
                'x_max': 45
            },
        ],
        2: [
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '105001001A0FFC00_0',
                'y_min': 16,
                'y_max': 18
            },
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '10400100684A4B00_1',
                'y_min': 70,
                'y_max': 76
            },
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '10300100AF395C00_2',
                'y_min': 47,
                'y_max': 55
            },
            {
                'aoi': 'Germany_Training_Public',
                'tile_group': '10500500C4DD7000_0',
                'x_min': 32,
                'x_max': 38
            },
        ],
        3: [
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '105001001A0FFC00_0',
                'y_min': 19,
                'y_max': 22
            },
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '10400100684A4B00_1',
                'y_min': 88,
                'y_max': 94
            },
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '10300100AF395C00_2',
                'y_min': 56,
                'y_max': 65
            },
            {
                'aoi': 'Germany_Training_Public',
                'tile_group': '10500500C4DD7000_0',
                'x_min': 15,
                'x_max': 21
            },
        ],
        4: [
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '105001001A0FFC00_0',
                'y_min': 12,
                'y_max': 15
            },
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '10400100684A4B00_1',
                'y_min': 95,
                'y_max': 106
            },
            {
                'aoi': 'Louisiana-East_Training_Public',
                'tile_group': '10300100AF395C00_2',
                'y_min': 37,
                'y_max': 42
            },
            {
                'aoi': 'Germany_Training_Public',
                'tile_group': '10500500C4DD7000_0',
                'x_min': 22,
                'x_max': 27
            },
        ],
    }

    aois = [d for d in os.listdir(args.train_dir) if os.path.isdir(os.path.join(args.train_dir, d))]
    aois.sort()
    df_all_aois = []
    for aoi in aois:
        mapping_csv_path = get_mapping_csv(args, aoi)
        df = pd.read_csv(mapping_csv_path)
        df['aoi'] = aoi
        df_all_aois.append(df)
    df_all_aois = pd.concat(df_all_aois)
    df_all_aois = df_all_aois.reset_index(drop=True)

    for fold_id, mappings in fold_tiles_mapping.items():
        val_indices = []

        for m in mappings:
            aoi = m['aoi']
            tile_group = m['tile_group']
            x_min = m.get('x_min', -1)
            y_min = m.get('y_min', -1)
            x_max = m.get('x_max', 10000)
            y_max = m.get('y_max', 10000)

            for i, row in df_all_aois.iterrows():
                if row['aoi'] != aoi:
                    continue

                file_name = row['pre-event image']  # e.g., 104001006504F400_0_27_41.tif
                file_name, _ = os.path.splitext(file_name)
                splits = file_name.split('_')

                if f'{splits[0]}_{splits[1]}' != tile_group:
                    continue

                x = int(splits[2])
                if x < x_min or x > x_max:
                    continue

                y = int(splits[3])
                if y < y_min or y > y_max:
                    continue

                val_indices.append(i)

        train_indices = [i for i in df_all_aois.index if i not in val_indices]

        val_df = df_all_aois.loc[val_indices]
        train_df = df_all_aois.loc[train_indices]
        assert len(val_df) + len(train_df) == len(df_all_aois)

        train_df = shuffle(train_df, random_state=args.seed)
        val_df = shuffle(val_df, random_state=args.seed)

        val_df.to_csv(os.path.join(out_dir, f'val_{fold_id}.csv'), index=False)
        train_df.to_csv(os.path.join(out_dir, f'train_{fold_id}.csv'), index=False)

        print(f'fold={fold_id}, n_val: {len(val_df)}, n_train: {len(train_df)}')


if __name__ == '__main__':
    main()
