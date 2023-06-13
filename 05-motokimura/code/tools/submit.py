import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--building', required=True)
    parser.add_argument('--road', required=True)
    parser.add_argument('--artifact_dir', default='/wdata')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--out')
    return parser.parse_args()


def main():
    args = parse_args()

    building_df = pd.read_csv(
        os.path.join(args.building, 'solution.csv')
    )
    assert (building_df[building_df.Object == 'Road'].WKT_Pix == 'LINESTRING EMPTY').all()
    building_df = building_df[building_df.Object == 'Building']

    road_df = pd.read_csv(
        os.path.join(args.road, 'solution.csv')
    )
    assert (road_df[road_df.Object == 'Building'].WKT_Pix == 'POLYGON EMPTY').all()
    road_df = road_df[road_df.Object == 'Road']

    df = pd.concat([building_df, road_df])
    print(df.head(15))

    exp_building = os.path.basename(os.path.normpath(args.building)).replace('exp_', '')
    exp_road = os.path.basename(os.path.normpath(args.road)).replace('exp_', '')
    assert exp_building == exp_road, (exp_building, exp_road)
    out_dir = f'exp_{exp_building}'
    if args.val:
        out_dir = os.path.join(args.artifact_dir, '_val/submissions', out_dir)
    else:
        out_dir = os.path.join(args.artifact_dir, 'submissions', out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'solution.csv')
    df.to_csv(out_path, index=False)
    print(f'saved {out_path}')

    if args.out is not None:
        df.to_csv(args.out, index=False)
        print(f'saved {args.out}')


if __name__ == '__main__':
    main()
