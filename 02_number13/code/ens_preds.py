from skimage import io
import os
import glob
import argparse
from tqdm import tqdm
import pandas as pd


def cmaps(args):
    masks = args.masks
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    dir_1_masks = glob.glob(masks[0]+'/*.png')
    print(f'num masks {len(dir_1_masks)}')
    for item in tqdm(dir_1_masks):
        out_path = os.path.join(out_dir, os.path.basename(item))
        m = (io.imread(item) > 0).astype('uint8')
        for other_dir in masks[1::]:
            m += (io.imread(os.path.join(other_dir, os.path.basename(item))) > 0).astype('uint8')
        m = (m/len(masks)*255).astype('uint8')
        io.imsave(out_path, m, check_contrast=False)


# simple averaging
def avg_maps(args):
    masks = args.masks
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    dir_1_masks = glob.glob(masks[0]+'/*.png')
    print(f'num masks {len(dir_1_masks)}')
    for item in tqdm(dir_1_masks):
        out_path = os.path.join(out_dir, os.path.basename(item))
        m = io.imread(item)/255
        for other_dir in masks[1::]:
            m += io.imread(os.path.join(other_dir, os.path.basename(item)))/255
        m = m/len(masks)
        m = (m > args.thres)
        m = m*255
        io.imsave(out_path, m.astype('uint8'), check_contrast=False)


# averaging via temperature shaping
def temp_maps(args):
    t = 0.5
    masks = args.masks
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    dir_1_masks = glob.glob(masks[0]+'/*.png')
    dir_1_masks = [item for item in dir_1_masks if not 'flood_' in item]
    print(f'num masks {len(dir_1_masks)}')
    for item in tqdm(dir_1_masks):
        out_path = os.path.join(out_dir, os.path.basename(item))
        m = io.imread(item)/255
        for other_dir in masks[1::]:
            mf = io.imread(os.path.join(other_dir, os.path.basename(item)))/255
            m += mf**t
        m = m/len(masks)
        m = (m > args.thres)
        m = m*255
        io.imsave(out_path, m.astype('uint8'), check_contrast=False)


def join_and_filter(road_df, build_df, sol_out):
    df = df_build.append(df_road)
    df.to_csv(sol_out, index=False)
    flooded_perc = df['Flooded'].apply(lambda x: bool(x=='True')).sum()/df.shape[0]*100
    print('percentage with floods->', flooded_perc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--masks', nargs='+', required=False)
    parser.add_argument('-o', '--out_dir', required=False)
    parser.add_argument('-t', '--thres', default=0.5, type=float, required=False)
    parser.add_argument('--road_wkt', type=str)
    parser.add_argument('--build_wkt', type=str)
    parser.add_argument('--solution_out', type=str)
    parser.add_argument('--postprocess_wkt',  action='store_true', help='postprocess wkts...')
    args = parser.parse_args()
    if not args.postprocess_wkt:
        #avg_maps(args)
        temp_maps(args)
    else:
        df_build = pd.read_csv(args.build_wkt)
        df_road = pd.read_csv(args.road_wkt)
        join_and_filter(df_road, df_build, args.solution_out)