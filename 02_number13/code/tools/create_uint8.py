from apls_tools import convert_to_8Bit
import os
import glob
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='full test')
    parser.add_argument('--in_dir', required=True, type=str, help='input dir')
    parser.add_argument('--out_dir', required=True, type=str, help='output dir')

    args = parser.parse_args()
    g = glob.glob(args.in_dir + '/*.tif')
    
    print ('Num images  : ', len(g))
    err = []
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    for item in tqdm(g):
        outpath = os.path.join(args.out_dir, os.path.basename(item))
        try:
            convert_to_8Bit(item, outpath)
        except:
            print('error ' , item)
            err.append(item)
    print('done.....')
    print('errors->', err)
