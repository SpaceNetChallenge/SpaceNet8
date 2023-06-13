import argparse
import os
import ssl

import torch

ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', '-i', required=True)
    parser.add_argument('--out_dir', '-o', required=True)
    return parser.parse_args()


def prepare(args, model_name):
    for fold in ['fold0', 'fold1', 'fold2', 'fold3']:
        model = torch.load(os.path.join(args.in_dir, model_name, fold, f'{fold}_best.pth'))
        state_dict = model.state_dict()
        out_dir = os.path.join(args.out_dir, model_name, fold)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(out_dir, f'{fold}_best.pth'))


def main():
    args = parse_args()
    prepare(args, 'serx50_focal')
    prepare(args, 'r50a')


if __name__ == '__main__':
    main()
