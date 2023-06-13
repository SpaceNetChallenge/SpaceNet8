import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', required=True, type=int)
    parser.add_argument('--artifact_dir', default='/wdata')
    parser.add_argument('--verbose', '-v', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = os.path.join(args.artifact_dir, f'models/exp_{args.exp_id:05d}/best.ckpt')
    state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))

    keys = state_dict['callbacks'].keys()
    keys = [k for k in keys if k.startswith('ModelCheckpoint')]
    assert len(keys) == 1, keys

    results = state_dict['callbacks'][keys[0]]
    print('best epoch: ', state_dict['epoch'])
    print('best score: ', results['best_model_score'].item())
    if args.verbose:
        print('metric: ', results['monitor'])
        print('weight: ', results['best_model_path'])


if __name__ == '__main__':
    main()
