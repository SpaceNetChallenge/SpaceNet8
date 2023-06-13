import argparse
import json
import os

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# isort: off
from spacenet8_model.datasets import get_test_dataloader
from spacenet8_model.models import get_model
from spacenet8_model.utils.config import load_config
from spacenet8_model.utils.misc import get_flatten_classes, save_array_as_geotiff
from train_net import get_default_cfg_path
# isort: on


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_id',
        type=int,
        required=True
    )
    parser.add_argument(
        '--config',
        default=None,
        help='YAML config path. This will overwrite `configs/default.yaml`')
    parser.add_argument(
        '--artifact_dir',
        default='/wdata'
    )
    parser.add_argument(
        '--device',
        default='cuda')
    parser.add_argument(
        '--val',
        action='store_true'
    )
    parser.add_argument(
        '--tta_hflip_channels',
        type=int,
        nargs='*',
        default=[]
    )
    parser.add_argument(
        '--tta_vflip_channels',
        type=int,
        nargs='*',
        default=[]
    )
    parser.add_argument(
        '--use_ema_weight',
        action='store_true'
    )
    parser.add_argument(
        '--use_swa_weight',
        action='store_true'
    )
    parser.add_argument(
        '--override_model_dir')
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='overwrite configs (e.g., General.fp16=true, etc.)')
    return parser.parse_args()


def load_test_config(args):
    model_dir = os.path.join(args.artifact_dir, 'models') if (args.override_model_dir is None) else args.override_model_dir
    config_exp_path = os.path.join(model_dir, f'exp_{args.exp_id:05d}/config.yaml')
    config_exp: DictConfig = OmegaConf.load(config_exp_path)
    task: str = config_exp.task

    default_cfg_path: str = get_default_cfg_path(task)
    
    cfg_paths = [config_exp_path]
    if args.config is not None:
        cfg_paths.append(args.config)

    config: DictConfig = load_config(
        default_cfg_path,
        cfg_paths,
        update_dotlist=args.opts
    )
    return config


def crop_center(pred, crop_wh):
    _, h, w = pred.shape
    crop_w, crop_h = crop_wh
    assert w >= crop_w
    assert h >= crop_h

    left = (w - crop_w) // 2
    right = crop_w + left
    top = (h - crop_h) // 2
    bottom = crop_h + top

    return pred[:, top:bottom, left:right]


def prepare_test_dataloaders(config, args):
    test_to_val = args.val
    classes = np.array(get_flatten_classes(config))
    n_classses = len(classes)

    # prepare dataloaders, flipping flags, and weights for averaging
    test_dataloaders, flags_hflip, flags_vflip, weights = [], [], [], []

    # default dataloader (w/o tta)
    test_dataloaders.append(get_test_dataloader(config, test_to_val=test_to_val))
    flags_hflip.append(False)
    flags_vflip.append(False)
    weights.append([1] * n_classses)

    # dataloader w/ tta horizontal flipping
    if len(args.tta_hflip_channels) > 0:
        print(f'horizontal flip TTA is enabled for classes: {classes[args.tta_hflip_channels]}')
        test_dataloaders.append(get_test_dataloader(config, test_to_val=test_to_val, tta_hflip=True))
        flags_hflip.append(True)
        flags_vflip.append(False)
        w = [1 if (i in args.tta_hflip_channels) else 0 for i in range(n_classses)]
        weights.append(w)

    # dataloader w/ tta vertical flipping
    if len(args.tta_vflip_channels) > 0:
        print(f'vertical flip TTA is enabled for classes: {classes[args.tta_vflip_channels]}')
        test_dataloaders.append(get_test_dataloader(config, test_to_val=test_to_val, tta_vflip=True))
        flags_hflip.append(False)
        flags_vflip.append(True)
        w = [1 if (i in args.tta_vflip_channels) else 0 for i in range(n_classses)]
        weights.append(w)

    # normalize weights
    weights = np.array(weights, dtype=float)  # shape: [1 + n_tta, n_classes]
    weights /= weights.sum(axis=0, keepdims=True)
    weights = weights[:, :, np.newaxis, np.newaxis]  # shape: [1 + n_tta, n_classes, 1, 1]

    return test_dataloaders, flags_hflip, flags_vflip, weights


def main():
    args = parse_args()
    assert (not args.use_ema_weight) or (not args.use_swa_weight)

    config: DictConfig = load_test_config(args)

    model_dir = os.path.join(args.artifact_dir, 'models') if (args.override_model_dir is None) else args.override_model_dir
    model = get_model(config, model_dir)

    ckpt_fn = 'best.ckpt'
    if args.use_swa_weight:
        print('using SWA weight')
        ckpt_fn = 'last.ckpt'
    ckpt_path = os.path.join(model_dir, f'exp_{args.exp_id:05d}', ckpt_fn)
    print(f'loading {ckpt_path}')
    state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if args.use_ema_weight:
        print('using EMA weight')
        state_dict = state_dict['callbacks']['EMA']['ema_state_dict']
    else:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    out_root = '_val/preds' if args.val else 'preds'
    out_root = os.path.join(args.artifact_dir, out_root, f'exp_{args.exp_id:05d}')
    print(f'going to save prediction results under {out_root}')

    os.makedirs(out_root, exist_ok=True)

    # dump meta
    meta = {
        'groups': list(config.Class.groups),
        'classes': {
            g: list(cs) for g, cs in config.Class.classes.items()
        }
    }
    with open(os.path.join(out_root, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    
    test_dataloaders, flags_hflip, flags_vflip, weights = prepare_test_dataloaders(config, args)

    for batches in tqdm(zip(*test_dataloaders),
                        total=len(test_dataloaders[0])):
        # prepare buffers for image file name and predicted array
        batch_size = len(batches[0]['image'])
        output_paths = [None] * batch_size
        preds_averaged = np.zeros(shape=[
            batch_size,
            len(get_flatten_classes(config)), 1300, 1300  # TODO
        ])

        for dataloader_idx, batch in enumerate(batches):
            images = batch['image'].to(args.device)
            batch_pre_paths = batch['pre_path']
            batch_orig_heights = batch['original_height']
            batch_orig_widths = batch['original_width']

            n_input_post_images = config.Model.n_input_post_images
            assert n_input_post_images in [0, 1, 2]
            images_post_a = None
            images_post_b = None
            if n_input_post_images == 1:
                images_post_a = batch['image_post_a'].to(args.device)
            elif n_input_post_images == 2:
                images_post_a = batch['image_post_a'].to(args.device)
                images_post_b = batch['image_post_b'].to(args.device)

            with torch.no_grad(): 
                batch_preds = model(images, images_post_a, images_post_b)
            batch_preds = torch.sigmoid(batch_preds)
            batch_preds = batch_preds.cpu().numpy()

            for i in range(images.shape[0]):
                pred = batch_preds[i]
                pre_path = batch_pre_paths[i]
                orig_h = batch_orig_heights[i].item()
                orig_w = batch_orig_widths[i].item()

                # set pred=0 on black pixels in the pre-image
                image = images[i].cpu().numpy()  # 3,H,W
                nodata_mask = np.sum(image, axis=0) == 0  # H,W
                pred[:, nodata_mask] = 0.0

                # set flooded_pred=0 on black pixels in the post-images
                nodata_mask = np.zeros(shape=[orig_h, orig_w], dtype=bool)
                if images_post_a is not None:
                    image = images_post_a[i].cpu().numpy()
                    nodata_mask = np.sum(image, axis=0) == 0  # H,W
                if images_post_b is not None:
                    image = images_post_b[i].cpu().numpy()
                    nodata_mask = nodata_mask & (np.sum(image, axis=0) == 0)  # H,W 
                classes = get_flatten_classes(config)
                for class_index, class_name in enumerate(classes):
                    if class_name in ['flood_building', 'flood_road', 'flood']:
                        pred[class_index, nodata_mask] = 0.0

                # flip (only when flipping tta is applied)
                if flags_vflip[dataloader_idx]:
                    pred = pred[:, ::-1, :]
                if flags_hflip[dataloader_idx]:
                    pred = pred[:, :, ::-1]

                pred = crop_center(pred, crop_wh=(orig_w, orig_h))

                # store predictions into the buffer
                preds_averaged[i] += pred * weights[dataloader_idx]

                aoi = os.path.basename(os.path.dirname(os.path.dirname(pre_path)))
                filename = os.path.basename(pre_path)
                out_dir = os.path.join(out_root, aoi)
                os.makedirs(out_dir, exist_ok=True)
                output_path = os.path.join(os.path.join(out_dir, filename))

                if dataloader_idx == 0:
                    output_paths[i] = output_path
                else:
                    assert output_paths[i] == output_path

        for output_path, pred, pre_path in zip(output_paths, preds_averaged, batch_pre_paths):
            assert pred.min() >= 0
            assert pred.max() <= 1
            pred_8bit = (pred * 255).astype(np.uint8)
            save_array_as_geotiff(pred_8bit, pre_path, output_path)


if __name__ == '__main__':
    main()
