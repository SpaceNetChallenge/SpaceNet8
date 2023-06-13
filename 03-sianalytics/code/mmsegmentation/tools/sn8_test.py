import argparse
import os
from glob import glob

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis.sn8_inference import inference_scene
from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--image-path', help='file path for inference',
        default='/data/datasets/SpaceNet8/Testing/Louisiana-West_Test_Public')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--logit', action='store_true',
        help='Save logits instead of the segmentation outputs.')
    parser.add_argument(
        '--post', action='store_true',
        help='Inference on POST image')
    parser.add_argument(
        '--post-select', default=0, type=int,
        help='the file to select as post')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified',
    )
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local-path', type=str, default='ckpt')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():

    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    cfg.data.test = cfg.data.val

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher != 'none':
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    else:
        distributed = False

    image_path = args.image_path

    test_transforms = [tfm.get('type') for tfm in cfg.data.test.pipeline[1].transforms]
    if 'ReflectPad' not in test_transforms:
        index = test_transforms.index('Pad')
        cfg.data.test.pipeline[1].transforms[index] = dict(type='ReflectPad', size=(1440, 1440))

    if cfg.model.test_cfg.mode == 'whole':
        train_lines = cfg.data.train
        if isinstance(train_lines, list):
            train_transforms = [tfm.get('type') for tfm in train_lines[0].pipeline]
            index = train_transforms.index('RandomCrop')
            crop_size = train_lines[0].pipeline[index]['crop_size']
        else:
            train_transforms = [tfm.get('type') for tfm in train_lines.pipeline]
            index = train_transforms.index('RandomCrop')
            crop_size = train_lines.pipeline[index]['crop_size']

        # cfg.model.test_cfg = dict(mode='slide', crop_size=crop_size, stride=(768, 768))
        cfg.model.test_cfg.mode = 'slide'
        cfg.model.test_cfg.crop_size = crop_size
        cfg.model.test_cfg.stride = (512,512)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = revert_sync_batchnorm(model)
        model = MMDataParallel(model, device_ids=[torch.cuda.current_device()])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    mapping_csv = glob(os.path.join(image_path, '*label_image_mapping.csv'))[0]

    inference_scene(
        model=model,
        test_cfg=cfg.model.test_cfg,
        test_pipeline=cfg.data.test.pipeline,
        image_path=image_path,
        mapping_csv=mapping_csv,
        out_path=args.out,
        save_logit=args.logit,
        use_post=args.post,
        post_select=args.post_select,
    )


if __name__ == '__main__':
    main()
