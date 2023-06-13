import os
import torch
import torch.nn
from torch import optim, nn
import json
from tqdm import tqdm
import dataloaders as ds
import numpy as np
import random
import torch.nn.functional as F
from loss import MixedLoss, BCE
import segmentation_models_pytorch as smp
from collections import OrderedDict
from models import UnetMhead, UnetSiamese, MAnetMhead, UnetSiamese_Mhead, MAnetSiamese_Mhead
#from utils import *
import argparse


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def optimizers_schedulers(model, config, optim_weights=None, scheduler_weights=None):
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0.0001) # test weight_decay param??
    if optim_weights is not None:
        optimizer.load_state_dict(optim_weights)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=config['lr_steps'], gamma=0.2)
    if scheduler_weights is not None:
        scheduler.load_state_dict(scheduler_weights)
    return optimizer, scheduler


def save_model(out_file, model,
               optimizer=None, epoch=0, scheduler=None, score=None):
    model_ = OrderedDict()
    model_['state_dict'] = model.state_dict()
    if optimizer is not None:
        model_['optimizer_state_dict'] = optimizer.state_dict()
    model_['epoch'] = int(epoch)
    if scheduler is not None:
        model_['scheduler_state_dict'] = scheduler.state_dict()
    model_['score_iou'] = score
    torch.save(model_, out_file)


def load_pretrained_weights(model, weights):
    w = torch.load(weights, map_location='cpu')
    pretrained_dict = w['state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print(f'Loaded weights from ', weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--resume', default='', required=False, type=str, help='resume from checkpoint')
    parser.add_argument('--model_name', default='', required=False, type=str, help='model_name')
    parser.add_argument('--batch_size', required=False, default=8, type=int, help='batch_size')
    parser.add_argument('--weights', required=False, default='', type=str, help='model weights')
    parser.add_argument('--out_dir', required=False, default='', type=str, help='model weights out dir')
    parser.add_argument('--tr', required=True, default='foundation', type=str,  help='train foundation or flood')
    parser.add_argument('--gpu', default=0, type=int, help="device id")
    parser.add_argument('--net', default='unet', type=str, help="unet or manet")
    parser.add_argument('--train_csvs', default='train_val_csvs/sn8_data_train_fold1.csv', type=str, help="train_csv")
    parser.add_argument('--val_csvs', default='train_val_csvs/sn8_data_val_fold1.csv', type=str, help="val_csv")

    args = parser.parse_args()
    seed_everything(13)

    training_type = args.tr
    config = json.load(open('configs/config_aws.json'))
    model_name = config['model_name'] if args.model_name == '' else args.model_name
    config['model_name'] = model_name
    output_dir = args.out_dir
    net = args.net
    device = f'cuda:{args.gpu}'
    assert args.tr in ['foundation', 'flood', 'previous']
    assert args.net in ['unet', 'manet']
    batch_size = args.batch_size
    if not config['batch_size'] == batch_size:
        config['batch_size'] = batch_size
    print(f'Training...{args.tr} spacenet with batch_size {batch_size}')

    if training_type == 'foundation':
        import trainer_ms as tsm_mixed
        data_to_load = ["preimg", "building", "road"]
        config['sn8:train_csv'] = args.train_csvs
        config['sn8:val_csv'] = args.val_csvs
        gens = ds.get_generators_sn8(config, data_to_load=data_to_load)

        if args.net == 'unet':
            model = UnetMhead(encoder_name=model_name, encoder_weights=None)
        else:
            model = MAnetMhead(encoder_name=model_name, encoder_weights=None)

        model.load_state_dict(torch.load(args.weights, map_location='cpu')['state_dict'])
        model.to(device)
        print('previous spacenet model loaded')
        trainer_train = tsm_mixed.TrainEpoch
        trainer_val = tsm_mixed.ValidEpoch
        loss = MixedLoss()

    if training_type == 'previous':
        import trainer_pm as tpm
        gens = ds.get_generators(config, use_offnadir=False, task='both')
        if net == 'unet':
            print('training unet..')
            model = UnetMhead(encoder_name=model_name)
        else:
            print('training manet..')
            model = MAnetMhead(encoder_name=model_name)
        trainer_train = tpm.TrainEpoch
        trainer_val = tpm.ValidEpoch
        loss = MixedLoss()

    if training_type == 'flood':
        import trainer_ms as tsm_mixed
        trainer_train = tsm_mixed.TrainEpochSiamese
        trainer_val = tsm_mixed.ValidEpochSiamese
        data_to_load = ["preimg", "postimg", "flood"]
        config['sn8:train_csv'] = args.train_csvs
        config['sn8:val_csv'] = args.val_csvs
        gens = ds.get_generators_sn8(config, data_to_load=data_to_load, do_imbalanced=True)
        if args.net == 'unet':
            model = UnetSiamese_Mhead(encoder_name=model_name, encoder_weights=None)
        else:
            model = MAnetSiamese_Mhead(encoder_name=model_name, encoder_weights=None)
        load_pretrained_weights(model, args.weights)
        model.to(device)
        loss = [MixedLoss(), BCE()]
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5)
    ]
    optimizer_state = None
    resume_epoch = None
    if args.resume!='':
        states = torch.load(args.resume, map_location='cpu')
        optimizer_state = states['optimizer_state_dict']
        model.load_state_dict(states['state_dict'])
        model.to(device)
        resume_epoch = states['epoch']+1
        print(f'Resuming...{os.path.basename(args.resume)} from {resume_epoch}')

    optimizer, scheduler = optimizers_schedulers(model, config)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print('Loaded optimizer state for resumption...')

    train_epoch = trainer_train(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = trainer_val(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    max_score = 0
    best_epoch = 0
    gamma = 0.2
    monitor_score = 'iou_score'
    loss_name = loss[0].__name__ if isinstance(loss, list) else loss.__name__
    score_name = 'iou_score'
    start_epoch = 0 if resume_epoch is None else resume_epoch
    # train model for 45 epochs for previous and foundation spacenet data,
    # For flood label using siamese train 100 epochs (only 0.75 of segmentation loss is used)
    num_epoch = 45 if training_type != 'flood' else 100
    gamma = 0.2 if training_type != 'flood' else 0.5
    for i in range(start_epoch, num_epoch):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(gens['training_generator'])
        valid_logs = valid_epoch.run(gens['val_generator'])
        if max_score < valid_logs[monitor_score]:
            max_score = valid_logs[monitor_score]
            save_model(f'{output_dir}/best_model.pth', model,
                       train_epoch.optimizer, i, scheduler, score=valid_logs[monitor_score])
            print('Model saved! New Best')
            best_epoch = i
        if training_type!='flood':
            if i == 25:
                for p in optimizer.param_groups:
                    p['lr'] = p['lr'] * gamma
                print('Decrease  learning rate ..')

            if i == 35:
                for p in optimizer.param_groups:
                    p['lr'] = p['lr'] * gamma
                print('Decrease  learning rate ..')
        else:
            if i % 30 == 0 and i > 5:
                save_model(f'{output_dir}/model_e_{i}.pth', model,
                           train_epoch.optimizer, i, scheduler, score=valid_logs[monitor_score])
                for p in optimizer.param_groups:
                    p['lr'] = p['lr'] * gamma
                print('Decrease  learning rate t', i)
        save_model(f'{output_dir}/{args.tr}_{model_name}_last.pth', model,
            train_epoch.optimizer, i, scheduler, score=valid_logs[monitor_score])

    save_model(f'{output_dir}/{args.tr}_{model_name}_last.pth', model,
               None, epoch=i, scheduler=None, score=valid_logs[monitor_score])
    print(f'done....best epoch..{best_epoch}..')
