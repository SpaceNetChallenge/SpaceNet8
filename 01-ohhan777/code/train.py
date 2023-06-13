# Korea Aerospace Research Insitute(KARI) AI Lab, 2022. 
# 
# This code has been written with reference to the following.
# - YOLOv5(https://github.com/ultralytics/yolov5).
# - HRNet-Semantic Segmentation(https://github.com/HRNet/HRNet-Semantic-Segmentation) 
# - NVIDIA semantic segmenation(https://github.com/NVIDIA/semantic-segmentation).
# 
import os
import argparse
import yaml
import math
import torch
import time
from tqdm import tqdm
import numpy as np
from loss.sn8_loss import BuildingRoadFloodLoss
from torchvision import models, datasets, transforms
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from torch.cuda import amp
from pathlib import Path
import torch.backends.cudnn as cudnn
import requests
from utils.datasets import create_dataloader
from utils.loggers import Loggers
from utils.general import (print_args, LOGGER, colorstr, one_cycle, increment_path,
                           check_yaml, methods, check_suffix, init_seeds, intersect_dicts,
                           strip_optimizer, get_latest_run, download, AverageMeter, check_version)
from utils.metrics import fitness
from utils.torch_utils import select_device, de_parallel, is_main_process, EarlyStopping, is_distributed, get_rank, get_world_size, reduce_tensor, torch_distributed_zero_first
from utils.callbacks import Callbacks
from copy import deepcopy
from datetime import datetime
import val
from models.config import update_config
from models.seg_hrnet_ocr import get_seg_model


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(hyp, opt, device, callbacks):
    save_dir, epochs, batch_size, resume, weights, noval, nosave, workers = Path(opt.save_dir), opt.epochs, opt.batch_size,\
                                                                opt.resume, opt.weights, opt.noval, opt.nosave, opt.workers

    # Directories
    save_dir = Path('./runs/train')
    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best, best_building, best_road, best_flood = w / 'last.pt', w / 'best.pt', \
                        w / 'best_building.pt', w / 'best_road.pt', w / 'best_flood.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)

    if is_main_process():
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir/ 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Loggers
    data_dict = None
    if is_main_process():
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    cuda = device.type != 'cpu'    
    init_seeds(1 + RANK)
    # TODO: data_dict

    # Model
    check_suffix(weights, '.pt')  # check weights
    # HR-Net+OCR
    with torch_distributed_zero_first(RANK):
        if not os.path.exists('./models/hrnetv2_w48_imagenet_pretrained.pth'):
            url = 'http://ohhan.net/wordpress/wp-content/uploads/2022/08/hrnetv2_w48_imagenet_pretrained.pth'
            download(url, out_path='./models')
    cfg = update_config('./models/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml')
    model = get_seg_model(cfg)
    
  
    # Dataloaders and Datasets
    data_to_load = ["preimg", "postimg", "building", "roadspeed", "flood"]
    train_loader, train_dataset = create_dataloader('./', data_to_load, task="train", batch_size=batch_size // WORLD_SIZE,
                                                    hyp=hyp, augment=False, cache=False, rank=LOCAL_RANK, workers=workers)
    
    val_loader, _ = create_dataloader('./', data_to_load, task="val", batch_size=batch_size // WORLD_SIZE,
                                                    hyp=hyp, augment=False, cache=False, rank=LOCAL_RANK, workers=workers)
    nb = len(train_loader)  # number of batches
    loss_func = BuildingRoadFloodLoss(device)
    model = model.to(device)
    
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    if is_main_process():
        LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    if opt.optimizer == 'Adam':
        optimizer = Adam(g[2], lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    elif opt.optimizer == 'AdamW':
        optimizer = AdamW(g[2], lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g[2], lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g[0], 'weight_decay': hyp['weight_decay']})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1]})  # add g1 (BatchNorm2d weights)
    if is_main_process():
        LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                    f"{len(g[1])} weight (no decay), {len(g[0])} weight, {len(g[2])} bias")
    del g

    # Scheduler
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    # if opt.cos_lr:
    #     lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: math.pow(1 -x / epochs, hyp['poly_exp'])    
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  

    # TODO: EMA

    # Resume (load checkpoint and apply it to the model)
    start_epoch, best_fitness, best_mean_iou, best_epoch = 0, 0.0, 0.0, 0
    best_building_iou, best_road_iou, best_flood_miou = 0.0, 0.0, 0.0
    if resume:
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict())  # intersect
        model.load_state_dict(csd, strict=False)  # load

        if is_main_process():
            LOGGER.info(colorstr('yellow', ('Resuming training from %s saved at %s (last epoch %d)') 
                                            % (weights, ckpt['date'], ckpt['epoch'])))
            LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report  

        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
            best_mean_iou = ckpt['best_mean_iou']

            best_building_iou = ckpt['best_building_iou']
            best_road_iou = ckpt['best_road_iou']
            best_flood_miou = ckpt['best_flood_miou']
        
        start_epoch = ckpt['epoch'] + 1
        assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'        
        del ckpt, csd
    
     # DP (Data-Parallel) mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        if is_main_process():
            LOGGER.info("DP mode is enabled, but DDP is preferred for best performance.\n",
                        "To start DDP, torchrun --nproc_per_node=<# of GPUS> train.py" )
        #model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        if is_main_process():
            LOGGER.info('Using SyncBatchNorm()')
    
    # DDP mode
    if cuda and RANK != -1:
        if check_version(torch.__version__, '1.11.0'):
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
        else:
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


    # Start training 
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    last_opt_step = -1
    results = [0, 0, 0, 0, 0]   # building IoU, Road mIOU, building pix_acc, Road pix_acc, val loss
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    callbacks.run('on_train_start')
    if is_main_process():
        LOGGER.info(f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n' \
                    f"Logging results to {colorstr('bold', save_dir)}\n" \
                    f"Starting training for {epochs} epochs...")
        
    for epoch in range(start_epoch, epochs):  # epoch ---------------------------------------------------------------
        t1 = time.time()
        callbacks.run('on_train_epoch_start', epoch=epoch)
        model.train()
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)

        avg_loss = AverageMeter()
                   
        for i, data in enumerate(train_loader):
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)
            preimgs, postimgs, building_labels, _, roadspeed_labels, flood_labels, _ = data
            preimgs = preimgs.to(device, non_blocking=True)
            postimgs = postimgs.to(device, non_blocking=True)
            building_labels = building_labels.unsqueeze(1).float().to(device)
            roadspeed_labels = roadspeed_labels.to(device)
            flood_labels = flood_labels.to(device)
            

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Forward
            with amp.autocast(enabled=cuda):
                preds = model(preimgs, postimgs)    # [builing(no ocr), builing(ocr), road(no ocr), road(ocr)]
                loss = loss_func(preds, building_labels, roadspeed_labels, flood_labels)  # loss scaled by batch_size
                
                if RANK != -1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)  # gradient summed between devices in DDP mode
                    

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                last_opt_step = ni

            # Log
            if is_main_process() and i % 10 == 0:
                avg_loss.update(loss.item())    # update average loss
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                msg = 'Epoch: [{}/{}] Iter:[{}/{}], GPU: {}, ' \
                      'lr: {}, Loss: {:.6f}' .format(
                      epoch, epochs, i, nb, mem, [x['lr'] for x in optimizer.param_groups], avg_loss.average())
                LOGGER.info(msg)
            
            # debug mode 
            if opt.debug and i == 10:  
               break

            # end batch --------------------------------------------------------------------------------------
        
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
            

        if is_main_process():
            callbacks.run('on_train_epoch_end', epoch=epoch)
        
        # validation
        final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
        if not noval or final_epoch:  # Calculate mIOU, pix_acc, val loss
            results = val.run(model=model, batch_size=batch_size, workers=workers, hyp=hyp, dataloader=val_loader, 
                                  save_dir=save_dir, callbacks=callbacks, loss_func=loss_func)

        if is_main_process():
            fi = fitness(np.array(results[:3]))
            if fi > best_fitness:
                best_fitness = fi 
                best_epoch = epoch    
                best_mean_iou = fi
                LOGGER.info(colorstr('yellow','bold','[Best so far]'))

            log_vals = [loss.item()] + list(results[:8]) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
            LOGGER.info('Best mIoU=%.4f (epoch=%d)' % (best_mean_iou, best_epoch))
            LOGGER.info('[Best] Building IoU=%.4f, Road IoU=%.4f, Flood mIoU=%.4f' % (best_building_iou, best_road_iou, best_flood_miou) )

            # Save model
            if (not nosave) or final_epoch:  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness, 
                        'best_mean_iou' : best_mean_iou,
                        'best_building_iou' : best_building_iou,
                        'best_road_iou' : best_road_iou,
                        'best_flood_miou' : best_flood_miou,
                        'model': deepcopy(de_parallel(model)).half(), 
                        'optimizer': optimizer.state_dict(), 
                        'wandb_id' : loggers.wandb.wandb_run.id if loggers.wandb else None,                   
                        'date': datetime.now().isoformat()}
                
                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    best_mean_iou = fi
                    torch.save(ckpt, best)
                if results[0] > best_building_iou:
                    best_building_iou = results[0]
                    torch.save(ckpt, best_building)
                if results[1] > best_road_iou:
                    best_road_iou = results[1]
                    torch.save(ckpt, best_road)
                if results[2] > best_flood_miou:
                    best_flood_miou = results[2]
                    torch.save(ckpt, best_flood)

                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)
            LOGGER.info(f'Epoch {epoch} completed in {(time.time() - t1):.3f} seconds.')

            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    if is_main_process():
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')

        # for f in last, best:
        #     if f.exists():
        #         if f is best:
        #             s = str(f).replace('best.pt', 'best_stripped.pt')
        #             strip_optimizer(f, s) # strip optimizers
        #             LOGGER.info(f'\nValidating {s}...')
        #             results = val.run(s, batch_size=batch_size, workers=workers, hyp=hyp, 
        #                             save_dir=save_dir, callbacks=callbacks)

        # callbacks.run('on_train_end', last, best, epoch, results)
        # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
                
                    
    torch.cuda.empty_cache()
    # return results
    return 0


def main(opt, callbacks=Callbacks()):
    if is_main_process():
        print_args(FILE.stem, opt)
    # Resume
    if opt.resume: # resume an interrupted run
        epochs = opt.epochs
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.weights, opt.resume, opt.epochs = ckpt, True, epochs  # reinstate
    else:
        opt.hyp, opt.weights, opt.project = \
            check_yaml(opt.hyp), str(opt.weights), str(opt.project)
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if RANK != -1:
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(RANK)
        device = torch.device('cuda', RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    
    # Train
    train(opt.hyp, opt, device, callbacks)
    
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'weights.pt', help='initial weights path')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=ROOT/'runs', help='save to project/runs/name')
    parser.add_argument('--name', default='exp', help='save to project/runs/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.spacenet8.yaml', help='hyperparameters path')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--patience', type=int, default=50, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--debug', action='store_true', help='debug mode (training is early stopped every epoch)')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)