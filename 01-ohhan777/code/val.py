import argparse
import json
from logging.config import valid_ident
import os
import sys
from pathlib import Path
from threading import Thread
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from loss.sn8_loss import BuildingRoadFloodLoss, get_confusion_matrix, get_iou, get_pix_acc
from utils.datasets import create_dataloader
from utils.callbacks import Callbacks
from utils.plots import write_gt_pred_images, write_pred_images, write_pred_tiffs
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.general import LOGGER, AverageMeter, colorstr, print_args, check_version, increment_path, FullModel
from utils.plots import write_gt_pred_images, write_pred_images, write_pred_tiffs
from utils.torch_utils import is_main_process, select_device, time_sync, reduce_tensor, torch_distributed_zero_first
from torchvision import models
import torch.nn.functional as F
from models.config import update_config
from models.seg_hrnet_ocr import get_seg_model

# DDP-related settings
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


@torch.no_grad()
def run(weights=None,
        batch_size=32,
        task='val',
        device='',
        workers=8,
        hyp=None,
        half=False,  # use FP16 half-precision inference
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        model=None,
        dataloader=None,
        save_dir=Path(''),
        callbacks=Callbacks(),
        loss_func=None,
        pngs=False,
        tiffs=False,
        local_rank=0
        ):

    # Initialize/load model and set device
    training = model is not None
    if training:
        device = next(model.parameters()).device      
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()

    else: # called directly
        # Directories
        cuda = device.type != 'cpu'
    
        # Load model
        cfg = update_config('./models/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml')
        model = get_seg_model(cfg)
             
        model = model.to(device)

        # TODO: task=['train', 'val', 'test'], currently works for task='val'
        if task == 'val':
            data_to_load = ["preimg", "postimg", "building", "roadspeed", "flood"]
        
            dataloader, dataset = create_dataloader('./data/SN8_floods', data_to_load, task, batch_size=batch_size // WORLD_SIZE,
                                                        hyp=hyp, augment=False, cache=False, rank=LOCAL_RANK, workers=workers)
            loss_func = BuildingRoadFloodLoss(device)
        
        if task == 'test':
            data_to_load = ["pre-event image","post-event image 1"]
            dataloader, dataset = create_dataloader('./data/SN8_floods/Louisiana-West_Test_Public', data_to_load, task='test', batch_size=batch_size // WORLD_SIZE,
                                                        hyp=hyp, augment=False, cache=False, rank=LOCAL_RANK, workers=workers)

       

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()

        # Load checkpoint and apply it to the model
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        LOGGER.info(colorstr('yellow', ('Loaded the model from %s saved at %s (last epoch %d)') 
                                            % (weights, ckpt['date'], ckpt['epoch'])))
        model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)  # load



    # Directory
    w = Path(save_dir) 
    w.mkdir(parents=True, exist_ok=True)  # make dir
    

    # Configure
    model.eval()
    building_iou = AverageMeter()
    building_pix_acc = AverageMeter()
    road_iou = AverageMeter()
    road_pix_acc = AverageMeter()
    
    avg_loss = AverageMeter()
    num_classes = 8  # number of classes (road)
            
    seen = 0  # number of images seen
    dt = [0.0, 0.0, 0.0]
    nb = len(dataloader)   # number of batches
    callbacks.run('on_val_start')
    np.seterr(invalid='ignore')
    roadspeed_confusion_matrix = np.zeros(
        (num_classes, num_classes))
    
    flood_confusion_matrix = np.zeros(
        (5, 5))
    
    for i, data in enumerate(dataloader):
        callbacks.run('on_val_batch_start')
        if task in ('train', 'trainval', 'val'):
            preimgs, postimgs, building_labels, _, roadspeed_labels, flood_labels, preimg_filenames = data
            building_labels = building_labels.unsqueeze(1).float().to(device)  # (B, 1, H, W)
            roadspeed_labels = roadspeed_labels.to(device)                     # (B, 8, H, W)
            flood_labels = flood_labels.to(device)                             # (B, 4, H, W)
        if task == 'test':
            preimgs, postimgs, _, _, _, _, preimg_filenames = data

        seen += preimgs.shape[0]
        size = preimgs.size()
        t1 = time_sync()
        preimgs = preimgs.to(device)
        preimgs = preimgs.half() if half else preimgs.float()    # half precision (fp16)
        postimgs = postimgs.to(device)
        postimgs = postimgs.half() if half else postimgs.float()    # half precision (fp16)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        preds = model(preimgs, postimgs)

        if task in ('train', 'trainval', 'val'):
            loss = loss_func(preds, building_labels, roadspeed_labels, flood_labels) 
        
            if RANK != -1:
                dist.all_reduce(loss)   # in DDP mode, losses in all processes are summed

            avg_loss.update(loss.item())

            # roadspeed_target 
            road_targets = roadspeed_labels[:,7,:,:]
            roadspeed_targets = roadspeed_labels.float()
            roadspeed_targets[:,7,:,:] = 0.99
            roadspeed_targets = roadspeed_targets.argmax(dim=1).long()  # (B, H, W) # roadspeed, 0-6 speed, 7 background (no road)
           

        # building
        building_preds = preds[1]
        building_preds = F.interpolate(input=building_preds, size=size[-2:], mode='bilinear', align_corners=False) # (B, 1, H, W)
        building_preds = building_preds.sigmoid().round().long().squeeze(dim=1) # (B, H, W)
               
        # road   
        roadspeed_preds = preds[3]   
        roadspeed_preds = F.interpolate(input=roadspeed_preds, size=size[-2:], mode='bilinear', align_corners=False)  # (B, 8, H, W) 
        roadspeed_preds = F.softmax(roadspeed_preds, dim=1)
        roadspeed_sn_preds = roadspeed_preds.clone() # CRESI-compatible format for TIFF saving
        roadspeed_preds = roadspeed_preds.argmax(dim=1) # (B, H, W)
        roadspeed_sn_preds[:,7,:,:] = torch.where(roadspeed_preds==7, 0.0, 1.0) 

        # flood
        flood_preds = preds[5]
        flood_preds = F.interpolate(input=flood_preds, size=size[-2:], mode='bilinear', align_corners=False)  # (B, 5, H, W) 
        flood_preds = F.softmax(flood_preds, dim=1)  
        flood_preds = flood_preds.argmax(dim=1)  # (B, H, W)

          
        if task in ('train', 'val'):
            # building
            building_targets = building_labels.squeeze(1).long()  # (B, H, W)
            building_intersection = (building_preds & building_targets).sum()
            building_union = (building_preds | building_targets).sum().float()
            building_iou.update((building_intersection/(building_union + 1e-7)).item())
            building_pix_acc.update((building_preds == building_targets).sum().float()/(building_targets.numel()))
            # road
            roadspeed_confusion_matrix += get_confusion_matrix(roadspeed_targets, roadspeed_preds, size, num_classes)
            road_iou.update(get_iou(road_targets, roadspeed_sn_preds[:,7,:,:].long()))
            road_pix_acc.update(get_pix_acc(road_targets, roadspeed_sn_preds[:,7,:,:].long()))
            # flood
            b, _, h, w = flood_labels.shape
            flood_targets = torch.cat([torch.zeros(b,1,h,w).to(device), flood_labels], dim=1).argmax(dim=1).long() # (B, H, W)
            flood_confusion_matrix += get_confusion_matrix(flood_targets, flood_preds, size, 5)

        
        
        dt[1] += time_sync() - t2
        
         
        t3 = time_sync()
            
        

        if is_main_process():
            if i == 0:
                LOGGER.info(colorstr('validation: '))
            if i % 5 == 0:
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)               
                LOGGER.info('            [%d/%d] GPU: %s, loss: %.4f' % (i, nb, mem, avg_loss.average())) 
            if not (pngs or tiffs) and (i < 5):
                if task in ('train', 'val'):
                    write_gt_pred_images(preimgs, building_targets, roadspeed_targets, building_preds, roadspeed_preds, save_dir, preimg_filenames)
                else:
                    write_pred_images(preimgs, building_preds, roadspeed_preds, save_dir, preimg_filenames)
        if task in ('train', 'val'):
            if pngs:
                write_gt_pred_images(preimgs, building_targets, roadspeed_targets, building_preds, roadspeed_preds, save_dir, preimg_filenames)
        else:
            if pngs:
                write_pred_images(preimgs, building_preds, roadspeed_preds, save_dir, preimg_filenames)
        if tiffs:
            write_pred_tiffs(preimg_filenames, building_preds, roadspeed_sn_preds, flood_preds, save_dir)

        

        callbacks.run('on_val_batch_end')
       
    # Metrics
    if task in ('train', 'trainval', 'val') and (RANK != -1):   # in DDP mode, confusion matrices in all processes are summed
        roadspeed_confusion_matrix = torch.from_numpy(roadspeed_confusion_matrix).to(device)
        dist.all_reduce(roadspeed_confusion_matrix, op=dist.ReduceOp.SUM)
        roadspeed_confusion_matrix = roadspeed_confusion_matrix.cpu().numpy()

        flood_confusion_matrix = torch.from_numpy(flood_confusion_matrix).to(device)
        dist.all_reduce(flood_confusion_matrix, op=dist.ReduceOp.SUM)
        flood_confusion_matrix = flood_confusion_matrix.cpu().numpy()

    if task in ('train', 'trainval', 'val'):
        # roadspeed
        roadspeed_pix_acc = np.diag(roadspeed_confusion_matrix).sum() / roadspeed_confusion_matrix.sum()
        roadspeed_acc_cls = np.diag(roadspeed_confusion_matrix) / roadspeed_confusion_matrix.sum(axis=1)
        roadspeed_acc_cls = np.nanmean(roadspeed_acc_cls)
        divisor = roadspeed_confusion_matrix.sum(axis=1) + roadspeed_confusion_matrix.sum(axis=0) - \
                        np.diag(roadspeed_confusion_matrix)
        roadspeed_class_iou = np.diag(roadspeed_confusion_matrix) / divisor
        roadspeed_mean_iou = np.nansum(roadspeed_class_iou) / num_classes 

        # flood    
        flood_pix_acc = np.diag(flood_confusion_matrix).sum() / flood_confusion_matrix.sum()
        flood_acc_cls = np.diag(flood_confusion_matrix) / flood_confusion_matrix.sum(axis=1)
        flood_acc_cls = np.nanmean(flood_acc_cls)
        divisor = flood_confusion_matrix.sum(axis=1) + flood_confusion_matrix.sum(axis=0) - \
                        np.diag(flood_confusion_matrix)
        flood_class_iou = np.diag(flood_confusion_matrix) / divisor
        flood_mean_iou = np.nansum(flood_class_iou) / 5 

    dt[2] += time_sync() - t3

    #callbacks.run('on_val_end')

    model.float()  # return to float32 for training

    # Print results
    if is_main_process():
        if task in ('train', 'trainval', 'val'):
            LOGGER.info(('[Building] IoU=%.4g, Pixel Accuracy=%.4g') % (building_iou.average(), building_pix_acc.average()))
            LOGGER.info(('[Road] IoU=%.4g, Roadspeed mIoU=%.4g, Pixel Accuracy=%.4g, Pixel Accuracy(roadspeed)=%.4g') % (road_iou.average(), roadspeed_mean_iou, road_pix_acc.average(), roadspeed_pix_acc))
            LOGGER.info('        class IoU: [' + ', '.join([('%.4f' % (x)) for x in roadspeed_class_iou]) + ']')
            LOGGER.info(('[Flood] Flood mIoU=%.4g, Pixel Accuracy=%.4g') % (flood_mean_iou, flood_pix_acc))
            LOGGER.info('        class IoU: [' + ', '.join([('%.4f' % (x)) for x in flood_class_iou]) + ']')
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms confusion matrix calculation per image' % t)

    
    if task in ('train', 'trainval', 'val'):
        return (building_iou.average(), road_iou.average(), flood_mean_iou, roadspeed_mean_iou, building_pix_acc.average(), road_pix_acc.average(), avg_loss.average())
    else:
        return (0, 0, 0, 0, 0, 0, 0)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'weights.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--pngs', action='store_true', help='store PNGs')
    parser.add_argument('--tiffs', action='store_true', help='store TIFFs')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt):


    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if RANK != -1:
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(RANK)
        device = torch.device('cuda', RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    opt.device = device

    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    else:
        # TODO: python val.py --task speed --weights weights0.pt weights1.pt ...
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        for opt.weights in weights:
            run(**vars(opt), plots=False)

    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)