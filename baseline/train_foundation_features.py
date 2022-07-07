import csv
import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from datasets.datasets import SN8Dataset
from core.losses import focal, soft_dice_loss
import models.pytorch_zoo.unet as unet
from models.other.unet import UNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",
                        type=str,
                        required=True)
    parser.add_argument("--val_csv",
                         type=str,
                         required=True)
    parser.add_argument("--save_dir",
                         type=str,
                         required=True)
    parser.add_argument("--model_name",
                         type=str,
                         required=True)
    parser.add_argument("--lr",
                         type=float,
                        default=0.0001)
    parser.add_argument("--batch_size",
                         type=int,
                        default=2)
    parser.add_argument("--n_epochs",
                         type=int,
                         default=50)
    parser.add_argument("--gpu",
                        type=int,
                        default=0)
    args = parser.parse_args()
    return args
    
def write_metrics_epoch(epoch, fieldnames, train_metrics, val_metrics, training_log_csv):
    epoch_dict = {"epoch":epoch}
    merged_metrics = {**epoch_dict, **train_metrics, **val_metrics}
    with open(training_log_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(merged_metrics)
        
def save_model_checkpoint(model, checkpoint_model_path): 
    torch.save(model.state_dict(), checkpoint_model_path)

models = {
    'resnet34': unet.Resnet34_upsample,
    'resnet50': unet.Resnet50_upsample,
    'resnet101': unet.Resnet101_upsample,
    'seresnet50': unet.SeResnet50_upsample,
    'seresnet101': unet.SeResnet101_upsample,
    'seresnet152': unet.SeResnet152_upsample,
    'seresnext50': unet.SeResnext50_32x4d_upsample,
    'seresnext101': unet.SeResnext101_32x4d_upsample,
    'unet':UNet
}

if __name__ == "__main__":
    args = parse_args()
    train_csv = args.train_csv
    val_csv = args.val_csv
    save_dir = args.save_dir
    model_name = args.model_name
    initial_lr = args.lr
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    gpu = args.gpu

    now = datetime.now() 
    date_total = str(now.strftime("%d-%m-%Y-%H-%M"))
    
    img_size = (1300,1300)
    
    soft_dice_loss_weight = 0.25 # road loss
    focal_loss_weight = 0.75 # road loss
    road_loss_weight = 0.5
    building_loss_weight = 0.5

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    SEED=12
    torch.manual_seed(SEED)

    assert(os.path.exists(save_dir))
    save_dir = os.path.join(save_dir, f"{model_name}_lr{'{:.2e}'.format(initial_lr)}_bs{batch_size}_{date_total}")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.chmod(save_dir, 0o777)
    checkpoint_model_path = os.path.join(save_dir, "model_checkpoint.pth")
    best_model_path = os.path.join(save_dir, "best_model.pth")
    training_log_csv = os.path.join(save_dir, "log.csv")

    # init the training log
    with open(training_log_csv, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'lr', 'train_tot_loss', 'train_bce', 'train_dice', 'train_focal', 'train_road_loss',
                                     'val_tot_loss', 'val_bce', 'val_dice', 'val_focal', 'val_road_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    train_dataset = SN8Dataset(train_csv,
                            data_to_load=["preimg","building","roadspeed"],
                            img_size=img_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=batch_size)
    val_dataset = SN8Dataset(val_csv,
                            data_to_load=["preimg","building","roadspeed"],
                            img_size=img_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=4, batch_size=batch_size)

    #model = models["resnet34"](num_classes=[1, 8], num_channels=3)
    if model_name == "unet":
        model = UNet(3, [1,8], bilinear=True)
    else:
        model = models[model_name](num_classes=[1, 8], num_channels=3)
    
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    bceloss = nn.BCEWithLogitsLoss()

    best_loss = np.inf
    for epoch in range(n_epochs):
        print(f"EPOCH {epoch}")

        ### Training ##
        model.train()
        train_loss_val = 0
        train_focal_loss = 0
        train_soft_dice_loss = 0
        train_bce_loss = 0
        train_road_loss = 0
        train_building_loss = 0
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()

            preimg, postimg, building, road, roadspeed, flood = data

            preimg = preimg.cuda().float()
            roadspeed = roadspeed.cuda().float()
            building = building.cuda().float()

            building_pred, road_pred = model(preimg)
            bce_l = bceloss(building_pred, building)
            y_pred = F.sigmoid(road_pred)

            focal_l = focal(y_pred, roadspeed)
            dice_soft_l = soft_dice_loss(y_pred, roadspeed)

            road_loss = (focal_loss_weight * focal_l + soft_dice_loss_weight * dice_soft_l)
            building_loss = bce_l
            loss = road_loss_weight * road_loss + building_loss_weight * building_loss

            train_loss_val+=loss
            train_focal_loss += focal_l
            train_soft_dice_loss += dice_soft_l
            train_bce_loss += bce_l
            train_road_loss += road_loss
            loss.backward()
            optimizer.step()

            print(f"    {str(np.round(i/len(train_dataloader)*100,2))}%: TRAIN LOSS: {(train_loss_val*1.0/(i+1)).item()}", end="\r")
        print()
        train_tot_loss = (train_loss_val*1.0/len(train_dataloader)).item()
        train_tot_focal = (train_focal_loss*1.0/len(train_dataloader)).item()
        train_tot_dice = (train_soft_dice_loss*1.0/len(train_dataloader)).item()
        train_tot_bce = (train_bce_loss*1.0/len(train_dataloader)).item()
        train_tot_road_loss = (train_road_loss*1.0/len(train_dataloader)).item()
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        train_metrics = {"lr":current_lr, "train_tot_loss":train_tot_loss,
                         "train_bce":train_tot_bce, "train_focal":train_tot_focal,
                         "train_dice":train_tot_dice, "train_road_loss":train_tot_road_loss}

        # validation
        model.eval()
        val_loss_val = 0
        val_focal_loss = 0
        val_soft_dice_loss = 0
        val_bce_loss = 0
        val_road_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                preimg, postimg, building, road, roadspeed, flood = data
                preimg = preimg.cuda().float()
                roadspeed = roadspeed.cuda().float()
                building = building.cuda().float()

                building_pred, road_pred = model(preimg)
                bce_l = bceloss(building_pred, building)
                y_pred = F.sigmoid(road_pred)

                focal_l = focal(y_pred, roadspeed)
                dice_soft_l = soft_dice_loss(y_pred, roadspeed)

                road_loss = (focal_loss_weight * focal_l + soft_dice_loss_weight * dice_soft_l)
                building_loss = bce_l
                loss = road_loss_weight * road_loss + building_loss_weight * building_loss

                val_focal_loss += focal_l
                val_soft_dice_loss += dice_soft_l
                val_bce_loss += bce_l
                val_loss_val += loss
                val_road_loss += road_loss

                print(f"    {str(np.round(i/len(val_dataloader)*100,2))}%: VAL LOSS: {(val_loss_val*1.0/(i+1)).item()}", end="\r")

        print()        
        val_tot_loss = (val_loss_val*1.0/len(val_dataloader)).item()
        val_tot_focal = (val_focal_loss*1.0/len(val_dataloader)).item()
        val_tot_dice = (val_soft_dice_loss*1.0/len(val_dataloader)).item()
        val_tot_bce = (val_bce_loss*1.0/len(val_dataloader)).item()
        val_tot_road_loss = (val_road_loss*1.0/len(val_dataloader)).item()
        val_metrics = {"val_tot_loss":val_tot_loss,"val_bce":val_tot_bce,
                       "val_focal":val_tot_focal, "val_dice":val_tot_dice, "val_road_loss":val_tot_road_loss}

        write_metrics_epoch(epoch, fieldnames, train_metrics, val_metrics, training_log_csv)

        save_model_checkpoint(model, checkpoint_model_path)

        epoch_val_loss = val_metrics["val_tot_loss"]
        if epoch_val_loss < best_loss:
            print(f"    loss improved from {np.round(best_loss, 6)} to {np.round(epoch_val_loss, 6)}. saving best model...")
            best_loss = epoch_val_loss
            save_model_checkpoint(model, best_model_path)