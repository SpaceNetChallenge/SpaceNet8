import os
import argparse
import time

from osgeo import gdal
from osgeo import osr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.nn as nn

import models.pytorch_zoo.unet as unet
from models.other.unet import UNet
from datasets.datasets import SN8Dataset
from utils.utils import write_geotiff

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                         type=str,
                         required=True)
    parser.add_argument("--model_name",
                         type=str,
                         required=True)
    parser.add_argument("--in_csv",
                       type=str,
                       required=True)
    parser.add_argument("--save_fig_dir",
                         type=str,
                         required=False,
                         default=None)
    parser.add_argument("--save_preds_dir",
                         type=str,
                         required=False,
                         default=None)
    parser.add_argument("--gpu",
                         type=int,
                         required=False,
                         default=0)
    args = parser.parse_args()
    return args

def make_prediction_png_roads_buildings(image, predictions, save_figure_filename):
    bldg_pred = predictions[0][0]
    road_pred = predictions[1]
    
    # seperate the binary road preds and speed preds
    binary_road_pred = road_pred[-1]
    
    speed_pred = np.argmax(road_pred[:-1], axis=0)
    
    roadspeed_shape = road_pred.shape
    tempspeed = np.zeros(shape=(roadspeed_shape[0]+1,roadspeed_shape[1],roadspeed_shape[2]))
    tempspeed[1:] = road_pred
    road_pred = tempspeed
    road_pred = np.argmax(road_pred, axis=0)
    
    combined_pred = np.zeros(shape=bldg_pred.shape, dtype=np.uint8)
    combined_pred = np.where(bldg_pred==1, 1, combined_pred)
    combined_pred = np.where(binary_road_pred==1, 2, combined_pred)
    
    
    raw_im = np.moveaxis(image, 0, -1) # now it is channels last
    raw_im = raw_im/np.max(raw_im)
    
    grid = [[raw_im, combined_pred, speed_pred]]
    
    nrows = len(grid)
    ncols = len(grid[0])
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*4,nrows*4))
    for row in range(nrows):
        for col in range(ncols):
            ax = axs[col]
            ax.axis('off')
            if row==0 and col==0:
                ax.imshow(grid[row][col])
            elif row==0 and col in [3,4]:
                combined_mask_cmap = colors.ListedColormap(['black', 'green', 'blue', 'red',
                                                            'purple', 'orange', 'yellow', 'brown',
                                                            'pink'])
                ax.imshow(grid[row][col], cmap=combined_mask_cmap, interpolation='nearest', origin='upper',
                                  norm=colors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6, 7, 8], combined_mask_cmap.N))
            if row==0 and col in [1,2]:
                combined_mask_cmap = colors.ListedColormap(['black', 'red', 'blue'])
                ax.imshow(grid[row][col],
                          interpolation='nearest', origin='upper',
                          cmap=combined_mask_cmap,
                          norm=colors.BoundaryNorm([0, 1, 2, 3], combined_mask_cmap.N))
            # if row==1 and col == 1:
            #     ax.imshow(grid[0][0])
            #     mask = np.where(combined_gt==0, np.nan, combined_gt)
            #     ax.imshow(mask, cmap='gist_rainbow_r', alpha=0.6)
            # if row==1 and col == 2:
            #     ax.imshow(grid[0][0])
            #     mask = np.where(combined_pred==0, np.nan, combined_pred)
            #     ax.imshow(mask, cmap='gist_rainbow_r', alpha=0.6)
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(save_figure_filename)
    plt.close(fig)
    plt.close('all')

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
    model_path = args.model_path
    in_csv = args.in_csv
    save_fig_dir = args.save_fig_dir
    save_preds_dir = args.save_preds_dir
    gpu = args.gpu
    model_name = args.model_name

    img_size = (1300,1300)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if model_name == "unet":
        model = UNet(3, [1,8], bilinear=True)
    else:
        model = models[model_name](num_classes=[1, 8], num_channels=3)
    val_dataset = SN8Dataset(in_csv,
                        data_to_load=["preimg"],
                        img_size=img_size)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    #criterion = nn.BCEWithLogitsLoss()

    predictions = np.zeros((len(val_dataset),2,8,img_size[0],img_size[1]))
    running_tp = [0,0] 
    running_fp = [0,0]
    running_fn = [0,0]
    running_union = [0,0]

    filenames = [[], []]
    precisions = [[], []]
    recalls = [[], []]
    f1s = [[], []]
    ious = [[], []]
    positives = [[], []]

    model.eval()
    val_loss_val = 0
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            current_image_filename = val_dataset.get_image_filename(i)
            print("evaluating: ", i, os.path.basename(current_image_filename))
            preimg, postimg, building, road, roadspeed, flood = data
            preimg = preimg.cuda().float()
            
            building_pred, roadspeed_pred = model(preimg)
            
            roadspeed_pred = torch.sigmoid(roadspeed_pred)
            building_pred = torch.sigmoid(building_pred)
            
            preimg = preimg.cpu().numpy()[0] # index at 0 so we have (C,H,W)
            
            building_prediction = building_pred.cpu().numpy()[0][0] # index so shape is (H,W) for buildings
            building_prediction = np.rint(building_prediction).astype(int)
            road_prediction = roadspeed_pred.cpu().numpy()[0] # index so we have (C,H,W)
            roadspeed_prediction = np.rint(road_prediction).astype(int)
            
            predictions[i,0,0] = building_prediction
            predictions[i,1,:] = roadspeed_prediction

            ### save prediction
            if save_preds_dir is not None:
                road_pred_arr = (road_prediction * 255).astype(np.uint8) # to be compatible with the SN5 eval and road speed prediction, need to mult by 255
                ds = gdal.Open(current_image_filename)
                geotran = ds.GetGeoTransform()
                xmin, xres, rowrot, ymax, colrot, yres = geotran
                raster_srs = osr.SpatialReference()
                raster_srs.ImportFromWkt(ds.GetProjectionRef())
                ds = None
                output_tif = os.path.join(save_preds_dir, os.path.basename(current_image_filename.replace(".tif","_roadspeedpred.tif")))
                nchannels, nrows, ncols = road_pred_arr.shape
                write_geotiff(output_tif, ncols, nrows,
                            xmin, xres, ymax, yres,
                            raster_srs, road_pred_arr)
                            
                building_pred_arr = np.array([(building_prediction * 255).astype(np.uint8)])
                output_tif = os.path.join(save_preds_dir, os.path.basename(current_image_filename.replace(".tif","_buildingpred.tif")))
                nchannels, nrows, ncols = road_pred_arr.shape
                write_geotiff(output_tif, ncols, nrows,
                            xmin, xres, ymax, yres,
                            raster_srs, building_pred_arr)
            

            if save_fig_dir is not None:
                #if save_preds_dir is not None: # for some reason, seg fault when doing both of these. maybe file saving or something is interfering. so sleep for a little
                #    time.sleep(2) 
                save_figure_filename = os.path.join(save_fig_dir, os.path.basename(current_image_filename)[:-4]+"_pred.png")
                make_prediction_png_roads_buildings(preimg, predictions[i], save_figure_filename)

    print("Evaluation End!")
