import os
import time
import argparse

from osgeo import gdal
from osgeo import osr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.nn as nn

import models.pytorch_zoo.unet as unet
from datasets.datasets import SN8Dataset
from models.other.unet import UNetSiamese
from utils.utils import write_geotiff
from models.other.siamunetdif import SiamUnet_diff

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
                        help="saves model predictions as .pngs for easy viewing.",
                         type=str,
                         required=False,
                         default=None)
    parser.add_argument("--save_preds_dir",
                        help="saves model predictions as .tifs",
                         type=str,
                         required=False,
                         default=None)
    parser.add_argument("--gpu",
                         type=int,
                         required=False,
                         default=0)
    args = parser.parse_args()
    return args

def make_prediction_png(image, postimage, prediction, save_figure_filename):
    #raw_im = image[:,:,:3]
    #raw_im = np.asarray(raw_im[:,:,::-1], dtype=np.float32)
    raw_im = np.moveaxis(image, 0, -1) # now it is channels last
    raw_im = raw_im/np.max(raw_im)
    post_im = np.moveaxis(postimage, 0, -1)
    post_im = post_im/np.max(post_im)
    
    #gt = np.asarray(gt*255., dtype=np.uint8)
    #pred = np.asarray(prediction*255., dtype=np.uint8)
    
    combined_mask_cmap = colors.ListedColormap(['black', 'red', 'blue', 'green', 'yellow'])

    grid = [[raw_im, prediction],[post_im, 0,]]

    fig, axs = plt.subplots(2, 3, figsize=(12,8))
    for row in range(2):
        for col in range(2):
            ax = axs[row][col]
            ax.axis('off')
            if row==0 and col == 0:
                theim = ax.imshow(grid[row][col])
            elif row==1 and col == 0:
                theim = ax.imshow(grid[row][col])
            elif row==0 and col in [1,2]:
                ax.imshow(grid[row][col],
                          interpolation='nearest', origin='upper',
                          cmap=combined_mask_cmap,
                          norm=colors.BoundaryNorm([0, 1, 2, 3, 4, 5], combined_mask_cmap.N))
            elif row==1 and col == 1:
                ax.imshow(grid[0][0])
                mask = np.where(prediction==0, np.nan, 1)
                ax.imshow(mask, cmap='gist_rainbow_r', alpha=0.6)
            elif row==1 and col == 2:
                ax.imshow(grid[0][0])
                mask = np.where(prediction==0, np.nan, 1)
                ax.imshow(mask, cmap='gist_rainbow_r', alpha=0.6)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(save_figure_filename, dpi=95)
    plt.close(fig)
    plt.close('all')
                
    
models = {
    'resnet34_siamese': unet.Resnet34_siamese_upsample,
    'resnet34': unet.Resnet34_upsample,
    'resnet50': unet.Resnet50_upsample,
    'resnet101': unet.Resnet101_upsample,
    'seresnet50': unet.SeResnet50_upsample,
    'seresnet101': unet.SeResnet101_upsample,
    'seresnet152': unet.SeResnet152_upsample,
    'seresnext50': unet.SeResnext50_32x4d_upsample,
    'seresnext101': unet.SeResnext101_32x4d_upsample,
    'unet_siamese_dif':SiamUnet_diff,
    'unet_siamese': UNetSiamese
}

if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    in_csv = args.in_csv
    model_name = args.model_name
    save_fig_dir = args.save_fig_dir
    save_preds_dir = args.save_preds_dir
    gpu = args.gpu

    num_classes = 5
    img_size = (1300,1300)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    val_dataset = SN8Dataset(in_csv,
                            data_to_load=["preimg","postimg"],
                            img_size=img_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    if model_name == "unet_siamese":
        model = UNetSiamese(3, num_classes, bilinear=True)
    else:
        model = models[model_name](num_classes=num_classes, num_channels=3)

    model.load_state_dict(torch.load(model_path))
    model.cuda()

    #criterion = nn.BCEWithLogitsLoss()

    predictions = np.zeros((len(val_dataset),img_size[0],img_size[1]))
    gts = np.zeros((len(val_dataset),img_size[0],img_size[1]))

    # we need running numbers for each class: [no flood bldg, flood bldg, no flood road, flood road]
    classes = ["non-flooded building", "flooded building", "non-flooded road", "flooded road"]
    running_tp = [0, 0, 0, 0]
    running_fp = [0, 0, 0, 0]
    running_fn = [0, 0, 0, 0]
    running_union = [0, 0, 0, 0]

    filenames = []
    precisions = [[],[],[],[]]
    recalls = [[],[],[],[]]
    f1s = [[],[],[],[]]
    ious = [[],[],[],[]]
    positives = [[],[],[],[]]

    model.eval()
    val_loss_val = 0
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            
            current_image_filename = val_dataset.get_image_filename(i)
            print("evaluating: ", i, os.path.basename(current_image_filename))
            preimg, postimg, building, road, roadspeed, flood = data

            preimg = preimg.cuda().float() #siamese
            postimg = postimg.cuda().float() #siamese
            

            flood_pred = model(preimg, postimg) # siamese resnet34 with stacked preimg+postimg input
            flood_pred = torch.nn.functional.softmax(flood_pred, dim=1).cpu().numpy()[0] # (5, H, W)
            #for i in flood_pred:
            #    plt.imshow(i)
            #    plt.colorbar()
            #    plt.show()
            
            flood_prediction = np.argmax(flood_pred, axis=0) # (H, W)
            #plt.imshow(flood_pred)
            #plt.colorbar()
            #plt.show()
            
            #flood_pred = torch.softmax(flood_pred)
            #flood_pred = torch.sigmoid(flood_pred)
            
            #print(flood_pred.shape)
            
            ### save prediction
            if save_preds_dir is not None:
                ds = gdal.Open(current_image_filename)
                geotran = ds.GetGeoTransform()
                xmin, xres, rowrot, ymax, colrot, yres = geotran
                raster_srs = osr.SpatialReference()
                raster_srs.ImportFromWkt(ds.GetProjectionRef())
                ds = None
                output_tif = os.path.join(save_preds_dir, os.path.basename(current_image_filename.replace(".tif","_floodpred.tif")))
                nrows, ncols = flood_prediction.shape
                write_geotiff(output_tif, ncols, nrows,
                            xmin, xres, ymax, yres,
                            raster_srs, [flood_prediction])
            
            preimg = preimg.cpu().numpy()[0] # index at 0 so we have (C,H,W)
            postimg = postimg.cpu().numpy()[0]
            
            
            #flood_prediction = flood_pred.cpu().numpy()[0] # index so shape is (C,H,W) for buildings
            #flood_prediction = np.append(np.zeros(shape=(1,flood_shape[2],flood_shape[3])), flood_prediction, axis=0) # for focal loss 
            #flood_prediction = np.argmax(flood_prediction, axis=0)
            #flood_prediction = np.rint(flood_prediction).astype(int)

                           
            current_image_filename = val_dataset.files[i]["preimg"]
            filenames.append(current_image_filename)
            if save_fig_dir != None:
                save_figure_filename = os.path.join(save_fig_dir, os.path.basename(current_image_filename)[:-4]+"_pred.png")
                make_prediction_png(preimg, postimg, flood_prediction, save_figure_filename)

    print('Inference End!')
