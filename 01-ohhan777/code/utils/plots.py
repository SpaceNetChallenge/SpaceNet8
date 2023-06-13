import os
import cv2
import torch
import numpy as np
from osgeo import gdal
from osgeo import osr

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()   # https://color-hex.org/
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7',
               '0000FF')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

def write_gt_pred_images(imgs, building_targets, road_targets, building_preds, road_preds, save_dir, filenames):
      
    size = imgs.shape[0]
    for k in range(size):
        # Plot image grid with labels
        if isinstance(imgs, torch.Tensor):
            img = imgs[k].cpu().float().numpy()   # img: (C, H, W)

        if isinstance(building_targets, torch.Tensor):
            building_label = building_targets[k].cpu().numpy()       # label: (H, W) 
            building_pred = building_preds[k].cpu().numpy()

        if isinstance(road_targets, torch.Tensor):
            road_label = road_targets[k].cpu().numpy()       # label: (H, W) 
            road_pred = road_preds[k].cpu().numpy()


        # de-normalize
        img = img.transpose(1,2,0)
        mean=[0.249566, 0.318912, 0.21801]
        std=[0.12903, 0.11784, 0.10739]
        img *= std
        img += mean
        img *= 255.0 

        img = img.astype(np.uint8)  # (C, H, W) -> (H, W, C)
        h, w, _ = img.shape
        target_label = np.zeros((h, w, 3), dtype=np.uint8)  # (H, W, 3)
        pred_label = np.zeros((h, w, 3), dtype=np.uint8)  # (H, W, 3)


        # building
        pos = (building_label == 1)
        target_label[pos] = list(colors(10, bgr=True))
        pos = (building_pred == 1) 
        pred_label[pos] = list(colors(10, bgr=True))

        # roadspeed
        for i in range(7):
            pos = (road_label == i)
            target_label[pos] = list(colors(i, bgr=True))
            pos = (road_pred == i) 
            pred_label[pos] = list(colors(i, bgr=True))

        target_img = cv2.addWeighted(target_label, 0.5, img, 1, 0)
        target_img = np.concatenate((target_img, np.zeros((h, 2, 3))), axis=1)  # vertical seperate line
        pred_img = cv2.addWeighted(pred_label, 0.5, img, 1, 0)

        filename = filenames[k]

        img = np.concatenate((target_img, pred_img), axis=1)
        new_filename = os.path.basename(filename).replace('.tif', '_gt_pred.png')
        cv2.imwrite(os.path.join(save_dir, new_filename), img)


def write_pred_images(imgs, building_preds, road_preds, save_dir, filenames):
      
    size = imgs.shape[0]
    for k in range(size):
        # Plot image grid with labels
        if isinstance(imgs, torch.Tensor):
            img = imgs[k].cpu().float().numpy()   # img: (C, H, W)

        if isinstance(building_preds, torch.Tensor):
            building_pred = building_preds[k].cpu().numpy()

        if isinstance(road_preds, torch.Tensor):
            road_pred = road_preds[k].cpu().numpy()


        # de-normalize
        img = img.transpose(1,2,0)
        mean=[0.249566, 0.318912, 0.21801]
        std=[0.12903, 0.11784, 0.10739]
        img *= std
        img += mean
        img *= 255.0 

        img = img.astype(np.uint8)  # (C, H, W) -> (H, W, C)
        h, w, _ = img.shape
        pred_label = np.zeros((h, w, 3), dtype=np.uint8)  # (H, W, 3)


        # building
        pos = (building_pred == 1) 
        pred_label[pos] = list(colors(10, bgr=True))

        # roadspeed
        for i in range(7):
            pos = (road_pred == i)
            pred_label[pos] = list(colors(i, bgr=True))

        pred_img = cv2.addWeighted(pred_label, 0.5, img, 1, 0)

        filename = filenames[k]

        new_filename = os.path.basename(filename).replace('.tif', '_pred.png')
        cv2.imwrite(os.path.join(save_dir, new_filename), pred_img)



def write_pred_tiffs(preimg_filenames, building_preds, roadspeed_sn_preds, flood_preds, save_dir):
    size = len(preimg_filenames)
    for k in range(size):
        ds = gdal.Open(preimg_filenames[k])
        geotran = ds.GetGeoTransform()
        xmin, xres, _, ymax, _, yres = geotran
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(ds.GetProjectionRef())
        ds = None
        building_pred_img = np.array([building_preds[k].cpu().numpy() * 255]).astype(np.uint8)
        filename = os.path.join(save_dir, os.path.basename(preimg_filenames[k]).replace('.tif', '_buildingpred.tif'))
        _, nrows, ncols = building_pred_img.shape
        write_geotiff(filename, ncols, nrows, xmin, xres, ymax, yres, raster_srs, building_pred_img)

        roadspeed_sn_pred_img = (roadspeed_sn_preds[k].cpu().numpy() * 255).astype(np.uint8)
        filename = os.path.join(save_dir, os.path.basename(preimg_filenames[k]).replace('.tif', '_roadspeedpred.tif'))
        _, nrows, ncols = roadspeed_sn_pred_img.shape
        write_geotiff(filename, ncols, nrows, xmin, xres, ymax, yres, raster_srs, roadspeed_sn_pred_img)

        flood_pred_img = flood_preds[k].cpu().numpy() # (H, W)
        nrows, ncols = flood_pred_img.shape
        filename = os.path.join(save_dir, os.path.basename(preimg_filenames[k]).replace('.tif', '_floodpred.tif'))
        write_geotiff(filename, ncols, nrows, xmin, xres, ymax, yres, raster_srs, [flood_pred_img])


def write_best_pred_tiffs(preimg_filenames, item, building_preds, roadspeed_sn_preds, flood_preds, save_dir):
    size = len(preimg_filenames)
    for k in range(size):
        ds = gdal.Open(preimg_filenames[k])
        geotran = ds.GetGeoTransform()
        xmin, xres, _, ymax, _, yres = geotran
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(ds.GetProjectionRef())
        ds = None
        if item == "building":
            building_pred_img = np.array([building_preds[k].cpu().numpy() * 255]).astype(np.uint8)
            filename = os.path.join(save_dir, os.path.basename(preimg_filenames[k]).replace('.tif', '_buildingpred.tif'))
            _, nrows, ncols = building_pred_img.shape
            write_geotiff(filename, ncols, nrows, xmin, xres, ymax, yres, raster_srs, building_pred_img)
        elif item == "road":
            roadspeed_sn_pred_img = (roadspeed_sn_preds[k].cpu().numpy() * 255).astype(np.uint8)
            filename = os.path.join(save_dir, os.path.basename(preimg_filenames[k]).replace('.tif', '_roadspeedpred.tif'))
            _, nrows, ncols = roadspeed_sn_pred_img.shape
            write_geotiff(filename, ncols, nrows, xmin, xres, ymax, yres, raster_srs, roadspeed_sn_pred_img)
        elif item == "flood":
            flood_pred_img = flood_preds[k].cpu().numpy() # (H, W)
            nrows, ncols = flood_pred_img.shape
            filename = os.path.join(save_dir, os.path.basename(preimg_filenames[k]).replace('.tif', '_floodpred.tif'))
            write_geotiff(filename, ncols, nrows, xmin, xres, ymax, yres, raster_srs, [flood_pred_img])


def write_pred_tiff(preimg_filename, building_pred, roadspeed_pred, flood_pred, save_dir):
    
    
    ds = gdal.Open(preimg_filename)
    geotran = ds.GetGeoTransform()
    xmin, xres, _, ymax, _, yres = geotran
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(ds.GetProjectionRef())
    ds = None
    building_pred_img = np.array([building_pred * 255]).astype(np.uint8)
    filename = os.path.join(save_dir, os.path.basename(preimg_filename).replace('.tif', '_buildingpred.tif'))
    _, nrows, ncols = building_pred_img.shape
    write_geotiff(filename, ncols, nrows, xmin, xres, ymax, yres, raster_srs, building_pred_img)

    roadspeed_sn_pred_img = (roadspeed_pred * 255).astype(np.uint8)
    filename = os.path.join(save_dir, os.path.basename(preimg_filename).replace('.tif', '_roadspeedpred.tif'))
    _, nrows, ncols = roadspeed_sn_pred_img.shape
    write_geotiff(filename, ncols, nrows, xmin, xres, ymax, yres, raster_srs, roadspeed_sn_pred_img)

    flood_pred_img = flood_pred # (H, W)
    nrows, ncols = flood_pred_img.shape
    filename = os.path.join(save_dir, os.path.basename(preimg_filename).replace('.tif', '_floodpred.tif'))
    write_geotiff(filename, ncols, nrows, xmin, xres, ymax, yres, raster_srs, [flood_pred_img])


        

def write_geotiff(output_tif, ncols, nrows,
                  xmin, xres,ymax, yres,
                 raster_srs, label_arr):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif, ncols, nrows, len(label_arr), gdal.GDT_Byte)
    out_ds.SetGeoTransform((xmin, xres, 0, ymax, 0, yres))
    out_ds.SetProjection(raster_srs.ExportToWkt())
    for i in range(len(label_arr)):
        outband = out_ds.GetRasterBand(i+1)
        outband.WriteArray(label_arr[i])
        #outband.SetNoDataValue(0)
        outband.FlushCache()
    out_ds = None


def write_diff_image(img, building_pred, flood_building_pred, road_pred, flood_road_pred, save_dir, filename):
      
    h, w, _ = img.shape
    pred_label = np.zeros((h, w, 3), dtype=np.uint8)  # (H, W, 3)


    # compare
    # building
    pos = ((building_pred == 1) & (flood_building_pred == 1)) 
    pred_label[pos] = list(colors(20, bgr=True))  # blue

    pos = ((building_pred == 1) & (flood_building_pred == 0))
    pred_label[pos] = list(colors(0, bgr=True))   # red

    pos = ((building_pred == 0) & (flood_building_pred == 1))
    pred_label[pos] = list(colors(4, bgr=True))   # yellow


    # road
    pos = ((road_pred == 1) & (flood_road_pred == 1)) 
    pred_label[pos] = list(colors(12, bgr=True))  # blue

    pos = ((road_pred == 1) & (flood_road_pred == 0))
    pred_label[pos] = list(colors(0, bgr=True))   # red

    pos = ((road_pred == 0) & (flood_road_pred == 1))
    pred_label[pos] = list(colors(4, bgr=True))   # yellow
         

    pred_img = cv2.addWeighted(pred_label, 0.7, img, 1, 0)

    
    cv2.imwrite(os.path.join(save_dir, filename), pred_img)


