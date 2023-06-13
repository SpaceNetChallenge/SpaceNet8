import os
import glob
import numpy as np
from osgeo import gdal
from utils.plots import write_diff_image
from utils.plots import write_pred_tiff
from skimage.morphology import erosion
import csv
import copy
import cv2

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()   # https://color-hex.org/
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7',
               '0000FF', '00FF00')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

def open_tiff(path):
    ds = gdal.Open(path)
    return ds.ReadAsArray()   

def assign_flood_class(img, flood_map, building_pred, road_pred):
    new_flood_pred = np.zeros(img.shape[:2])  # (H, W)
    pos = (building_pred == 1)
    new_flood_pred[pos] = 1 + flood_map[pos] 
    pos = (road_pred == 1)
    new_flood_pred[pos] = 3 + flood_map[pos]
    return new_flood_pred

def get_warped_ds(post_image_filename: str) -> gdal.Dataset:
    """ gdal warps (resamples) the post-event image to the same spatial resolution as the pre-event image and masks 
    
    SN8 labels are created from referencing pre-event image. Spatial resolution of the post-event image does not match the spatial resolution of the pre-event imagery and therefore the labels.
    In order to align the post-event image with the pre-event image and mask labels, we must resample the post-event image to the resolution of the pre-event image. Also need to make sure
    the post-event image covers the exact same spatial extent as the pre-event image. this is taken care of in the the tiling"""
    ds = gdal.Warp("", post_image_filename,
                    format='MEM', width=1300, height=1300,
                    resampleAlg=gdal.GRIORA_Bilinear,
                    outputType=gdal.GDT_Byte)
    return ds


data_types = ["preimg","postimg","flood","building","road","roadspeed"]
#data_types = ["preimg","postimg"]
dict_template = {}
list_path = './data/sn8_data_train.csv'
pred_dir = './train_tifs'
save_dir = './train_pngs'
preimg_dirs = ['./data/SN8_floods/Germany_Training_Public/PRE-event', './data/SN8_floods/Louisiana-East_Training_Public/PRE-event']
postimg_dirs = ['./data/SN8_floods/Germany_Training_Public/POST-event', './data/SN8_floods/Louisiana-East_Training_Public/POST-event']

for i in data_types:
    dict_template[i] = None
files = []
with open(list_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile) 
    for row in reader:
        in_data = copy.copy(dict_template)
        for j in data_types:
            in_data[j]=row[j]
        files.append(in_data)


for i, data_dict in enumerate(files):
    print(f'[{i+1}/{len(files)}]')
    if os.path.exists(data_dict["preimg"]):
        preimg = open_tiff(data_dict["preimg"]).transpose(1,2,0) 
        postimg = get_warped_ds(data_dict["postimg"]).ReadAsArray().transpose(1,2,0) if 'postimg' in data_types else None
        building_mask = open_tiff(data_dict["building"]) if 'building' in data_types else None
        road_mask = open_tiff(data_dict["road"]) if 'road' in data_types else None
        roadspeed_label = open_tiff(data_dict["roadspeed"]).transpose(1,2,0) if 'roadspeed' in data_types else None
        flood_label = open_tiff(data_dict["flood"]).transpose(1,2,0) if 'flood' in data_types else None
        basename = os.path.basename(data_dict["preimg"])
        preimg_name = os.path.join(pred_dir, os.path.basename(data_dict["preimg"]))
        flood_pred_path = preimg_name.replace('.tif', '_floodpred.tif')
        
        if os.path.exists(flood_pred_path):
        #   preimg         postimg
        # post-with-gt post-with-pred
            pre_post_img = np.concatenate((preimg, postimg), axis=1)

            # # gt 
            # nonflood_both_pred = flood_label[:,:,0] | flood_label[:,:,2]
            # flood_both_pred = flood_label[:,:,1] | flood_label[:,:,3]
    
            # # coloring flood and nonflood
            # flood_both_label = np.zeros((1300, 1300, 3), dtype=np.uint8)  # (H, W, 3)
            # nonflood_both_label = np.zeros((1300, 1300, 3), dtype=np.uint8)  # (H, W, 3)
            # pos = (flood_both_pred == 1) 
            # flood_both_label[pos] = list(colors(20, bgr=True)) # blue
            # flood_img = cv2.addWeighted(flood_both_label, 0.5, postimg, 1, 0)
            # pos = (nonflood_both_pred == 1) 
            # nonflood_both_label[pos] = list(colors(21, bgr=True)) # green
            # post_with_gt = cv2.addWeighted(nonflood_both_label, 0.5, flood_img, 1, 0)

            # pred 
            flood_pred = open_tiff(flood_pred_path)
            flood_both_pred = np.where((flood_pred == 2) | (flood_pred == 4), 1, 0)
            nonflood_both_pred = np.where((flood_pred == 1) | (flood_pred == 3), 1, 0)

            # coloring flood and nonflood
            flood_both_label = np.zeros((1300, 1300, 3), dtype=np.uint8)  # (H, W, 3)
            nonflood_both_label = np.zeros((1300, 1300, 3), dtype=np.uint8)  # (H, W, 3)
            pos = (flood_both_pred == 1) 
            flood_both_label[pos] = list(colors(20, bgr=True)) # red
            flood_img = cv2.addWeighted(flood_both_label, 0.5, postimg, 1, 0)
            pos = (nonflood_both_pred == 1) 
            nonflood_both_label[pos] = list(colors(21, bgr=True)) # blue
            post_with_pred = cv2.addWeighted(nonflood_both_label, 0.5, flood_img, 1, 0)

            #gt_pred_img = np.concatenate((post_with_gt, post_with_pred), axis=1)

            # conacatenate and save
            #img = np.concatenate((pre_post_img, gt_pred_img), axis=0)
            img = np.concatenate((pre_post_img, post_with_pred), axis=1)
            new_filename = os.path.basename(flood_pred_path).replace('.tif', '_pred.png')
            cv2.imwrite(os.path.join(save_dir, new_filename), img)

    # preimg postimg flood img    
        # preimg and postimg
        # pre_post_img = np.concatenate((preimg, postimg), axis=1)

        # # pred 
        # flood_pred = open_tiff(flood_pred_path)
        # flood_both_pred = np.where((flood_pred == 1) | (flood_pred == 3), 1, 0)
        # nonflood_both_pred = np.where((flood_pred == 2) | (flood_pred == 4), 1, 0)
        
        # # coloring flood and nonflood
        # flood_both_label = np.zeros((1300, 1300, 3), dtype=np.uint8)  # (H, W, 3)
        # nonflood_both_label = np.zeros((1300, 1300, 3), dtype=np.uint8)  # (H, W, 3)
        # pos = (flood_both_pred == 1) 
        # flood_both_label[pos] = list(colors(0, bgr=True)) # red
        # flood_img = cv2.addWeighted(flood_both_label, 0.5, postimg, 1, 0)
        # pos = (nonflood_both_pred == 1) 
        # nonflood_both_label[pos] = list(colors(20, bgr=True)) # blue
        # flood_both_img = cv2.addWeighted(nonflood_both_label, 0.5, flood_img, 1, 0)

        # # conacatenate and save
        # img = np.concatenate((pre_post_img, flood_both_img), axis=1)
        # new_filename = os.path.basename(flood_pred_path).replace('.tif', '_pred.png')
        # cv2.imwrite(os.path.join(save_dir, new_filename), img)


        # # gt 
        # flood_both_pred = flood_label[:,:,0] | flood_label[:,:,2]
        # nonflood_both_pred = flood_label[:,:,1] | flood_label[:,:,3]
        
        # # coloring flood and nonflood
        # flood_both_label = np.zeros((1300, 1300, 3), dtype=np.uint8)  # (H, W, 3)
        # nonflood_both_label = np.zeros((1300, 1300, 3), dtype=np.uint8)  # (H, W, 3)
        # pos = (flood_both_pred == 1) 
        # flood_both_label[pos] = list(colors(0, bgr=True)) # red
        # flood_img = cv2.addWeighted(flood_both_label, 0.5, postimg, 1, 0)
        # pos = (nonflood_both_pred == 1) 
        # nonflood_both_label[pos] = list(colors(20, bgr=True)) # blue
        # flood_both_img = cv2.addWeighted(nonflood_both_label, 0.5, flood_img, 1, 0)

        # # conacatenate and save
        # img = np.concatenate((pre_post_img, flood_both_img), axis=1)
        # new_filename = os.path.basename(flood_pred_path).replace('.tif', '_gt.png')
        # cv2.imwrite(os.path.join(save_dir, new_filename), img)

    # Building and road respectively
        # # pred 
        # flood_building_pred = np.where(flood_pred == 1, 1, 0)
        # flood_road_pred = np.where(flood_pred == 3, 1, 0)
        # building_label = np.zeros((1300, 1300, 3), dtype=np.uint8)  # (H, W, 3)
        # road_label = np.zeros((1300, 1300, 3), dtype=np.uint8)  # (H, W, 3)
        # pre_post_img = np.concatenate((preimg, postimg), axis=1)
        # # flood
        # pos = (flood_building_pred == 1) 
        # building_label[pos] = list(colors(0, bgr=True)) # red
        # flood_img = cv2.addWeighted(building_label, 0.5, postimg, 1, 0)
        # pos = (flood_road_pred == 1) 
        # road_label[pos] = list(colors(20, bgr=True)) # blue
        # flood_img = cv2.addWeighted(road_label, 0.5, flood_img, 1, 0)
        # img = np.concatenate((pre_post_img, flood_img), axis=1)
        # new_filename = os.path.basename(flood_pred_path).replace('.tif', '_pred.png')
        # cv2.imwrite(os.path.join(pred_dir, new_filename), img)

        # # gt 
        # flood_building = flood_label[:,:,0] 
        # flood_road = flood_label[:,:,2]
        # building_label = np.zeros((1300, 1300, 3), dtype=np.uint8)  # (H, W, 3)
        # road_label = np.zeros((1300, 1300, 3), dtype=np.uint8)  # (H, W, 3)
        # pre_post_img = np.concatenate((preimg, postimg), axis=1)
        # # flood
        # pos = (flood_building == 1) 
        # building_label[pos] = list(colors(0, bgr=True)) # red
        # flood_img = cv2.addWeighted(building_label, 0.5, postimg, 1, 0)
        # pos = (flood_road == 1) 
        # road_label[pos] = list(colors(20, bgr=True)) # blue
        # flood_img = cv2.addWeighted(road_label, 0.5, flood_img, 1, 0)
        # img = np.concatenate((pre_post_img, flood_img), axis=1)
        # new_filename = os.path.basename(flood_pred_path).replace('.tif', '_gt.png')
        # cv2.imwrite(os.path.join(pred_dir, new_filename), img)








