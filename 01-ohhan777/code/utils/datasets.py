import os
import csv
import glob
from osgeo import gdal
from pathlib import Path
import copy
import numpy as np
import random
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from utils.torch_utils import torch_distributed_zero_first
from utils.augmentations import Albumentations, letterbox, random_perspective, augment_hsv, random_scale, random_crop



def create_dataloader(path, data_to_load, task, batch_size, hyp=None, augment=False,
                      cache=False, pad=0.0, rank=-1, workers=8,
                      shuffle=True):
    #with torch_distributed_zero_first(rank):
    dataset = LoadImagesAndLabels(path, data_to_load, task, batch_size, augment=augment,
                                      hyp=hyp, cache_imgs=cache)
    batch_size = min(batch_size, len(dataset))
    num_devices = torch.cuda.device_count()
    num_workers = min([os.cpu_count() // max(num_devices, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
                        num_workers=num_workers, sampler=sampler, pin_memory=True, drop_last=False)
    return loader, dataset


class LoadImagesAndLabels(Dataset):

    def __init__(self, path, data_to_load, task, batch_size=16, augment=False,
                 hyp=None, cache_imgs=False):

        self.augment = augment
        self.hyp = hyp
        self.root = path
        self.task = task
        if task == 'train':
            list_path = '/wdata/sn8_data_train.csv'
            self.multi_scale = True
            self.flip = True
            self.crop_size = (600, 600)
        elif task == 'val':
            list_path = '/wdata/sn8_data_val.csv'
            self.multi_scale = False
            self.flip = False
            self.crop_size = (600, 600)
        elif task == 'trainval':
            list_path = ['/wdata/sn8_data_train.csv', '/wdata/sn8_data_val.csv']
            self.multi_scale = True
            self.flip = True
            self.crop_size = (600, 600)
        elif task == 'test':
            if os.path.exists(os.path.join(path, 'MysteryCity_Test_Private/MysteryCity_Test_Private_label_image_mapping.csv')):
                list_path = os.path.join(path, 'MysteryCity_Test_Private/MysteryCity_Test_Private_label_image_mapping.csv')
            else:
                list_path = os.path.join(path, 'Louisiana-West_Test_Public/Louisiana-West_Test_Public_label_image_mapping.csv')
            
            self.multi_scale = False
            self.flip = False
            self.crop_size = (1300, 1300)
        self.task = task
        self.list_path = list_path    
        self.albumentation = Albumentations()
        self.list_path = list_path
        self.all_data_types = ["preimg", "postimg", "building", "road", "roadspeed", "flood"]
        
        self.data_to_load = data_to_load
        self.img_size = (1300, 1300)

        self.mean=[0.249566, 0.318912, 0.21801]
        self.std=[0.12903, 0.11784, 0.10739]
        
        self.files = []

        
        dict_template = {}
        if task in ["train", "trainval", "val"]:
            for i in self.all_data_types:
                dict_template[i] = None
            
            if isinstance(list_path, str):
                with open(list_path, newline='') as csvfile:
                    reader = csv.DictReader(csvfile) 
                    for row in reader:
                        in_data = copy.copy(dict_template)
                        for j in self.data_to_load:
                            in_data[j]=row[j]
                        self.files.append(in_data)
            elif isinstance(list_path, list):
                for l in list_path:
                    with open(l, newline='') as csvfile:
                        reader = csv.DictReader(csvfile) 
                        for row in reader:
                            in_data = copy.copy(dict_template)
                            for j in self.data_to_load:
                                in_data[j]=row[j]
                            self.files.append(in_data)        
        else:
            test_data_types = ["label","pre-event image","post-event image 1","post-event image 2"]
            for i in test_data_types:
                dict_template[i] = None

            with open(list_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile) 
                for row in reader:
                    in_data = copy.copy(dict_template)
                    for j in self.data_to_load:
                        in_data[j]=row[j]
                    self.files.append(in_data)
            # preimg_dir = os.path.join(path, 'PRE-event')
            # preimg_files = glob.glob(os.path.join(preimg_dir, '*.tif'))
            # postimg_dir = os.path.join(path, 'POST-event')
            # for preimg_file in preimg_files:
            #     in_data = copy.copy(dict_template)
            #     in_data["preimg"] = preimg_file
            #     in_data["postimg"] = os.path.join(postimg_dir, os.path.basename(preimg_file))
            #     self.files.append(in_data)
            
            
        #print("loaded", len(self.files), "image filepaths")
        
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_dict = self.files[index]
        #print(data_dict)
        
        #assert 'preimg' in self.data_to_load, print('preimg should be in the data_to_load list')
        if self.task in ["train", "trainval", "val"]:
            preimg_path = data_dict["preimg"]
            postimg_path = data_dict["postimg"]
        else:
            preimg_path = data_dict["pre-event image"]
            postimg_path = data_dict["post-event image 1"]
            preimg_path = os.path.join(os.path.dirname(self.list_path), 'PRE-event/' + preimg_path)
            postimg_path = os.path.join(os.path.dirname(self.list_path), 'POST-event/' + postimg_path)
        preimg = self.open_geotiff(preimg_path).transpose(1,2,0) 
        postimg = self.get_warped_ds(postimg_path).ReadAsArray().transpose(1,2,0) if 'postimg' in self.data_to_load or 'post-event image 1' in self.data_to_load else None
        building_mask = self.open_geotiff(data_dict["building"]) if 'building' in self.data_to_load else None
        road_mask = self.open_geotiff(data_dict["road"]) if 'road' in self.data_to_load else None
        roadspeed_label = self.open_geotiff(data_dict["roadspeed"]).transpose(1,2,0) if 'roadspeed' in self.data_to_load else None
        flood_label = self.open_geotiff(data_dict["flood"]).transpose(1,2,0) if 'flood' in self.data_to_load else None

        preimg, postimg, building_mask, road_mask, \
                 roadspeed_label, flood_label = self.gen_sample(preimg, postimg, building_mask, road_mask, \
                                                           roadspeed_label, flood_label, \
                                                           self.multi_scale, self.flip) 
        
          
        return preimg, postimg, building_mask, road_mask, roadspeed_label, flood_label, preimg_path


    def gen_sample(self, pre_img, post_img, building_mask, road_mask, roadspeed_label, flood_label, multi_scale=True, is_flip=True):
        if multi_scale:
            rand_scale = 0.8 + random.randint(0, 4) / 10.0   #0.8-1.2
            pre_img, post_img, building_mask, road_mask, roadspeed_label, flood_label = self.multi_scale_aug(pre_img, post_img, \
                                              building_mask, road_mask, roadspeed_label, flood_label, rand_scale=rand_scale)

        if self.task in ('train', 'trainval'):
            pre_img = self.albumentation(pre_img)
            if post_img is not None:
                post_img = self.albumentation(post_img)

        pre_img = self.input_transform(pre_img)
        post_img = self.input_transform(post_img)
        building_mask = self.label_transform(building_mask)
        road_mask = self.label_transform(road_mask)
        roadspeed_label = self.label_transform(roadspeed_label)
        flood_label = self.label_transform(flood_label)

        pre_img = pre_img.transpose(2, 0, 1)
        post_img = post_img.transpose(2, 0, 1) if post_img is not None else None
        roadspeed_label = roadspeed_label.transpose(2, 0, 1) if roadspeed_label is not None else None
        flood_label = flood_label.transpose(2, 0, 1) if flood_label is not None else None



        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            pre_img = pre_img[:, :, ::flip]
            post_img = post_img[:, :, ::flip] if post_img is not None else None 
            building_mask = building_mask[:, ::flip] if building_mask is not None else None
            road_mask = road_mask[:, ::flip].copy() if road_mask is not None else None
            roadspeed_label = roadspeed_label[:, :, ::flip] if roadspeed_label is not None else None
            flood_label = flood_label[:, :, ::flip] if flood_label is not None else None

        # NoneType to number and copy() 
        post_img = post_img.copy() if post_img is not None else 0
        building_mask = building_mask.copy() if building_mask is not None else 0
        road_mask = road_mask.copy() if road_mask is not None else 0
        roadspeed_label = roadspeed_label.copy() if roadspeed_label is not None else 0
        flood_label = flood_label.copy() if flood_label is not None else 0

        return pre_img.copy(), post_img, building_mask, road_mask, roadspeed_label, flood_label
    
    def multi_scale_aug(self, pre_img, post_img, building_mask, road_mask, roadspeed_label, flood_label,
                        rand_scale=1, rand_crop=True):
        long_size = np.int(1300 * rand_scale + 0.5)
        h, w = pre_img.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        pre_img = cv2.resize(pre_img, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if post_img is not None:
            post_img = cv2.resize(post_img, (new_w, new_h),
                            interpolation=cv2.INTER_LINEAR)
        if building_mask is not None:
            building_mask = cv2.resize(building_mask, (new_w, new_h),
                                interpolation=cv2.INTER_NEAREST)
        if road_mask is not None:                        
            road_mask = cv2.resize(road_mask, (new_w, new_h),
                                interpolation=cv2.INTER_NEAREST)
        if roadspeed_label is not None:
            roadspeed_label = cv2.resize(roadspeed_label, (new_w, new_h),
                                interpolation=cv2.INTER_NEAREST)
        if flood_label is not None:
            flood_label = cv2.resize(flood_label, (new_w, new_h),
                                interpolation=cv2.INTER_NEAREST)
        if rand_crop:
            pre_img, post_img, building_mask, road_mask, roadspeed_label, flood_label = self.rand_crop(pre_img, post_img, \
                                                                                                       building_mask, road_mask, \
                                                                                                       roadspeed_label, flood_label)

        return pre_img, post_img, building_mask, road_mask, roadspeed_label, flood_label

    def open_geotiff(self, path):
        ds = gdal.Open(path)
        return ds.ReadAsArray()    


    def rand_crop(self, pre_img, post_img, building_mask, road_mask, roadspeed_label, flood_label):
        h, w = pre_img.shape[:-1]
        pre_img = self.pad_image(pre_img, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        new_h, new_w = pre_img.shape[:-1]        
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        pre_img = pre_img[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        if post_img is not None:
            post_img = self.pad_image(post_img, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
            post_img = post_img[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        if building_mask is not None:
            building_mask = self.pad_image(building_mask, h, w, self.crop_size,
                               (0,))
            building_mask = building_mask[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        if road_mask is not None: 
            road_mask = self.pad_image(road_mask, h, w, self.crop_size,
                               (0,))
            road_mask = road_mask[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        if roadspeed_label is not None:
            roadspeed_label = self.pad_image(roadspeed_label, h, w, self.crop_size,
                               (0,))
            roadspeed_label = roadspeed_label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        if flood_label is not None:
            flood_label = self.pad_image(flood_label, h, w, self.crop_size,
                               (0,))                       
            flood_label = flood_label[y:y+self.crop_size[0], x:x+self.crop_size[1]]        

        return pre_img, post_img, building_mask, road_mask, roadspeed_label, flood_label

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image


    def input_transform(self, image):
        if image is None:
            return image
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        if label is None:
            return None
        return np.array(label).astype('int32')

            
    # @staticmethod
    # def collate_fn(batch):
    #     img, label = zip(*batch)  # transposed
    #     return torch.stack(img, 0), torch.stack(label, 0)    



    def get_warped_ds(self, post_image_filename: str) -> gdal.Dataset:
        """ gdal warps (resamples) the post-event image to the same spatial resolution as the pre-event image and masks 
        
        SN8 labels are created from referencing pre-event image. Spatial resolution of the post-event image does not match the spatial resolution of the pre-event imagery and therefore the labels.
        In order to align the post-event image with the pre-event image and mask labels, we must resample the post-event image to the resolution of the pre-event image. Also need to make sure
        the post-event image covers the exact same spatial extent as the pre-event image. this is taken care of in the the tiling"""
        ds = gdal.Warp("", post_image_filename,
                       format='MEM', width=self.img_size[1], height=self.img_size[0],
                       resampleAlg=gdal.GRIORA_Bilinear,
                       outputType=gdal.GDT_Byte)
        return ds



