import numpy as np
import os
import sys
from glob import glob
import json
import torch
from skimage import io
import tifffile
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.utils.data
import random
import torchvision
import cv2
import csv
import pathlib
import copy
from typing import List, Tuple
from torchvision import transforms as T
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler
from albumentations import Compose, RandomSizedCrop, HorizontalFlip, VerticalFlip, RGBShift, RandomBrightnessContrast, \
    RandomGamma, OneOf, RandomRotate90, PadIfNeeded, Transpose, RandomCrop, Rotate, ShiftScaleRotate, ColorJitter, Resize
from matplotlib import pyplot as plt

is_torchvision_installed = True
np.random.seed(13)
#MAPPING_SPEED = {15: 1, 19: 1, 20: 1, 22: 2, 25: 2, 26: 2, 30: 2, 34: 3, 35: 3, 41: 4, 45: 4, 49: 4, 55: 5, 65: 6}
MAPPING_SPEED = {15: 1, 19: 1, 20: 1, 22: 2, 25: 2, 26: 3, 30: 3, 34: 4, 35: 4, 41: 5, 45: 5, 49: 6, 55: 7, 65: 8}
max_speed_dim = max(MAPPING_SPEED.values())

debug = False
crop_size = (512, 512) if not debug else (16, 16)
resize_size = (512, 512) # 640 640
width, height = crop_size
rand_crop = T.RandomCrop(crop_size, pad_if_needed=True)

aug_val_center = T.CenterCrop(crop_size)
aug_val_five = T.FiveCrop(crop_size[0])
has_offnadir = False

tx = [
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0),
    OneOf([RandomSizedCrop(min_max_height=(int(height * 0.8), int(height * 1.2)), w2h_ratio=1., height=height,
                           width=width, p=0.9),
           RandomCrop(height=height, width=width, p=0.1)], p=1),
    Rotate(limit=10, p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0),
    HorizontalFlip(),
    RandomRotate90(),
    Transpose(),
    OneOf([RGBShift(), RandomBrightnessContrast(), RandomGamma(), ColorJitter()], p=0.5),
]
if not has_offnadir:
    tx.append(VerticalFlip())


def belta_loaders(root_dir, img_dirname, label=None, add_prefix='', add_postfix=''):
    root_dir = os.path.normpath(root_dir)
    subdirs = glob(root_dir + '/*')
    images_masks = []
    for subdir in subdirs:
        city_images_masks = []
        masks = glob(os.path.join(subdir, 'masks') + '/*.png')
        for maskpath in masks:
            basename = os.path.basename(maskpath).replace('.png', '.tif')
            basename = basename if add_prefix == '' else add_prefix + basename
            if add_postfix == '':
                basename = basename
            else:
                bs = basename.split('_')
                basename = '_'.join(bs[0:-1]) + add_postfix + bs[-1]
            impath = os.path.join(subdir, img_dirname, basename)
            # actual image not available or image with no annotation
            if os.path.isfile(impath):
                city_images_masks.append((impath, maskpath, label))
            else:
                print(maskpath, impath)
        images_masks.extend(city_images_masks)
    print(f'num samples in {os.path.basename(root_dir)} = {len(images_masks)}')
    return images_masks


def get_images_mask_paths(config):
    c = config
    path_root = c['root_dirs']
    all_paths = []
    if os.path.isdir(path_root['buildings']):
        buildings_paths = belta_loaders(path_root['buildings'], c['buildings_imgdir'], label='buildings')
        all_paths.extend(buildings_paths)
    if os.path.isdir(path_root['roads']):
        roads_path = belta_loaders(path_root['roads'],  c['roads_imgdir'], label='roads')
        all_paths.extend(roads_path)

    for k in ['offnadir.1', 'offnadir.2']:
        if k in path_root.keys():
            if os.path.isdir(path_root[k]):
                off_path = belta_loaders(path_root[k],  c['offnadir_imgdir'], label='buildings',
                                 add_prefix=c['offnadir_prefix_add'])
                all_paths.extend(off_path)
    if 'sp5_roads' in path_root.keys():
        if os.path.isdir(path_root['sp5_roads']):
            roads_path_sp5 = belta_loaders(path_root['sp5_roads'], c['sp5_imgdir'], label='roads',
                                           add_postfix=c['sp5_postfix_add'])
            all_paths.extend(roads_path_sp5)

    ix = np.arange(0, len(all_paths))
    np.random.shuffle(ix)
    all_paths = np.asarray(all_paths)[ix]
    bar = int(len(all_paths) * 0.9)
    train_paths, val_paths = all_paths[0:bar], all_paths[bar::]
    return train_paths, val_paths


# Based on https://github.com/galatolofederico/pytorch-balanced-batch/blob/master/sampler.py
class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max * len(self.keys)


class SpacenetDataset(Dataset):
    def __init__(self, paths, train=True, task='both', tr=None):
        super(SpacenetDataset, self).__init__()
        self.paths = paths
        self.labels = [int(item[-1]=='roads') for item in paths]
        if tr is None:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            self.tr = T.Compose([T.ToTensor(), T.Normalize(mean=self.mean, std=self.std)])
        else:
            self.tr = tr
        print('Using normalization ', self.tr)
        self.train = train
        self.aug_train = Compose(tx)
        self.task = task

    def __len__(self):
        return len(self.paths)

    def augment(self, img, mask):
        augmented = self.aug_train(image=img, mask=mask)
        return augmented['image'], augmented['mask']

    def random_crop(self, img, mask):
        i, j, h, w = rand_crop.get_params(img, output_size=crop_size)
        image = T.functional.crop(img, i, j, h, w)
        mask = T.functional.crop(mask, i, j, h, w)
        return image, mask

    def map_roads(self, mask):
        for k in np.unique(mask):
            if k == 0:
                continue
            mask[mask == k] = MAPPING_SPEED[k]  # replace with find nearest?
        return mask

    def __getitem__(self, idx):
        img_path,mask_path,label = self.paths[idx]
        image = io.imread(img_path)
        mask = io.imread(mask_path)
        mask = mask if not np.max(mask) == 255 else (mask/255).astype('uint8')
        image, mask = self.augment(image, mask)
        label = 0 if label == 'buildings' else 1
        if label==1:
            mask = (mask>0)
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        mask = np.transpose(mask, (2, 0, 1)).astype('float32')
        image = self.tr(Image.fromarray(image))
        if self.task == 'both':
            return image, mask, label
        else:
            return image, mask


def get_generators(config, use_offnadir=True,
                   task='both'):
    def filter_offnadir(paths):
        return [item for item in paths if not 'Atlanta' in item[0]]

    c = config
    train_paths, val_paths = get_images_mask_paths(config)
    assert task in ['buildings', 'roads', 'both']
    if task == 'buildings':
        train_paths = [item for item in train_paths if int(item[-1] == 'buildings')]
        val_paths = [item for item in val_paths if int(item[-1] == 'buildings')]
    if task == 'roads':
        train_paths = [item for item in train_paths if int(item[-1] == 'roads')]
        val_paths = [item for item in val_paths if int(item[-1] == 'roads')]
    if not use_offnadir:
        train_paths = filter_offnadir(train_paths)
        val_paths = filter_offnadir(val_paths)

    print(f'Final num train samples {len(train_paths)}....Final num val samples {len(val_paths)}')
    tr = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) \
        if c['model_name'] == 'inceptionresnetv2' else None
    train_dataset = SpacenetDataset(train_paths, train=True, task=task, tr=tr)
    val_dataset = SpacenetDataset(val_paths, train=False, task=task, tr=tr)
    bs_train = BalancedBatchSampler(train_dataset, labels=torch.Tensor([int(item[-1] == 'roads') for item in train_paths]))
    bs_val = BalancedBatchSampler(val_dataset, labels=torch.Tensor([int(item[-1] == 'roads') for item in val_paths]))

    training_generator = DataLoader(train_dataset, num_workers=c['num_workers'], sampler=bs_train,
                                    batch_size=c['batch_size'], pin_memory=True, drop_last=True)
    val_generator = DataLoader(val_dataset, batch_size=c['batch_size'], pin_memory=True,
                               sampler=bs_val,
                               num_workers=c['num_workers'], drop_last=True)

    gens = {'training_generator': training_generator, 'val_generator': val_generator}
    return gens


# Based on spacenet8 baseline -> https://github.com/SpaceNetChallenge/SpaceNet8
# with addition of labeling flood/non-flood based on segment features as outlined in report
# applies deterministic augmenation to multiple images(pre ,post) and labels(buildings , road ,flood)
class SN8Dataset(Dataset):
    def __init__(self, config,
                 train=True,
                 data_to_load: List[str] = ["preimg", "postimg", "building", "road", "roadspeed", "flood"],
                 img_size: Tuple[int, int] = (1300, 1300),
                 aug_val_type='center',
                 tr=None):
        """ pytorch dataset for spacenet-8 data. loads images from a csv that contains filepaths to the images

        Parameters:
        ------------
        csv_filename (str): absolute filepath to the csv to load images from. the csv should have columns: preimg, postimg, building, road, roadspeed, flood.
            preimg column contains filepaths to the pre-event image tiles (.tif)
            postimg column contains filepaths to the post-event image tiles (.tif)
            building column contains the filepaths to the binary building labels (.tif)
            road column contains the filepaths to the binary road labels (.tif)
            roadspeed column contains the filepaths to the road speed labels (.tif)
            flood column contains the filepaths to the flood labels (.tif)
        data_to_load (list): a list that defines which of the images and labels to load from the .csv.
        img_size (tuple): the size of the input pre-event image in number of pixels before any augmentation occurs.

        """
        if tr is None:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            self.tr = T.Compose([T.ToTensor(), T.Normalize(mean=self.mean, std=self.std)])
        else:
            self.tr = tr
        print('Using normalization ', self.tr)
        if 'preimg' in data_to_load and 'postimg' in data_to_load:
            additional_targets = {'image0': 'image'}
            for i in range(len(data_to_load)-3):
                additional_targets[f'mask{i}'] = 'mask'

            self.aug_train = Compose(tx, additional_targets=additional_targets)
        else:
            additional_targets = {}
            for i in range(len(data_to_load)-2):
                additional_targets[f'mask{i}'] = 'mask'
            self.aug_train = Compose(tx, additional_targets=additional_targets)
        self.aug_val_type = aug_val_type
        if not 'flood' in data_to_load:
            if aug_val_type == 'center':
                self.aug_val = aug_val_center
            if aug_val_type == 'five':
                self.aug_val = aug_val_five
        else:
            self.aug_val = T.Resize(resize_size)
        self.config = config
        self.root_dir = config['sn8:root_dir']
        self.train = train
        if self.train:
            self.csv_filename = config['sn8:train_csv']
        else:
            self.csv_filename = config['sn8:val_csv']

        self.all_data_types = ["preimg", "postimg", "building", "road", "roadspeed", "flood"]

        self.img_size = img_size
        self.data_to_load = data_to_load

        self.files = []

        dict_template = {}
        for i in self.all_data_types:
            dict_template[i] = None

        with open(self.csv_filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                in_data = copy.copy(dict_template)
                for j in self.data_to_load:
                    in_data[j] = row[j]
                for k, v in in_data.items():
                    if v is not None:
                        if k in ['preimg', 'postimg']:
                            p = pathlib.Path(v)
                            event = os.path.basename(p.parent)
                            if 'postimg' in k:
                                event = 'POST-event-warped'
                            city = os.path.basename(p.parent.parent)
                            impath = os.path.join(self.root_dir, city, event, os.path.basename(v))
                            in_data[k] = impath
                        if k in ['building', 'road', 'roadspeed', 'flood']:
                            city = os.path.basename(pathlib.Path(v).parent.parent.parent)
                            maskpath = os.path.join(self.root_dir, city, 'annotations',
                                                    self.config['sn8:mask_dir_name'],
                                                  os.path.basename(v).replace('.tif', '.png'))
                            if not os.path.isfile(maskpath):
                                assert os.path.isfile(maskpath)
                            in_data[k] = maskpath

                self.files.append(in_data)

        print("loaded", len(self.files), "image filepaths")
        if 'flood' in data_to_load:
            self.labels = np.asarray([1 - int((io.imread(self.files[i]['flood']).sum(-1) == 0).all()) for i in tqdm(range(len(self.files)))])
            print('Num images with flood::', sum(self.labels == 1))
            print('Num images with non-flood::', sum(self.labels == 0 ))

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.files)

    def augment_train(self, img, mask):
        if len(img)==1:
            if len(mask)==1:
                augmented = self.aug_train(image=img[0], mask=mask[0])
            elif len(mask)==2:
                augmented = self.aug_train(image=img[0], mask=mask[0], mask0=mask[1])
            elif len(mask)==3:
                augmented = self.aug_train(image=img[0], mask=mask[0], mask0=mask[1], mask1=mask[2])
        else:
            if len(mask)==1:
                augmented = self.aug_train(image=img[0], image0=img[1], mask=mask[0])
            elif len(mask)==2:
                augmented = self.aug_train(image=img[0], image0=img[1], mask=mask[0], mask0=mask[1])
            elif len(mask)==3:
                augmented = self.aug_train(image=img[0], image0=img[1], mask=mask[0], mask0=mask[1], mask1=mask[2])
        return augmented

    def augment_val(self, img, mask):
        c_croped_img = []
        c_croped_mask = []
        for item in img:
            c_croped_img.append(self.aug_val(Image.fromarray(item)))

        for item in mask:
            c_croped_mask.append(np.asarray(self.aug_val(Image.fromarray(item))))

        return c_croped_img, c_croped_mask

    def __getitem__(self, index):
        data_dict = self.files[index]
        returned_images = []
        returned_masks = []
        for k in self.data_to_load:
            if k in ['preimg', 'postimg']:
                returned_images.append(io.imread(data_dict[k]))
            else:
                returned_masks.append((io.imread(data_dict[k])>0).astype('uint8'))

        if self.train:
            augmented = self.augment_train(returned_images, returned_masks)
            returned_images = []
            returned_masks = []
            for k in augmented.keys():
                if 'image' in k:
                    returned_images.append(self.tr(Image.fromarray(augmented[k])))
                else:
                    mask = augmented[k]
                    if len(mask.shape) == 2:
                        mask = mask[:, :, np.newaxis]
                    returned_masks.append(np.transpose(mask, (2, 0, 1)).astype('float32'))


        else:
            returned_images, returned_masks = self.augment_val(returned_images, returned_masks)
            if self.aug_val_type =='center':
                returned_images = [self.tr(item) for item in returned_images]
            else:
                rtemp = []
                for item in returned_images:
                    r = []
                    for crop in item:
                        r.append(self.tr(crop))
                    rtemp.append(torch.stack(r, dim=0))
                returned_images = rtemp
            if self.aug_val_type == 'center':
                returned_masks = [np.transpose(item if not len(item.shape) == 2 else
                                                item[:, :, np.newaxis], (2, 0, 1)).astype('float32')
                                                for item in returned_masks]
            else:
                rtemp = []

                for item in returned_masks:
                    r = []
                    for crop in item:
                        r.append(T.PILToTensor()(crop))
                    rtemp.append(torch.stack(r, dim=0))
                returned_masks = rtemp

        if 'flood' in self.data_to_load:
            #returned_masks.append(torch.Tensor([self.labels[index]]))
            lx = np.sum(returned_masks[-1]>0)
            returned_masks.append(torch.Tensor([int(lx > 0)]))
        returned_images.extend(returned_masks)
        return returned_images

    def get_image_filename(self, index: int) -> str:
        """ return pre-event image absolute filepath at index """
        data_dict = self.files[index]
        return data_dict["preimg"]


# the flooded/non-flooded images in Spacenet8 is imbalanced so we will use
# use imbalanced sampler here.
def get_generators_sn8(config, data_to_load, do_imbalanced=False):
    c = config
    tr = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  \
            if c['model_name'] == 'inceptionresnetv2' else None
    train_dataset = SN8Dataset(config, train=True, data_to_load=data_to_load, tr=tr)
    val_dataset = SN8Dataset(config, train=False, data_to_load=data_to_load, tr=tr)
    if 'flood' in data_to_load and do_imbalanced:
        print('Using imbalanced sampler')
        training_generator = DataLoader(train_dataset, num_workers=c['num_workers'],
                                        sampler=ImbalancedDatasetSampler(train_dataset),
                                        batch_size=c['batch_size'], pin_memory=True)
    else:
        training_generator = DataLoader(train_dataset, num_workers=c['num_workers'], shuffle=True,
                                    batch_size=c['batch_size'], pin_memory=True)
    val_generator = DataLoader(val_dataset, batch_size=c['batch_size'], pin_memory=True,
                               shuffle=False,
                               num_workers=c['num_workers'])
    gens = {'training_generator': training_generator, 'val_generator': val_generator}
    return gens


if __name__ == '__main__':
    # do some debugs for paths and image loading
    config = json.load(open('configs/config_aws.json'))
    gens = get_generators(config, use_offnadir=True)
    for i, (data) in enumerate(gens['training_generator']):
        print (len(data))
        print (data[-1])
        if i==3:
            break
