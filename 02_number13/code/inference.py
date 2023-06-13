import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
from PIL import Image
import cv2
from skimage.exposure import rescale_intensity
import json
from functools import partial
from skimage.measure import label


def predict_patches_foundation(image_array, model, model_input_shape=(512, 512, 3), step_size=384, num_cls=1, rm_cutoff=0):
    '''Run inference on a large image by tiling.'''
    x_steps = int(np.ceil((image_array.shape[1]-model_input_shape[1])/step_size)) + 1
    y_steps = int(np.ceil((image_array.shape[0]-model_input_shape[0])/step_size)) + 1
    training_arr = np.empty((x_steps*y_steps, model_input_shape[0], model_input_shape[1], image_array.shape[2]))
    raw_inference_arr = np.empty((x_steps*y_steps, image_array.shape[0], image_array.shape[1], num_cls))
    raw_inference_arr[:] = np.nan
    counter = 0
    subarr_indices = []
    y_ind = 0
    while y_ind < image_array.shape[0] + step_size - model_input_shape[0] - 1:
        if y_ind + model_input_shape[0] > image_array.shape[0]:
            y_ind = image_array.shape[0] - model_input_shape[0]
        x_ind = 0
        while x_ind < image_array.shape[1] + step_size - model_input_shape[1] - 1:
            if x_ind + model_input_shape[0] > image_array.shape[1]:
                x_ind = image_array.shape[1] - model_input_shape[1]
            training_arr[counter, :, :, :] = image_array[y_ind:y_ind+model_input_shape[0],
                                             x_ind:x_ind+model_input_shape[1], :]
            subarr_indices.append((y_ind, x_ind))
            x_ind += step_size
            counter += 1
        y_ind += step_size
    training_arr = torch.tensor(np.transpose(training_arr,(0,3,1,2))).to(device).float()

    def load_to_raw_(predictions):
        for idx in range(len(subarr_indices)):
            raw_inference_arr[
            idx,
            subarr_indices[idx][0]:subarr_indices[idx][0] + model_input_shape[0],
            subarr_indices[idx][1]:subarr_indices[idx][1] + model_input_shape[1]
            ] = predictions[idx, :, :, :]
        return np.nanmean(raw_inference_arr, axis=0)
    preds = model.predict(training_arr)
    # if rm_cutoff:
    #    labels = label(final_preds)
    #    labs, cts = np.unique(labels, return_counts=True)
    #    labels[np.isin(labels, labs[cts < rm_cutoff])] = 0
    #    final_preds = labels > 0

    if len(preds)==2:
        predictions_buildings, predictions_roads = preds

        predictions_buildings, predictions_roads = torch.sigmoid(predictions_buildings), torch.sigmoid(predictions_roads)
        predictions_buildings, predictions_roads = predictions_buildings.detach().cpu().numpy(), predictions_roads.detach().cpu().numpy()

        predictions_buildings = np.transpose(predictions_buildings, (0,2,3,1))
        predictions_roads = np.transpose(predictions_roads, (0,2,3,1))
        predictions_buildings = load_to_raw_(predictions_buildings)
        predictions_roads = load_to_raw_(predictions_roads)
        return predictions_buildings[:, :, 0], predictions_roads[:, :, 0]
    else:
        preds = load_to_raw_(np.transpose(torch.sigmoid(preds).detach().cpu().numpy(), (0, 2,3, 1)))[:, :, 0]
        return preds, preds


def predict_patches_flood(image_array_pre, image_array_post, model, model_input_shape=(512, 512, 3), step_size=384, num_cls=1, rm_cutoff=0):
    '''Run inference on a large image by tiling.'''
    image_array = image_array_pre
    x_steps = int(np.ceil((image_array.shape[1]-model_input_shape[1])/step_size)) + 1
    y_steps = int(np.ceil((image_array.shape[0]-model_input_shape[0])/step_size)) + 1
    training_arr_pre = np.empty((x_steps*y_steps, model_input_shape[0], model_input_shape[1], image_array.shape[2]))
    training_arr_post = np.empty((x_steps*y_steps, model_input_shape[0], model_input_shape[1], image_array.shape[2]))

    raw_inference_arr = np.empty((x_steps*y_steps, image_array.shape[0], image_array.shape[1], num_cls))
    raw_inference_arr[:] = np.nan
    counter = 0
    subarr_indices = []
    y_ind = 0
    while y_ind < image_array.shape[0] + step_size - model_input_shape[0] - 1:
        if y_ind + model_input_shape[0] > image_array.shape[0]:
            y_ind = image_array.shape[0] - model_input_shape[0]
        x_ind = 0
        while x_ind < image_array.shape[1] + step_size - model_input_shape[1] - 1:
            if x_ind + model_input_shape[0] > image_array.shape[1]:
                x_ind = image_array.shape[1] - model_input_shape[1]
            training_arr_pre[counter, :, :, :] = image_array_pre[y_ind:y_ind+model_input_shape[0],
                                                 x_ind:x_ind+model_input_shape[1], :]
            training_arr_post[counter, :, :, :] = image_array_post[y_ind:y_ind+model_input_shape[0],
                                                  x_ind:x_ind+model_input_shape[1], :]
            subarr_indices.append((y_ind, x_ind))
            x_ind += step_size
            counter += 1
        y_ind += step_size

    training_arr_pre = torch.tensor(np.transpose(training_arr_pre,(0,3,1,2))).to(device).float()
    training_arr_post = torch.tensor(np.transpose(training_arr_post,(0,3,1,2))).to(device).float()

    predictions_build, predictions_road,  cls = model.predict(training_arr_pre, training_arr_post)
    predictions_build, predictions_road,  cls = torch.sigmoid(predictions_build), torch.sigmoid(predictions_road), \
                                                torch.sigmoid(cls).squeeze()

    #if cls is not None:
    #    for i, (p, c) in enumerate(zip(predictions_build, cls)):
    #        if c < 0.7:
    #            predictions_build[i] = torch.zeros(p.shape, device=p.device)
    #            #predictions_road[i] = torch.zeros(p.shape, device=p.device)

    predictions_build = np.transpose(predictions_build.detach().cpu().numpy(), (0,2,3,1))
    predictions_road = np.transpose(predictions_road.detach().cpu().numpy(), (0,2,3,1))

    def load_to_raw_(predictions):
        for idx in range(len(subarr_indices)):
            raw_inference_arr[
            idx,
            subarr_indices[idx][0]:subarr_indices[idx][0] + model_input_shape[0],
            subarr_indices[idx][1]:subarr_indices[idx][1] + model_input_shape[1]
            ] = predictions[idx, :, :, :]
        return np.nanmean(raw_inference_arr, axis=0)
    predictions_build = load_to_raw_(predictions_build)
    predictions_road = load_to_raw_(predictions_road)
    #if rm_cutoff:
    #    labels = label(final_preds)
    #    labs, cts = np.unique(labels, return_counts=True)
    #    labels[np.isin(labels, labs[cts < rm_cutoff])] = 0
    #    final_preds = labels > 0
    predictions_flood = np.stack([predictions_build, predictions_road], axis=-1).squeeze()
    return predictions_flood


# based on https://www.kaggle.com/competitions/severstal-steel-defect-detection/discussion/107716
def sharpen(p, t=0.5):
    if t!=0:
        return p**t
    else:
        return p


def test_foundation(model, test_dataset,  pred_dir,  task='both', thres=None, tta=False):
    model.eval()
    print('images in test = ', test_dataset.__len__())
    winsize = 448
    with torch.set_grad_enabled(False):
        for image_id, images in tqdm(test_dataset):
                num_augment = 0
                filename_building = os.path.join(pred_dir, 'building_'+image_id+'.png')
                filename_road = os.path.join(pred_dir, 'road_'+image_id+'.png')
                merged_preds_buildings, merged_preds_roads = predict_patches_foundation(images.moveaxis(0, -1),
                                                                                        model, (512,512,3), winsize, num_cls=1, rm_cutoff=0)
                num_augment += 1
                if tta:
                    mb, mr = predict_patches_foundation(torch.flip(images, dims=[2]).moveaxis(0, -1),
                                                        model, (512, 512, 3), winsize, num_cls=1, rm_cutoff=0)
                    mb, mr = np.fliplr(mb), np.fliplr(mr)
                    merged_preds_buildings += sharpen(mb)
                    merged_preds_roads += sharpen(mr)
                    num_augment += 1

                    mb, mr = predict_patches_foundation(torch.flip(images, dims=[1]).moveaxis(0, -1),
                                                        model, (512, 512, 3), winsize, num_cls=1, rm_cutoff=0)
                    mb, mr = np.flipud(mb), np.flipud(mr)
                    merged_preds_buildings += sharpen(mb)
                    merged_preds_roads += sharpen(mr)
                    num_augment += 1

                merged_preds_buildings = merged_preds_buildings/num_augment
                merged_preds_roads = merged_preds_roads/num_augment

                merged_preds_roads = None if task == 'building' else merged_preds_roads
                merged_preds_buildings = None if task=='road' else merged_preds_buildings

                if merged_preds_roads is not None:
                    merged_preds_roads = (merged_preds_roads * 255).astype('uint8') if thres is None else \
                        ((merged_preds_roads>thres) * 255).astype('uint8')
                if merged_preds_buildings is not None:
                    merged_preds_buildings = (merged_preds_buildings * 255).astype('uint8') if thres is None else \
                        ((merged_preds_buildings > thres) * 255).astype('uint8')
                if merged_preds_buildings is not None:
                    if merged_preds_buildings.shape[0] == 1344:
                        merged_preds_buildings = merged_preds_buildings[22: -22, 22: -22]
                    io.imsave(filename_building, merged_preds_buildings,  check_contrast=False)
                if merged_preds_roads is not None:
                    if merged_preds_roads.shape[0] == 1344:
                        merged_preds_roads = merged_preds_roads[22:-22, 22:-22]
                    io.imsave(filename_road, merged_preds_roads,  check_contrast=False)


def test_flood(model, test_dataset,  pred_dir,  thres=None, tta=False):
    model.eval()
    print('images in test = ', test_dataset.__len__())
    winsize = 384
    labels = {}
    with torch.set_grad_enabled(False):
        for image_id, pre_images, post_images in tqdm(test_dataset):
            num_augment = 0
            filename_flood = os.path.join(pred_dir, 'flood_'+image_id+'.png')
            merged_preds_flood = predict_patches_flood(pre_images.moveaxis(0, -1), post_images.moveaxis(0, -1),
                                                       model, (512, 512, 3), winsize, num_cls=1, rm_cutoff=0)
            num_augment += 1
            #merged_preds_flood_blur = predict_patches_flood(pre_images_blur.moveaxis(0, -1), post_images.moveaxis(0, -1),
            #                                           model, (512, 512, 3), 384, num_cls=1, rm_cutoff=0)
            #merged_preds_flood = np.mean([merged_preds_flood, merged_preds_flood_blur], axis=0)
            if tta:
                m1 = predict_patches_flood(torch.flip(pre_images, dims=[2]).moveaxis(0, -1),
                                                       torch.flip(post_images, dims=[2]).moveaxis(0, -1),
                                                       model, (512, 512, 3), winsize, num_cls=1, rm_cutoff=0)
                m1 = np.fliplr(m1)
                merged_preds_flood += sharpen(m1)
                num_augment += 1

            merged_preds_flood = merged_preds_flood/num_augment
            if merged_preds_flood.shape[0]==1344:
                merged_preds_flood = merged_preds_flood[22:-22, 22:-22, :]
            if thres is None:
                merged_preds_flood = (merged_preds_flood * 255).astype(np.uint8)
            else:
                merged_preds_flood = (merged_preds_flood > thres)
                merged_preds_flood = (merged_preds_flood*255).astype(np.uint8)
            if merged_preds_flood.shape[-1] == 2:
                merged_preds_flood = cv2.merge((merged_preds_flood, np.zeros((1300, 1300, 1), dtype=np.uint8)))
            #merged_preds_flood = postprocess_flood(merged_preds_flood, image_id, pred_dir)
            io.imsave(filename_flood, merged_preds_flood, check_contrast=False)


class FoundationTestDataset(Dataset):
    def __init__(self, image_paths, tr=None):
        self.image_paths = image_paths
        if tr is None:
            self.tr3 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean= [0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])])
        else:
            self.tr3 = tr
        print('Using normalization..', self.tr3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.tr3(Image.fromarray(io.imread(self.image_paths[idx])))
        image = transforms.Pad(22, padding_mode='reflect')(image)
        return os.path.basename(self.image_paths[idx]).replace('.tif', ''), image


class FloodTestDataset(Dataset):
    def __init__(self, image_paths, resize=None, tr=None):
        self.image_paths = image_paths # tuples
        if tr is None:
            self.tr3 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean= [0.485, 0.456, 0.406],
                                                                                   std=[0.229, 0.224, 0.225])])
        else:
            self.tr3 = tr
        print('Using normalization..', self.tr3)
        self.resize = (resize, resize)

    def __len__(self):
        return len(self.image_paths)

    # Make composite through simple median averaging
    # of post-distaer images if more than one
    def composite(self, img1, img2):
        img_s = np.vstack([img1[np.newaxis], img2[np.newaxis]])
        img_g = np.median(img_s, axis=0)
        #img_g = np.zeros((1300, 1300, 3), dtype=np.float32)
        #for i in range(3):
        #    img_g[:, :, i] = geometric_median(img_s[:, :, :, i].reshape((2, 1300*1300))).reshape((1300, 1300))
        img_m = rescale_intensity(img_g, out_range=np.uint8).astype('uint8')
        return img_m

    def __getitem__(self, idx):
        image_pre = Image.fromarray(io.imread(self.image_paths[idx][0]))
        post_paths = self.image_paths[idx][1]
        if len(post_paths) == 1:
            imp = io.imread(post_paths[0])
        else:
            # make composite if more than one post-images
            imp = self.composite(io.imread(post_paths[0]), io.imread(post_paths[1]))
        image_post = Image.fromarray(imp)
        if self.resize[0] is not None:
            image_pre, image_post = image_pre.resize(self.resize), image_post.resize(self.resize)
        image_pre = self.tr3(image_pre)
        image_post = self.tr3(image_post) # do gdal warping beforehand and store somewhere /wdata/city/POST-event-warped
        image_pre, image_post = transforms.Pad(22, padding_mode='reflect')(image_pre), transforms.Pad(22, padding_mode='reflect')(image_post)
        assert image_pre.shape == image_post.shape
        return os.path.basename(self.image_paths[idx][0]).replace('.tif', ''), image_pre, image_post


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--test_dir_pre', required=False, help='directory to read pre flood images')
    parser.add_argument('--test_dir_post', required=False, default='',  help='directory to read post flood images')
    parser.add_argument('--flood',  action='store_true', help='predict foundation or flood')
    parser.add_argument('--mapping_csv', required=False, type=str, help='the mapping file from pre to post event')
    parser.add_argument('--model_path', required=True, type=str, help='model weights path')
    parser.add_argument('--pred_dir', required=False, help='output pred')
    parser.add_argument('--thres', type=float, default=None, help='thres val')
    parser.add_argument('--task', default='both', required=False, help='which task building road or both')
    parser.add_argument('--val_csv', required=False, type=str, default='', help='val csv')
    parser.add_argument('--out_file', required=False, type=str, default='', help='classificatoin result')
    parser.add_argument('--model_name',  default='resnet50', required=True, help='which model . will be required if not directly loadable')
    parser.add_argument('--gpu', default=0, type=int, help="device id")
    parser.add_argument('--tta',  action='store_true', help='do_tta or not')
    parser.add_argument('--net', default='unet', type=str, help="unet or manet")

    args = parser.parse_args()
    device = f'cuda:{args.gpu}'
    model_path = args.model_path
    thres = args.thres
    pred_dir = args.pred_dir
    os.makedirs(pred_dir, exist_ok=True)
    if args.model_name == 'inceptionresnetv2':
        tr = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    else:
        tr = None
    if not args.flood:

        mw = torch.load(args.model_path, map_location='cpu')
        from models import UnetMhead, MAnetMhead
        if args.net =='unet':
            model = UnetMhead(encoder_name=args.model_name, encoder_weights=None, in_channels=3, classes=1)
            model.load_state_dict(mw['state_dict'])
        else:
            model = MAnetMhead(encoder_name=args.model_name, encoder_weights=None)
            model.load_state_dict(mw['state_dict'])

        model = model.to(device)
        images = glob.glob(args.test_dir_pre+'/*.tif')
        print(f'Num images {len(images)}')
        test_dataset = FoundationTestDataset(images, tr=tr)
        test_foundation(model, test_dataset,  pred_dir,  task=args.task, thres=thres, tta=args.tta)

    if args.flood:
        os.makedirs(pred_dir, exist_ok=True)
        assert os.path.isdir(args.test_dir_pre) and os.path.isdir(args.test_dir_post)
        assert os.path.isfile(args.mapping_csv)
        mapping = pd.read_csv(args.mapping_csv).to_dict(orient='records')
        available_paths = []
        for item in mapping:
            pre = os.path.join(args.test_dir_pre, item['pre-event image'])
            posts = []
            p1 = os.path.join(args.test_dir_post, item['post-event image 1'])
            p2 = os.path.join(args.test_dir_post, str(item['post-event image 2']))
            if os.path.isfile(p1):
                posts.append(p1)
            if os.path.isfile(p2):
                posts.append(p2)
            available_paths.append([pre, posts])

        from models import UnetSiamese_Mhead, MAnetSiamese_Mhead
        if args.net=='unet':
            model = UnetSiamese_Mhead(encoder_name=args.model_name, encoder_weights=None, in_channels=3, num_classes=1)
        else:
            model = MAnetSiamese_Mhead(encoder_name=args.model_name, encoder_weights=None)
        model.load_state_dict(torch.load(args.model_path, map_location='cpu')['state_dict'])
        model = model.to(device)
        test_dataset = FloodTestDataset(available_paths, tr=tr)
        print(f'Num matched pair availbale..{len(available_paths)}')
        test_flood(model, test_dataset,  pred_dir,  thres=thres, tta=args.tta)
