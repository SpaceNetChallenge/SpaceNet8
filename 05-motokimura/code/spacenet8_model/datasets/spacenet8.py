import os

import numpy as np
import pandas as pd
import torch
from skimage import io

# isort: off
from spacenet8_model.utils.misc import get_flatten_classes
# isort: on


class SpaceNet8Dataset(torch.utils.data.Dataset):
    def __init__(self, config, is_train, transform=None):
        self.pre_paths, self.post1_paths, self.post2_paths, self.mask_paths, self.post1_mses, self.post2_mses \
            = self.get_file_paths(config, is_train)
        self.masks_to_load = self.get_mask_types_to_load(config)

        self.config = config
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.pre_paths)

    def __getitem__(self, idx):
        pre = io.imread(self.pre_paths[idx])  # HWC
        h, w = pre.shape[:2]
        
        mask = self.load_mask(idx)  # HWC
        h_mask, w_mask = mask.shape[:2]
        assert h_mask == h
        assert w_mask == w

        sample = dict(image=pre, mask=mask)

        post_images = self.load_post_images(idx)
        for k in post_images:
            h_post, w_post = post_images[k].shape[:2]
            assert h == h_post
            assert w == w_post
        sample.update(post_images)

        if self.transform is not None:
            sample = self.transform(**sample)

        # convert format HWC -> CHW
        sample['image'] = np.moveaxis(sample['image'], -1, 0)
        sample['mask'] = np.moveaxis(sample['mask'], -1, 0)
        for k in post_images:
            sample[k] = np.moveaxis(sample[k], -1, 0)

        return sample

    def get_file_paths(self, config, is_train):
        split = 'train' if is_train else 'val'
        path = os.path.join(config.Data.artifact_dir,
                            config.Data.fold_ver,
                            f'{split}_{config.fold_id}.csv')
        df = pd.read_csv(path)

        # prepare image and mask paths
        pre_paths = []
        post1_paths = []
        post2_paths = []
        building_3channel_paths = []
        building_flood_paths = []
        road_paths = []
        post1_mses = []
        post2_mses = []
        for i, row in df.iterrows():
            aoi = row['aoi']

            pre_fn = row['pre-event image']
            if pre_fn in config.Data.pre_image_blacklist:
                print(f'removed {pre_fn} from {"train set" if is_train else "val set"}')
                continue

            pre = os.path.join(config.Data.train_dir, aoi, 'PRE-event', pre_fn)
            assert os.path.exists(pre), pre
            pre_paths.append(pre)

            post1 = os.path.join(config.Data.artifact_dir, 'warped_posts_train', aoi, row['post-event image 1'])
            assert os.path.exists(post1), post1
            post1_paths.append(post1)

            post2 = row['post-event image 2']
            if isinstance(post2, str):
                post2 = os.path.join(config.Data.artifact_dir, 'warped_posts_train', aoi, post2)
                assert os.path.exists(post2), post2
            else:
                post2 = None
            post2_paths.append(post2)

            mask_filename, _ = os.path.splitext(pre_fn)
            mask_filename = f'{mask_filename}.png'

            building_3channel = os.path.join(config.Data.artifact_dir, 'masks_building_3channel', aoi, mask_filename)
            assert os.path.exists(building_3channel), building_3channel
            building_3channel_paths.append(building_3channel)

            building_flood = os.path.join(config.Data.artifact_dir, 'masks_building_flood', aoi, mask_filename)
            assert os.path.exists(building_flood), building_flood
            building_flood_paths.append(building_flood)

            road = os.path.join(config.Data.artifact_dir, 'masks_road', aoi, mask_filename)
            assert os.path.exists(road), road
            road_paths.append(road)

            post1_mses.append(row['mse1'])
            post2_mses.append(row['mse2'])

        if is_train and config.Data.use_mosaic_for_train:
            path = os.path.join(config.Data.artifact_dir, f'mosaics/train_{config.fold_id}/mosaics.csv')
            df = pd.read_csv(path)
            for i, row in df.iterrows():
                # assuming images in the blacklist are already removed
                pre_paths.append(row['pre'])
                post1_paths.append(row['post1'])

                post2 = row['post2']
                if not isinstance(post2, str):
                    post2 = None
                post2_paths.append(post2)

                building_3channel_paths.append(row['building_3channel'])
                building_flood_paths.append(row['building_flood'])
                road_paths.append(row['road'])

                post1_mses.append(row['mse1'])
                post2_mses.append(row['mse2'])

        mask_paths = {
            'building_3channel': building_3channel_paths,
            'building_flood': building_flood_paths,
            'road': road_paths,
        }
        return pre_paths, post1_paths, post2_paths, mask_paths, post1_mses, post2_mses

    def get_mask_types_to_load(self, config):
        mask_types = []
        cs = get_flatten_classes(config)

        if ('building' in cs) or ('building_border' in cs) or ('building_contact' in cs):
            mask_types.append('building_3channel')

        if ('flood' in cs) or ('flood_building' in cs) or ('not_flood_building' in cs):
            mask_types.append('building_flood')

        if ('road' in cs) or ('road_junction' in cs) or ('flood' in cs) or ('flood_road' in cs) or ('not_flood_road' in cs):
            mask_types.append('road')

        assert len(mask_types) > 0
        return mask_types

    def load_mask(self, idx):
        # load mask images
        masks = {}
        for mask_type in self.masks_to_load:
            masks[mask_type] = io.imread(self.mask_paths[mask_type][idx])

        # prepare target mask
        target_mask = []
        for g in self.config.Class.groups:
            target_mask.extend(
                self.prepare_multichannel_mask(masks, class_names=list(self.config.Class.classes[g]))
            )
        target_mask = np.stack(target_mask)  # CHW
        return target_mask.transpose(1, 2, 0)  # HWC

    def prepare_multichannel_mask(self, masks, class_names):
        assert len(class_names) == len(set(class_names)), class_names

        if '_background' in class_names:
            assert class_names.index('_background') == len(class_names) - 1, class_names
            class_names.remove('_background')
            assert len(class_names) > 0, class_names
            use_background = True
        else:
            use_background = False

        target_mask = []
        for c in class_names:
            target_mask.append(self.prepare_binary_mask(masks, class_name=c))

        if use_background:
            background_mask = (np.stack(target_mask).sum(axis=0) == 0).astype(np.float32)
            target_mask.append(background_mask)

        return target_mask

    def prepare_binary_mask(self, masks, class_name):
        c = class_name
        # building
        if c == 'building':
            mask = masks['building_3channel'][:, :, 0]
        elif c == 'building_border':
            mask = masks['building_3channel'][:, :, 1]
        elif c == 'building_contact':
            mask = masks['building_3channel'][:, :, 2]
        # road
        elif c == 'road':
            mask = masks['road'][:, :, 0] + masks['road'][:, :, 1]
        elif c == 'road_junction':
            mask = masks['road'][:, :, 2]
        # flood
        elif c == 'flood':
            mask = masks['building_flood'][:, :, 0] + masks['road'][:, :, 0]
        elif c == 'flood_building':
            mask = masks['building_flood'][:, :, 0]
        elif c == 'flood_road':
            mask = masks['road'][:, :, 0]
        elif c == 'not_flood_building':
            mask = masks['building_flood'][:, :, 1]
        elif c == 'not_flood_road':
            mask = masks['road'][:, :, 1]
        else:
            raise ValueError(class_name)

        return (mask > 0).astype(np.float32)

    def load_post_images(self, idx):
        post_select_method = self.config.Model.train_post_select_method if self.is_train else self.config.Model.test_post_select_method
        if self.is_train:
            assert post_select_method in ['always_post1', 'less_mse', 'less_mse_prob'], post_select_method
        else:
            assert post_select_method in ['always_post1', 'less_mse'], post_select_method
        return load_post_images(
            self.post1_paths[idx],
            self.post2_paths[idx],
            self.post1_mses[idx],
            self.post2_mses[idx],
            self.config.Model.n_input_post_images,
            post_select_method=post_select_method
        )


class SpaceNet8TestDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None, test_to_val=False):
        self.pre_paths, self.post1_paths, self.post2_paths, self.post1_mses, self.post2_mses \
            = self.get_file_paths(config, test_to_val)

        self.config = config
        self.transform = transform

    def __len__(self):
        return len(self.pre_paths)

    def __getitem__(self, idx):
        pre = io.imread(self.pre_paths[idx])  # HWC
        h, w = pre.shape[:2]
        sample = dict(image=pre)

        post_images = self.load_post_images(idx)
        for k in post_images:
            h_post, w_post = post_images[k].shape[:2]
            assert h == h_post
            assert w == w_post
        sample.update(post_images)

        if self.transform is not None:
            sample = self.transform(**sample)

        # convert format HWC -> CHW
        sample['image'] = np.moveaxis(sample['image'], -1, 0)
        for k in post_images:
            sample[k] = np.moveaxis(sample[k], -1, 0)

        # add image metadata
        meta = {
            'pre_path': self.pre_paths[idx],
            'original_height': h,
            'original_width': w
        }
        sample.update(meta)

        return sample

    def get_file_paths(self, config, test_to_val=False):
        if test_to_val:
            csv_path = os.path.join(config.Data.artifact_dir, f'{config.Data.fold_ver}/val_{config.fold_id}.csv')
            data_root = config.Data.train_dir
            post_image_dir = os.path.join(config.Data.artifact_dir, 'warped_posts_train')
        else:
            csv_path = os.path.join(config.Data.artifact_dir, 'test.csv')
            data_root = config.Data.test_dir
            post_image_dir = os.path.join(config.Data.artifact_dir, 'warped_posts_test')
        df = pd.read_csv(csv_path)

        # prepare image paths
        pre_paths = []
        post1_paths = []
        post2_paths = []
        post1_mses = []
        post2_mses = []
        for i, row in df.iterrows():
            aoi = row['aoi']

            pre = os.path.join(data_root, aoi, 'PRE-event', row['pre-event image'])
            assert os.path.exists(pre), pre
            pre_paths.append(pre)

            post1 = os.path.join(post_image_dir, aoi, row['post-event image 1'])
            assert os.path.exists(post1), post1
            post1_paths.append(post1)

            post2 = row['post-event image 2']
            if isinstance(post2, str):
                post2 = os.path.join(post_image_dir, aoi, post2)
                assert os.path.exists(post2), post2
            else:
                post2 = None
            post2_paths.append(post2)

            post1_mses.append(row['mse1'])
            post2_mses.append(row['mse2'])

        return pre_paths, post1_paths, post2_paths, post1_mses, post2_mses

    def load_post_images(self, idx):
        post_select_method = self.config.Model.test_post_select_method
        assert post_select_method in ['always_post1', 'less_mse'], post_select_method
        return load_post_images(
            self.post1_paths[idx],
            self.post2_paths[idx],
            self.post1_mses[idx],
            self.post2_mses[idx],
            self.config.Model.n_input_post_images,
            post_select_method=post_select_method
        )


def load_post_images(post1_path, post2_path, post1_mse, post2_mse, n_input_post_images, post_select_method):
    if n_input_post_images == 0:
        return {}

    elif n_input_post_images == 1:
        if post2_path is None:
            # if post-2 image does not exist, return post-1 image
            return {'image_post_a': io.imread(post1_path)}

        if post_select_method == 'always_post1':
            return {'image_post_a': io.imread(post1_path)}
        elif post_select_method == 'less_mse':
            if post1_mse <= post2_mse:
                return {'image_post_a': io.imread(post1_path)}
            else:
                return {'image_post_a': io.imread(post2_path)}
        elif post_select_method == 'less_mse_prob':
            max_mse_diff = 900.0
            if post1_mse <= post2_mse:
                post2_prob = (1.0 - min((post2_mse - post1_mse) / max_mse_diff, 1.0)) * 0.5
                post1_prob = 1.0 - post2_prob
                assert post1_prob >= post2_prob, (post1_prob, post2_prob)
            else:
                post1_prob = (1.0 - min((post1_mse - post2_mse) / max_mse_diff, 1.0)) * 0.5
                post2_prob = 1.0 - post1_prob
                assert post2_prob > post1_prob, (post1_prob, post2_prob)
            post_path = np.random.choice([post1_path, post2_path], size=1, p=[post1_prob, post2_prob])[0]
            return {'image_post_a': io.imread(post_path)}
        else:
            raise ValueError(post_select_method)

    elif n_input_post_images == 2:
        post1 = io.imread(post1_path)
        if post2_path is None:
            # if post-2 image does not exist, just copy post-1 as post-2 image
            post2 = post1.copy()
        else:
            post2 = io.imread(post2_path)
        return {
            'image_post_a': post1,
            'image_post_b': post2
        }

    else:
        raise ValueError(n_input_post_images)
