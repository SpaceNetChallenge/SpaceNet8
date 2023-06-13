from osgeo import gdal
from PIL import Image
from multiprocessing import Pool

from tqdm import tqdm
import os
import numpy as np

PATCH_SIZE = 512
OVERLAY = 256

def make_segmentation(img_path, gt_path, train='train'):

    basename = os.path.splitext(os.path.basename(img_path))[0]

    img_src_ds = gdal.Open(img_path).ReadAsArray()
    gt_src_ds = gdal.Open(gt_path).ReadAsArray()

    channel, height, width = img_src_ds.shape

    idx = 0
    for col in range(0, height - OVERLAY, PATCH_SIZE - OVERLAY):
        for row in range(0, width - OVERLAY, PATCH_SIZE - OVERLAY):
            img_patch = img_src_ds[:, col:col+PATCH_SIZE, row:row+PATCH_SIZE]
            img_patch = np.transpose(img_patch, [1, 2, 0])
            gt_patch = gt_src_ds[col:col+PATCH_SIZE, row:row+PATCH_SIZE]
            gt_patch = np.array(gt_patch / 255.0, dtype=np.uint8)

            img_patch = Image.fromarray(img_patch, 'RGB')
            gt_patch = Image.fromarray(gt_patch).convert('P')
            gt_patch.putpalette(np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8))

            idx += 1

            img_path = os.path.join(
                    f'/nas/k8s/shared/dataset-artifacts/inria',
                    f'SEG/building/img/{train}/{basename}_{idx}')
            gt_path = os.path.join(
                    f'/nas/k8s/shared/dataset-artifacts/inria',
                    f'SEG/building/gt/{train}/{basename}_{idx}')

            os.makedirs(img_path, exist_ok=True)
            os.makedirs(gt_path, exist_ok=True)

            gt_patch.save(os.path.join(gt_path, '01.png'))
            img_patch.save(os.path.join(img_path, '01.png'))

def map_function(data):
    make_segmentation(**data)


def main():

    train_root = '/nas/Dataset/inria/AerialImageDataset/train/'
    train_image_root = os.path.join(train_root, 'images')
    train_input_args = list()

    scene_list = os.listdir(os.path.join(train_image_root))
    train_list = scene_list[:int(len(scene_list) * 0.8)]
    valid_list = scene_list[int(len(scene_list) * 0.8):]

    train_input_args = list()
    for listdir in train_list:

        assert os.path.exists(os.path.join(train_image_root, listdir))
        assert os.path.exists(os.path.join(train_root, 'gt', listdir))

        train_input_args.append({
            'img_path': os.path.join(train_image_root, listdir),
            'gt_path': os.path.join(train_root, 'gt', listdir),
            'train': 'train'})

    valid_input_args = list()
    for listdir in valid_list:

        assert os.path.exists(os.path.join(train_image_root, listdir))
        assert os.path.exists(os.path.join(train_root, 'gt', listdir))

        valid_input_args.append({
            'img_path': os.path.join(train_image_root, listdir),
            'gt_path': os.path.join(train_root, 'gt', listdir),
            'train': 'test'})

    pool = Pool(10)
    for _ in tqdm(pool.imap_unordered(map_function, train_input_args), total=len(train_input_args)):
        pass
    pool.close()
    pool.join()

    pool = Pool(10)
    for _ in tqdm(pool.imap_unordered(map_function, valid_input_args), total=len(valid_input_args)):
        pass
    pool.close()
    pool.join()




    '''
    for listdir in os.listdir(os.path.join(train_image_root)):

        assert os.path.exists(os.path.join(train_image_root, listdir))
        assert os.path.exists(os.path.join(train_root, 'gt', listdir))

        train_input_args.append({
            'img_path': os.path.join(train_image_root, listdir),
            'gt_path': os.path.join(train_root, 'gt', listdir),
            'train': 'train'})
    '''

    '''
    make_segmentation(
            '/nas/Dataset/inria/AerialImageDataset/train/images/austin1.tif',
            '/nas/Dataset/inria/AerialImageDataset/train/gt/austin1.tif',
            train='train')
            '''

if __name__ == '__main__':
    main()
