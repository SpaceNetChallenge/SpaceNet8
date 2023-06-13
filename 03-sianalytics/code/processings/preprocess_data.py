import csv
import copy
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

from osgeo import gdal
from multiprocessing import Pool

SAVE_DIR='/data/SpaceNet8'
DATA_TO_LOAD = ["preimg", "postimg", "building", "road", "flood"]


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--prefix', default='mmstyle/train',
        help='directory to save preprocessed images')
    parser.add_argument('--csv', default='/data/SpaceNet8/train.csv',
        help='path containing train csvs')
    parser.add_argument('--twoclasseslabel', action='store_true')

    args = parser.parse_args()

    return args


def get_warped_ds(img_size, post_image_filename: str) -> gdal.Dataset:
    """ gdal warps (resamples) the post-event image to the same spatial resolution as the pre-event image and masks

    SN8 labels are created from referencing pre-event image. Spatial resolution of the post-event image does not match the spatial resolution of the pre-event imagery and therefore the labels.
    In order to align the post-event image with the pre-event image and mask labels, we must resample the post-event image to the resolution of the pre-event image. Also need to make sure
    the post-event image covers the exact same spatial extent as the pre-event image. this is taken care of in the the tiling"""
    ds = gdal.Warp("", post_image_filename,
                    format='MEM', width=img_size[1], height=img_size[0],
                    resampleAlg=gdal.GRIORA_Bilinear,
                    outputType=gdal.GDT_Byte)
    return ds


def create_mmstyle(filedict):
    idx = filedict['idx']
    all_data_types = filedict['all_data_types']
    rootdir = filedict['rootdir']
    twoclasseslabel = filedict['twoclasseslabel']
    savepngname = str(idx).zfill(3) + '.png'
    data_dict = filedict

    returned_data = []
    for key_type in all_data_types:
        filepath = data_dict[key_type]
        if filepath is not None:
            # need to resample postimg to same spatial resolution/extent as preimg and labels.
            if key_type == "postimg":
                ds = get_warped_ds((1300,1300), data_dict["postimg"])
            else:
                ds = gdal.Open(filepath)

            image = ds.ReadAsArray()

            ds = None
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = image.transpose(1,2,0)
                pil_image = Image.fromarray(image)
                pil_image.save(os.path.join(rootdir, key_type, savepngname))

            if len(image.shape)==2: # add a channel axis if read image is only shape (H,W).
                labels= Image.fromarray(image).convert('P')
                labels.putpalette(np.array([[0,0,0], [255, 255, 255]], dtype=np.uint8))
                labels.save(os.path.join(rootdir, key_type, savepngname))

            elif key_type == 'flood' and image is not None:
                mask_building_noflood = image[0] == 1
                mask_building_flood = image[1] == 1

                mask_road_noflood = image[2] == 1
                mask_road_flood = image[3] == 1

                label = np.zeros((image.shape[1], image.shape[2]))
                if twoclasseslabel:
                    label[mask_building_flood] = 1
                    label[mask_building_noflood] = 1
                    label[mask_road_flood] = 2
                    label[mask_road_noflood] = 2

                    label = label.astype('uint8')
                    labels= Image.fromarray(label).convert('P')
                    labels.putpalette(np.array(
                        [[0,0,0],
                        [255, 0, 0],
                        [0, 255, 0],
                        ], dtype=np.uint8))
                    labels.save(os.path.join(rootdir, key_type, savepngname))

                else:
                    label[mask_road_flood] = 4
                    label[mask_road_noflood] = 3

                    label[mask_building_flood] = 2
                    label[mask_building_noflood] = 1

                    label = label.astype('uint8')
                    labels= Image.fromarray(label).convert('P')
                    labels.putpalette(np.array(
                        [[0,0,0],
                        [255, 0, 0],
                        [0, 0, 255],
                        [0, 255, 0],
                        [0, 255, 255]
                        ], dtype=np.uint8))
                    labels.save(os.path.join(rootdir, key_type, savepngname))

            else:
                returned_data.append(f"{key_type}_{image.shape}")
        else:
            returned_data.append(f"{key_type}_0")


def map_function(filedict):
    create_mmstyle(filedict)


def main(prefix, csv_filename, data_to_load, twoclasseslabel=False):
    rootdir = os.path.join(SAVE_DIR, prefix)
    os.makedirs(rootdir, exist_ok=True)

    files = []
    all_data_types = ["preimg", "postimg", "building", "road", "roadspeed", "flood"]
    dict_template = {}
    for i in all_data_types:
        dict_template[i] = None

    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            in_data = copy.copy(dict_template)
            for j in data_to_load:
                in_data[j]=row[j]
            files.append(in_data)

    print("Lengths of files: {}".format(len(files)))

    # Make Folders
    for data_key in data_to_load:
        os.makedirs(os.path.join(rootdir, data_key), exist_ok=True)

    input_args = []
    for idx, f in enumerate(files):
        f['idx'] = idx
        f['all_data_types'] = all_data_types
        f['rootdir'] = rootdir
        f['twoclasseslabel'] = twoclasseslabel
        input_args.append(f)

    pool = Pool(20)
    for _ in tqdm(pool.imap_unordered(map_function, input_args), total=len(files)):
        pass
    pool.close()
    pool.join()

    # rename files
    os.rename(os.path.join(rootdir, 'preimg'), os.path.join(rootdir, 'pre'))
    os.rename(os.path.join(rootdir, 'postimg'), os.path.join(rootdir, 'post'))
    os.rename(os.path.join(rootdir, 'flood'), os.path.join(rootdir, 'ann'))

if __name__ == '__main__':
    args = parse_args()
    main(args.prefix, args.csv, DATA_TO_LOAD, args.twoclasseslabel)