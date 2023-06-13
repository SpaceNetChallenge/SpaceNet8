#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:56:31 2017

@author: avanetten
"""

import argparse
import os
import sys
import time

from tqdm import tqdm

# add apls path and import apls_tools
path_apls_src = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_apls_src)
import apls_tools_road as apls_tools
import glob
import cv2


###############################################################################
def create_masks(path_data, buffer_meters=2, is_SN3=True,
                 burnValue=150, make_plots=True, overwrite_ims=False,
                 output_mask_path='',
                 header=['name', 'im_file', 'im_vis_file', 'mask_file',
                         'mask_vis_file']):
    t0 = time.time()
    # set paths
    path_labels = os.path.join(path_data, 'geojson_roads_speed/')
    # output directories
    path_masks = output_mask_path
    # image directories
    if is_SN3:
        #old directory RGB-PanSharpen-u8
        path_images_vis = os.path.join(path_data, 'PS-RGB')
    else:
        path_images_vis = os.path.join(path_data, 'PS-RGB') \
            if os.path.isdir(os.path.join(path_data, 'PS-RGB')) else os.path.join(path_data, 'PS-RGB-u8')
    outfile_list = []
    im_files = os.listdir(path_images_vis)
    nfiles = len(im_files)
    unavailabel = []
    for i, im_name in enumerate(tqdm(im_files)):
        if not im_name.endswith('.tif'):
            continue

        # define files

        name_root = os.path.basename(im_name)

        im_file_vis = os.path.join(path_images_vis, im_name)
       
        lab_name = name_root.replace('.tif', '.geojson').split('_')
        label_file = os.path.join(path_labels,
                                  ''.join(['_'.join(lab_name[0:-1]), '_geojson_roads_speed_', lab_name[-1]]))
        label_file_tot = label_file.replace('PS-RGB_', '')
        if os.path.isfile(label_file_tot):
            mask_file = os.path.join(path_masks, name_root.replace('.tif', '.png'))
            if not os.path.exists(mask_file) or overwrite_ims:
                mask, gdf_buffer = apls_tools.get_road_buffer(label_file_tot,
                                                              im_file_vis,
                                                              mask_file,
                                                              buffer_meters=buffer_meters,
                                                              burnValue=burnValue,
                                                              bufferRoundness=6,
                                                              plot_file='',
                                                              figsize=(6, 6),
                                                              fontsize=8,
                                                              dpi=500,
                                                              show_plot=False,
                                                              verbose=False)

                cv2.imwrite(mask_file, mask)

        else:
            unavailabel.append(im_name)
    print(len(unavailabel), '  Unavilable out of ', nfiles)
    t4 = time.time()
    print("Time to run create_masks():", t4 - t0, "seconds")


###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str, required=True,
                        help='Folder containing imagery and geojson labels')
    parser.add_argument('--output_mask_path', required=True, type=str,
                        help='csv of dataframe containing image and mask locations')
    parser.add_argument('--buffer_meters', default=2, type=float,
                        help='Buffer distance (meters) around graph')
    parser.add_argument('--burnValue', default=255, type=int,
                        help='Value of road pixels (for plotting)')
    parser.add_argument('--overwrite_ims', default=1, type=int,
                        help='Switch to overwrite 8bit images and masks')
    parser.add_argument('--is_SN5', action='store_true', help='SN3 or SN5')

    args = parser.parse_args()
    output_mask_path = args.output_mask_path
    if not os.path.isdir(output_mask_path):
        os.makedirs(output_mask_path, exist_ok=True)
    is_SN5 = args.is_SN5
    print('is_SN5->', is_SN5)
    dir_path = args.path_data
    print('doing...', dir_path)
    create_masks(dir_path,
                 buffer_meters=args.buffer_meters,
                 is_SN3=not is_SN5,
                 burnValue=args.burnValue,
                 output_mask_path=output_mask_path,
                 make_plots=0,
                 overwrite_ims=1)


###############################################################################
if __name__ == "__main__":
    main()
