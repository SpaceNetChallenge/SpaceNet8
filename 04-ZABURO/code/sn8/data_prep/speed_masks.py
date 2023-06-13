#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:03:15 2019

@author: avanetten
"""

import math
import os

import numpy as np
import pandas as pd
import skimage.io
from osgeo import gdal

from sn8.data_prep import road_speed

# also see ipynb/_speed_data.prep.ipynb


###############################################################################
def convert_array_to_multichannel(in_arr, n_channels=7, burnValue=255, append_total_band=False, verbose=False):
    """Take input array with multiple values, and make each value a unique
    channel.  Assume a zero value is background, while value of 1 is the
    first channel, 2 the second channel, etc."""

    h, w = in_arr.shape[:2]
    # scikit image wants it in this format by default
    out_arr = np.zeros((n_channels, h, w), dtype=np.uint8)
    # out_arr = np.zeros((h,w,n_channels), dtype=np.uint8)

    for band in range(n_channels):
        val = band + 1
        band_out = np.zeros((h, w), dtype=np.uint8)
        if verbose:
            print("band:", band)
        band_arr_bool = np.where(in_arr == val)
        band_out[band_arr_bool] = burnValue
        out_arr[band, :, :] = band_out
        # out_arr[:,:,band] = band_out

    if append_total_band:
        tot_band = np.zeros((h, w), dtype=np.uint8)
        band_arr_bool = np.where(in_arr > 0)
        tot_band[band_arr_bool] = burnValue
        tot_band = tot_band.reshape(1, h, w)
        out_arr = np.concatenate((out_arr, tot_band), axis=0).astype(np.uint8)

    if verbose:
        print("out_arr.shape:", out_arr.shape)
    return out_arr


###############################################################################
def CreateMultiBandGeoTiff(OutPath, Array):
    """
    Author: Jake Shermeyer
    Array has shape:
        Channels, Y, X?
    """
    driver = gdal.GetDriverByName("GTiff")
    DataSet = driver.Create(OutPath, Array.shape[2], Array.shape[1], Array.shape[0], gdal.GDT_Byte, ["COMPRESS=LZW"])
    for i, image in enumerate(Array, 1):
        DataSet.GetRasterBand(i).WriteArray(image)
    del DataSet

    return OutPath


def get_parameters(continuous: bool, output_conversion_csv: str):
    ###########################################################################
    # CONTINUOUS
    ###########################################################################
    if continuous:
        min_road_burn_val = 0
        min_speed_contin = 0
        max_speed_contin = 65
        mask_max = 255
        # placeholder variables for binned case
        channel_value_mult, n_channels, channel_burnValue, append_total_band = 0, 0, 0, 0

        #######################################################################
        def speed_to_burn_func(speed):
            """Convert speed estimate to mask burn value between
            0 and mask_max"""
            bw = mask_max - min_road_burn_val
            burn_val = min(
                min_road_burn_val + bw * ((speed - min_speed_contin) / (max_speed_contin - min_speed_contin)), mask_max
            )
            return max(burn_val, min_road_burn_val)

        speed_arr_contin = np.arange(min_speed_contin, max_speed_contin + 1, 1)
        burn_val_arr = [speed_to_burn_func(s) for s in speed_arr_contin]
        d = {"burn_val": burn_val_arr, "speed": speed_arr_contin}
        df_s = pd.DataFrame(d)

        # make conversion dataframe (optional)
        if not os.path.exists(output_conversion_csv):
            print("Write burn_val -> speed conversion to:", output_conversion_csv)
            df_s.to_csv(output_conversion_csv)
        else:
            print("path already exists, not overwriting...", output_conversion_csv)

    ###########################################################################
    # BINNED
    ###########################################################################
    else:
        min_speed_bin = 1
        max_speed_bin = 65
        bin_size_mph = 10.0
        channel_burnValue = 255
        channel_value_mult = 1
        append_total_band = True

        #######################################################################
        def speed_to_burn_func(speed_mph):
            """bin every 10 mph or so
            Convert speed estimate to appropriate channel
            bin = 0 if speed = 0"""
            return int(int(math.ceil(speed_mph / bin_size_mph)) * channel_value_mult)

        # determine num_channels
        n_channels = int(speed_to_burn_func(max_speed_bin))
        print("n_channels:", n_channels)
        # update channel_value_mult
        channel_value_mult = int(255 / n_channels)

        # make conversion dataframe
        speed_arr_bin = np.arange(min_speed_bin, max_speed_bin + 1, 1)
        burn_val_arr = np.array([speed_to_burn_func(s) for s in speed_arr_bin])
        d = {"burn_val": burn_val_arr, "speed": speed_arr_bin}
        df_s_bin = pd.DataFrame(d)
        # add a couple columns, first the channel that the speed corresponds to
        channel_val = (burn_val_arr / channel_value_mult).astype(int) - 1
        print("channel_val:", channel_val)
        df_s_bin["channel"] = channel_val
        # burn_uni = np.sort(np.unique(burn_val_arr))
        # print ("burn_uni:", burn_uni)
        if not os.path.exists(output_conversion_csv):
            print("Write burn_val -> speed conversion to:", output_conversion_csv)
            df_s_bin.to_csv(output_conversion_csv)
        else:
            print("path already exists, not overwriting...", output_conversion_csv)

    return speed_to_burn_func, channel_value_mult, n_channels, channel_burnValue, append_total_band


def speed_mask(
    geojson_path: str,
    image_path: str,
    mask_path_out: str,
    mask_path_out_md: str,
    speed_to_burn_func,
    buffer_distance_meters,
    verbose=True,
    # below here is all variables for binned speed
    channel_value_mult=1,
    n_channels=8,
    channel_burnValue=255,
    append_total_band=True,
) -> None:
    buffer_roundness = 1
    mask_burn_val_key = "burnValue"
    dissolve_by = "inferred_speed_mps"  # 'speed_m/s'
    bin_conversion_key = "inferred_speed_mph"  # 'speed_mph'
    # resave_pkl = False  # True

    road_speed.create_speed_gdf(
        image_path,
        geojson_path,
        mask_path_out,
        speed_to_burn_func,
        mask_burn_val_key=mask_burn_val_key,
        buffer_distance_meters=buffer_distance_meters,
        buffer_roundness=buffer_roundness,
        dissolve_by=dissolve_by,
        bin_conversion_key=bin_conversion_key,
        verbose=verbose,
    )

    # If Binning...
    if mask_path_out_md != "":
        # Convert array to a multi-channel image
        mask_bins = skimage.io.imread(mask_path_out)
        mask_bins = (mask_bins / channel_value_mult).astype(int)
        if verbose:
            print("mask_bins.shape:", mask_bins.shape)
            print("np unique mask_bins:", np.unique(mask_bins))
            # print ("mask_bins:", mask_bins)
        # define mask_channels
        if np.max(mask_bins) == 0:
            h, w = skimage.io.imread(mask_path_out).shape[:2]
            # h, w = cv2.imread(mask_path_out, 0).shape[:2]
            if append_total_band:
                mask_channels = np.zeros((n_channels + 1, h, w)).astype(np.uint8)
            else:
                mask_channels = np.zeros((n_channels, h, w)).astype(np.uint8)

        else:
            mask_channels = convert_array_to_multichannel(
                mask_bins,
                n_channels=n_channels,
                burnValue=channel_burnValue,
                append_total_band=append_total_band,
                verbose=verbose,
            )
        if verbose:
            print("mask_channels.shape:", mask_channels.shape)
            print("mask_channels.dtype:", mask_channels.dtype)

        # write to file
        # skimage version...
        # skimage.io.imsave(mask_path_out_md, mask_channels, compress=1)  # , plugin='tifffile')
        # gdal version
        CreateMultiBandGeoTiff(mask_path_out_md, mask_channels)
