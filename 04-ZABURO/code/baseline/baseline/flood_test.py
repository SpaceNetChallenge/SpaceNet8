import argparse
import os
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import skimage.io
import torch
from osgeo import gdal, osr
from torch.utils.data import Dataset

import models.pytorch_zoo.unet as unet
from models.other.unet import UNetSiamese
from utils.utils import write_geotiff


class SN8Dataset(Dataset):
    def __init__(
        self,
        pre_image_paths: Sequence[str],
        post_image_paths: Sequence[str],
        img_size: Tuple[int, int] = (1300, 1300),
    ) -> None:
        assert len(pre_image_paths) == len(post_image_paths)
        self.pre_image_paths = pre_image_paths
        self.post_image_paths = post_image_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.pre_image_paths)

    def __getitem__(self, index: int):
        preimg = np.moveaxis(skimage.io.imread(self.pre_image_paths[index]), -1, 0)
        postimg = gdal.Warp(
            "",
            self.post_image_paths[index],
            format="MEM",
            width=self.img_size[1],
            height=self.img_size[0],
            resampleAlg=gdal.GRIORA_Bilinear,
            outputType=gdal.GDT_Byte,
        ).ReadAsArray()
        return preimg, postimg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--in_csv", type=str, required=True)
    parser.add_argument(
        "--save_preds_dir",
        help="saves model predictions as .tifs",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


models = {
    "resnet34_siamese": unet.Resnet34_siamese_upsample,
    "resnet34": unet.Resnet34_upsample,
    "resnet50": unet.Resnet50_upsample,
    "resnet101": unet.Resnet101_upsample,
    "seresnet50": unet.SeResnet50_upsample,
    "seresnet101": unet.SeResnet101_upsample,
    "seresnet152": unet.SeResnet152_upsample,
    "seresnext50": unet.SeResnext50_32x4d_upsample,
    "seresnext101": unet.SeResnext101_32x4d_upsample,
    "unet_siamese": UNetSiamese,
}


if __name__ == "__main__":
    args = parse_args()
    in_csv = args.in_csv
    model_name = args.model_name
    save_preds_dir = args.save_preds_dir

    num_classes = 5

    df_test = pd.read_csv(in_csv)
    val_dataset = SN8Dataset(
        pre_image_paths=[
            os.path.join(args.data_dir, "PRE-event", name)
            for name in df_test["pre-event image"]
        ],
        post_image_paths=[
            os.path.join(args.data_dir, "POST-event", name)
            for name in df_test["post-event image 1"]
        ],
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    if model_name == "unet_siamese":
        model = UNetSiamese(3, num_classes, bilinear=True)
    else:
        model = models[model_name](num_classes=num_classes, num_channels=3)

    model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    model.eval()

    with torch.no_grad():
        for i, (preimg, postimg) in enumerate(val_dataloader):
            preimg = preimg.cuda().float()  # siamese
            postimg = postimg.cuda().float()  # siamese

            # siamese resnet34 with stacked preimg+postimg input
            flood_pred = model(preimg, postimg)
            flood_pred = (
                torch.nn.functional.softmax(flood_pred, dim=1).cpu().numpy()[0]
            )  # (5, H, W)

            flood_prediction = np.argmax(flood_pred, axis=0)  # (H, W)

            image_path = val_dataset.pre_image_paths[i]

            ds = gdal.Open(image_path)
            geotran = ds.GetGeoTransform()
            xmin, xres, rowrot, ymax, colrot, yres = geotran
            raster_srs = osr.SpatialReference()
            raster_srs.ImportFromWkt(ds.GetProjectionRef())
            ds = None
            output_tif = os.path.join(
                save_preds_dir,
                os.path.basename(image_path.replace(".tif", "_floodpred.tif")),
            )

            nrows, ncols = flood_prediction.shape
            write_geotiff(
                output_tif,
                ncols,
                nrows,
                xmin,
                xres,
                ymax,
                yres,
                raster_srs,
                [flood_prediction],
            )
