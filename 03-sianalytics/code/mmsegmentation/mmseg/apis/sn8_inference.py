import os

import numpy as np
import torch
from tqdm import tqdm

try:
    from osgeo import gdal
    from osgeo import osr
except ImportError:
    gdal = None
    osr = None

from mmseg.datasets.builder import build_dataloader, build_dataset

def write_geotiff(original_img_path, img_write_path, im_array, logit=False):
    ds = gdal.Open(original_img_path)
    geotran = ds.GetGeoTransform()
    xmin, xres, rowrot, ymax, colrot, yres = geotran
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(ds.GetProjectionRef())
    ds = None

    if logit:
        im_array = (im_array*255).astype('uint8')

    nrows, ncols = im_array.shape[-2:]
    if len(im_array.shape) == 2:
        im_array = [im_array]

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(img_write_path, ncols, nrows, len(im_array),
                           gdal.GDT_Byte)
    out_ds.SetGeoTransform((xmin, xres, 0, ymax, 0, yres))
    out_ds.SetProjection(raster_srs.ExportToWkt())
    for i in range(len(im_array)):
        outband = out_ds.GetRasterBand(i + 1)
        outband.WriteArray(im_array[i])
        outband.FlushCache()
    out_ds = None

def inference_scene(
    model,
    test_cfg,
    test_pipeline,
    image_path,
    mapping_csv,
    out_path,
    save_logit=False,
    use_post=False,
    post_select=0,
    bs_size=2,
):

    assert out_path, 'show dir must be specified'
    assert test_pipeline[0]['type'] in ['LoadImageFromFile', 'LoadImageFromPair']


    if test_pipeline[0]['type'] == 'LoadImageFromPair':
        if use_post:
            raise RuntimeError("POST flag can only be used in single "
                               "image inference.")

        test_pipeline[0]['type'] = 'LoadTIFImageFromPair'
        if 'pre_filename' not in test_pipeline[-1]['transforms'][-1]['meta_keys']:
            test_pipeline[-1]['transforms'][-1]['meta_keys'].append('pre_filename')

        cfg = dict(
            type='SpaceNet8TestDataset',
            img_dir=image_path,
            mapping_csv=mapping_csv,
            pipeline=test_pipeline,
            test_mode=True,
        )
    else:
        if 'filename' not in test_pipeline[-1]['transforms'][-1]['meta_keys']:
            test_pipeline[-1]['transforms'][-1]['meta_keys'].append('filename')
        if use_post:
            test_pipeline[0]['type'] = 'LoadPOSTTIFImageFromPair'
            cfg = dict(
                type='SpaceNet8TestDataset',
                img_dir=image_path,
                mapping_csv=mapping_csv,
                pipeline=test_pipeline,
                post_select=post_select,
                test_mode=True,
            )

        else:
            cfg = dict(
                type='SpaceNet8TestSingleDataset',
                img_dir=image_path,
                mapping_csv=mapping_csv,
                pipeline=test_pipeline,
                test_mode=True,
            )

    dataset = build_dataset(cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=bs_size,
        workers_per_gpu=bs_size,
        dist=False,
        shuffle=False)

    model.eval()
    scene_names = []
    for data in tqdm(data_loader):

        with torch.no_grad():
            if save_logit:
                seg_map = model(return_loss=False, logit=True, **data)
            else:
                seg_map = model(return_loss=False, **data)

        for i, d in enumerate(data['img_metas'][0].data[0]):
            scene_path = d.get('pre_filename', d.get('filename'))

            scene_name = os.path.basename(scene_path).replace('.tif', '_floodpred.tif')

            dst_path = os.path.join(out_path, scene_name)
            os.makedirs(out_path, exist_ok=True)
            write_geotiff(
                original_img_path=scene_path,
                img_write_path=dst_path,
                im_array=seg_map[i],
                logit=save_logit)

            scene_names.append(scene_name)
    return scene_names
