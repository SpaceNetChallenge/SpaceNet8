# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']

        if 'tif' in filename:
            gt_semantic_seg = (mmcv.imread(filename)[..., 0]>0).astype('uint8')
        else:
            img_bytes = self.file_client.get(filename)
            gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
        if len(gt_semantic_seg.shape) > 2:
            gt_semantic_seg = (gt_semantic_seg[..., 0]>1).astype(np.uint8)

        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromPair(LoadImageFromFile):
    """Load an image from before/after image pairs."""

    def __init__(self, keys=['post', 'pre'], **kwargs):
        super().__init__(**kwargs)
        self.keys = keys

    def load_image(self, results, key='post'):
        filename = results['img_info'][f'{key}_filename']
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'], filename)
        results[f'{key}_filename'] = filename
        results[f'{key}_ori_filename'] = results['img_info'][f'{key}_filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        return img

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        img = []
        for key in self.keys:
            img.append(self.load_image(results, key=key))
        img = np.concatenate(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0

        if len(img.shape) < 3:
            raise ValueError
        num_channels = img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results


@PIPELINES.register_module()
class LoadTIFImageFromPair(LoadImageFromPair):
    """Load an image from before/after image pairs."""

    def load_image(self, results, key='post'):
        from osgeo import gdal
        filename = results['img_info'][f'{key}_filename']
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'], filename)
        results[f'{key}_filename'] = filename
        results[f'{key}_ori_filename'] = results['img_info'][f'{key}_filename']
        if key == 'post':
            ds = gdal.Warp(
                '',
                filename,
                format='MEM',
                width=self.img_size[1],
                height=self.img_size[0],
                resampleAlg=gdal.GRIORA_Bilinear,
                outputType=gdal.GDT_Byte)
        else:
            ds = gdal.Open(filename)

        img = ds.ReadAsArray().transpose(1, 2, 0)[:, :, ::-1]

        if key == 'pre':
            self.img_size = img.shape[:2]

        return img

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        imgs = dict()
        # pre image should be loaded first
        for key in ['pre', 'post']:
            imgs[key] = self.load_image(results, key=key)
        # rollback to the original order
        imgs = [imgs[key] for key in self.keys]
        img = np.concatenate(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0

        if len(img.shape) < 3:
            raise ValueError
        num_channels = img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results


@PIPELINES.register_module()
class LoadPOSTTIFImageFromPair(LoadTIFImageFromPair):
    """Load an image from before/after image pairs."""

    def get_extent(self, prename):
        from osgeo import gdal
        src = gdal.Open(prename)
        geoTransform = src.GetGeoTransform()
        minx = geoTransform[0]
        maxy = geoTransform[3]
        maxx = minx + geoTransform[1] * src.RasterXSize
        miny = maxy + geoTransform[5] * src.RasterYSize
        geo = [minx, miny, maxx, maxy]
        return geo

    def load_image(self, results, key='post'):
        from osgeo import gdal    
        filename = results['img_info'][f'{key}_filename']
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'], filename)
        results[f'{key}_filename'] = filename
        results[f'{key}_ori_filename'] = results['img_info'][f'{key}_filename']
        if key == 'post':
            prename = results
            ds = gdal.Warp(
                '',
                filename,
                format='MEM',
                width=self.img_size[1],
                height=self.img_size[0],
                outputBounds=self.get_extent(self.pre_imgname),
                resampleAlg=gdal.GRIORA_Bilinear,
                outputType=gdal.GDT_Byte)
        else:
            ds = gdal.Open(filename)

        img = ds.ReadAsArray().transpose(1, 2, 0)[:, :, ::-1]

        if key == 'pre':
            self.img_size = img.shape[:2]
            self.pre_imgname = filename

        return img

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        imgs = dict()
        # pre image should be loaded first
        for key in ['pre', 'post']:
            imgs[key] = self.load_image(results, key=key)

        # get post image only
        img = imgs['post']
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        # file name should be PRE-image
        results['filename'] = results['pre_filename']

        if len(img.shape) < 3:
            raise ValueError
        num_channels = img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results
