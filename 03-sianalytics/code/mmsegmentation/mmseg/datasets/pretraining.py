import os.path as osp

import mmcv 
from mmcv.utils import print_log
from mmseg.datasets import DATASETS
from mmseg.utils import get_root_logger

from .custom import CustomDataset

#### Building

class BaseDataset(CustomDataset):
    def __init__(self, filter_empty_gt=True, **kwargs):
        self.filter_empty_gt = filter_empty_gt
        super().__init__(**kwargs)
        assert osp.exists(self.img_dir)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []

        for img in self.file_client.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=img_suffix,
                recursive=True):
            img_info = dict(filename=img)
            if ann_dir is not None:
                seg_map = img.replace(img_suffix, seg_map_suffix)
                img_info['ann'] = dict(seg_map=seg_map)
                if self.filter_empty_gt and not  \
                    (mmcv.imread(osp.join(ann_dir, seg_map)).any()):
                    continue
            img_infos.append(img_info)
        img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

@DATASETS.register_module()
class INRIABuildingDataset(BaseDataset):
    ## mmseg_style
    # - img_dir
    # - - train - *.png
    # - - test - *.png
    # - ann_dir
    # - - train - *.png
    # - - test - *.png

    CLASSES = ('bg', 'building')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir)


@DATASETS.register_module()
class MassachusettsBuildingDataset(BaseDataset):
    ## original format
    # - train 
    # - - images - *.tiff
    # - - masks - *.tif
    # - val 
    # - - images - *.tiff
    # - - masks - *.tif
    # - test
    # - - images - *.tiff
    # - - masks - *.tif

    CLASSES = ('bg', 'building')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.tiff', seg_map_suffix='.tif', **kwargs)
        assert osp.exists(self.img_dir)


@DATASETS.register_module()
class xView2BuildingDataset(BaseDataset):
    # not yet
    ## mmseg_style
    # - img_dir
    # - - train - *.png
    # - - test - *.png
    # - ann_dir
    # - - train - *.png
    # - - test - *.png

    CLASSES = ('bg', 'building')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir)


    CLASSES = ('bg', 'building')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []

        for img in self.file_client.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=img_suffix,
                recursive=True):
            if 'post' in img:
                continue
            img_info = dict(filename=img)
            if ann_dir is not None:
                seg_map = img.replace(img_suffix, seg_map_suffix)
                img_info['ann'] = dict(seg_map=seg_map)
                if self.filter_empty_gt and not  \
                    (mmcv.imread(osp.join(ann_dir, seg_map)).any()):
                    continue
            img_infos.append(img_info)
        img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

#### ROAD

@DATASETS.register_module()
class MassachusettsRoadDataset(BaseDataset):
    ## original format
    # - train 
    # - - images - *.tiff
    # - - masks - *.tif
    # - val 
    # - - images - *.tiff
    # - - masks - *.tif
    # - test
    # - - images - *.tiff
    # - - masks - *.tif

    CLASSES = ('bg', 'building')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.tiff', seg_map_suffix='.tif', **kwargs)
        assert osp.exists(self.img_dir)

@DATASETS.register_module()
class DeepGlobeRoadDataset(BaseDataset):
    ## original format
    # - train 
    # - - *sat.jpg
    # - - *mask.png
    # - valid
    # - - *sat.jpg
    # - test
    # - - *sat.jpg

    CLASSES = ('bg', 'road')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='sat.jpg', seg_map_suffix='mask.png', **kwargs)
        assert osp.exists(self.img_dir)
