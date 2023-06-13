import os.path as osp

import mmcv
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose, LoadAnnotations, MergeClasses


@DATASETS.register_module()
class SpaceNet8Dataset(CustomDataset):
    """SpaceNet8 dataset for semantic segmentation based change detection. An
    example of file structure is as followed.

    .. code-block:: none

        ├── data
        │   ├── SpaceNet8
        │   │   ├── train
        │   │   │   ├── ann
        │   │   │   │   ├── 0.png
        │   │   │   │   ├── 1.png
        │   │   │   ├── post
        │   │   │   │   ├── 0.png
        │   │   │   │   ├── 1.png
        │   │   │   ├── pre
        │   │   │   │   ├── 0.png
        │   │   │   │   ├── 1.png
        │   │   ├── test
        │   │   │   ├── ann
        │   │   │   │   ├── 0.png
        │   │   │   │   ├── 1.png
        │   │   │   ├── post
        │   │   │   │   ├── 0.png
        │   │   │   │   ├── 1.png
        │   │   │   ├── pre
        │   │   │   │   ├── 0.png
        │   │   │   │   ├── 1.png

    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        ann_dir (str, optional): Path to annotation directory. Default: None
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
    """

    CLASSES = ('background', 'non-flooded building', 'flooded building',
               'non-flooded road', 'flooded road')

    PALETTE = [[0, 0, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 255, 255]]

    def __init__(self, filter_empty_gt=False, **kwargs):
        self.filter_empty_gt = filter_empty_gt
        super(SpaceNet8Dataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory."""
        if split is not None:
            raise ValueError('split must be None value')

        img_infos = []
        for img in self.file_client.list_dir_or_file(
                dir_path=osp.join(img_dir, 'ann'),
                list_dir=False,
                suffix=img_suffix,
                recursive=True):
            img_info = dict()
            img_info['filename'] = img
            for key in ('post', 'pre'):
                img_info[f'{key}_filename'] = osp.join(img_dir, key, img)
            if ann_dir is not None:
                seg_map = osp.join(ann_dir, 'ann',
                                   img).replace(img_suffix, seg_map_suffix)
                img_info['ann'] = dict(seg_map=seg_map)
                if self.filter_empty_gt and not (mmcv.imread(seg_map).any()):
                    continue
            img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos


@DATASETS.register_module()
class SpaceNet8BuildingDataset(SpaceNet8Dataset):
    CLASSES = ('background', 'building')
    PALETTE = [[0, 0, 0], [255, 0, 0]]
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory."""
        if split is not None:
            raise ValueError('split must be None value')

        img_infos = []
        for img in self.file_client.list_dir_or_file(
                dir_path=osp.join(img_dir),
                list_dir=False,
                suffix=img_suffix,
                recursive=True):
            img_info = dict()
            img_info['filename'] = img
            for key in ('post', 'pre'):
                img_info[f'{key}_filename'] = osp.join(img_dir, key, img)
            if ann_dir is not None:
                seg_map = osp.join(ann_dir,
                                   img).replace(img_suffix, seg_map_suffix)
                img_info['ann'] = dict(seg_map=seg_map)
                if self.filter_empty_gt and not (mmcv.imread(seg_map).any()):
                    continue
            img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos


@DATASETS.register_module()
class SpaceNet8RoadDataset(SpaceNet8Dataset):
    CLASSES = ('background', 'road')
    PALETTE = [[0, 0, 0], [0, 255, 255]]
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory."""
        if split is not None:
            raise ValueError('split must be None value')

        img_infos = []
        for img in self.file_client.list_dir_or_file(
                dir_path=osp.join(img_dir),
                list_dir=False,
                suffix=img_suffix,
                recursive=True):
            img_info = dict()
            img_info['filename'] = img
            for key in ('post', 'pre'):
                img_info[f'{key}_filename'] = osp.join(img_dir, key, img)
            if ann_dir is not None:
                seg_map = osp.join(ann_dir,
                                   img).replace(img_suffix, seg_map_suffix)
                img_info['ann'] = dict(seg_map=seg_map)
                if self.filter_empty_gt and not (mmcv.imread(seg_map).any()):
                    continue
            img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos


@DATASETS.register_module()
class SpaceNet8FoundationDataset(SpaceNet8Dataset):
    CLASSES = ('background', 'building', 'road')
    PALETTE = [[0, 0, 0], [0, 255, 255], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(SpaceNet8FoundationDataset, self).__init__(**kwargs)
        self.gt_seg_map_loader = Compose([LoadAnnotations(), MergeClasses()])


@DATASETS.register_module()
class SpaceNet8PairRoadDataset(SpaceNet8Dataset):
    CLASSES = ('background', 'road')
    PALETTE = [[0, 0, 0], [0, 255, 255], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(SpaceNet8PairRoadDataset, self).__init__(**kwargs)
        self.gt_seg_map_loader = Compose([LoadAnnotations(), MergeClasses(classes=[[3,4]])])


@DATASETS.register_module()
class SpaceNet8PairBuildingDataset(SpaceNet8Dataset):
    CLASSES = ('background', 'building')
    PALETTE = [[0, 0, 0], [0, 255, 255], [255, 0, 0]]

    def __init__(self, **kwargs):
        super(SpaceNet8PairBuildingDataset, self).__init__(**kwargs)
        self.gt_seg_map_loader = Compose([LoadAnnotations(), MergeClasses(classes=[[1,2]])])

