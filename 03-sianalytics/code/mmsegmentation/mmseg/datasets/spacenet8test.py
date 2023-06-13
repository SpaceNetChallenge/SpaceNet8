import csv
import os.path as osp
import warnings
from glob import glob

from mmcv.utils import print_log

from .builder import DATASETS
from .custom import CustomDataset
from mmseg.utils import get_root_logger
from mmseg.datasets.pipelines import LoadImageFromFile


@DATASETS.register_module()
class SpaceNet8TestDataset(CustomDataset):
    """xView2 dataset for semantic segmentation based change detection. An
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

    def __init__(
        self,
        pipeline,
        img_dir,
        mapping_csv,
        ann_dir=None,
        test_mode=True,
        filter_empty_gt=False,
        ignore_index=255,
        reduce_zero_label=False,
        post_select=0,
    ):
        self.filter_empty_gt = filter_empty_gt
        self.annos = []
        self.post_select = post_select
        with open(mapping_csv) as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.annos.append(r['label'])

        super().__init__(
            pipeline,
            img_dir,
            img_suffix='.png',
            ann_dir=ann_dir,
            seg_map_suffix='.png',
            test_mode=test_mode,
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label)
        assert osp.exists(mapping_csv), f'{mapping_csv} file should be exists!'

    def load_annotations(
        self,
        img_dir,
        img_suffix,
        ann_dir,
        seg_map_suffix,
        split,
        return_labels=False,
    ):
        """Load annotation from directory."""

        data_infos = []
        pre_images = glob(osp.join(img_dir, 'PRE-event', '*.tif'))
        post_images = glob(osp.join(img_dir, 'POST-event', '*.tif'))

        if return_labels:
            bldgs = glob(
                osp.join(img_dir, 'annotations', 'masks', 'building*.tif'))
            roads = glob(
                osp.join(img_dir, 'annotations', 'masks', 'road*.tif'))
            floods = glob(
                osp.join(img_dir, 'annotations', 'masks', 'flood*.tif'))
            roadspeeds = glob(
                osp.join(img_dir, 'annotations', 'masks', 'roadspeed*.tif'))

        for i in self.annos:
            tileid = osp.basename(i).split('.')[0]
            pre_im = [j for j in pre_images if f'_{tileid}.tif' in j][0]
            post_im = [j for j in post_images if f'_{tileid}.tif' in j][self.post_select]

            img_info = dict()
            img_info['pre_filename'] = pre_im
            img_info['post_filename'] = post_im

            if return_labels:
                building = [
                    j for j in bldgs
                    if 'building_' in j and f'_{tileid}.tif' in j
                ][0]
                road = [
                    j for j in roads if 'road_' in j and f'_{tileid}.tif' in j
                ][0]
                flood = [
                    j for j in floods
                    if 'flood_' in j and f'_{tileid}.tif' in j
                ][0]
                speed = [
                    j for j in roadspeeds
                    if 'roadspeed_' in j and f'_{tileid}.tif' in j
                ][0]

                img_info['ann_building'] = building
                img_info['ann_road'] = road
                img_info['ann_flood'] = flood
                img_info['ann_speed'] = speed

            data_infos.append(img_info)

        print_log(f'Loaded {len(data_infos)} images', logger=get_root_logger())
        return data_infos

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        return results['gt_seg_map']

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default.')

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results['gt_seg_map']

@DATASETS.register_module()
class SpaceNet8TestSingleDataset(SpaceNet8TestDataset):
    """ SpaceNet8Dataset for PRE-event images """
    def load_annotations(
        self,
        img_dir,
        img_suffix,
        ann_dir,
        seg_map_suffix,
        split,
        return_labels=False,
    ):
        """Load annotation from directory."""

        data_infos = []
        pre_images = glob(osp.join(img_dir, 'PRE-event', '*.tif'))
        post_images = glob(osp.join(img_dir, 'POST-event', '*.tif'))

        if return_labels:
            bldgs = glob(
                osp.join(img_dir, 'annotations', 'masks', 'building*.tif'))
            roads = glob(
                osp.join(img_dir, 'annotations', 'masks', 'road*.tif'))
            floods = glob(
                osp.join(img_dir, 'annotations', 'masks', 'flood*.tif'))
            roadspeeds = glob(
                osp.join(img_dir, 'annotations', 'masks', 'roadspeed*.tif'))

        for i in self.annos:
            tileid = osp.basename(i).split('.')[0]
            pre_im = [j for j in pre_images if f'_{tileid}.tif' in j][0]
            post_im = [j for j in post_images if f'_{tileid}.tif' in j][0]

            img_info = dict()
            img_info['pre_filename'] = pre_im
            img_info['filename'] = pre_im
            img_info['post_filename'] = post_im

            if return_labels:
                building = [
                    j for j in bldgs
                    if 'building_' in j and f'_{tileid}.tif' in j
                ][0]
                road = [
                    j for j in roads if 'road_' in j and f'_{tileid}.tif' in j
                ][0]
                flood = [
                    j for j in floods
                    if 'flood_' in j and f'_{tileid}.tif' in j
                ][0]
                speed = [
                    j for j in roadspeeds
                    if 'roadspeed_' in j and f'_{tileid}.tif' in j
                ][0]

                img_info['ann_building'] = building
                img_info['ann_road'] = road
                img_info['ann_flood'] = flood
                img_info['ann_speed'] = speed

            data_infos.append(img_info)

        print_log(f'Loaded {len(data_infos)} images', logger=get_root_logger())
        return data_infos
