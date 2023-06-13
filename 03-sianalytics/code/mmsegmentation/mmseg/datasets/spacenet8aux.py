import os.path as osp
from mmseg.datasets import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RoadDataset(CustomDataset):

    CLASSES = ('bg', 'road')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(RoadDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir)


@DATASETS.register_module()
class BuildingRoadDataset(CustomDataset):

    CLASSES = ('bg', 'building', 'road')

    PALETTE = [[0, 0, 0], [255, 255, 255], [0, 255, 255]]

    def __init__(self, **kwargs):
        super(BuildingRoadDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir)
