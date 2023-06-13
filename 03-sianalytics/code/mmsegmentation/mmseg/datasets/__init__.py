# Copyright (c) OpenMMLab. All rights reserved.
from .ade import ADE20KDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
from .coco_stuff import COCOStuffDataset
from .custom import CustomDataset
from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import (ConcatDataset, MultiImageMixDataset,
                               RepeatDataset)
from .drive import DRIVEDataset
from .hrf import HRFDataset
from .isaid import iSAIDDataset
from .isprs import ISPRSDataset
from .loveda import LoveDADataset
from .night_driving import NightDrivingDataset
from .pascal_context import PascalContextDataset, PascalContextDataset59
from .potsdam import PotsdamDataset
from .spacenet8 import SpaceNet8Dataset, SpaceNet8RoadDataset, SpaceNet8BuildingDataset, SpaceNet8FoundationDataset
from .spacenet8test import SpaceNet8TestDataset, SpaceNet8TestSingleDataset
from .stare import STAREDataset
from .voc import PascalVOCDataset
from .spacenet8aux import BuildingRoadDataset, RoadDataset
from .pretraining import DeepGlobeRoadDataset, INRIABuildingDataset, MassachusettsBuildingDataset, MassachusettsRoadDataset, xView2BuildingDataset

__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
    'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
    'PascalContextDataset59', 'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset',
    'STAREDataset', 'DarkZurichDataset', 'NightDrivingDataset',
    'COCOStuffDataset', 'LoveDADataset', 'MultiImageMixDataset',
    'iSAIDDataset', 'ISPRSDataset', 'PotsdamDataset', 'SpaceNet8Dataset',
    'SpaceNet8TestSingleDataset', 'SpaceNet8TestDataset', 'BuildingRoadDataset',
    'RoadDataset',
]
