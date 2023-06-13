_base_ = 'road_1300_louisiana.py'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', ratio_range=(0.8, 1.2)),
    dict(type='RandomRotate', prob=0.5, degree=(-45, 45)),
    dict(type='RandomCrop', crop_size=(768, 768)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=['img_shape', 'ori_shape', 'pad_shape'])
]
data = dict(train=[
    dict(
        type='SpaceNet8RoadDataset',
        img_dir='/nas/Dataset/SpaceNet8/mmstyle/louisiana_east/pre',
        ann_dir='/nas/Dataset/SpaceNet8/mmstyle/louisiana_east/road',
        filter_empty_gt=True,
        pipeline=train_pipeline)
])
model = dict(
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)))
