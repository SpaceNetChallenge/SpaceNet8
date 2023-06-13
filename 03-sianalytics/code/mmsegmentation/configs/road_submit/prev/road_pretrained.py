train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', ratio_range=(0.8, 1.2)),
    dict(type='RandomRotate', prob=0.5, degree=(-45, 45)),
    dict(type='RandomCrop', crop_size=(1024, 1024)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomShadow', prob=0.3),
    dict(type='RandomFog', prob=0.3),
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
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1300, 1300),
        img_ratios=None,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'img_shape', 'ori_shape', 'pad_shape', 'flip',
                    'flip_direction'
                ])
        ])
]
data = dict(
    dist=True,
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=[
        dict(
            type='SpaceNet8RoadDataset',
            img_dir='/nas/Dataset/SpaceNet3/mmstyle/img_dir/train',
            ann_dir='/nas/Dataset/SpaceNet3/mmstyle/ann_dir/train',
            pipeline=train_pipeline),
    ],
    val=dict(
        type='SpaceNet8RoadDataset',
        img_dir='/nas/Dataset/SpaceNet8/mmstyle/val/pre',
        ann_dir='/nas/Dataset/SpaceNet8/mmstyle/val/road',
        pipeline=test_pipeline),
    test=dict(
        type='SpaceNet8RoadDataset',
        img_dir='/nas/Dataset/SpaceNet8/mmstyle/val/pre',
        ann_dir='/nas/Dataset/SpaceNet8/mmstyle/val/road',
        pipeline=test_pipeline))
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='MlflowLoggerHook', exp_name='spacenet8', by_epoch=False)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
fp16 = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=120)
checkpoint_config = dict(interval=12)
evaluation = dict(interval=12, metric='mIoU', pre_eval=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=(
                'https://download.openmmlab.com/'
                'mmsegmentation/v0.5/pretrain/swin/'
                'swin_base_patch4_window7_224_22k_20220317-4f79f7c0.pth')),
        patch_norm=True),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='FocalLoss'),
            dict(type='DiceLoss'),
            dict(type='LovaszLoss', reduction='none')
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='FocalLoss', loss_weight=0.4),
            dict(type='DiceLoss', loss_weight=0.4),
            dict(type='LovaszLoss', reduction='none', loss_weight=0.4)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
gpu_ids = range(4)
