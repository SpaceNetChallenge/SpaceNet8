train_pipeline = [
    dict(type='LoadImageFromPair', keys=['post', 'pre'], to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='Resize', ratio_range=(0.8, 1.2)),
    dict(type='RandomRotate', prob=0.5, degree=(-45, 45)),
    dict(type='RandomCrop', crop_size=(1024, 1024)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675, 103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395, 57.375, 57.12, 58.395],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=['img_shape', 'ori_shape', 'pad_shape'])
]
test_pipeline = [
    dict(type='LoadImageFromPair', keys=['post', 'pre'], to_float32=True),
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
                mean=[103.53, 116.28, 123.675, 103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395, 57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            # dict(type='ReflectPad', size=(1440, 1440)),
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
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=[
        dict(
            type='SpaceNet8Dataset',
            img_dir='/data/SpaceNet8/mmstyle/train',
            ann_dir='/data/SpaceNet8/mmstyle/train',
            pipeline=train_pipeline),
    ],
    val=dict(
        type='SpaceNet8Dataset',
        img_dir='/data/SpaceNet8/mmstyle/train',
        ann_dir='/data/SpaceNet8/mmstyle/train',
        pipeline=test_pipeline),
    test=dict(
        type='SpaceNet8Dataset',
        img_dir='/data/SpaceNet8/mmstyle/train',
        ann_dir='/data/SpaceNet8/mmstyle/train',
        pipeline=test_pipeline))
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=1.5e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
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
    type='ChangeDetector',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=(
                'https://download.openmmlab.com/'
                'mmsegmentation/v0.5/pretrain/swin/'
                'swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth')),
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        num_classes=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='FocalLoss'),
            dict(type='DiceLoss'),
            dict(type='LovaszLoss', reduction='none')
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='FocalLoss', loss_weight=0.4),
            dict(type='DiceLoss', loss_weight=0.4),
            dict(type='LovaszLoss', reduction='none', loss_weight=0.4)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))
gpu_ids = range(4)
