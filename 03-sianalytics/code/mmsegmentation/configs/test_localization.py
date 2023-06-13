_base_ = 'test_spacenet8.py'

model = dict(auxiliary_head=[
    dict(
        type='FCNHead',
        in_channels=1536,
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
    dict(
        type='LocalizationHead',
        target=[1, 2, 3, 4],  # for obj
        decode_head=dict(
            type='SiameseHead',
            decode_head=dict(
                type='UPerHead',
                in_channels=[192, 384, 768, 1536],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=512,
                num_classes=2,
                norm_cfg=dict(type='SyncBN', requires_grad=True)),
            channels=1024,
            num_classes=2),
        channels=1024,
        num_classes=2,
        align_corners=False,
        loss_decode=[
            dict(type='FocalLoss'),
            dict(type='DiceLoss'),
            dict(type='LovaszLoss', reduction='none')
        ])
])
