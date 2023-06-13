_base_ = 'test_spacenet8.py'

model = dict(
    decode_head=dict(
        _delete_=True,
        type='SiameseHead',
        decode_head=dict(
            type='UPerHead',
            in_channels=[192, 384, 768, 1536],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            num_classes=5,
            norm_cfg=dict(type='SyncBN', requires_grad=True)),
        channels=1024,
        num_classes=5,
        align_corners=False,
        loss_decode=[
            dict(type='FocalLoss'),
            dict(type='DiceLoss'),
            dict(type='LovaszLoss', reduction='none')
        ]))
