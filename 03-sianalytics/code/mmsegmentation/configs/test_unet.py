_base_ = 'test_spacenet8.py'

model = dict(
    backbone=dict(use_stem=True),
    decode_head=dict(
        _delete_=True,
        type='UnetHead',
        in_channels=[384, 384, 768, 1536, 3072],
        in_index=[0, 1, 2, 3, 4],
        channels=64,
        num_classes=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        dec_num_convs=(1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        scale_factors=(2, 2, 2, 1),
        align_corners=False,
        loss_decode=[
            dict(type='FocalLoss'),
            dict(type='DiceLoss'),
            dict(type='LovaszLoss', reduction='none')
        ]),
    auxiliary_head=dict(in_index=3))
