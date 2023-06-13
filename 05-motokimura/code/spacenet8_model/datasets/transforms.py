import albumentations as albu


def get_transforms(config, is_train):
    if is_train:
        transforms = [
            albu.RandomCrop(
                width=config.Transform.train_random_crop_size[0],
                height=config.Transform.train_random_crop_size[1],
                always_apply=True),
        ]
        # flip
        if config.Transform.train_random_flip_prob > 0:
            transforms.append(albu.Flip(p=config.Transform.train_random_flip_prob))
        # horizontal flip
        if config.Transform.train_random_horizontal_flip_prob > 0:
            transforms.append(albu.HorizontalFlip(p=config.Transform.train_random_horizontal_flip_prob))
        # vertical flip
        if config.Transform.train_random_vertical_flip_prob > 0:
            transforms.append(albu.VerticalFlip(p=config.Transform.train_random_vertical_flip_prob))
        # rotate90
        if config.Transform.train_random_rotate90_prob > 0:
            transforms.append(albu.RandomRotate90(p=config.Transform.train_random_rotate90_prob))
    else:
        transforms = [
            albu.PadIfNeeded(
                pad_height_divisor=32,
                pad_width_divisor=32,
                min_height=None,
                min_width=None,
                always_apply=True,
                border_mode=0,  # 0: cv2.BORDER_CONSTANT
                value=0,
                mask_value=0
            ),
        ]
    return albu.Compose(
        transforms,
        additional_targets={
            'image_post_a': 'image',
            'image_post_b': 'image'
        }
    )


def get_test_transforms(config, tta_hflip=False, tta_vflip=False):
    transforms = [
        albu.PadIfNeeded(
            pad_height_divisor=32,
            pad_width_divisor=32,
            min_height=None,
            min_width=None,
            always_apply=True,
            border_mode=0,  # 0: cv2.BORDER_CONSTANT
            value=0,
        ),
    ]

    # tta flipping
    if tta_hflip:
        transforms.append(albu.HorizontalFlip(always_apply=True))
    if tta_vflip:
        transforms.append(albu.VerticalFlip(always_apply=True))

    return albu.Compose(
        transforms,
        additional_targets={
            'image_post_a': 'image',
            'image_post_b': 'image'
        }
    )
