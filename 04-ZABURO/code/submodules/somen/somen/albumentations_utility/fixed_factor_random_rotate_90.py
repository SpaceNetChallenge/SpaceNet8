# import albumentations.augmentations.functional as F
# import numpy as np
# from albumentations.core.transforms_interface import DualTransform


# class FixedFactorRandomRotate90(DualTransform):
#     """Rotate the input by 90 degrees zero or more times.

#     Args:
#         p (float): probability of applying the transform. Default: 0.5.

#     Targets:
#         image, mask, bboxes, keypoints

#     Image types:
#         uint8, float32
#     """

#     def __init__(self, factor, *args, **kwargs):
#         super(FixedFactorRandomRotate90, self).__init__(*args, **kwargs)
#         self.factor = factor

#     def apply(self, img, **params):
#         return np.ascontiguousarray(np.rot90(img, self.factor))

#     def get_params(self):
#         return {}

#     def apply_to_bbox(self, bbox, **params):
#         return F.bbox_rot90(bbox, **params)

#     def apply_to_keypoint(self, keypoint, **params):
#         return F.keypoint_rot90(keypoint, **params)

#     def get_transform_init_args_names(self):
#         return ()
