# Obsolete version of Alubmentations.ToTensor
# https://github.com/albumentations-team/albumentations/pull/907

from albumentations.core.transforms_interface import BasicTransform
from albumentations.pytorch.transforms import img_to_tensor, mask_to_tensor


class ToTensorV1(BasicTransform):
    def __init__(self, num_classes=1, sigmoid=True, normalize=None):
        super().__init__(always_apply=True, p=1.0)
        self.num_classes = num_classes
        self.sigmoid = sigmoid
        self.normalize = normalize

    def __call__(self, *args, force_apply=True, **kwargs):
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        kwargs.update({"image": img_to_tensor(kwargs["image"], self.normalize)})
        if "mask" in kwargs.keys():
            kwargs.update({"mask": mask_to_tensor(kwargs["mask"], self.num_classes, sigmoid=self.sigmoid)})

        for k, _v in kwargs.items():
            if self._additional_targets.get(k) == "image":
                kwargs.update({k: img_to_tensor(kwargs[k], self.normalize)})
            if self._additional_targets.get(k) == "mask":
                kwargs.update({k: mask_to_tensor(kwargs[k], self.num_classes, sigmoid=self.sigmoid)})
        return kwargs

    @property
    def targets(self):
        raise NotImplementedError

    def get_transform_init_args_names(self):
        return "num_classes", "sigmoid", "normalize"
