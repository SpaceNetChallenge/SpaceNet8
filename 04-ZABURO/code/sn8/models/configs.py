from dataclasses import dataclass
from typing import Literal, Tuple, Type, TypeVar

import segmentation_models_pytorch as smp
from timm.models import swin_transformer
from torch import nn
from torch.utils import model_zoo

from sn8.models.multi_task_unet_siamese import MultiTaskUnetSiamese
from sn8.models.swin_transformer import swin_encoders
from sn8.models.unet_siamese import UnetSiamese

T = TypeVar("T")


def _build_core(model_cls: Type[T], encoder_name: str, encoder_weights: str, **kwargs) -> T:
    if encoder_name in swin_encoders:
        url = swin_transformer.default_cfgs[encoder_name]["url"]
        encoder_weights = None

    model = model_cls(encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs)

    if encoder_name in swin_encoders:
        model.encoder.load_state_dict(model_zoo.load_url(url, map_location="cpu")["model"], strict=False)

    return model


@dataclass
class UnetConfig:
    class_name: Literal["Unet"]
    encoder_name: str = "resnet50"
    encoder_weights: str = "imagenet"

    def build(self, classes: int = 8) -> nn.Module:
        return _build_core(smp.Unet, self.encoder_name, self.encoder_weights, in_channels=3, classes=classes)


@dataclass
class UnetSiameseConfig:
    class_name: Literal["UnetSiamese"]
    encoder_name: str = "resnet50"
    encoder_weights: str = "imagenet"

    def build(self, classes: int = 8) -> nn.Module:
        return _build_core(UnetSiamese, self.encoder_name, self.encoder_weights, in_channels=3, classes=classes)


@dataclass
class MultiTaskUnetSiameseConfig:
    class_name: Literal["MultiTaskUnetSiamese"]
    encoder_name: str = "resnet50"
    encoder_weights: str = "imagenet"
    insert_branch_block: bool = True
    flood_grad_mul: float = 1.0

    def build(self, classes: Tuple[int, int, int] = (8, 2, 1)) -> nn.Module:
        return _build_core(
            MultiTaskUnetSiamese,
            self.encoder_name,
            self.encoder_weights,
            in_channels=3,
            classes=classes,
            insert_branch_block=self.insert_branch_block,
            flood_grad_mul=self.flood_grad_mul,
        )
