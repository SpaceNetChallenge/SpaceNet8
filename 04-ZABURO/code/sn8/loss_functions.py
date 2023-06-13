from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from somen.pytorch_utility import losses
from torch import Tensor

from sn8.datasets.sn5_multi_class_segmentation_dataset import OutputType


class RoadLossFn:
    def __init__(
        self,
        output_type: OutputType,
        focal_label_smoothing: float,
        junction_loss_weight: float,
        bce_weight: float = 0.0,
        focal_weight: float = 0.75,
        dice_weight: float = 0.25,
    ) -> None:
        self.output_type = output_type
        self.focal_label_smoothing = focal_label_smoothing
        self.junction_loss_weight = junction_loss_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def _core(self, output: Tensor, target: Tensor) -> Dict[str, Any]:
        if self.focal_label_smoothing > 0:
            smoothed_target = target + torch.where(
                target >= 0.5, -self.focal_label_smoothing, self.focal_label_smoothing
            )
            bce = F.binary_cross_entropy_with_logits(output, smoothed_target)
            focal = losses.binary_focal_loss(output, smoothed_target)
        else:
            bce = F.binary_cross_entropy_with_logits(output, target)
            focal = losses.binary_focal_loss(output, target)
        dice = losses.dice_loss(output.sigmoid(), target)
        loss = self.bce_weight * bce + self.focal_weight * focal + self.dice_weight * dice
        return {"bce": bce.item(), "focal": focal.item(), "dice": dice.item(), "loss": loss}

    def __call__(self, output: Tensor, target: Tensor) -> Dict[str, Any]:
        ret = self._core(output[:, :8], target[:, :8])  # speed or binary

        if self.output_type == "speed-and-junction":
            junction_loss = self._core(output[:, 8:], target[:, 8:])
            ret["junction_bce"] = junction_loss["bce"]
            ret["junction_focal"] = junction_loss["focal"]
            ret["junction_dice"] = junction_loss["dice"]
            ret["loss"] = ret["loss"] + self.junction_loss_weight * junction_loss["loss"]

        return ret


class BuildingLossFn:
    def __init__(self, bce_weight: float = 1.0, focal_weight: float = 0.0, dice_weight: float = 1.0) -> None:
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def __call__(self, output: Tensor, target: Tensor) -> Dict[str, Any]:
        bce = F.binary_cross_entropy_with_logits(output, target)
        focal = losses.binary_focal_loss(output, target)
        dice = losses.dice_loss(output.sigmoid(), target)
        loss = self.bce_weight * bce + self.focal_weight * focal + self.dice_weight * dice
        return {"bce": bce.item(), "focal": focal.item(), "dice": dice.item(), "loss": loss}


class FloodLossFn:
    def __init__(
        self, bce_weight: float, focal_weight: float, dice_weight: float, image_level_bce_weight: float
    ) -> None:
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.image_level_bce_weight = image_level_bce_weight

    def __call__(self, output: Tensor, target: Tensor, flood_exists: Optional[Tensor] = None) -> Dict[str, Any]:
        assert output.shape[1] == 1
        bce = F.binary_cross_entropy_with_logits(output[:, 0], target[:, 1], reduction="none")
        bce = (bce * target[:, 0]).sum() / (target[:, 0].sum() + 1e-6)

        focal = losses.binary_focal_loss(output[:, 0], target[:, 1], reduction="none")
        focal = (focal * target[:, 0]).sum() / (target[:, 0].sum() + 1e-6)

        dice = losses.dice_loss(output[:, 0].sigmoid() * target[:, 0], target[:, 1])

        loss = self.bce_weight * bce + self.focal_weight * focal + self.dice_weight * dice
        ret = {"bce": bce.item(), "focal": focal.item(), "dice": dice.item()}

        assert (self.image_level_bce_weight > 0.0) == (flood_exists is not None)
        if flood_exists is not None:
            assert self.image_level_bce_weight > 0.0
            assert flood_exists.shape == (output.shape[0],)
            flood_exists = flood_exists.float()
            # flood が一つもないところは画像全体が False になるように促す
            image_level_output = output.reshape(output.shape[0], output.shape[2] * output.shape[3]).mean(dim=1)
            image_level_bce = F.binary_cross_entropy_with_logits(
                image_level_output, flood_exists.float(), reduction="none"
            )
            image_level_bce = (image_level_bce * (1 - flood_exists)).sum() / ((1 - flood_exists).sum() + 1e-6)

            ret["image_level_bce"] = image_level_bce.item()
            loss = loss + self.image_level_bce_weight * image_level_bce

        ret["loss"] = loss
        return ret
