from somen.pytorch_utility.losses.basics import mae_loss, mse_loss, nll_loss, rmspe_loss
from somen.pytorch_utility.losses.segmentations import binary_focal_loss, dice_loss
from somen.types import LossFnType, SupportedObjectiveLiteral


def get_loss_fn(objective: SupportedObjectiveLiteral) -> LossFnType:
    if objective == "mse":
        return mse_loss
    elif objective == "nll":
        return nll_loss
    elif objective == "rmspe":
        return rmspe_loss
    elif objective == "mae":
        return mae_loss
    elif objective == "binary_focal":
        return binary_focal_loss
    elif objective == "dice":
        return dice_loss
    else:
        raise ValueError


__all__ = ["get_loss_fn", "mse_loss", "nll_loss", "rmspe_loss", "mae_loss", "binary_focal_loss", "dice_loss"]
