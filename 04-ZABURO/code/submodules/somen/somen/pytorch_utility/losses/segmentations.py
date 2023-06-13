import torch.nn.functional as F
from torch import Tensor


def _check_if_sigmoid_is_applied(output: Tensor) -> None:
    if not ((0 <= output) & (output <= 1)).all():
        raise RuntimeError("Pass through the loss function after applying the sigmoid")


def binary_focal_loss(output: Tensor, target: Tensor, gamma: float = 2, reduction: str = "mean") -> Tensor:
    """Based on paper `Focal Loss for Dense Object Detection` - https://arxiv.org/abs/1708.02002"""
    target = target.reshape(output.shape)

    bce = F.binary_cross_entropy_with_logits(output, target)
    pt = (-bce).exp().clamp(min=0, max=1)

    loss = (1 - pt) ** gamma * bce
    if reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    else:
        raise NotImplementedError


def binary_reduced_focal_loss(output: Tensor, target: Tensor) -> Tensor:
    raise NotImplementedError


def dice_loss(output: Tensor, target: Tensor, smoothing: float = 1e-5) -> Tensor:
    _check_if_sigmoid_is_applied(output)

    batch_size = output.size()[0]
    dice_target = target.contiguous().view(batch_size, -1).float()
    dice_output = output.contiguous().view(batch_size, -1)
    intersection = (dice_output * dice_target).sum(dim=1)
    union = dice_output.sum(dim=1) + dice_target.sum(dim=1)
    return (1 - (2 * intersection + smoothing) / (union + smoothing)).mean()


def jaccard_index_loss(output: Tensor, target: Tensor) -> Tensor:
    raise NotImplementedError
