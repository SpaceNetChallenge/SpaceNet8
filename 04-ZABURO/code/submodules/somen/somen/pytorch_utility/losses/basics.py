import torch.nn.functional as F
from torch import Tensor


def mse_loss(output: Tensor, target: Tensor) -> Tensor:
    target = target.reshape(output.shape)
    return F.mse_loss(output, target)


def nll_loss(output: Tensor, target: Tensor) -> Tensor:
    target = target.reshape(output.shape[:1] + output.shape[2:])
    return F.nll_loss(output, target)


def rmspe_loss(output: Tensor, target: Tensor, eps: float = 1e-12) -> Tensor:
    target = target.reshape(output.shape)
    return (((output - target) / target) ** 2).mean().clamp(min=eps).sqrt()


def mae_loss(output: Tensor, target: Tensor) -> Tensor:
    target = target.reshape(output.shape)
    return (output - target).abs().mean()


def smooth_l1_loss(output: Tensor, target: Tensor) -> Tensor:
    raise NotImplementedError
