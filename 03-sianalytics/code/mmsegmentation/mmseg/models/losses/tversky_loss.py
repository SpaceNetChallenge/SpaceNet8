import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss


@weighted_loss
def tversky_loss(pred,
                 target,
                 smooth=1,
                 class_weight=None,
                 alpha=0.3,
                 beta=0.7,
                 ignore_index=255):
    num_classes = pred.shape[1]
    one_hot_target = F.one_hot(
        torch.clamp(target.long(), 0, num_classes - 1),
        num_classes=num_classes)
    valid_mask = (target != ignore_index).long()
    assert pred.shape[0] == one_hot_target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            tversky_loss = binary_tversky_loss(
                pred[:, i],
                one_hot_target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                alpha=alpha,
                beta=beta,
            )
            if class_weight is not None:
                tversky_loss *= class_weight[i]
            total_loss += tversky_loss
    return total_loss / num_classes


@weighted_loss
def binary_tversky_loss(pred,
                        target,
                        valid_mask,
                        smooth=1,
                        alpha=0.3,
                        beta=0.7):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    TP = torch.sum(torch.mul(pred, target) * valid_mask, dim=1)
    FP = torch.sum(torch.mul(pred, 1 - target) * valid_mask, dim=1)
    FN = torch.sum(torch.mul(1 - pred, target) * valid_mask, dim=1)
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - tversky


@LOSSES.register_module()
class TverskyLoss(nn.Module):

    def __init__(
        self,
        smooth=1,
        reduction='mean',
        class_weight=None,
        loss_weight=1.0,
        alpha=0.3,
        beta=0.7,
        loss_name='loss_tversky',
    ):
        super(TverskyLoss, self).__init__()

        self.smooth = smooth
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = reduction_override or self.reduction
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)

        loss = self.loss_weight * tversky_loss(
            pred,
            target,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            class_weight=class_weight,
            alpha=self.alpha,
            beta=self.beta,
            **kwargs,
        )
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
