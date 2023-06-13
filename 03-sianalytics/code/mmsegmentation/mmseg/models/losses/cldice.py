import mmcv

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


def soft_erode(img):
    if len(img.shape)==4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape)==5:
        p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1  =  soft_open(img)
    skel  =  F.relu(img-img1)
    for j in range(iter_):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img-img1)
        skel  =  skel +  F.relu(delta-skel*delta)
    return skel


@weighted_loss
def soft_cldice_loss(pred,
                     target,
                     valid_mask,
                     iteration=3,
                     smooth=1e-3):
    skel_pred = soft_skel(pred, iteration)
    skel_true = soft_skel(target, iteration)
    valid_mask = valid_mask.unsqueeze(1)
    
    tprec = (torch.sum(torch.multiply(skel_pred, target)[:,1:,...] * valid_mask)+smooth)/(torch.sum(skel_pred[:,1:,...] * valid_mask)+smooth)    
    tsens = (torch.sum(torch.multiply(skel_true, pred)[:,1:,...] * valid_mask)+smooth)/(torch.sum(skel_true[:,1:,...] * valid_mask)+smooth)
    cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
    return cl_dice


@LOSSES.register_module()
class CLDiceLoss(nn.Module):
    """CL-Dice Loss.
    
    """
    def __init__(
        self,
        iteration=3,
        smooth=1e-3,
        reduction='mean',
        class_weight=None,
        loss_weight=1.0,
        ignore_index=255,
        loss_name='loss_cldice',
    ):
        super(CLDiceLoss, self).__init__()
        
        self.iteration = iteration
        self.smooth = smooth
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = reduction_override or self.reduction
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes
        )
        one_hot_target =  torch.permute(one_hot_target, (0, 3, 1, 2)).float()
        valid_mask = (target != self.ignore_index).long()
        
        loss = self.loss_weight * soft_cldice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            iteration=self.iteration,
            smooth=self.smooth,
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

