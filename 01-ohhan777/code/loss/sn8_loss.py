import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.rmi import RMILoss


def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.long()]               
        true_1_hot = true_1_hot.permute(0, 3, 1, 2)                # (B, H, W, C) to (B, C, H, W)
        probas = F.softmax(logits, dim=1)
        
    true_1_hot = true_1_hot.type(logits.type()).contiguous()
    dims = (0,) + tuple(range(2, true.ndimension()))        # dims = (0, 2, 3)
    intersection = torch.sum(probas * true_1_hot, dims)     # intersection w.r.t. the class
    cardinality = torch.sum(probas + true_1_hot, dims)      # cardinality w.r.t. the class
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


class CrossEntropyLoss2d(torch.nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, ignore_index=255,
                 reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = torch.nn.NLLLoss(weight, reduction=reduction,
                                   ignore_index=ignore_index)

    def forward(self, inputs, targets, do_rmi=None):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class CrossEntropy(nn.Module):
    def __init__(self, bce_pos_weight=None, ce_weight=None, ignore_label=-1):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label

        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=bce_pos_weight
        )
        
        self.sigmoid = nn.Sigmoid()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=ce_weight,
            ignore_index=ignore_label
        )

    def _forward1(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                    h, w), mode='bilinear', align_corners=False)
        
        loss = self.bce_loss(score, target)
      
        return loss
    

    def _forward2(self, score, target):
        
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                    h, w), mode='bilinear', align_corners=False)
        
        score = self.sigmoid(score)
        loss = self.ce_loss(score, target)
        
        return loss
    
   
    def forward(self, score, target1, target2):
        # building
        weights1 = [0.4, 1]
        assert len(weights1) == len(score[:2])
        
        total_loss = sum([w * self._forward1(x, target1) for (w, x) in zip(weights1, score[:2])]) 
        
        # road
        weights2 = [0.4, 1]
        assert len(weights2) == len(score[2:4])

        total_loss += sum([w * self._forward2(x, target2) for (w, x) in zip(weights2, score[2:4])]) 
        return total_loss


class BuildingRoadFloodLoss(nn.Module):
    def __init__(self, device):
        super(BuildingRoadFloodLoss, self).__init__()
        self.device = device
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.rmi_roadspeed_loss = RMILoss(num_classes=8)
        self.rmi_flood_loss = RMILoss(num_classes=5)
        
    def _forward1(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                    h, w), mode='bilinear', align_corners=False)
        
        loss = self.bce_loss(score, target)
      
        return loss

    def _forward2(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                    h, w), mode='bilinear', align_corners=False)
        
        loss = self.rmi_roadspeed_loss(score, target)
      
        return loss

    def _forward3(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                    h, w), mode='bilinear', align_corners=False)
        
        loss = self.rmi_flood_loss(score, target)
      
        return loss

    def forward(self, score, building_target, road_target, flood_target):
        # building    
        # building_target should be a form of [B, 1, H, W] (float).
        building_weights = [0.4, 1]
        assert len(building_weights) == len(score[:2])
        
        building_loss = sum([w * self._forward1(x, building_target) for (w, x) in zip(building_weights, score[:2])])
        
        # road
        # input target (B, C, H, W) : C (0-7), 0~6 road speed, 7: binary segmentation (1: road, 0: no road)
        # we want target (B, H, W ) : [0-7] 0~6 road speed, 7: no road (background)
        road_target = road_target.float()
        road_target[:,7,:,:] = 0.99
        road_target = road_target.argmax(dim=1).long()   # (B, H, W)

        road_weights = [0.4, 1.0]
        assert len(road_weights) == len(score[2:4])
        
        roadspeed_loss = sum([w * self._forward2(x, road_target) for (w, x) in zip(road_weights, score[2:4])])


        # flood
        # input target (B, C, H, W) : C (0-3), 0: no flooded building, 1: flooded building, 2: no-flooded road, 3: flooded building
        # we want target (B, H, W) : [0-4] 0: background, 1: no flooded building, 2: flooded building, 3: no-flooded road, 4: flooded building
        b, _, h, w = flood_target.shape
        flood_target = torch.cat([torch.zeros(b,1,h,w).to(self.device), flood_target], dim=1).argmax(dim=1).long() # (B, H, W)
        

        flood_weights = [0.4, 1.0]
        assert len(flood_weights) == len(score[4:6])
        
        flood_loss = sum([w * self._forward3(x, flood_target) for (w, x) in zip(flood_weights, score[4:6])])
    
        total_loss = 0.6 * building_loss + 0.4 * roadspeed_loss + 0.4 * flood_loss

        return total_loss 

 

class BuildingLoss(nn.Module):
    def __init__(self, bce_pos_weight=None):
        super(BuildingLoss, self).__init__()
        
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=bce_pos_weight
        )

        self.sigmoid = nn.Sigmoid()
        

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                    h, w), mode='bilinear', align_corners=False)
        
        loss = self.bce_loss(score, target)
      
        return loss

    def forward(self, score, target):
         # building
        weights1 = [0.4, 1]
        assert len(weights1) == len(score[:2])
        
        building_loss = sum([w * self._forward(x, target) for (w, x) in zip(weights1, score[:2])]) 
        return building_loss

class RoadLoss(nn.Module):
    def __init__(self, bce_pos_weight=None, ce_weight=None):
        super(RoadLoss, self).__init__()
        
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=bce_pos_weight
        )
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
        

    def _forward1(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                    h, w), mode='bilinear', align_corners=False)
        
        loss = self.ce_loss(score, target)
      
        return loss

    def _forward2(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                    h, w), mode='bilinear', align_corners=False)
        
        loss = self.bce_loss(score, target)
      
        return loss

    def forward(self, score, target):
        # input target (B, C, H, W) : C (0-7), 0~6 road speed, 7: binary segmentation (1: road, 0: no road)
        # we want two types of targets, target 1 (B, H, W ) : C (0-7), 0~6 road speed, 7: no road (background)
        # target 2 (B, 1, H, W) : 0 no road, 1 road
        target1 = target.float()
        target1[:,7,:,:] = 0.99
        target1 = target1.argmax(dim=1).long()   # (B, H, W)
        target2 = target[:,7,:,:].unsqueeze(1).float()   # (B, 1, H, W)

        weights1 = [0.4, 1.0]
        assert len(weights1) == len(score[:2])
        
        roadspeed_loss = sum([w * self._forward1(x, target1) for (w, x) in zip(weights1, score[:2])])
        road_loss = self._forward2(score[2], target2)
        total_loss = 0.4 * roadspeed_loss + 0.6 * road_loss

        return total_loss 



class RoadLoss2(nn.Module):
    def __init__(self):
        super(RoadLoss2, self).__init__()
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.rmi_loss = RMILoss(num_classes=8)

    def _forward1(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                    h, w), mode='bilinear', align_corners=False)
        
        loss = self.rmi_loss(score, target)
      
        return loss

    def _forward2(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                    h, w), mode='bilinear', align_corners=False)
        
        loss = self.bce_loss(score, target)
      
        return loss

    def forward(self, score, target):
        # input target (B, C, H, W) : C (0-7), 0~6 road speed, 7: binary segmentation (1: road, 0: no road)
        # we want two types of targets, target 1 (B, H, W ) : C (0-7), 0~6 road speed, 7: no road (background)
        # target 2 (B, 1, H, W) : 0 no road, 1 road
        target1 = target.float()
        target1[:,7,:,:] = 0.99
        target1 = target1.argmax(dim=1).long()   # (B, H, W)
        target2 = target[:,7,:,:].unsqueeze(1).float()   # (B, 1, H, W)

        weights1 = [0.4, 1.0]
        assert len(weights1) == len(score[:2])
        
        roadspeed_loss = sum([w * self._forward1(x, target1) for (w, x) in zip(weights1, score[:2])])
        road_loss = self._forward2(score[2], target2)
        total_loss = 0.99 * roadspeed_loss + 0.01 * road_loss

        return total_loss 



def get_confusion_matrix(targets, preds, size, num_classes, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    preds = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    mask = (targets >= 0) & (targets < num_classes)
    
    confusion_mtx = np.bincount(num_classes * targets[mask].astype(int) + preds[mask],
                       minlength=num_classes ** 2)
    confusion_mtx = confusion_mtx.reshape(num_classes, num_classes)
    return confusion_mtx


def get_iou(targets, preds):
    # building_targets = building_targets.squeeze(1).long()  # (B, H, W)
    intersection = (preds & targets).sum()
    union = (preds | targets).sum().float()
    iou = intersection/(union + 1e-7)
    # building_pix_acc.update((building_preds == building_targets).sum().float()/(building_targets.numel()))
    # roadspeed_targets = roadspeed_targets.float()
    # roadspeed_targets[:,7,:,:] = 0.99
    # roadspeed_targets = roadspeed_targets.argmax(1).long()  # (B, H, W)  last channel is background
    # road_preds = torch.where(road_preds[:,7,:,:]>0.0, road_preds[:,:7,:,:].argmax(1), 7)
    return iou.item()

def get_pix_acc(targets, preds):
    pix_acc = (preds == targets).sum().float()/targets.numel()
    return pix_acc.item()