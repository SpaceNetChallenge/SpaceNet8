from typing import Sequence

import torch
import torch.nn as nn

from .. import builder
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class LocalizationHead(BaseDecodeHead):

    def __init__(self, decode_head, target, **kwargs):
        super(LocalizationHead, self).__init__(
            in_channels=-1, in_index=-1, input_transform=None, **kwargs)

        if not isinstance(target, Sequence):
            target = [target]
        self.target = target

        decode_head = builder.build_head(decode_head)
        if decode_head.dropout is not None:
            decode_head.dropout = nn.Identity()
        decode_head.conv_seg = nn.Identity()
        self.decode_head = decode_head

    def forward(self, inputs):
        """Placeholder of forward function."""
        output = self.decode_head.forward(inputs)
        output = self.cls_seg(output)
        return output

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        gt_semantic_seg_ = torch.zeros_like(gt_semantic_seg)
        for target in self.target:
            gt_semantic_seg_ |= gt_semantic_seg == target
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg_)
        return losses
