import torch.nn as nn

from ..backbones.unet import BasicConvBlock
from ..builder import HEADS
from ..utils import UpConvBlock
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class UnetHead(BaseDecodeHead):

    def __init__(self,
                 dec_num_convs=(1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 scale_factors=(2, 2, 2, 2),
                 upsample_cfg=dict(type='InterpConv'),
                 **kwargs):
        super(UnetHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.scale_factors = scale_factors

        self.decoder = nn.ModuleList()

        num_layers = len(self.in_index)
        for i, skip_channels in enumerate(self.in_channels[:-1]):
            in_channels = self.channels * 2**(i + 1)
            if i == num_layers - 2:
                in_channels = self.in_channels[-1]

            upsample_cfg.update(scale_factor=scale_factors[::-1][i])
            self.decoder.append(
                UpConvBlock(
                    conv_block=BasicConvBlock,
                    in_channels=in_channels,
                    skip_channels=skip_channels,
                    out_channels=self.channels * 2**i,
                    num_convs=dec_num_convs[i],
                    stride=1,
                    dilation=dec_dilations[i],
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    upsample_cfg=upsample_cfg))

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        output = inputs[-1]
        for i in reversed(range(len(self.decoder))):
            output = self.decoder[i](inputs[i], output)
        return output

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
