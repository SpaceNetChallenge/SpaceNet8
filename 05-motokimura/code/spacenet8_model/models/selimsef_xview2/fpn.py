"""
from selimsef's 2nd place solution for xView2 challenge
https://github.com/selimsef/xview2_solution/blob/master/models/fpn.py
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d, Sequential, UpsamplingBilinear2d

from .unet import (Conv1x1, Conv3x3, ConvReLu3x3, DensenetUnet, DPNUnet,
                   Resnet, ResneXt, encoder_params)


class FPN(nn.Module):
    def __init__(self, inner_filters, filters):
        super().__init__()
        self.lateral4 = Conv1x1(filters[-1], 256)
        self.lateral3 = Conv1x1(filters[-2], 256)
        self.lateral2 = Conv1x1(filters[-3], 256)
        self.lateral1 = Conv1x1(filters[-4], 256)

        self.smooth5 = Conv3x3(256, inner_filters)
        self.smooth4 = Conv3x3(256, inner_filters)
        self.smooth3 = Conv3x3(256, inner_filters)
        self.smooth2 = Conv3x3(256, inner_filters)

    def forward(self, encoder_results: list):
        x = encoder_results[0]
        lateral4 = self.lateral4(x)
        lateral3 = self.lateral3(encoder_results[1])
        lateral2 = self.lateral2(encoder_results[2])
        lateral1 = self.lateral1(encoder_results[3])

        m5 = lateral4
        m4 = lateral3 + F.upsample(m5, scale_factor=2, mode="nearest")
        m3 = lateral2 + F.upsample(m4, scale_factor=2, mode="nearest")
        m2 = lateral1 + F.upsample(m3, scale_factor=2, mode="nearest")

        p5 = self.smooth5(m5)
        p4 = self.smooth4(m4)
        p3 = self.smooth3(m3)
        p2 = self.smooth2(m2)

        return p2, p3, p4, p5


class FPNSegmentation(nn.Module):

    def __init__(self, inner_filters, filters, forward_pyramid_features=False, upsampling_mode="nearest"):
        super().__init__()
        self.fpn = FPN(inner_filters, filters)
        self.forward_pyramid_features = forward_pyramid_features
        seg_filters = inner_filters // 2
        output_filters = seg_filters
        for i in range(2, 6):
            self.add_module("level{}".format(i),
                            nn.Sequential(ConvReLu3x3(inner_filters, seg_filters),
                                          ConvReLu3x3(seg_filters, seg_filters)))
        self.upsampling = upsampling_mode
        self.aggregator = Sequential(
            Conv3x3(seg_filters * 4, output_filters),
            BatchNorm2d(output_filters),
            nn.ReLU()
        )

    def forward(self, encoder_results: list):
        pyramid_features = self.fpn(encoder_results)
        outputs = [
            self.level2(pyramid_features[0]),
            F.upsample(self.level3(pyramid_features[1]), scale_factor=2, mode=self.upsampling),
            F.upsample(self.level4(pyramid_features[2]), scale_factor=4, mode=self.upsampling),
            F.upsample(self.level5(pyramid_features[3]), scale_factor=8, mode=self.upsampling),
        ]
        x = torch.cat(outputs, dim=1)
        x = self.aggregator(x)
        return x


class ResneXtFPN(ResneXt):
    def __init__(self, seg_classes, backbone_arch):
        super().__init__(seg_classes, backbone_arch)
        self.fpn = FPNSegmentation(inner_filters=256, filters=encoder_params[backbone_arch]["filters"])
        self.up = UpsamplingBilinear2d(scale_factor=4)
        self.final = Conv1x1(in_channels=128, out_channels=seg_classes)
        self.dropout = nn.Dropout2d(p=0.15)
        _initialize_weights(self.fpn)
        _initialize_weights(self.final)


    def forward(self, x):
        enc_results = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i > 0:
                enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
        seg = self.fpn(list(reversed(enc_results)))
        seg = self.up(seg)
        seg = self.dropout(seg)
        x = self.final(seg)
        return x


def _initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight.data = nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class ResnetFPN(Resnet):
    def __init__(self, seg_classes, backbone_arch):
        super().__init__(seg_classes, backbone_arch)
        self.fpn = FPNSegmentation(inner_filters=128, filters=encoder_params[backbone_arch]["filters"])
        self.up = UpsamplingBilinear2d(scale_factor=4)
        self.final = Conv1x1(in_channels=128, out_channels=seg_classes)
        self.dropout = nn.Dropout2d(p=0.5)
        _initialize_weights(self.fpn)
        _initialize_weights(self.final)

    def forward(self, x):
        enc_results = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i > 0:
                enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
        seg = self.fpn(list(reversed(enc_results)))
        seg = self.up(seg)
        seg = self.dropout(seg)
        x = self.final(seg)
        return x


class DPNFPN(DPNUnet):
    def __init__(self, seg_classes, backbone_arch):
        super().__init__(seg_classes, backbone_arch)
        self.fpn = FPNSegmentation(inner_filters=128, filters=encoder_params[backbone_arch]["filters"])
        self.up = UpsamplingBilinear2d(scale_factor=4)
        self.final = Conv1x1(in_channels=128, out_channels=seg_classes)

    def forward(self, x):
        enc_results = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i > 0:
                enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
        seg = self.fpn(list(reversed(enc_results)))
        seg = self.up(seg)
        x = self.final(seg)
        return x

class DensenetFPN(DensenetUnet):
    def __init__(self, seg_classes, backbone_arch):
        super().__init__(seg_classes, backbone_arch)
        self.fpn = FPNSegmentation(inner_filters=128, filters=encoder_params[backbone_arch]["filters"])
        self.up = UpsamplingBilinear2d(scale_factor=4)
        self.final = Conv1x1(in_channels=128, out_channels=seg_classes)

    def forward(self, x):
        enc_results = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i > 0:
                enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
        seg = self.fpn(list(reversed(enc_results)))
        seg = self.up(seg)
        x = self.final(seg)
        return x

