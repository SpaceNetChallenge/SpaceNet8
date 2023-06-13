"""
from selimsef's 2nd place solution for xView2 challenge
https://github.com/selimsef/xview2_solution/blob/master/models/siamese_unet.py
"""

import os
import sys
from functools import partial

import torch.hub
from efficientnet_pytorch import EfficientNet, get_model_params
from torch.nn import (Dropout2d, ModuleList, Sequential, Upsample,
                      UpsamplingBilinear2d)
from torch.utils import model_zoo

from . import resnet
from .densenet import densenet121, densenet161
from .dpn import dpn92
from .resnet import resnext50_32x4d, resnext101_32x8d
from .senet import SCSEModule, se_resnext50_32x4d, senet154

encoder_params = {

    'resnext101':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [48, 96, 192, 192],
         'last_upsample': 48,
         'init_op': partial(resnext101_32x8d, in_channels=3),
         # using weights from WSL
         'url': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth'},
    'resnext50':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [64, 96, 128, 128],
         'last_upsample': 64,
         'init_op': partial(resnext50_32x4d, in_channels=3),
         'url': "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth"},
    'densenet161':
        {'filters': [96, 384, 768, 2112, 2208],
         'decoder_filters': [64, 96, 192, 256],
         'last_upsample': 64,
         'url': None,
         'init_op': partial(densenet161, in_channels=3)},
    'resnet34':
        {'filters': [64, 64, 128, 256, 512],
         'decoder_filters': [64, 128, 256, 512],
         'last_upsample': 64,
         'init_op': partial(resnet.resnet34, in_channels=3),
         'url': resnet.model_urls['resnet34']},
    'densenet121':
        {'filters': [64, 256, 512, 1024, 1024],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'url': None,
         'init_op': partial(densenet121, in_channels=3)},
    'dpn92':
        {'filters': [64, 336, 704, 1552, 2688],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'init_op': dpn92,
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth'},
    'seresnext50':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [48, 92, 192, 256],
         'init_op': se_resnext50_32x4d,
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth'},
    'senet154':
        {'filters': [128, 256, 512, 1024, 2048],
         'decoder_filters': [48, 92, 192, 256],
         'init_op': senet154,
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth'},
    'efficientnet-b2':
        {"filters": (32, 24, 48, 120, 352),
         "stage_idxs": (5, 8, 16, 23),
         'last_upsample': 48,
         'decoder_filters': [48, 96, 192, 256],
         'init_op': partial(EfficientNet.from_pretrained, "efficientnet-b2"),
         'url': None},
    'efficientnet-b3':
        {"filters": (40, 32, 48, 136, 384),
         "stage_idxs": (5, 8, 18, 26),
         'last_upsample': 48,
         'decoder_filters': [48, 96, 192, 256],
         'init_op': partial(EfficientNet.from_pretrained, "efficientnet-b3"),
         'url': None},
    'efficientnet-b4':
        {"filters": (48, 32, 56, 160, 448),
         "stage_idxs": (6, 10, 22, 32),
         'last_upsample': 48,
         'decoder_filters': [48, 96, 192, 256],
         'init_op': partial(EfficientNet.from_pretrained, "efficientnet-b4"),
         'url': None},
}

import torch
import torch.nn.functional as F
from torch import nn


class BasicConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, activation=nn.ReLU, bias=True):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                            bias=bias)
        self.use_act = activation is not None
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class Conv1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=None, bias=bias)


class Conv3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=None)


class ConvReLu1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=nn.ReLU)


class ConvReLu3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=nn.ReLU)


class BasicUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.ReLU, mode='nearest'):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * 1
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1)
        self.use_act = activation is not None
        self.mode = mode
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = F.upsample(x, scale_factor=2, mode=self.mode)
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize_encoder(self, model, model_url, num_channels_changed=False):
        if os.path.isfile(model_url):
            pretrained_dict = torch.load(model_url)
        else:
            pretrained_dict = model_zoo.load_url(model_url)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if num_channels_changed:
            model.state_dict()[self.first_layer_params_names[0] + '.weight'][:, :3, ...] = pretrained_dict[
                self.first_layer_params_names[0] + '.weight'].data
            # init RGB channels for post disaster image as well
            model.state_dict()[self.first_layer_params_names[0] + '.weight'][:, 3:6, ...] = pretrained_dict[
                self.first_layer_params_names[0] + '.weight'].data
            skip_layers = self.first_layer_params_names
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               not any(k.startswith(s) for s in skip_layers)}
        model.load_state_dict(pretrained_dict, strict=False)

    @property
    def first_layer_params_names(self):
        return ['conv1.conv']


class SiameseEncoderDecoder(AbstractModel):
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34', shared=False):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = False
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck
        if not hasattr(self, 'use_bilinear_4x'):
            self.use_bilinear_4x = False

        self.filters = encoder_params[encoder_name]['filters']
        self.decoder_filters = encoder_params[encoder_name].get('decoder_filters', self.filters[:-1])
        self.last_upsample_filters = encoder_params[encoder_name].get('last_upsample', self.decoder_filters[0] // 2)

        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.bottlenecks = nn.ModuleList([self.bottleneck_type(self.filters[-i - 2] * 2 + f, f) for i, f in
                                          enumerate(reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])

        if self.first_layer_stride_two:
            self.last_upsample = Upsample(scale_factor=2)  # motokimura replaced UpsamplingBilinear2d with Upsample to use deterministic algorithm
        self.final = self.make_final_classifier(
            self.last_upsample_filters if self.first_layer_stride_two else self.decoder_filters[0], num_classes)
        self._initialize_weights()
        self.dropout = Dropout2d(p=0.1)
        self.shared = shared
        if shared:
            encoder = encoder_params[encoder_name]['init_op']()
            self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(self.filters))])
            if encoder_params[encoder_name]['url'] is not None:
                self.initialize_encoder(encoder, encoder_params[encoder_name]['url'], num_channels != 3)

        else:
            encoder1 = encoder_params[encoder_name]['init_op']()
            self.encoder_stages1 = nn.ModuleList([self.get_encoder(encoder1, idx) for idx in range(len(self.filters))])
            encoder2 = encoder_params[encoder_name]['init_op']()
            self.encoder_stages2 = nn.ModuleList([self.get_encoder(encoder2, idx) for idx in range(len(self.filters))])
            if encoder_params[encoder_name]['url'] is not None:
                self.initialize_encoder(encoder1, encoder_params[encoder_name]['url'], num_channels != 3)
                self.initialize_encoder(encoder2, encoder_params[encoder_name]['url'], num_channels != 3)

    def forward(self, input_x):
        enc_results1 = []
        enc_results2 = []
        # pre disaster
        x = input_x[:, :3, ...]
        for stage in (self.encoder_stages if self.shared else self.encoder_stages1):
            x = stage(x)
            enc_results1.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
        # post disaster
        x = input_x[:, 3:, ...]
        for stage in (self.encoder_stages if self.shared else self.encoder_stages2):
            x = stage(x)
            enc_results2.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, torch.cat([enc_results1[rev_idx - 1], enc_results2[rev_idx - 1]], dim=1))

        if self.first_layer_stride_two:
            x = self.last_upsample(x)
        x = self.dropout(x)
        f = self.final(x)
        return f

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_filters, num_classes, 1, padding=0)
        )

    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params(self):
        return _get_layers_params([self.encoder_stages[0]])

    @property
    def first_layer_params_names(self):
        raise NotImplementedError

    @property
    def layers_except_first_params(self):
        layers = get_slice(self.encoder_stages, 1, -1) + [self.bottlenecks, self.decoder_stages, self.final]
        return _get_layers_params(layers)


def _get_layers_params(layers):
    return sum((list(l.parameters()) for l in layers), [])


def get_slice(features, start, end):
    if end == -1:
        end = len(features)
    return [features[i] for i in range(start, end)]


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class SCSeResneXt(SiameseEncoderDecoder):

    def __init__(self, seg_classes, backbone_arch, reduction=2, mode='concat', num_channels=3, shared=False):
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = partial(ConvSCSEBottleneckNoBn, reduction=reduction, mode=mode)
        self.first_layer_stride_two = True
        self.concat_scse = mode == 'concat'

        super().__init__(seg_classes, num_channels, backbone_arch, shared=shared)
        self.last_upsample = self.decoder_block(
            self.decoder_filters[0] * 2 if self.concat_scse else self.decoder_filters[0],
            self.last_upsample_filters,
            self.last_upsample_filters)

    def calc_dec_filters(self, d_filters):
        return d_filters * 2 if self.concat_scse else d_filters

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        if self.concat_scse and layer + 1 < len(self.decoder_filters):
            in_channels *= 2

        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return encoder.layer0
        elif layer == 1:
            return nn.Sequential(
                encoder.pool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

    @property
    def first_layer_params_names(self):
        return ['layer0.conv1']


class ConvSCSEBottleneckNoBn(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, mode='concat'):
        print("bottleneck ", in_channels, out_channels)
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            SCSEModule(out_channels, reduction=reduction, mode=mode)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class Resnet(SiameseEncoderDecoder):
    def __init__(self, seg_classes, backbone_arch):
        self.first_layer_stride_two = True,
        super().__init__(seg_classes, 3, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


class ResneXt(SiameseEncoderDecoder):
    def __init__(self, seg_classes, backbone_arch, shared=False):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, 3, backbone_arch, shared=shared)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

    @property
    def first_layer_params_names(self):
        return ['conv1']


class DPNUnet(SiameseEncoderDecoder):
    def __init__(self, seg_classes, backbone_arch='dpn92', num_channels=3, shared=False):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, num_channels, backbone_arch, shared=shared)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.blocks['conv1_1'].conv,  # conv
                encoder.blocks['conv1_1'].bn,  # bn
                encoder.blocks['conv1_1'].act,  # relu
            )
        elif layer == 1:
            return nn.Sequential(
                encoder.blocks['conv1_1'].pool,  # maxpool
                *[b for k, b in encoder.blocks.items() if k.startswith('conv2_')]
            )
        elif layer == 2:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv3_')])
        elif layer == 3:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv4_')])
        elif layer == 4:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv5_')])

    @property
    def first_layer_params_names(self):
        return ['features.conv1_1.conv']


class DPNUnetFixed(DPNUnet):


    def forward(self, input_x):
        enc_results1 = []
        enc_results2 = []
        # pre disaster
        x = input_x[:, :3, ...]
        for stage in (self.encoder_stages if self.shared else self.encoder_stages1):
            x = stage(x)
            if isinstance(x, (list, tuple)):
                enc_results1.append(F.relu(torch.cat(x, dim=1), inplace=True))
            else:
                enc_results1.append(x)
        # post disaster
        x = input_x[:, 3:, ...]
        for stage in (self.encoder_stages if self.shared else self.encoder_stages2):
            x = stage(x)
            if isinstance(x, (list, tuple)):
                enc_results2.append(F.relu(torch.cat(x, dim=1), inplace=True))
            else:
                enc_results2.append(x)

        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, torch.cat([enc_results1[rev_idx - 1], enc_results2[rev_idx - 1]], dim=1))

        if self.first_layer_stride_two:
            x = self.last_upsample(x)
        x = self.dropout(x)
        f = self.final(x)
        return f

class DensenetUnet(SiameseEncoderDecoder):
    def __init__(self, seg_classes, backbone_arch='densenet161', shared=True):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, 3, backbone_arch, shared=shared)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.features.conv0,  # conv
                encoder.features.norm0,  # bn
                encoder.features.relu0  # relu
            )
        elif layer == 1:
            return nn.Sequential(encoder.features.pool0, encoder.features.denseblock1)
        elif layer == 2:
            return nn.Sequential(encoder.features.transition1, encoder.features.denseblock2)
        elif layer == 3:
            return nn.Sequential(encoder.features.transition2, encoder.features.denseblock3)
        elif layer == 4:
            return nn.Sequential(encoder.features.transition3, encoder.features.denseblock4, encoder.features.norm5,
                                 nn.ReLU())


class SEUnet(SiameseEncoderDecoder):
    def __init__(self, seg_classes, backbone_arch='senet154', shared=False):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, 3, backbone_arch, shared=shared)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return encoder.layer0
        elif layer == 1:
            return nn.Sequential(
                encoder.pool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


class IRV2Unet(SiameseEncoderDecoder):
    def __init__(self, seg_classes, backbone_arch='inceptionresnetv2', num_channels=3):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, num_channels, backbone_arch)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (1, 1)
            if isinstance(m, nn.MaxPool2d):
                m.padding = (1, 1)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return Sequential(encoder.conv2d_1a, encoder.conv2d_2a, encoder.conv2d_2b)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool_3a,
                encoder.conv2d_3b,
                encoder.conv2d_4a
            )
        elif layer == 2:
            return nn.Sequential(
                encoder.maxpool_5a,
                encoder.mixed_5b,
                encoder.repeat
            )
        elif layer == 3:
            return nn.Sequential(
                encoder.mixed_6a,
                encoder.repeat_1,
            )
        elif layer == 4:
            return nn.Sequential(
                encoder.mixed_7a,
                encoder.repeat_2,
                encoder.block8,
                encoder.conv2d_7b,
            )

    @property
    def first_layer_params_names(self):
        return ['conv2d_1a.conv']

    def initialize_encoder(self, model, model_url, num_channels_changed=False):
        del model.last_linear
        super().initialize_encoder(model, model_url, num_channels_changed)


class EfficientUnet(SiameseEncoderDecoder):
    def __init__(self, seg_classes, backbone_arch='efficientnet-b2', shared=False):
        self.first_layer_stride_two = True
        self._stage_idxs = encoder_params[backbone_arch]['stage_idxs']
        super().__init__(seg_classes, 3, backbone_arch, shared=shared)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(encoder._conv_stem, encoder._bn0, encoder._swish)
        elif layer == 1:
            return Sequential(*encoder._blocks[:self._stage_idxs[0]])
        elif layer == 2:
            return Sequential(*encoder._blocks[self._stage_idxs[0]:self._stage_idxs[1]])
        elif layer == 3:
            return Sequential(*encoder._blocks[self._stage_idxs[1]:self._stage_idxs[2]])
        elif layer == 4:
            return Sequential(*encoder._blocks[self._stage_idxs[2]:])

    def forward(self, input_x):
        enc_results1 = []
        enc_results2 = []
        # pre disaster
        x = input_x[:, :3, ...]
        block_idx = 0
        drop_connect_rate = 0.0
        for i, stage in enumerate(self.encoder_stages if self.shared else self.encoder_stages1):
            if i > 0:
                for block in stage:
                    block_idx += 1
                    drop_connect_rate *= float(block_idx) / self._stage_idxs[-1]
                    x = block(x, drop_connect_rate=drop_connect_rate)
            else:
                x = stage(x)
            enc_results1.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
        # post disaster
        x = input_x[:, 3:, ...]
        block_idx = 0
        drop_connect_rate = 0.0

        for i, stage in enumerate(self.encoder_stages if self.shared else self.encoder_stages2):
            if i > 0:
                for block in stage:
                    block_idx += 1
                    drop_connect_rate *= float(block_idx) / self._stage_idxs[-1]
                    x = block(x, drop_connect_rate=drop_connect_rate)
            else:
                x = stage(x)
            enc_results2.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, torch.cat([enc_results1[rev_idx - 1], enc_results2[rev_idx - 1]], dim=1))

        if self.first_layer_stride_two:
            x = self.last_upsample(x)
        x = self.dropout(x)
        f = self.final(x)
        return f


setattr(sys.modules[__name__], 'resnext_seamese_unet', partial(ResneXt))
setattr(sys.modules[__name__], 'resnext_seamese_unet_shared', partial(ResneXt, shared=True))
setattr(sys.modules[__name__], 'scseresnext_seamese_unet_shared', partial(SCSeResneXt, shared=True))
setattr(sys.modules[__name__], 'densenet_seamese_unet', partial(DensenetUnet))
setattr(sys.modules[__name__], 'densenet_seamese_unet_shared', partial(DensenetUnet, shared=True))
setattr(sys.modules[__name__], 'resnet_seamese_unet', partial(Resnet))
setattr(sys.modules[__name__], 'dpn_seamese_unet', partial(DPNUnet))
setattr(sys.modules[__name__], 'dpn_seamese_unet_shared', partial(DPNUnet, shared=True))
setattr(sys.modules[__name__], 'dpn_fixed_seamese_unet_shared', partial(DPNUnetFixed, shared=True))
setattr(sys.modules[__name__], 'efficient_seamese_unet_shared', partial(EfficientUnet, shared=True))

__all__ = ['resnext_seamese_unet',
           'densenet_seamese_unet',
           'resnet_seamese_unet',
           'dpn_seamese_unet_shared',
           'resnext_seamese_unet_shared',
           'efficient_seamese_unet_shared',
           'scseresnext_seamese_unet_shared',
           'densenet_seamese_unet_shared',
           'dpn_fixed_seamese_unet_shared',
           ]

if __name__ == '__main__':
    import numpy as np

    d = DPNUnetFixed(5, backbone_arch="dpn92", shared=True)
    d.eval()
    with torch.no_grad():
        images = torch.from_numpy(np.zeros((1, 6, 256, 256), dtype="float32"))
        i = d(images)
    print(d)
    print(i.size())
