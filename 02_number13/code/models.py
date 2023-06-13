from segmentation_models_pytorch.base import initialization as init
import torch.nn as nn
import torch
from segmentation_models_pytorch.base.modules import Activation
from typing import Optional, Union, List
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import ClassificationHead
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.manet.decoder import *


# U-Net with two head for building and road
class UnetMhead(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 1
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None
            ,
        )
        # We use two heads one for building and the other for roads
        # instead of one head with two channel for buildings and road
        self.buildings_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes, activation=None, kernel_size=3)

        self.roads_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes, activation=None, kernel_size=3)

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.buildings_head)
        init.initialize_head(self.roads_head)

    def forward_body(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        return decoder_output

    def forward(self, x_build, x_road=None):
        decoder_output_build = self.forward_body(x_build)
        buildings = self.buildings_head(decoder_output_build)
        if x_road is None:
            roads = self.roads_head(decoder_output_build)
            return buildings, roads
        else:
            decoder_output_road = self.forward_body(x_road)
            roads = self.roads_head(decoder_output_road)
            return buildings, roads

    def predict(self, x_build, x_road=None):
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x_build, x_road)
        return x


class UnetSiamese(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        num_classes: int = 1,
        fuse='cat'
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None
            ,
        )

        if encoder_name in 'resnet34':
            ch = [3, 64, 64, 128, 256, 512]
        if encoder_name in  ['se_resnet50', 'resnet50', 'se_resnext50_32x4d']:
            ch = [3, 64, 256, 512, 1024, 2048]
        if encoder_name == 'timm-efficientnet-b0' or encoder_name == 'timm-efficientnet-b1':
            ch = [3, 32, 24, 40, 112, 320]
        if encoder_name == 'timm-efficientnet-b2':
            ch = [3, 32, 24, 48, 120, 352]
        if encoder_name == 'timm-efficientnet-b3':
            ch = [3, 40, 32, 48, 136, 384]
        self.fuse = fuse

        if self.fuse == 'cat':
            t = 2
        elif self.fuse == 'cat_add':
            t = 2
        elif self.fuse == 'add':
            t = 1

        # Pass the encoder outputs after concatenation through a Grouped Convolution layer
        self.projs_0 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[0] * t, out_channels=ch[0], kernel_size=(1, 1), groups=ch[0])])
        self.projs_1 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[1] * t, out_channels=ch[1], kernel_size=(1, 1), groups=ch[1])])
        self.projs_2 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[2] * t, out_channels=ch[2], kernel_size=(1, 1), groups=ch[2])])
        self.projs_3 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[3] * t, out_channels=ch[3], kernel_size=(1, 1), groups=ch[3])])
        self.projs_4 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[4] * t, out_channels=ch[4], kernel_size=(1, 1), groups=ch[4])])
        self.projs_5 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[5] * t, out_channels=ch[5], kernel_size=(1, 1), groups=ch[5])])
        self.projs = {0: self.projs_0, 1: self.projs_1, 2: self.projs_2, 3: self.projs_3,
                      4: self.projs_4, 5: self.projs_5}
        # the segmentation head
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes, activation=None, kernel_size=3)
        # the classification head for determining flooded/non-flooded image
        self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1],  classes=1)
        #self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        for i in range(6):
            init.initialize_decoder(self.projs[i])
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x1, x2):
        enc_1 = self.encoder(x1)
        enc_2 = self.encoder(x2)
        final_features = []
        for i in range(0, len(enc_1)):
            if self.fuse == 'cat':
                enc_fusion = torch.cat([enc_1[i], enc_2[i]], dim=1)
                final_features.append(self.projs[i][0](enc_fusion))
            elif self.fuse == 'add':
                enc_fusion = enc_1[i] + enc_2[i]
                #final_features.append(self.projs[i][0](enc_fusion))
                final_features.append(enc_fusion)

            elif self.fuse == 'cat_add':
                enc_fusion_1 = torch.cat([enc_1[i], enc_2[i]], dim=1)
                enc_fusion_2 = enc_1[i] + enc_2[i]
                enc_fusion = self.projs[i][0](enc_fusion_1) + enc_fusion_2
                final_features.append(enc_fusion)

        decoder_output = self.decoder(*final_features)
        change = self.segmentation_head(decoder_output)
        if self.classification_head is not None:
            cls = self.classification_head(final_features[-1])
            return change, cls
        else:
            return change, None

    def predict(self, x1, x2):
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x1, x2)
        return x


class MAnetMhead(nn.Module):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_pab_channels: int = 64,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = MAnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            pab_channels=decoder_pab_channels,
        )

        self.buildings_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.roads_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "manet-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.buildings_head)
        init.initialize_head(self.roads_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x_build, x_road=None):
        features = self.encoder(x_build)
        decoder_output_build = self.decoder(*features)

        buildings = self.buildings_head(decoder_output_build)
        if x_road is None:
            roads = self.roads_head(decoder_output_build)
        else:
            features = self.encoder(x_road)
            decoder_output_road = self.decoder(*features)
            roads = self.roads_head(decoder_output_road)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return buildings, roads, labels

        return buildings, roads

    def predict(self, x_build, x_road=None):
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x_build, x_road)
        return x


# U-Net Siamese that handles two heads (buildings and roads)
class UnetSiamese_Mhead(nn.Module):
    def __init__(
            self,
            encoder_name: str = "resnet50",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            in_channels: int = 3,
            num_classes: int = 1,
            fuse='cat'
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None
            ,
        )

        if encoder_name in ['resnet34']:
            ch = [3, 64, 64, 128, 256, 512]
        if encoder_name in ['se_resnet50', 'resnet50', 'se_resnext50_32x4d']:
            ch = [3, 64, 256, 512, 1024, 2048]
        if encoder_name in ['inceptionresnetv2']:
            ch = [3, 64, 192, 320, 1088, 1536]
        if encoder_name == 'timm-efficientnet-b0' or encoder_name == 'timm-efficientnet-b1':
            ch = [3, 32, 24, 40, 112, 320]
        if encoder_name == 'timm-efficientnet-b2':
            ch = [3, 32, 24, 48, 120, 352]
        if encoder_name == 'timm-efficientnet-b3':
            ch = [3, 40, 32, 48, 136, 384]
        if encoder_name == 'timm-efficientnet-b4':
            ch = [3, 48, 32, 56, 160, 448]
        if encoder_name == 'timm-efficientnet-b5':
            ch = [3, 48, 40, 64, 176, 512]
        if encoder_name in ['vgg13_bn']:
            ch = [64, 128, 256, 512, 512, 512]

        self.fuse = fuse

        if self.fuse == 'cat':
            t = 2
        elif self.fuse == 'cat_add':
            t = 2
        elif self.fuse == 'add':
            t = 1

        # Pass the encoder outputs after concatenation through a Grouped Convolution layer
        self.projs_0 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[0] * t, out_channels=ch[0], kernel_size=(1, 1), groups=ch[0])])
        self.projs_1 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[1] * t, out_channels=ch[1], kernel_size=(1, 1), groups=ch[1])])
        self.projs_2 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[2] * t, out_channels=ch[2], kernel_size=(1, 1), groups=ch[2])])
        self.projs_3 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[3] * t, out_channels=ch[3], kernel_size=(1, 1), groups=ch[3])])
        self.projs_4 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[4] * t, out_channels=ch[4], kernel_size=(1, 1), groups=ch[4])])
        self.projs_5 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[5] * t, out_channels=ch[5], kernel_size=(1, 1), groups=ch[5])])

        self.projs = {0: self.projs_0, 1: self.projs_1, 2: self.projs_2, 3: self.projs_3,
                      4: self.projs_4, 5: self.projs_5}

        self.buildings_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes, activation=None, kernel_size=3)

        self.roads_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes, activation=None, kernel_size=3)

        self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1],  classes=1)
        #self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.roads_head)
        init.initialize_head(self.buildings_head)
        for i in range(6):
            init.initialize_decoder(self.projs[i])
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x1, x2):
        enc_1 = self.encoder(x1)
        enc_2 = self.encoder(x2)
        final_features = []
        for i in range(0, len(enc_1)):
            if self.fuse == 'cat':
                enc_fusion = torch.cat([enc_1[i], enc_2[i]], dim=1)
                final_features.append(self.projs[i][0](enc_fusion))
            elif self.fuse == 'add':
                enc_fusion = enc_1[i] + enc_2[i]
                #final_features.append(self.projs[i][0](enc_fusion))
                final_features.append(enc_fusion)

            elif self.fuse == 'cat_add':
                enc_fusion_1 = torch.cat([enc_1[i], enc_2[i]], dim=1)
                enc_fusion_2 = enc_1[i] + enc_2[i]
                enc_fusion = self.projs[i][0](enc_fusion_1) + enc_fusion_2
                final_features.append(enc_fusion)

        decoder_output = self.decoder(*final_features)
        change_build = self.buildings_head(decoder_output)
        change_road = self.roads_head(decoder_output)
        if self.classification_head is not None:
            cls = self.classification_head(final_features[-1])
            return change_build, change_road, cls
        else:
            return change_build, change_road, None

    def predict(self, x1, x2):
        if self.training:
            self.eval()
        with torch.no_grad():
            seg_b, seg_r,  cls  = self.forward(x1, x2)
        return seg_b, seg_r,  cls


class MAnetSiamese_Mhead(nn.Module):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_pab_channels: int = 64,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
            fuse = 'cat'
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = MAnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            pab_channels=decoder_pab_channels,
        )

        self.buildings_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.roads_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if encoder_name in ['resnet34']:
            ch = [3, 64, 64, 128, 256, 512]
        if encoder_name in ['se_resnet50', 'resnet50', 'se_resnext50_32x4d']:
            ch = [3, 64, 256, 512, 1024, 2048]
        if encoder_name in ['inceptionresnetv2']:
            ch = [3, 64, 192, 320, 1088, 1536]
        if encoder_name == 'timm-efficientnet-b0' or encoder_name == 'timm-efficientnet-b1':
            ch = [3, 32, 24, 40, 112, 320]
        if encoder_name == 'timm-efficientnet-b2':
            ch = [3, 32, 24, 48, 120, 352]
        if encoder_name == 'timm-efficientnet-b3':
            ch = [3, 40, 32, 48, 136, 384]
        if encoder_name == 'timm-efficientnet-b4':
            ch = [3, 48, 32, 56, 160, 448]
        if encoder_name == 'timm-efficientnet-b5':
            ch = [3, 48, 40, 64, 176, 512]
        if encoder_name in ['vgg13_bn']:
            ch = [64, 128, 256, 512, 512, 512]

        self.fuse = fuse

        if self.fuse == 'cat':
            t = 2
        elif self.fuse == 'cat_add':
            t = 2
        elif self.fuse == 'add':
            t = 1

        self.projs_0 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[0] * t, out_channels=ch[0], kernel_size=(1, 1), groups=ch[0])])
        self.projs_1 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[1] * t, out_channels=ch[1], kernel_size=(1, 1), groups=ch[1])])
        self.projs_2 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[2] * t, out_channels=ch[2], kernel_size=(1, 1), groups=ch[2])])
        self.projs_3 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[3] * t, out_channels=ch[3], kernel_size=(1, 1), groups=ch[3])])
        self.projs_4 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[4] * t, out_channels=ch[4], kernel_size=(1, 1), groups=ch[4])])
        self.projs_5 = torch.nn.ModuleList(
            [torch.nn.Conv2d(in_channels=ch[5] * t, out_channels=ch[5], kernel_size=(1, 1), groups=ch[5])])
        self.projs = {0: self.projs_0, 1: self.projs_1, 2: self.projs_2, 3: self.projs_3,
                      4: self.projs_4, 5: self.projs_5}

        self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1],  classes=1)
        self.name = "manet-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.buildings_head)
        init.initialize_head(self.roads_head)
        for i in range(6):
            init.initialize_decoder(self.projs[i])
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x1, x2):
        enc_1 = self.encoder(x1)
        enc_2 = self.encoder(x2)
        final_features = []
        for i in range(0, len(enc_1)):
            if self.fuse == 'cat':
                enc_fusion = torch.cat([enc_1[i], enc_2[i]], dim=1)
                final_features.append(self.projs[i][0](enc_fusion))
            elif self.fuse == 'add':
                enc_fusion = enc_1[i] + enc_2[i]
                #final_features.append(self.projs[i][0](enc_fusion))
                final_features.append(enc_fusion)

            elif self.fuse == 'cat_add':
                enc_fusion_1 = torch.cat([enc_1[i], enc_2[i]], dim=1)
                enc_fusion_2 = enc_1[i] + enc_2[i]
                enc_fusion = self.projs[i][0](enc_fusion_1) + enc_fusion_2
                final_features.append(enc_fusion)

        decoder_output = self.decoder(*final_features)
        change_build = self.buildings_head(decoder_output)
        change_road = self.roads_head(decoder_output)
        if self.classification_head is not None:
            cls = self.classification_head(final_features[-1])
            return change_build, change_road, cls
        else:
            return change_build, change_road, None

    def predict(self, x1, x2):
        if self.training:
            self.eval()
        with torch.no_grad():
            seg_b, seg_r,  cls  = self.forward(x1, x2)
        return seg_b, seg_r,  cls


if __name__ == '__main__':
    #model = UnetMhead()
    #print(model(torch.zeros(2,3,32,32), torch.zeros(2,3,32,32))[0].shape)
    model = UnetSiamese_Mhead(encoder_name='vgg13_bn', fuse='cat')
    print(model(torch.zeros(2,3,32,32), torch.zeros(2,3,32,32)))
