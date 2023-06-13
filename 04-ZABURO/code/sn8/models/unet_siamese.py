from typing import List, Optional, Union

import segmentation_models_pytorch.base.modules as md
import torch
import torch.nn as nn
from segmentation_models_pytorch.base.initialization import initialize_decoder
from segmentation_models_pytorch.decoders.unet.model import Unet


class FuseBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            2 * in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=2 * in_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetSiamese(Unet):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            aux_params=aux_params,
        )

        self.fuse = FuseBlock(decoder_channels[-1], decoder_channels[-1], decoder_use_batchnorm, decoder_attention_type)
        initialize_decoder(self.fuse)

        self.name = "u-siamese-{}".format(encoder_name)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=0)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        x1, x2 = decoder_output.reshape(2, -1, *decoder_output.shape[1:])
        decoder_output = self.fuse(x1, x2)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
