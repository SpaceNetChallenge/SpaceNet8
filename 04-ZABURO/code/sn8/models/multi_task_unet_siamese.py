from typing import List, Optional, Tuple, Union

import segmentation_models_pytorch.base.modules as md
import torch
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base.initialization import initialize_decoder, initialize_head
from segmentation_models_pytorch.decoders.unet.model import Unet
from torch import nn

from sn8.models.unet_siamese import FuseBlock


class BranchBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__(
            md.Conv2dReLU(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            ),
            md.Attention(attention_type, in_channels=2 * in_channels),
            md.Conv2dReLU(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            ),
            md.Attention(attention_type, in_channels=out_channels),
        )


class MultiTaskUnetSiamese(Unet):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: Tuple[int, int, int] = (8, 2, 1),
        activation: Optional[Union[str, callable]] = None,
        insert_branch_block: bool = True,
        flood_grad_mul: float = 1.0,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=1,
            activation=activation,
            aux_params=None,
        )
        delattr(self, "segmentation_head")

        if insert_branch_block:
            self.road_branch = BranchBlock(
                decoder_channels[-1], decoder_channels[-1], decoder_use_batchnorm, decoder_attention_type
            )
            self.building_branch = BranchBlock(
                decoder_channels[-1], decoder_channels[-1], decoder_use_batchnorm, decoder_attention_type
            )
            initialize_decoder(self.road_branch)
            initialize_decoder(self.building_branch)
        else:
            self.road_branch = nn.Identity()
            self.building_branch = nn.Identity()

        self.flood_fuse = FuseBlock(
            decoder_channels[-1], decoder_channels[-1], decoder_use_batchnorm, decoder_attention_type
        )
        initialize_decoder(self.flood_fuse)

        self.road_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes[0],
            activation=activation,
            kernel_size=3,
        )
        self.building_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes[1],
            activation=activation,
            kernel_size=3,
        )
        self.flood_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes[2],
            activation=activation,
            kernel_size=3,
        )
        initialize_head(self.road_head)
        initialize_head(self.building_head)
        initialize_head(self.flood_head)

        self.name = "u-multitask-siamese-{}".format(encoder_name)
        self.flood_grad_mul = flood_grad_mul

    def forward(self, x1, x2=None):
        if x2 is None:
            features = self.encoder(x1)
            decoder_output = self.decoder(*features)

            road_masks = self.road_head(self.road_branch(decoder_output))
            building_masks = self.building_head(self.building_branch(decoder_output))

            # Dummy calculation for find_unused_parameters in DDP
            flood_masks = self.flood_head(self.flood_fuse(decoder_output[:0], decoder_output[:0]))
        else:
            x = torch.cat([x1, x2], dim=0)
            features = self.encoder(x)
            decoder_output = self.decoder(*features)

            x1, x2 = decoder_output.reshape(2, -1, *decoder_output.shape[1:])

            road_masks = self.road_head(self.road_branch(x1))
            building_masks = self.building_head(self.building_branch(x1))

            if self.flood_grad_mul == 0.0:
                x1, x2 = x1.detach(), x2.detach()
            elif self.flood_grad_mul != 1.0:
                x1 = self.flood_grad_mul * x1 + (1 - self.flood_grad_mul) * x1.detach()
                x2 = self.flood_grad_mul * x2 + (1 - self.flood_grad_mul) * x2.detach()
            flood_masks = self.flood_head(self.flood_fuse(x1, x2))

        return {"road_output": road_masks, "building_output": building_masks, "flood_output": flood_masks}
