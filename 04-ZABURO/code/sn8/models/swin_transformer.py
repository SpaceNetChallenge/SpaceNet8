import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock
from segmentation_models_pytorch.encoders import encoders
from segmentation_models_pytorch.encoders._base import EncoderMixin
from timm.models.swin_transformer import SwinTransformer, window_partition
from torch import Tensor, nn


def _mod_corner_pad(x: Tensor, mods: Union[int, Tuple[int, ...]], dims: Tuple[int, ...] = (2, 3)) -> Tensor:
    if isinstance(mods, int):
        mods = (mods,) * len(dims)

    pads = [0] * (2 * x.dim())
    for i, mod in zip(dims, mods):
        pads[2 * i + 1] = (mod - x.shape[i] % mod) % mod
    return F.pad(x, pads)


class SwinTransformerBaseEncoder(SwinTransformer, EncoderMixin):
    def __init__(self, depth: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._depth = depth

        self._out_channels = [self.patch_embed.proj.in_channels, self.embed_dim]
        self.norm0 = nn.LayerNorm(self.embed_dim)

        for i, layer in enumerate(self.layers):
            # attn_mask は入力された画像の解像度に合わせて実行時に作成する
            # state_dict の対象から除くため一度削除して単なる attribute として持つ
            for block in layer.blocks:
                attn_mask = block.attn_mask
                delattr(block, "attn_mask")
                block.attn_mask = attn_mask

            out_channel = self.embed_dim * 2 ** i
            self._out_channels.append(out_channel)
            self.add_module(f"norm{i}", nn.LayerNorm(out_channel))

    @property
    def out_channels(self):
        return self._out_channels[: self._depth + 1]

    def get_stages(self):
        raise NotImplementedError

    def _get_partial_patch_embed(self, x: Tensor) -> Tensor:
        # avg_pool2d(f(x), 2) == self.patch_embed.proj となるような関数 f
        assert self.patch_embed.patch_size == (4, 4)
        outs = []
        for i in range(0, 4, 2):
            for j in range(0, 4, 2):
                outs.append(
                    F.conv2d(
                        x[..., i:, j:],
                        self.patch_embed.proj.weight[..., i : i + 2, j : j + 2] * 4,
                        self.patch_embed.proj.bias,
                        stride=4,
                    )
                )
        outs = torch.stack(outs, dim=2)
        outs = F.pixel_shuffle(outs.view(outs.shape[0], -1, *outs.shape[-2:]), 2)
        return outs

    def _make_attn_mask(self, H: int, W: int, window_size: int, shift_size: int) -> Tensor:
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        cnt = 0
        for h in (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        ):
            for w in (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            ):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, window_size)  # num_win, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: Tensor) -> Sequence[Tensor]:
        features = [x]
        if len(features) == self._depth + 1:
            return features

        x = _mod_corner_pad(x, self.patch_embed.patch_size)
        x = self._get_partial_patch_embed(x)
        features.append(self.norm0(x.transpose(1, 3)).transpose(1, 3))
        if len(features) == self._depth + 1:
            return features

        x = F.avg_pool2d(x, 2)
        resolution = x.shape[-2:]
        if self.patch_embed.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.patch_embed.norm(x)

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed

        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            for block in layer.blocks:
                if resolution[0] % block.window_size != 0 or resolution[1] % block.window_size != 0:
                    x = x.reshape(x.shape[0], *resolution, x.shape[-1])
                    x = _mod_corner_pad(x, block.window_size, dims=(1, 2))
                    resolution = x.shape[1:3]
                    x = x.reshape(x.shape[0], -1, x.shape[-1])

                if block.shift_size > 0 and block.input_resolution != resolution:
                    attn_mask = self._make_attn_mask(resolution[0], resolution[0], block.window_size, block.shift_size)
                    block.attn_mask = attn_mask.to(x.device)

                block.input_resolution = resolution
                x = block(x)

            norm = getattr(self, f"norm{i}")
            features.append(norm(x).reshape(x.shape[0], *resolution, x.shape[-1]).permute(0, 3, 1, 2))

            if len(features) == self._depth + 1:
                return features

            if layer.downsample is not None:
                if resolution[0] % 2 != 0 or resolution[1] % 2 != 0:
                    x = x.reshape(x.shape[0], *resolution, x.shape[-1])
                    x = _mod_corner_pad(x, 2, dims=(1, 2))
                    resolution = x.shape[1:3]
                    x = x.reshape(x.shape[0], -1, x.shape[-1])

                layer.downsample.input_resolution = resolution
                x = layer.downsample(x)
                resolution = (resolution[0] // 2, resolution[1] // 2)

        return features


swin_encoders = {
    "swin_tiny_patch4_window7_224": {
        "encoder": SwinTransformerBaseEncoder,
        "pretrained_settings": {},
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            "patch_size": 4,
            "window_size": 7,
            "embed_dim": 96,
            "depths": (2, 2, 6, 2),
            "num_heads": (3, 6, 12, 24),
        },
    },
    "swin_small_patch4_window7_224": {
        "encoder": SwinTransformerBaseEncoder,
        "pretrained_settings": {},
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            "patch_size": 4,
            "window_size": 7,
            "embed_dim": 96,
            "depths": (2, 2, 18, 2),
            "num_heads": (3, 6, 12, 24),
        },
    },
}
encoders.update(swin_encoders)


def _patch_unet_decoder_block_to_be_scale_adaptive() -> None:
    # Unet に入力される encoder の解像度が2の倍数でなくても動くようにする

    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        if skip is None or x.shape[-2] * 2 == skip.shape[-2] and x.shape[-1] * 2 == skip.shape[-1]:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear")

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

    DecoderBlock.forward = forward
    warnings.warn("segmentation_models_pytorch.decoders.unet.decoder.DecoderBlock was patched!")


_patch_unet_decoder_block_to_be_scale_adaptive()
