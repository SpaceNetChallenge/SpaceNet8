# TODO: 出典を明記
# https://www.kaggle.com/c/lish-moa/discussion/202256
# https://www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/274970

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from typing_extensions import Literal

from somen.pytorch_utility.models.get_act import get_act


@dataclass
class _ConvBlockConfig:
    out_channels: int
    kernel_size: int
    dropout: float
    bias: bool = True
    norm_dim: Optional[int] = None
    activation: Optional[str] = "relu"

    def get_conv(self, in_channels: int) -> Tuple[nn.Module, int]:
        act = get_act(self.activation)
        conv = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Dropout(self.dropout),
            nn.utils.weight_norm(
                nn.Conv1d(in_channels, self.out_channels, self.kernel_size, padding="same", bias=self.bias),
                dim=self.norm_dim,  # type: ignore
            ),
            act,
        )
        return conv, self.out_channels


@dataclass
class Tabular1DCNNConfig:
    # example: example/optiver/1st_place/configs/1dcnn.yaml
    class_name: Literal["Tabular1DCNN"]
    num_dim: int
    cat_sizes: Sequence[int]
    cat_emb_dims: Sequence[int]
    input_dropout: float
    input_channel: int
    input_length: int
    input_celu: bool
    conv1: _ConvBlockConfig
    conv2: _ConvBlockConfig
    conv3: Optional[_ConvBlockConfig]
    conv4: Optional[_ConvBlockConfig]
    dense_dropout: float
    dense_act: Optional[str]
    output_dim: int


class Tabular1DCNN(nn.Module):
    def __init__(self, config: Tabular1DCNNConfig) -> None:
        super().__init__()
        self.config = config

        self._embs = nn.ModuleList(
            [nn.Embedding(size, dim) for size, dim in zip(config.cat_sizes, config.cat_emb_dims)]
        )

        x_dim = config.num_dim + sum(config.cat_emb_dims)
        self._tabular_to_sequence = nn.Sequential(
            nn.BatchNorm1d(x_dim),
            nn.Dropout(config.input_dropout),
            nn.utils.weight_norm(nn.Linear(x_dim, config.input_channel * config.input_length), dim=None),  # type: ignore
            nn.CELU(0.06) if config.input_celu else nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(config.input_channel, config.input_length)),
        )

        in_channels, in_length = config.input_channel, config.input_length

        self._conv1, in_channels = config.conv1.get_conv(in_channels)
        self._pool1 = nn.AdaptiveAvgPool1d(in_length // 2)
        in_length = in_length // 2

        self._conv2, in_channels = config.conv2.get_conv(in_channels)

        self._conv3: Optional[nn.Module]
        if config.conv3 is not None:
            self._conv3, in_channels = config.conv3.get_conv(in_channels)
        else:
            self._conv3 = None

        self._conv4: Optional[nn.Module]
        if config.conv4 is not None:
            self._conv4, in_channels = config.conv4.get_conv(in_channels)
        else:
            self._conv4 = None

        self._pool2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        in_length = int((in_length + 2 * 1 - 1 * (4 - 1) - 1) / 2 + 1)

        assert in_length >= 1

        self._flatten = nn.Flatten()
        in_channels = in_channels * in_length

        self._dense = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Dropout(config.dense_dropout),
            nn.utils.weight_norm(nn.Linear(in_channels, config.output_dim), dim=0),
            get_act(config.dense_act),
        )

    def forward(self, x_num: Tensor, x_cat: Optional[Tensor] = None) -> Tensor:
        assert x_num.dim() == 2

        if x_cat is None:
            assert len(self.config.cat_sizes) == 0
            x = x_num
        else:
            assert x_num.dim() == x_cat.dim()
            x_cat = torch.cat([emb(x_cat[..., i]) for i, emb in enumerate(self._embs)], dim=-1)
            x = torch.cat([x_num, x_cat], dim=-1)

        x = self._tabular_to_sequence(x)
        x = x.reshape(x.shape[0], self.config.input_channel, self.config.input_length)

        x = self._conv1(x)
        x = self._pool1(x)
        x = self._conv2(x)
        x_skip = x

        if self._conv3 is not None:
            x = self._conv3(x)
        if self._conv4 is not None:
            x = self._conv4(x)

        if x is not x_skip:
            x = x * x_skip

        x = self._pool2(x)
        x = self._flatten(x)

        x = self._dense(x)
        return x
