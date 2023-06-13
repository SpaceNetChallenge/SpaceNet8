from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from typing_extensions import Literal


@dataclass
class MLPConfig:
    class_name: Literal["MLP"]
    num_dim: int
    cat_sizes: Sequence[int]
    cat_emb_dims: Sequence[int]
    hidden_dim: int
    output_dim: int
    dropout_cat: float
    dropout_hidden: float
    bn: bool


class MLP(nn.Module):
    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config

        self._embs = nn.ModuleList(
            [nn.Embedding(size, dim) for size, dim in zip(config.cat_sizes, config.cat_emb_dims)]
        )
        self._main = nn.Sequential(
            nn.Linear(config.num_dim + sum(config.cat_emb_dims), config.hidden_dim),
            nn.Dropout(config.dropout_hidden),
            nn.BatchNorm1d(config.hidden_dim) if config.bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout_hidden),
            nn.BatchNorm1d(config.hidden_dim) if config.bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, x_num: Tensor, x_cat: Optional[Tensor] = None) -> None:
        if x_cat is None:
            assert len(self.config.cat_sizes) == 0
            x = x_num
        else:
            assert x_num.dim() == x_cat.dim()
            x_cat = torch.cat([emb(x_cat[..., i]) for i, emb in enumerate(self._embs)], dim=-1)
            x_cat = torch.dropout(x_cat, self.config.dropout_cat, self.training)
            x = torch.cat([x_num, x_cat], dim=-1)

        y = self._main(x)
        return y
