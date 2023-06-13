from typing import Optional

from torch import nn


def get_act(act_str: Optional[str]) -> nn.Module:
    if act_str is None:
        return nn.Identity()
    elif act_str == "relu":
        return nn.ReLU()
    elif act_str == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError
