from pathlib import Path
from typing import Callable, Tuple, Union

import torch
from pytorch_pfn_extras.training.metrics import Batch as DictBatch
from torch import Tensor
from typing_extensions import Literal

PathLike = Union[str, Path]
DeviceLike = Union[torch.device, str]
IntervalType = Tuple[int, Literal["epoch", "iteration"]]
SupportedObjectiveLiteral = Literal["mse", "nll", "rmspe", "mae"]
LossFnType = Union[Callable[..., Tensor], Callable[..., DictBatch]]
