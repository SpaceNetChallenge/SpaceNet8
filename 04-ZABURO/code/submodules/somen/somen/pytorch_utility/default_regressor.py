import inspect
from typing import AbstractSet, Any, Mapping, Union

from pytorch_pfn_extras.training.metrics import Batch as DictBatch
from torch import Tensor, nn

from somen.pytorch_utility.losses import get_loss_fn
from somen.types import LossFnType, SupportedObjectiveLiteral


class DefaultRegressor(nn.Module):
    def __init__(self, model: nn.Module, objective: Union[SupportedObjectiveLiteral, LossFnType]) -> None:
        super().__init__()
        self.model = model
        self.model_keys = inspect.signature(self.model.forward).parameters.keys()

        if callable(objective):
            self.loss_fn: LossFnType = objective
        else:
            self.loss_fn = get_loss_fn(objective)

        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn_keys: AbstractSet[str] = inspect.signature(self.loss_fn.forward).parameters.keys()  # type: ignore
        else:
            self.loss_fn_keys = inspect.signature(self.loss_fn).parameters.keys()

    def forward(self, **kwargs: Mapping[str, Any]) -> DictBatch:
        model_ins = {key: value for key, value in kwargs.items() if key in self.model_keys}
        outs: Union[Tensor, DictBatch] = self.model(**model_ins)
        if isinstance(outs, Tensor):
            outs = {"output": outs}
        assert isinstance(outs, dict)

        loss_fn_ins: DictBatch = {
            key: value for key, value in {**kwargs, **outs}.items() if key in self.loss_fn_keys  # type: ignore
        }
        loss_outs = self.loss_fn(**loss_fn_ins)
        if isinstance(loss_outs, Tensor):
            outs["loss"] = loss_outs
        else:
            assert isinstance(loss_outs, dict)
            outs.update(loss_outs)

        return outs
