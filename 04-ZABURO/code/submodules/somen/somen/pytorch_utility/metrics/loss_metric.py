import inspect
from typing import Sequence

from pytorch_pfn_extras.training.metrics import Batch as DictBatch

from somen.pytorch_utility.losses import get_loss_fn
from somen.types import SupportedObjectiveLiteral


class LossMetric:
    def __init__(self, objectives: Sequence[SupportedObjectiveLiteral]) -> None:
        self.loss_functions = []
        for objective in objectives:
            f = get_loss_fn(objective)
            keys = inspect.signature(f).parameters.keys()
            self.loss_functions.append((objective, f, keys))

    def __call__(self, ins: DictBatch, outs: DictBatch) -> DictBatch:
        ins = {**ins, **outs}
        ret: DictBatch = {}
        for name, f, keys in self.loss_functions:
            res = f(**{key: value for key, value in ins.items() if key in keys})
            if isinstance(res, dict):
                for key, value in res.items():
                    ret[name + "/" + key] = value
            else:
                ret[name] = res
        return ret
