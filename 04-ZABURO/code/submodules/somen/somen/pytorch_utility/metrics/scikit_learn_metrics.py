import collections.abc
import functools
from typing import Any, Mapping, Sequence, Union

import torch
from pytorch_pfn_extras.training.metrics import Batch as DictBatch
from sklearn import metrics as sklean_metrics


class ScikitLearnMetrics:
    def __init__(
        self,
        metrics: Union[str, Sequence[str], Mapping[str, Mapping[str, Any]]],
        label_key: str = "target",
        output_key: str = "output",
    ) -> None:
        if isinstance(metrics, str):
            name = metrics
            metric_functions = {name: getattr(sklean_metrics, name)}
        elif isinstance(metrics, collections.abc.Sequence):
            metric_functions = {name: getattr(sklean_metrics, name) for name in metrics}
        else:
            metric_functions = {
                name: functools.partial(getattr(sklean_metrics, name), **kwargs) for name, kwargs in metrics.items()
            }
        self.metric_functions = metric_functions
        self.label_key = label_key
        self.output_key = output_key

    @torch.no_grad()
    def __call__(self, ins: DictBatch, outs: DictBatch) -> DictBatch:
        y_true = ins[self.label_key].detach().cpu().numpy()
        y_pred = outs[self.output_key].detach().cpu().numpy()
        return {name: f(y_true, y_pred) for name, f in self.metric_functions.items()}


class ScikitLearnProbMetrics(ScikitLearnMetrics):
    @torch.no_grad()
    def __call__(self, ins: DictBatch, outs: DictBatch) -> DictBatch:
        y_true = ins[self.label_key].detach().cpu().numpy()
        y_pred_logit = outs[self.output_key]

        if y_pred_logit.shape[-1] == 1:
            y_pred = torch.sigmoid(y_pred_logit).detach().cpu().numpy()
        else:
            y_pred = torch.softmax(y_pred_logit, dim=-1).detach().cpu().numpy()

        return {name: f(y_true, y_pred) for name, f in self.metric_functions.items()}
