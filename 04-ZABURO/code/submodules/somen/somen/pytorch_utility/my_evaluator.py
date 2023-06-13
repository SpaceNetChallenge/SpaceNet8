import queue
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from pytorch_pfn_extras import reporting
from pytorch_pfn_extras.handler import BaseHandler
from pytorch_pfn_extras.reporting import Observation
from pytorch_pfn_extras.training._evaluator import Evaluator, _progress_bar
from pytorch_pfn_extras.training.metrics import Batch as DictBatch
from pytorch_pfn_extras.training.metrics import MetricType
from torch import nn


class MyEvaluator(Evaluator):
    def __init__(
        self,
        handler: BaseHandler,
        models: Union[nn.Module, Mapping[str, nn.Module]],
        *,
        progress_bar: bool = False,
        metrics: Optional[Sequence[MetricType]] = None,
        macro_metrics: Optional[Sequence[MetricType]] = None,
        concatenated_ins: Optional[Sequence[Tuple[str, int]]] = None,
        concatenated_outs: Optional[Sequence[Tuple[str, int]]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(handler, models, progress_bar=progress_bar, metrics=metrics)
        self._macro_metrics = [] if macro_metrics is None else macro_metrics
        self._concatenated_ins = [("target", 0)] if concatenated_ins is None else concatenated_ins
        self._concatenated_outs = [("output", 0)] if concatenated_outs is None else concatenated_outs
        self._prefix = "" if name is None or name == "" else f"{name}/"

        self._ins_list: Optional[List[DictBatch]] = None
        self._outs_list: Optional[List[DictBatch]] = None

    def _complete_step(self, idx: int, outs: DictBatch, *, is_deferred: bool = False) -> None:
        c_idx = self._idxs.get()
        # Asure that iterations complete in order
        if c_idx != idx:
            raise RuntimeError(
                "Completed a not expected iteration. "
                "{} was expected but completion of {} happened".format(c_idx, idx)
            )
        x = self._inputs.get()
        observed = self._observed.get()

        # Store x and outs
        if len(self._macro_metrics) > 0:
            if self._outs_list is None or self._ins_list is None:
                raise RuntimeError("self._outs_list or self._ins_list is not initialized. Call self.run first.")

            self._ins_list.append({key: x[key].detach().cpu() for key, _ in self._concatenated_ins if key in x})
            self._outs_list.append({key: outs[key].detach().cpu() for key, _ in self._concatenated_outs if key in outs})

        with self._reporter.scope(observed):
            outs = self._process_metrics(x, outs)

            if len(self._macro_metrics) > 0 and (idx + 1) == self._eval_len:
                assert self._outs_list is not None and self._ins_list is not None
                macro_ins = {
                    key: torch.cat([x[key] for x in self._ins_list], dim=dim) for key, dim in self._concatenated_ins
                }
                macro_outs = {
                    key: torch.cat([x[key] for x in self._outs_list], dim=dim) for key, dim in self._concatenated_outs
                }
                for metric in self._macro_metrics:
                    outs.update(metric(macro_ins, macro_outs))

            self.handler.eval_post_step(self, idx, x, outs)

        self._summary.add(observed)
        self._update(idx)
        # On the last iteration, close the progress bar
        if self._idxs.qsize() == 0:
            self._pbar.__exit__(None, None, None)

    def run(self, loader: Iterable[Any], *, eval_len: Optional[int] = None) -> None:
        """Executes the evaluation loop.

        Args:
            loader (torch.utils.data.DataLoader):
                A data loader for evaluation.
            eval_len (int, optional):
                The number of iterations per one evaluation epoch.
        """
        if len(self._macro_metrics) > 0:
            self._ins_list = []
            self._outs_list = []

        # Note: setup_manager is done by the Trainer.
        self._idxs: "queue.Queue[int]" = queue.Queue()
        self._inputs: "queue.Queue[DictBatch]" = queue.Queue()
        self._observed: "queue.Queue[Observation]" = queue.Queue()

        if eval_len is None:
            eval_len = len(loader)  # type: ignore[arg-type]
        self._eval_len = eval_len

        self._summary = reporting.DictSummary()
        observation: Observation = {}
        self.handler.eval_loop_begin(self)
        self._pbar = _progress_bar("validation", self._progress_bar, eval_len)
        self._update = self._pbar.__enter__()
        loader_iter = iter(loader)
        with torch.no_grad():  # type: ignore[no-untyped-call]
            for idx in range(eval_len):
                try:
                    x = next(loader_iter)
                except StopIteration:
                    break
                self._idxs.put(idx)
                self._inputs.put(x)
                self._observed.put(observation)
                with self._reporter.scope(observation):
                    self.handler.eval_step(self, idx, x, self._complete_step)
                # Some of the DataLoaders might need an explicit break
                # since they could start cycling on their data
                if (idx + 1) == eval_len:
                    break
        # This will report to the trainer main reporter
        self.handler.eval_loop_end(self)
        reporting.report({f"{self._prefix}{key}": value for key, value in self._summary.compute_mean().items()})

        self._ins_list = None
        self._outs_list = None
