from typing import Any, Dict, Optional, Sequence

import numpy as np
from pytorch_pfn_extras import reporting
from pytorch_pfn_extras.handler import Handler
from pytorch_pfn_extras.training import Evaluator, Trainer
from torch import Tensor


class MyHandler(Handler):
    def consume_options(self, options: Dict[str, Any]) -> None:
        super().consume_options(options)
        self._eval_report_keys: Optional[Sequence[str]] = options.pop("eval_report_keys", None)
        self._train_report_keys: Optional[Sequence[str]] = options.pop("train_report_keys", None)
        # Consume this argument for backward compatibility
        options.pop("async", False)

    def eval_post_step(
        self,
        evaluator: Evaluator,
        batch_idx: int,
        batch: Any,
        outputs: Any,
    ) -> None:
        """A method called after each evaluation step.

        Args:
            evaluator (Evaluator): The evaluator.
            batch_idx (int): Number of iterations already finished.
            batch (dict of torch.Tensor): Input tensors of this batch.
            complete_fn (callable): A callback function called after
                training step.
        """
        # Context: Evaluator
        # Called after eval_step.
        for _, sm, rt in self._runtime_iterator(evaluator.models):
            rt.eval_post_step(evaluator, sm, batch_idx, batch, outputs)

        for name, out in outputs.items():
            if isinstance(out, Tensor) and out.dim() == 0:
                reporting.report({f"val/{name}": out.item()})
            elif np.isscalar(out):
                reporting.report({f"val/{name}": out})

    def train_post_step(self, trainer: Trainer, batch_idx: int, batch: Any, outputs: Any) -> None:
        """A method called after each training step.

        Args:
            trainer (Trainer): The trainer that calls this method.
            batch_idx (int): Number of iterations
            batch (dict of torch.Tensor): Input tensors of this batch.
            outputs (dict of torch.Tensor): Output tensors of this batch.
        """
        # Context: Trainer
        # Called after train_step.
        for _, sm, rt in self._runtime_iterator(trainer.models):
            rt.train_post_step(trainer, sm, batch_idx, batch, outputs)

        for name, out in outputs.items():
            if isinstance(out, Tensor) and out.dim() == 0:
                reporting.report({f"train/{name}": out.item()})
            elif np.isscalar(out):
                reporting.report({f"train/{name}": out})
