from typing import Any, Iterable

from pytorch_pfn_extras.training import Evaluator, Trainer
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol
from pytorch_pfn_extras.training.extension import PRIORITY_WRITER, Extension
from pytorch_pfn_extras.training.trigger import TriggerLike


class EvaluatorExt(Extension):

    priority: int = PRIORITY_WRITER

    def __init__(
        self, trainer: Trainer, evaluator: Evaluator, val_loader: Iterable[Any], trigger: TriggerLike = (1, "epoch")
    ) -> None:
        self.needs_model_state = True
        self.trigger = trigger
        self._trainer = trainer
        self._evaluator = evaluator
        self._val_loader = val_loader

        evaluator.handler.eval_setup(evaluator, val_loader)

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        evaluator = self._evaluator
        evaluator.handler.train_validation_begin(self._trainer, evaluator)
        evaluator.run(self._val_loader)
        evaluator.handler.train_validation_end(self._trainer, evaluator)
