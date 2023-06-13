from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol
from pytorch_pfn_extras.training.extension import Extension
from pytorch_pfn_extras.training.trigger import TriggerLike
from typing_extensions import Literal


class MyLRScheduler(Extension):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_steps: int,
        lr_scheduler: Optional[Literal["cos", "exp"]] = None,
        lr_scheduler_params: Optional[Mapping[str, Any]] = None,
        trigger: TriggerLike = (1, "epoch"),
    ) -> None:
        if lr_scheduler_params is None:
            lr_scheduler_params = {}

        self.trigger = trigger

        self._optimizer = optimizer
        self._default_lrs: Sequence[float] = [param_group["lr"] for param_group in optimizer.param_groups]
        self._max_steps = max_steps + lr_scheduler_params.get("step_offset", 0)
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_params = lr_scheduler_params

        self._step = 0

    def _set_lr(self, coef: float) -> None:
        for param_group, default_lr in zip(self._optimizer.param_groups, self._default_lrs):
            param_group["lr"] = coef * default_lr

    def _get_coef(self, cur_step: int) -> float:
        if self._lr_scheduler is None:
            return 1.0

        if self._lr_scheduler == "cos":
            return 0.5 * (1.0 + np.cos(np.pi * cur_step / self._max_steps))

        if self._lr_scheduler == "exp":
            return self._lr_scheduler_params.get("gamma", 0) ** cur_step

        raise RuntimeError(f"self._lr_scheduler is invalid: {self._lr_scheduler}")

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        self._step += 1

        cur_step = self._step + self._lr_scheduler_params.get("step_offset", 0)
        coef = self._get_coef(cur_step)

        warmup_steps = self._lr_scheduler_params.get("warmup_steps", 0)
        if cur_step < warmup_steps:
            alpha = cur_step / warmup_steps
            warmup_factor = self._lr_scheduler_params.get("warmup_factor", 0.1) * (1.0 - alpha) + alpha
            coef *= warmup_factor

        self._set_lr(coef)

    def state_dict(self) -> Dict[str, Any]:
        return {"_step": self._step}

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        self._step = to_load["_step"]
