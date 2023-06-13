import operator
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol
from pytorch_pfn_extras.training.extension import Extension
from pytorch_pfn_extras.training.trigger import TriggerLike


class BestValueSnapshot(Extension):
    def __init__(self, obj: Any, output_dir: Path, by: Sequence[Tuple[str, bool]], trigger: TriggerLike = (1, "epoch")):
        self.obj = obj
        self.output_dir = output_dir
        self.by = by
        self.trigger = trigger

        self._best_values = {key: -float("inf") if maximize else float("inf") for key, maximize in by}
        self._compare_ops = {key: operator.gt if maximize else operator.lt for key, maximize in by}

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        observation = manager.observation
        for key in self._best_values.keys():
            if key in observation:
                value = observation[key]
                if self._compare_ops[key](value, self._best_values[key]):
                    self._best_values[key] = float(value)  # type: ignore
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self.obj.state_dict(), self.output_dir / (key.replace("/", "_") + "_best.pt"))

    def state_dict(self) -> Dict[str, Any]:
        return {"_best_values": self._best_values}

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        self._best_values = to_load["_best_values"]
