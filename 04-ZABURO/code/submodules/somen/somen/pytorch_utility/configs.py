from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
from pytorch_pfn_extras.handler import BaseLogic
from pytorch_pfn_extras.training.trigger import TriggerLike

from somen.pytorch_utility.metrics.metric import Metric
from somen.types import DeviceLike, LossFnType, SupportedObjectiveLiteral


@dataclass
class TrainingConfig:
    # optimizer
    optimizer: Union[torch.optim.Optimizer, str] = "Adam"
    optimizer_params: Mapping[str, Any] = field(default_factory=dict)
    no_decay_name_patterns: Sequence[str] = field(default_factory=list)

    # lr scheduler
    lr_scheduler: Optional[str] = None
    lr_scheduler_params: Optional[Mapping[str, Any]] = None
    lr_scheduler_trigger: TriggerLike = (1, "epoch")

    # device
    device: DeviceLike = "cuda"

    # settings about distributed data parallel
    distributed: bool = False
    sync_batch_norm: bool = True
    find_unused_parameters: bool = True

    # data loader
    batch_size: int = 128
    batch_size_valid: Optional[int] = None  # If None, same as batch_size
    num_workers: int = 0
    pin_memory: bool = True
    sampler_seed: int = 0

    # trainer
    objective: Optional[Union[SupportedObjectiveLiteral, LossFnType]] = "mse"
    logic: Optional[BaseLogic] = None
    train_sampler: Optional[torch.utils.data.Sampler] = None
    progress_bar: bool = False

    # evaluator
    metrics: Optional[Sequence[Metric]] = None
    macro_metrics: Optional[Sequence[Metric]] = None
    macro_metrics_concatenated_ins: Optional[Sequence[Tuple[str, int]]] = None
    macro_metrics_concatenated_outs: Optional[Sequence[Tuple[str, int]]] = None
    save_best_by: Optional[Sequence[Tuple[str, bool]]] = (("val/loss", False),)
    eval_trigger: TriggerLike = (1, "epoch")

    # log report
    log_trigger: TriggerLike = (1, "epoch")
    print_report: bool = True

    # trainer snapshot
    take_trainer_snapshot: bool = True
    trainer_snapshot_trigger: TriggerLike = (1, "epoch")
    trainer_snapshot_n_saved: int = 1
    task_model_snapshot: bool = False
    model_snapshot_trigger: TriggerLike = (1, "epoch")
    model_snapshot_n_saved: int = -1

    # resume
    resume: bool = False

    # number of epochs
    nb_epoch: int = 100

    # debug settings
    debug_mode: bool = False
    debug_iter: int = 50
    enable_cprofile: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, str) and len(self.optimizer_params) > 0:
            # TODO: warning
            pass
