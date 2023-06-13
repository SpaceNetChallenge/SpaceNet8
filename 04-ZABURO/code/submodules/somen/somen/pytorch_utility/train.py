import cProfile
import io
import logging
import os
import pstats
import re
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import pytorch_pfn_extras as ppe
import torch
from pytorch_pfn_extras.handler import Logic
from pytorch_pfn_extras.runtime import runtime_registry
from pytorch_pfn_extras.training import Trainer
from pytorch_pfn_extras.training.extension import Extension
from pytorch_pfn_extras.training.extensions import (
    LogReport,
    LRScheduler,
    PrintReport,
    ProgressBar,
    observe_lr,
    snapshot,
)
from pytorch_pfn_extras.training.trigger import TriggerLike, get_trigger
from pytorch_pfn_extras.training.triggers import IntervalTrigger
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from somen.pytorch_utility.configs import TrainingConfig
from somen.pytorch_utility.default_regressor import DefaultRegressor
from somen.pytorch_utility.extensions.best_value_snapshot import BestValueSnapshot
from somen.pytorch_utility.extensions.evaluator_ext import EvaluatorExt
from somen.pytorch_utility.extensions.my_lr_scheduler import MyLRScheduler
from somen.pytorch_utility.my_evaluator import MyEvaluator
from somen.pytorch_utility.my_handler import MyHandler
from somen.types import PathLike

_logger = logging.getLogger(__name__)


def train(
    config: TrainingConfig,
    model: nn.Module,
    train_set: Optional[Dataset] = None,
    train_loader: Optional[DataLoader] = None,
    valid_sets: Optional[Union[Dataset, Mapping[str, Dataset]]] = None,
    valid_loaders: Optional[Union[DataLoader, Mapping[str, DataLoader]]] = None,
    working_dir: PathLike = "working/",
    collate_fn: Optional[Callable] = None,
    ext_extensions: Optional[Sequence[Extension]] = None,
) -> nn.Module:
    working_dir = Path(working_dir)

    if config.distributed:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        local_rank: Optional[int] = int(os.environ["LOCAL_RANK"])
        global_rank = torch.distributed.get_rank()
        is_main_process = global_rank == 0
        print(local_rank, global_rank, is_main_process)
    else:
        local_rank = None
        global_rank = None
        is_main_process = True

    # Move to device
    if config.distributed:
        assert local_rank is not None
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(config.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    if config.debug_mode:
        _logger.info("debug mode!!!")
        stop_trigger: Optional[TriggerLike] = (config.debug_iter, "iteration")
        log_trigger: TriggerLike = stop_trigger
        eval_trigger: TriggerLike = stop_trigger
        trainer_snapshot_trigger: TriggerLike = stop_trigger
        model_snapshot_trigger: TriggerLike = stop_trigger
        lr_scheduler_trigger: TriggerLike = stop_trigger
    else:
        stop_trigger = None
        log_trigger = config.log_trigger
        eval_trigger = config.eval_trigger
        trainer_snapshot_trigger = config.trainer_snapshot_trigger
        model_snapshot_trigger = config.model_snapshot_trigger
        lr_scheduler_trigger = config.lr_scheduler_trigger

    if len(config.no_decay_name_patterns) == 0:
        param_groups = model.parameters()
    else:
        no_decay_name_patterns = [re.compile(pattern) for pattern in config.no_decay_name_patterns]
        decay_params = {}
        no_decay_params = {}
        for name, param in model.named_parameters():
            for pattern in no_decay_name_patterns:
                if pattern.match(name):
                    no_decay_params[name] = param
                    break
            else:
                decay_params[name] = param

        _logger.info(f"weight_decay for the following parameters has been set to 0.0: {no_decay_params.keys()}")
        param_groups = [
            {"params": list(no_decay_params.values()), "weight_decay": 0.0},
            {"params": list(decay_params.values())},
        ]

    # Set up optimizer
    if isinstance(config.optimizer, torch.optim.Optimizer):
        optimizer = config.optimizer
    else:
        optimizer = getattr(torch.optim, config.optimizer)(param_groups, **config.optimizer_params)

    # Make a dataloader from train_set
    if train_loader is not None:
        assert train_set is None
    else:
        assert train_set is not None

        if config.train_sampler is None:
            if config.distributed:
                train_sampler: torch.utils.data.Sampler = torch.utils.data.distributed.DistributedSampler(
                    train_set, seed=config.sampler_seed
                )
            else:
                generator = torch.Generator()
                generator.manual_seed(config.sampler_seed)
                train_sampler = torch.utils.data.RandomSampler(train_set, generator=generator)  # type: ignore
        else:
            train_sampler = config.train_sampler

        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=config.pin_memory,
            sampler=train_sampler,
        )
    assert train_loader is not None

    # Setup regressor
    if config.objective is None:
        regressor: nn.Module = model
    else:
        regressor = DefaultRegressor(model, config.objective)

    if config.distributed:
        if config.sync_batch_norm:
            regressor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(regressor)
        regressor = DistributedDataParallel(
            regressor.to(device), device_ids=[device], find_unused_parameters=config.find_unused_parameters
        )

    ppe.to(regressor, device)

    # Setup extensions
    extensions = [] if ext_extensions is None else list(ext_extensions)

    if is_main_process:
        extensions.append(LogReport(trigger=log_trigger, filename="log.jsonl"))
        if config.print_report:
            extensions.append(PrintReport())
        if config.progress_bar:
            extensions.append(ProgressBar())
        extension = observe_lr(optimizer)
        extension.trigger = log_trigger
        extensions.append(extension)

    # LR scheduler
    if config.lr_scheduler is not None or config.lr_scheduler_params is not None:
        lr_scheduler_interval: IntervalTrigger = get_trigger(lr_scheduler_trigger)  # type: ignore
        if lr_scheduler_interval.unit == "iteration":
            max_steps = int(config.nb_epoch * len(train_loader) / lr_scheduler_interval.period)
        else:
            max_steps = int(config.nb_epoch / lr_scheduler_interval.period)

        if config.lr_scheduler is not None and hasattr(torch.optim.lr_scheduler, config.lr_scheduler):
            lr_scheduler_params = {} if config.lr_scheduler_params is None else config.lr_scheduler_params
            if config.lr_scheduler == "OneCycleLR" and "total_steps" not in lr_scheduler_params:
                lr_scheduler_params = {**lr_scheduler_params, "total_steps": max_steps}
            extensions.append(
                LRScheduler(
                    getattr(torch.optim.lr_scheduler, config.lr_scheduler)(optimizer, **lr_scheduler_params),
                    trigger=lr_scheduler_trigger,
                )
            )
        else:
            extensions.append(
                MyLRScheduler(
                    optimizer, max_steps, config.lr_scheduler, config.lr_scheduler_params, lr_scheduler_trigger  # type: ignore
                )
            )

    # Model snapshot
    if config.task_model_snapshot:
        snapshot_ext = snapshot(
            target=model,
            filename="model_iter_{.iteration}",
            n_retains=config.model_snapshot_n_saved,
            autoload=False,
            saver_rank=(0 if config.distributed else None),
        )
        snapshot_ext.trigger = model_snapshot_trigger  # type: ignore
        extensions.append(snapshot_ext)

    # Trainer snapshot and resume
    if config.take_trainer_snapshot:
        snapshot_ext = snapshot(
            n_retains=config.trainer_snapshot_n_saved,
            autoload=config.resume,
            saver_rank=(0 if config.distributed else None),
        )
        snapshot_ext.trigger = trainer_snapshot_trigger  # type: ignore
        extensions.append(snapshot_ext)

    # Build Trainer
    logic = Logic() if config.logic is None else config.logic
    handler = MyHandler(logic, runtime_registry.get_runtime_class_for_device_spec(device)(device, {}), {})
    trainer = Trainer(
        handler,
        evaluator=None,
        models=regressor,
        optimizers=optimizer,
        max_epochs=config.nb_epoch,
        extensions=extensions,
        out_dir=working_dir,
        stop_trigger=stop_trigger,
    )

    if valid_sets is not None:
        if isinstance(valid_sets, Dataset):
            valid_sets = {"": valid_sets}

        if config.distributed:
            # TODO: DistributedEvaluator
            _logger.warning(
                "DistributedEvaluator is still a work in progress. Only the main process will perform the evaluation."
            )

        batch_size_valid = config.batch_size if config.batch_size_valid is None else config.batch_size_valid

        for valid_name, valid_set in valid_sets.items():
            valid_loader = DataLoader(
                valid_set,
                batch_size=batch_size_valid,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=collate_fn,
                pin_memory=config.pin_memory,
            )

            evaluator = MyEvaluator(
                handler,
                models=regressor,
                progress_bar=config.progress_bar,
                metrics=config.metrics,  # type: ignore
                macro_metrics=config.macro_metrics,  # type: ignore
                concatenated_ins=config.macro_metrics_concatenated_ins,
                concatenated_outs=config.macro_metrics_concatenated_outs,
                name=valid_name,
            )
            evaluator_ext = EvaluatorExt(trainer, evaluator, valid_loader, eval_trigger)
            trainer.extend(evaluator_ext)

    if valid_loaders is not None:
        if not isinstance(valid_loaders, dict):
            valid_loaders = {"": valid_loaders}

        if config.distributed:
            # TODO: DistributedEvaluator
            _logger.warning(
                "DistributedEvaluator is still a work in progress. Only the main process will perform the evaluation."
            )

        for valid_name, valid_loader in valid_loaders.items():
            evaluator = MyEvaluator(
                handler,
                models=regressor,
                progress_bar=config.progress_bar,
                metrics=config.metrics,  # type: ignore
                macro_metrics=config.macro_metrics,  # type: ignore
                concatenated_ins=config.macro_metrics_concatenated_ins,
                concatenated_outs=config.macro_metrics_concatenated_outs,
                name=valid_name,
            )
            evaluator_ext = EvaluatorExt(trainer, evaluator, valid_loader, eval_trigger)
            trainer.extend(evaluator_ext)

    if (valid_sets is not None or valid_loaders is not None) and config.save_best_by is not None and is_main_process:
        trainer.extend(BestValueSnapshot(model, working_dir / "bests", config.save_best_by, trigger=eval_trigger))

    if config.enable_cprofile and is_main_process:
        _logger.info("cProfile is enabled.")
        pr = cProfile.Profile()
        pr.enable()

    _logger.info("Start training!")
    trainer.run(train_loader)

    if config.enable_cprofile and is_main_process:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats()
        _logger.info(s.getvalue())

        pr.dump_stats(working_dir / "train.cprofile")

    if config.distributed:
        torch.distributed.barrier()

    return model
