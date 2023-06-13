import argparse
import os
import shutil
import ssl

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging as SWA
from pytorch_lightning.loggers import TensorBoardLogger

# isort: off
from spacenet8_model.datasets import get_dataloader
from spacenet8_model.models import get_model
from spacenet8_model.utils.ema import EMA
from spacenet8_model.utils.config import load_config
from spacenet8_model.utils.wandb import get_wandb_logger
# isort: on

ssl._create_default_https_context = ssl._create_unverified_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        choices=['foundation', 'flood', 'foundation_xdxd_sn5', 'foundation_selimsef_xview2', 'flood_selimsef_xview2'],
        required=True
    )
    parser.add_argument(
        '--exp_id',
        type=int,
        default=99999
    )
    parser.add_argument(
        '--fold_id',
        type=int,
        default=0
    )
    parser.add_argument(
        '--pretrained',
        type=int,
        default=-1,
        help='exp_id of pretrained siamese branch'
    )
    parser.add_argument(
        '--pretrained_path'
    )
    parser.add_argument(
        '--config',
        default=None,
        help='YAML config path. This will overwrite `configs/default.yaml`')
    parser.add_argument(
        '--dry', action='store_true', help='dry-run mode')
    parser.add_argument(
        '--disable_wandb', action='store_true', help='disable W&B logger')
    parser.add_argument(
        '--override_model_dir')
    parser.add_argument(
        '--artifact_dir',
        default='/wdata'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='overwrite configs (e.g., General.fp16=true, etc.)')
    return parser.parse_args()


def get_default_cfg_path(task: str) -> str:
    if task == 'foundation':
        return 'configs/defaults/foundation.yaml'
    elif task == 'flood':
        return 'configs/defaults/flood.yaml'
    elif task == 'foundation_xdxd_sn5':
        return 'configs/defaults/foundation_xdxd_sn5.yaml'
    elif task == 'foundation_selimsef_xview2':
        return 'configs/defaults/foundation_selimsef_xview2.yaml'
    elif task == 'flood_selimsef_xview2':
        return 'configs/defaults/flood_selimsef_xview2.yaml'
    else:
        raise ValueError(task)


def main() -> None:
    args = parse_args()

    default_cfg_path: str = get_default_cfg_path(args.task)
    config: DictConfig = load_config(
        default_cfg_path,
        [] if args.config is None else [args.config],
        update_dotlist=args.opts,
        update_dict={
            'task': args.task,
            'exp_id': args.exp_id,
            'fold_id': args.fold_id,
            'pretrained': args.pretrained})

    seed_everything(config.General.seed + config.fold_id * 5555)

    model_dir = os.path.join(args.artifact_dir, 'models') if (args.override_model_dir is None) else args.override_model_dir
    output_dir = '_dry' if args.dry else f'exp_{args.exp_id:05d}'
    output_dir = os.path.join(model_dir, output_dir)
    if args.dry:
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=False)
    print(f'going to save training results under {output_dir}')

    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best',
        save_weights_only=False,
        save_top_k=1,
        monitor=config.General.val_metric_to_monitor,
        mode='max',
        save_last=True)
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    callbacks = [checkpoint_callback, lr_monitor_callback]

    assert (not config.General.enable_ema) or (not config.General.enable_swa)
    if config.General.enable_ema:
        print(f'enable EMA with momentum={config.General.ema_momentum}')
        callbacks.append(EMA(decay=1-config.General.ema_momentum))
    if config.General.enable_swa:
        print(f'enable SWA with lr={config.General.swa_lr} and epoch_start={config.General.swa_epoch_start}')
        callbacks.append(SWA(swa_epoch_start=config.General.swa_epoch_start, swa_lrs=config.General.swa_lr))

    loggers = [TensorBoardLogger(output_dir, name=None)]
    if (not args.dry) and (not args.disable_wandb):
        loggers.append(get_wandb_logger(config, args.exp_id))

    trainer = Trainer(
        max_epochs=2 if args.dry else config.General.epochs,
        callbacks=callbacks,
        logger=loggers,
        precision=16 if config.General.fp16 else 32,
        amp_backend=config.General.amp_backend,
        amp_level=config.General.amp_level,
        deterministic=config.General.deterministic,
        benchmark=config.General.benchmark,
        check_val_every_n_epoch=config.General.check_val_every_n_epoch,
        auto_select_gpus=False,
        default_root_dir=os.getcwd(),
        gpus=config.General.gpus,
        limit_train_batches=2 if args.dry else 1.0,
        limit_val_batches=2 if args.dry else 1.0,
    )

    model = get_model(config, model_dir, pretrained_exp_id=args.pretrained, pretrained_path=args.pretrained_path)

    train_dataloader = get_dataloader(config, is_train=True)
    val_dataloader = get_dataloader(config, is_train=False)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == '__main__':
    main()
