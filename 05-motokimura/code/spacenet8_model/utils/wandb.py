import os

import dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger


def get_wandb_logger(config: DictConfig,
                     exp_id: int) -> WandbLogger:
    dotenv.load_dotenv()  # load WANDB_API_KEY from .env file
    assert 'WANDB_API_KEY' in os.environ, \
        ('"WANDB_API_KEY" is empty. '
         'Create ".env" file with your W&B API key. '
         'See ".env.sample" for the file format')

    return WandbLogger(
        project='spacenet-8',
        group=f'exp_{exp_id:05d}',
        config=OmegaConf.to_container(config, resolve=True))
