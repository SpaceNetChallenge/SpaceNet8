from typing import Dict, List, Optional

from omegaconf import DictConfig, OmegaConf


def load_config(default_cfg_path: str,
                cfg_paths: Optional[List[str]] = [],
                update_dotlist: Optional[List[str]] = None,
                update_dict: Optional[Dict] = None) -> DictConfig:

    config = OmegaConf.load(default_cfg_path)
    for cfg_path in cfg_paths:
        optional_config = OmegaConf.load(cfg_path)
        config = OmegaConf.merge(config, optional_config)
    if update_dotlist is not None:
        update_config = OmegaConf.from_dotlist(update_dotlist)
        config = OmegaConf.merge(config, update_config)

    if update_dict is not None:
        update_config = OmegaConf.create(update_dict)
        config = OmegaConf.merge(config, update_config)

    OmegaConf.set_readonly(config, True)

    return config


def print_config(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
