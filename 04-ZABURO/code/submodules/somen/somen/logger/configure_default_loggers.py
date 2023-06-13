import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml

_default_config_path = Path(__file__).parent / "_default_logging_config.yaml"
_logger = logging.getLogger(__name__)


def configure_default_loggers(log_file_path: Optional[Path] = None) -> None:
    with _default_config_path.open("r") as fp:
        config_dict = yaml.safe_load(fp)

    if log_file_path is not None:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict["handlers"].update(
            {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": str(log_file_path),
                    "formatter": "plain-formatter",
                    "level": "DEBUG",
                }
            }
        )
        config_dict["root"]["handlers"].append("file")
        config_dict["loggers"]["__main__"]["handlers"].append("file")
        config_dict["loggers"]["somen"]["handlers"].append("file")

    logging.config.dictConfig(config_dict)
    _logger.info("Default Logger is successfully configured.")
