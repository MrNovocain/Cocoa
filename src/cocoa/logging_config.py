import logging
from logging.config import dictConfig

from .config import settings


def setup_logging() -> None:
    """Configure the root logger."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": level,
            }
        },
        "root": {
            "handlers": ["console"],
            "level": level,
        },
    }

    dictConfig(config)


__all__ = ["setup_logging"]
