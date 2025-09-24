from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from rich import reconfigure
from rich.logging import RichHandler
from rich.theme import Theme

_CONFIGURED: bool = False


def _level_from_env() -> int:
    lvl = os.getenv("TAB_LOG_LEVEL", "INFO").upper()
    return getattr(logging, lvl, logging.INFO)


def configure_logging(level: Optional[int] = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Gruvbox hard dark color palette
    gruvbox_theme = Theme(
        {
            "logging.level.debug": "#83a598",  # gruvbox blue
            "logging.level.info": "#b8bb26",  # gruvbox green
            "logging.level.warning": "#fabd2f",  # gruvbox yellow
            "logging.level.error": "#fb4934",  # gruvbox red
            "logging.level.critical": "bold #fb4934 on #1d2021",  # bold red on dark bg
            "logging.keyword": "#d3869b",  # gruvbox purple
            "logging.string": "#b8bb26",  # gruvbox green
        }
    )

    reconfigure(theme=gruvbox_theme)

    # Configure root logger to catch all messages
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
        # Rich handles formatting internally
        formatter = logging.Formatter(fmt="%(message)s", datefmt="[%X]")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.setLevel(level if level is not None else _level_from_env())

    _CONFIGURED = True


essential_get_logger = logging.getLogger


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return essential_get_logger(name)