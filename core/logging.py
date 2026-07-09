import logging
import sys
from pathlib import Path

from config import LOGS_DIR


# ==========================================================
# Log Directory
# ==========================================================

LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Formatter
# ==========================================================

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)s | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ==========================================================
# Setup Logger
# ==========================================================

def get_logger(name: str) -> logging.Logger:
    """
    Create a configured logger.

    Parameters
    ----------
    name
        Logger name (usually __name__).

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    # Console Handler

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler

    file_handler = logging.FileHandler(
        LOGS_DIR / "app.log",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger
