"""
samridhi/logger.py
==================
Structured rotating-file logger shared across all modules.

Usage in any module:
    from samridhi.logger import get_logger
    log = get_logger()
    log.info("...")

Outputs:
  - samridhi.log  (rotating, max 10 MB × 3 backups)
  - stderr        (WARNING and above only)
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from samridhi.config import LOG_FILE

_LOG_NAME    = "samridhi"
_MAX_BYTES   = 10 * 1024 * 1024   # 10 MB
_BACKUP_CNT  = 3
_FMT         = "%(asctime)s %(levelname)-8s [%(module)s] %(message)s"


def get_logger() -> logging.Logger:
    """
    Return the shared samridhi logger.
    Idempotent — handlers are attached only once.
    """
    logger = logging.getLogger(_LOG_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(_FMT)

    # Rotating file handler
    try:
        fh = RotatingFileHandler(
            str(LOG_FILE), maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_CNT, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass  # read-only filesystem — continue with console only

    # Console handler (WARNING+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger
