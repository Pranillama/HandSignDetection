"""Centralized logging setup for ASL hand sign detection pipeline.

This module provides a dual-handler logger configuration:
- Console handler at INFO level with a concise format (user-friendly)
- File handler at DEBUG level with a detailed timestamped log

Log files are created under the configured `log_dir` and named using the
pattern: `{logger_name}_{YYYYMMDD_HHMMSS}.log`.

Usage:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
    logger.info("Starting...")
"""

from datetime import datetime
from pathlib import Path
import logging
from typing import Optional


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Configure and return a logger with console and file handlers.

    The function will create the `log_dir` if it does not exist. If the
    directory cannot be created, the underlying OSError will propagate to the
    caller to signal an environmental issue (e.g., permission error).

    Args:
        name: Name for the logger (typically `__name__` from the calling module).
        log_dir: Directory where log files will be written. Defaults to "logs".

    Returns:
        Configured ``logging.Logger`` instance.

    Raises:
        OSError: If the log directory cannot be created.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)  # Let OSError propagate if it fails

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.log"
    file_path = log_path / filename

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding multiple handlers if the logger is configured already
    if not logger.handlers:
        # Console handler (INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

        # File handler (DEBUG)
        fh = logging.FileHandler(str(file_path))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        logger.addHandler(ch)
        logger.addHandler(fh)

    # Prevent log records from propagating to the root logger (avoids duplicate output)
    logger.propagate = False

    return logger
