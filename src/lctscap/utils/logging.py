"""Logging utilities: logger setup, metric logging, and seed management."""

import logging
import os
import random
import sys
from typing import Any, Dict, Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a named logger.

    Creates a logger with a stream handler (stdout) and optionally a
    file handler.  Uses a standardized format with timestamps.

    Args:
        name: logger name.
        log_file: optional path to a log file.
        level: logging level (default: INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        from lctscap.utils.io import ensure_dir
        from pathlib import Path

        ensure_dir(str(Path(log_file).parent))
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_metrics(
    metrics: Dict[str, Any],
    step: int,
    logger: logging.Logger,
    wandb_run: Optional[Any] = None,
) -> None:
    """Log metrics to both a logger and optionally to Weights & Biases.

    Args:
        metrics: dictionary of metric name -> value.
        step: current training step or epoch.
        logger: Logger instance for file/console output.
        wandb_run: optional wandb run object.  If provided, metrics are
                   also logged to W&B.
    """
    # Format for text logging
    parts = [f"step={step}"]
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            parts.append(f"{key}={value:.6f}")
        else:
            parts.append(f"{key}={value}")
    logger.info(" | ".join(parts))

    # Log to wandb if available
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch.

    Args:
        seed: integer seed value.
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    os.environ["PYTHONHASHSEED"] = str(seed)
