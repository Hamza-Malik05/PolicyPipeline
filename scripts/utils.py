"""
utils.py
────────
Shared utilities: config loading, logger setup, path resolution, timing.
"""

from __future__ import annotations

import logging
import time
import yaml
from pathlib import Path
from functools import wraps
from typing import Any

import colorlog

# ── Root of the project ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = None) -> dict:
    """Load config.yaml from the project root."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(relative: str) -> Path:
    """Resolve a relative path string from config against PROJECT_ROOT."""
    p = PROJECT_ROOT / relative
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logger(name: str, cfg: dict) -> logging.Logger:
    """
    Create a colour-coded logger that writes to both console and file.
    """
    log_cfg  = cfg.get("logging", {})
    level    = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = PROJECT_ROOT / log_cfg.get("file", "logs/pipeline.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    # Console handler — coloured
    fmt_console = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(name)s] %(levelname)s%(reset)s — %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        },
    )
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt_console)
    logger.addHandler(ch)

    # File handler — plain text
    fmt_file = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)

    return logger


def timer(logger: logging.Logger):
    """Decorator that logs wall-clock time for each stage function."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.info(f"▶  Starting: {fn.__name__}")
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"✔  Finished: {fn.__name__}  ({elapsed:.2f}s)")
            return result
        return wrapper
    return decorator


def vram_snapshot(logger: logging.Logger, tag: str = ""):
    """Log current GPU VRAM usage (requires PyTorch + CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.debug(f"VRAM {tag}: {alloc:.2f}/{total:.2f} GB used")
    except Exception:
        pass
