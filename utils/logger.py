from __future__ import annotations

import logging
from pathlib import Path

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None


class _TqdmStreamHandler(logging.StreamHandler):
    """Write console logs through ``tqdm.write`` so progress bars stay intact."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if tqdm is not None:
                tqdm.write(message)
            else:  # pragma: no cover
                super().emit(record)
        except Exception:  # pragma: no cover
            self.handleError(record)


def build_logger(log_dir: str | Path, name: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    console_formatter = logging.Formatter("%(message)s")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream = _TqdmStreamHandler()
    stream.setFormatter(console_formatter)
    file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger
