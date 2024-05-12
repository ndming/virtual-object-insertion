import logging, coloredlogs
from pathlib import Path


def get_logger(name, level='DEBUG') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    coloredlogs.install(level=level, logger=logger, 
        fmt='%(programname)s - %(levelname)s - %(message)s')
    return logger


def log_output(logger: logging.Logger, info: str, file: Path):
    logger.info(
        f"[>] {info}: \033[90m{file.resolve().relative_to(Path.cwd())}\033[0m")


def verify_path(logger: logging.Logger, file: Path):
    if not file.exists():
        relative_path = file.resolve().relative_to(Path.cwd())
        logger.error(f"could not locate: {relative_path}")
        raise FileNotFoundError(relative_path)