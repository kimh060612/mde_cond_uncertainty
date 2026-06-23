from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import wandb


class RankFilter(logging.Filter):
    """LogRecord에 distributed rank 정보를 추가한다."""

    def __init__(self, rank: int = 0) -> None:
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        return True


def setup_logger(
    name: str = "train",
    *,
    log_dir: Optional[str | Path] = None,
    filename: str = "train.log",
    level: int = logging.INFO,
    rank: int = 0,
    console_rank_zero_only: bool = True,
    file_rank_zero_only: bool = True,
    max_file_size_mb: int = 20,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Deep Learning 학습용 logger를 생성한다.

    Parameters
    ----------
    name:
        logger 이름.
    log_dir:
        로그 파일을 저장할 디렉터리. None이면 파일 로그를 만들지 않는다.
    filename:
        로그 파일 이름.
    level:
        logging.DEBUG, logging.INFO 등의 로그 레벨.
    rank:
        DDP 프로세스 rank. 단일 GPU 학습에서는 0.
    console_rank_zero_only:
        True이면 rank 0만 terminal에 출력한다.
    file_rank_zero_only:
        True이면 rank 0만 파일에 기록한다.
    max_file_size_mb:
        로그 파일 하나의 최대 크기.
    backup_count:
        rotation된 이전 로그 파일 개수.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 상위 root logger로 메시지가 다시 전달되어 중복 출력되는 것을 방지
    logger.propagate = False

    # Jupyter나 반복 호출 환경에서 handler 중복 추가 방지
    if logger.handlers:
        return logger

    rank_filter = RankFilter(rank)

    console_formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "rank=%(rank)d | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(name)s | "
            "rank=%(rank)d | "
            "%(filename)s:%(lineno)d | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Terminal handler
    if not console_rank_zero_only or rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(rank_filter)
        logger.addHandler(console_handler)

    # File handler
    if log_dir is not None and (not file_rank_zero_only or rank == 0):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_dir / filename,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(rank_filter)
        logger.addHandler(file_handler)

    return logger

def wandb_log_prefixed(
    wandb_run: wandb.Run, 
    prefix, 
    metrics, 
    step
):
    if wandb_run is None:
        return

    wandb_run.log(
        {
            f"{prefix}/{key}": value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        },
        step=step,
    )