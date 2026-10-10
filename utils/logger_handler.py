"""日志：控制台 INFO + 文件 DEBUG 双 handler。

沿用你 Agent项目 的 logger_handler 思路，两点改进：
    1. 文件 handler 显式 encoding="utf-8"（原版没加，写中文日志会有隐患）
    2. get_logger() 幂等 —— 重复调用返回同一个 logger，不会叠加 handler

先 import utils.console 是有意的：它会顺手修好全进程的中文编码，
否则 StreamHandler 输出中文日志会乱码。
"""

import logging
import os
from datetime import datetime

import utils.console  # noqa: F401  —— 副作用：修复 stdout/stderr 编码
from utils.path_tool import ensure_dir, get_abs_path

LOG_DIR = ensure_dir("logs")

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "deepresearch") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # 幂等：已经配过就直接返回
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(_FORMAT, datefmt="%H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    log_file = os.path.join(LOG_DIR, f"research_{datetime.now():%Y%m%d}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


if __name__ == "__main__":
    log = get_logger()
    log.debug("这条只进文件")
    log.info("这条同时进控制台和文件 —— 中文测试")
    log.warning("文件位于 %s", LOG_DIR)
