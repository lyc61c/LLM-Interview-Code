"""小张量测试使用单线程，避免 CPU 线程调度掩盖计算耗时。"""

import os

import torch

os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_sessionstart(session):
    torch.set_num_threads(1)
