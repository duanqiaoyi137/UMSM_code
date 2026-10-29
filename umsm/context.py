"""Run-wide context handed to every module (seed, RNG, profiler, caches, shared resources)."""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field

import numpy as np

from .profiling import NULL_PROFILER


def set_global_seed(seed: int) -> None:
    # 输入：seed；输出：None；同步 Python/NumPy/Torch 随机种子；继承：无（模块函数）。
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


@dataclass
class RunContext:
    seed: int = 42
    project_root: str = "."
    cache_dir: str = "cache"
    embedding_cache: bool = True
    profiler: object = NULL_PROFILER
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("umsm"))
    resources: dict = field(default_factory=dict)

    def __post_init__(self):
        # 输入：实例状态；输出：None；初始化或校验数据对象；继承：RunContext(object)。
        self.rng = np.random.default_rng(self.seed)

    def resolve(self, path: str) -> str:
        # 输入：path；输出：绝对路径字符串；继承：RunContext(object)。
        return path if os.path.isabs(path) else os.path.join(self.project_root, path)

    def reset_rng(self):
        # 输入：实例状态；输出：None；重置模块随机流；继承：RunContext(object)。
        """Start a module from the run seed, independent of preceding methods."""
        set_global_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)

    def shared(self, key, factory):
        # 输入：key、factory；输出：按 key 复用或创建的运行资源；继承：RunContext(object)。
        """Process-wide singleton resources, e.g. a loaded language model."""
        if key not in self.resources:
            self.resources[key] = factory()
        return self.resources[key]
