from __future__ import annotations

import numpy as np

from ..registry import PROPAGATIONS
from ..types import EdgeTable, ScoreSamples
from .base import UncertaintyPropagation, moments


@PROPAGATIONS.register("mean_std")
class MeanStd(UncertaintyPropagation):
    """Raw-score risk baseline: clip(mu - alpha * sigma, 0, 1); alpha >= 0."""

    def __init__(self, alpha: float = 0.5):
        # 输入：alpha；输出：None；初始化实例及校验参数；继承：MeanStd(UncertaintyPropagation)。
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = alpha

    def propagate(self, samples: ScoreSamples, ctx) -> EdgeTable:
        # 输入：samples、ctx；输出：EdgeTable（统计量与最终边证据）；继承：MeanStd(UncertaintyPropagation)，重写父类方法。
        df = moments(samples)
        df["w"] = (df["mu"] - self.alpha * df["sigma"]).clip(0.0, 1.0)
        return EdgeTable(df, samples.columns, meta={**samples.meta, "propagation": self.registry_name})


@PROPAGATIONS.register("expected")
class Expected(MeanStd):
    """Expected score only: w = mu (alpha = 0); on complete data w = s."""

    def __init__(self):
        # 输入：实例状态；输出：None；初始化实例及校验参数；继承：Expected(MeanStd)，重写父类方法。
        super().__init__(alpha=0.0)

    def params(self) -> dict:
        # 输入：实例状态；输出：组件构造参数字典（不包含内部缓存）；继承：Expected(MeanStd)。
        return {}
