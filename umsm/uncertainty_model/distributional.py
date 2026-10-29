"""Models that output a candidate distribution Q_x = {(v_k, p_k)} for every missing cell."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import UNCERTAINTY_MODELS
from .base import DistributionModel, top_k


def _marginal(series: pd.Series, k: int | None) -> list:
    # 输入：series、k；输出：list；继承：无（模块函数）。
    counts = series.dropna().value_counts(sort=True)
    return top_k(dict(zip(counts.index, counts.to_numpy(float))), k)


@UNCERTAINTY_MODELS.register("column_marginal")
class ColumnMarginal(DistributionModel):
    """Q_x = relative frequencies of the top-K observed values of the same column."""

    def __init__(self, top_k: int | None = None, rho: float = 0.95, k_max: int = 10):
        # 输入：top_k、rho、k_max；输出：None；初始化实例及校验参数；继承：ColumnMarginal(DistributionModel)，重写父类方法。
        super().__init__(rho, k_max)
        if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
            raise ValueError("top_k must be a positive integer or null")
        self.top_k = top_k

    def cell_distributions(self, frame, table, iset, ctx):
        # 输入：frame、table、iset、ctx；输出：缺失位置到候选分布的字典；继承：ColumnMarginal(DistributionModel)，重写父类方法。
        out = {}
        for c in frame.columns:
            miss = np.flatnonzero(frame[c].isna().to_numpy())
            if len(miss) == 0:
                continue
            q = _marginal(frame[c], None)  # truncate only after accounting for all probability mass
            for r in miss:
                out[(int(r), c)] = q
        return out


@UNCERTAINTY_MODELS.register("knn_tuple")
class KnnTuple(DistributionModel):
    """Q_x from the k most similar tuples of the same table (agreement on the other observed attributes).

    similarity(r, s) = #attributes where r and s are both observed and equal / #attributes where
    both are observed. Candidates are the values of the neighbours in the missing column,
    weighted by similarity; the top-K are kept. Cells without any similar tuple fall back to the
    column marginal.
    """

    def __init__(self, k: int = 10, top_k: int | None = None, min_similarity: float = 0.0,
                 rho: float = 0.95, k_max: int = 10):
        # 输入：k、top_k、min_similarity、rho、k_max；输出：None；初始化实例及校验参数；继承：KnnTuple(DistributionModel)，重写父类方法。
        super().__init__(rho, k_max)
        if k < 1 or (top_k is not None and top_k < 1):
            raise ValueError("k and top_k must be positive")
        self.k = k
        self.top_k = top_k
        self.min_similarity = min_similarity

    def cell_distributions(self, frame, table, iset, ctx):
        # 输入：frame、table、iset、ctx；输出：缺失位置到候选分布的字典；继承：KnnTuple(DistributionModel)，重写父类方法。
        cols = list(frame.columns)
        codes = np.stack([pd.factorize(frame[c], use_na_sentinel=True)[0] for c in cols], axis=1)
        observed = codes >= 0
        out = {}
        for ci, c in enumerate(cols):
            miss = np.flatnonzero(~observed[:, ci])
            if len(miss) == 0:
                continue
            donors = np.flatnonzero(observed[:, ci])
            if len(donors) == 0:
                continue
            others = [k for k in range(len(cols)) if k != ci]
            values = frame[c].to_numpy(object)
            fallback = _marginal(frame[c], None)
            d_codes, d_obs = codes[np.ix_(donors, others)], observed[np.ix_(donors, others)]
            for r in miss:
                r_codes, r_obs = codes[r, others], observed[r, others]
                both = d_obs & r_obs
                comparable = both.sum(axis=1)
                agree = ((d_codes == r_codes) & both).sum(axis=1)
                sim = np.where(comparable > 0, agree / np.maximum(comparable, 1), 0.0)
                order = np.argsort(-sim, kind="stable")[: self.k]
                order = order[sim[order] > self.min_similarity]
                if len(order) == 0:
                    out[(int(r), c)] = fallback
                    continue
                weights: dict = {}
                for d in order:
                    v = values[donors[d]]
                    weights[v] = weights.get(v, 0.0) + float(sim[d])
                out[(int(r), c)] = list(weights.items())
        return out
