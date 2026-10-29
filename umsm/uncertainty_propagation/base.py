"""Module 3: Uncertainty Propagation   (Q_i, Q_j, S_ij) -> (mu_ij, sigma_ij, w_ij)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..types import EdgeTable, ScoreSamples

KEY = ["col_i", "col_j"]
_NOT_PROPAGATED = {"col_i", "col_j", "table_i", "table_j", "state_i", "state_j", "sample_id", "prob", "score"}


class UncertaintyPropagation(ABC):
    """Combines the state probabilities with the scores of every state pair into edge evidence."""

    registry_name = "?"

    @abstractmethod
    def propagate(self, samples: ScoreSamples, ctx) -> EdgeTable:
        # 输入：samples、ctx；输出：EdgeTable（统计量与最终边证据）；继承：UncertaintyPropagation(ABC)。
        ...

    def params(self) -> dict:
        # 输入：实例状态；输出：组件构造参数字典（不包含内部缓存）；继承：UncertaintyPropagation(ABC)。
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


def moments(samples: ScoreSamples) -> pd.DataFrame:
    # 输入：samples；输出：含均值、方差和标准差的 DataFrame；继承：无（模块函数）。
    """MC mean and unbiased sample variance; weighted population moments for exact toy references.

    Other numeric features are propagated as expectations; pair order stays unchanged.
    """
    df = samples.df
    if df.empty:
        return pd.DataFrame(columns=["col_i", "col_j", "table_i", "table_j", "mu", "sigma", "variance",
                                     "n_states", "n_samples", "estimator", "edge_type"])
    g = df.groupby(KEY, sort=False)
    p = df["prob"].to_numpy(float) / g["prob"].transform("sum").to_numpy(float)
    s = df["score"].to_numpy(float)
    work = pd.DataFrame({"col_i": df["col_i"], "col_j": df["col_j"], "_ps": p * s})
    extra = [c for c in df.columns if c not in _NOT_PROPAGATED
             and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
    for c in extra:
        work[c] = p * df[c].astype(float).to_numpy()
    wg = work.groupby(KEY, sort=False)
    out = wg.sum()
    mu = out.pop("_ps")
    dev = s - mu.reindex(pd.MultiIndex.from_frame(df[KEY])).to_numpy()
    var = pd.Series(p * dev ** 2, index=pd.MultiIndex.from_frame(df[KEY])).groupby(level=[0, 1], sort=False).sum()
    first = g[["table_i", "table_j"]].first()
    idx = first.index
    counts = g.size().reindex(idx).to_numpy()
    estimator = samples.meta.get("estimator", "exact")
    if estimator == "mc":
        uniform = 1.0 / g["prob"].transform("size").to_numpy(float)
        if not np.allclose(p, uniform, atol=1e-12):
            raise ValueError("Monte Carlo score samples must have equal weights")
        # Empirical E[(S-mu)^2] uses 1/R; V1 requests the unbiased 1/(R-1) estimate.
        correction = counts / np.maximum(counts - 1, 1)
        var = var.reindex(idx) * correction
    res = first.assign(mu=mu.reindex(idx).to_numpy(),
                       sigma=np.sqrt(np.maximum(var.reindex(idx).to_numpy(), 0.0)),
                       variance=np.maximum(var.reindex(idx).to_numpy(), 0.0),
                       n_states=counts, n_samples=counts,
                       estimator=np.where(counts == 1, "deterministic", estimator))
    incomplete = set(samples.meta.get("incomplete_columns", []))
    types = [int(a in incomplete) + int(b in incomplete) for a, b in idx]
    res["edge_type"] = [["complete_complete", "complete_incomplete", "incomplete_incomplete"][n] for n in types]
    for c in extra:
        res[c] = out[c].reindex(idx).to_numpy()
    return res.reset_index()
