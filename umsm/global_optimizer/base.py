"""Module 4: Global Multi-Table Optimizer   G = (V, E, W, tau) -> Omega."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..types import EdgeTable, Matching


class GlobalOptimizer(ABC):
    """Turns uncertainty-aware edge weights into disjoint multi-table semantic groups.

    Every returned group must contain at most one column per table; only groups with >= 2
    columns are reported (singletons are unmatched columns).
    """

    registry_name = "?"

    @abstractmethod
    def optimize(self, graph: EdgeTable, ctx) -> Matching:
        # 输入：graph、ctx；输出：Matching（互斥且跨表的语义组）；继承：GlobalOptimizer(ABC)。
        ...

    def params(self) -> dict:
        # 输入：实例状态；输出：组件构造参数字典（不包含内部缓存）；继承：GlobalOptimizer(ABC)。
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


def gap_cut(sorted_desc: np.ndarray) -> int:
    # 输入：sorted_desc；输出：最大相邻分差处的截断位置；继承：无（模块函数）。
    """Algorithm 1 of the MTSM paper: index k of the largest gap between consecutive sorted values;
    the first k + 1 elements are kept."""
    if len(sorted_desc) < 2:
        return len(sorted_desc) - 1
    gaps = sorted_desc[:-1] - sorted_desc[1:]
    return int(np.argmax(gaps))  # first occurrence, like the strict '>' of the original loop


def select_edges(ranked: pd.DataFrame, selection: str, threshold: float | None = None,
                 top_k: int | None = None, components: tuple[str, ...] = ("name_score", "value_score"),
                 weight: str = "w") -> tuple[pd.DataFrame, dict]:
    # 输入：ranked、selection、threshold、top_k、components、weight；输出：(筛选后的边表, 选择过程诊断)；继承：无（模块函数）。
    """Candidate edges for the pairwise baselines, kept in descending weight order.

    selection:
        topk_gap  legacy Algorithm 1: K = max over w and the available component scores
        threshold w >= threshold
        top_k     first top_k pairs
        is_match  pairs flagged by the pair-table matcher in at least half of the data states
        all       every pair with w > 0
    """
    info = {"selection": selection}
    if selection == "topk_gap":
        ks = {weight: gap_cut(ranked[weight].to_numpy())}
        for c in components:
            if c in ranked.columns and ranked[c].notna().all():
                ks[c] = gap_cut(np.sort(ranked[c].to_numpy())[::-1])
        k = max(ks.values())
        info.update({"top_k": k + 1, "top_k_by": ks})
        return ranked.iloc[: k + 1], info
    if selection == "threshold":
        if threshold is None:
            raise ValueError("selection='threshold' requires a threshold")
        return ranked[ranked[weight] >= threshold], info
    if selection == "top_k":
        if top_k is None:
            raise ValueError("selection='top_k' requires top_k")
        return ranked.iloc[:top_k], info
    if selection == "is_match":
        if "is_match" not in ranked.columns:
            raise ValueError("selection='is_match' requires a matcher that outputs is_match")
        return ranked[ranked["is_match"].astype(float) >= 0.5], info
    if selection == "all":
        return ranked[ranked[weight] > 0], info
    raise ValueError(f"unknown selection {selection}")
