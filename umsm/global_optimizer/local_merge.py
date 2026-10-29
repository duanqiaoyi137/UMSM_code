"""Local merge baseline of the MTSM paper's case study (Sec. 4.2.4).

1. Local decisions: every table pair is decided independently. With local_decision='is_match'
   the matcher's own decision is used; with 'hungarian' a 1:1 assignment per table pair is taken
   and pairs below min_weight are dropped.
2. Columns linked by accepted correspondences form a graph; connected components are explored
   by BFS starting from the highest-weight edge. A column whose table is already present in
   the current component is removed from it (columns visited later lose).
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..registry import OPTIMIZERS
from ..types import EdgeTable, Matching
from .base import GlobalOptimizer


def hungarian_per_table_pair(df, min_weight: float, weight: str = "w"):
    # 输入：df、min_weight、weight；输出：允许未匹配的逐表对最优指派边表；继承：无（模块函数）。
    """Maximum signed evidence 1:1 matching, allowing either column to stay unmatched."""
    keep = []
    for _, block in df.groupby(["table_i", "table_j"], sort=False):
        rows = {c: k for k, c in enumerate(dict.fromkeys(block["col_i"]))}
        cols = {c: k for k, c in enumerate(dict.fromkeys(block["col_j"]))}
        m = np.full((len(rows), len(cols)), -np.inf)
        idx = {}
        for r in block.itertuples():
            m[rows[r.col_i], cols[r.col_j]] = getattr(r, weight)
            idx[(rows[r.col_i], cols[r.col_j])] = r.Index
        # One private dummy per source column permits a partial assignment. Thresholding
        # a forced assignment afterwards is wrong: below-threshold edges can displace a
        # good edge. With weight='a' and threshold 0 this uses the ILP logit evidence.
        gain = np.where(np.isfinite(m) & (m > min_weight), m - min_weight, -np.inf)
        augmented = np.concatenate([gain, np.zeros((len(rows), len(rows)))], axis=1)
        ri, ci = linear_sum_assignment(augmented, maximize=True)
        keep += [idx[(a, b)] for a, b in zip(ri, ci) if b < len(cols) and gain[a, b] > 0]
    return df.loc[sorted(keep)]


@OPTIMIZERS.register("local_merge")
class LocalMerge(GlobalOptimizer):
    """Pair-table decisions merged by connected components (BFS) with same-table conflict removal."""

    def __init__(self, local_decision: str = "hungarian", min_weight: float = 0.5,
                 weight: str = "w"):
        # 输入：local_decision、min_weight、weight；输出：None；初始化实例及校验参数；继承：LocalMerge(GlobalOptimizer)。
        if local_decision not in ("hungarian", "is_match"):
            raise ValueError("local_decision must be 'hungarian' or 'is_match'")
        self.local_decision = local_decision
        self.min_weight = min_weight
        if weight not in ("w", "a"):
            raise ValueError("weight must be 'w' or 'a'")
        if not np.isfinite(min_weight):
            raise ValueError("min_weight must be finite")
        self.weight = weight

    def optimize(self, graph: EdgeTable, ctx) -> Matching:
        # 输入：graph、ctx；输出：Matching（互斥且跨表的语义组）；继承：LocalMerge(GlobalOptimizer)，重写父类方法。
        df = graph.df
        if self.weight not in df or not np.isfinite(df[self.weight]).all():
            raise ValueError(f"local_merge requires finite {self.weight} weights")
        if self.local_decision == "is_match":
            if "is_match" not in df.columns:
                raise ValueError("local_decision='is_match' requires a matcher that outputs is_match")
            edges = df[df["is_match"].astype(float) >= 0.5]
        else:
            edges = hungarian_per_table_pair(df, self.min_weight, self.weight)
        edges = edges.sort_values(self.weight, ascending=False, kind="mergesort")

        table_of, adj = {}, defaultdict(list)
        for r in edges.itertuples(index=False):
            table_of[r.col_i], table_of[r.col_j] = r.table_i, r.table_j
            adj[r.col_i].append((getattr(r, self.weight), r.col_j))
            adj[r.col_j].append((getattr(r, self.weight), r.col_i))
        for v in adj:
            adj[v].sort(key=lambda x: -x[0])

        assigned: set[str] = set()
        groups = []
        for r in edges.itertuples(index=False):
            if r.col_i in assigned or r.col_j in assigned:
                continue
            comp, tables_in = [], set()
            queue, seen = deque([r.col_i, r.col_j]), {r.col_i, r.col_j}
            while queue:
                v = queue.popleft()
                if v in assigned or table_of[v] in tables_in:
                    continue
                comp.append(v)
                tables_in.add(table_of[v])
                assigned.add(v)
                for _, u in adj[v]:
                    if u not in seen:
                        seen.add(u)
                        queue.append(u)
            if len(comp) >= 2:
                groups.append(frozenset(comp))
            else:
                assigned.difference_update(comp)
        return Matching(groups, {"local_decision": self.local_decision, "weight": self.weight,
                                 "min_weight": self.min_weight, "n_local_edges": len(edges)})
