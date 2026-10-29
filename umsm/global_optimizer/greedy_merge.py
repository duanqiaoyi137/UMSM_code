"""Pairwise baseline of the original MTSM paper (Algorithm 1 TopK + Algorithm 2 greedy merge),
ported from group_edges_by_conditions() in matching/matching.py of Matching-main."""
from __future__ import annotations

from ..registry import OPTIMIZERS
from ..types import EdgeTable, Matching
from .base import GlobalOptimizer, select_edges


def greedy_groups(edges) -> list[set[str]]:
    # 输入：edges；输出：贪心合并后的列集合列表；继承：无（模块函数）。
    """Algorithm 2: visit edges by global rank; attach a column to the first group containing
    its partner unless that group already holds a column of the same table or the column was
    merged before; otherwise open a new group. Existing groups are never merged."""
    groups: list[set[str]] = []
    table_of: dict[str, str] = {}
    merged: set[str] = set()
    for i, j, ti, tj in edges:
        table_of[i], table_of[j] = ti, tj
        attached = False
        for g in groups:
            if i in g:
                if j not in merged and all(table_of[v] != tj for v in g):
                    g.add(j)
                    merged.add(j)
                attached = True
                break
            if j in g:
                if i not in merged and all(table_of[v] != ti for v in g):
                    g.add(i)
                    merged.add(i)
                attached = True
                break
        if not attached:
            groups.append({i, j})
            merged.update((i, j))
    return groups


@OPTIMIZERS.register("greedy_merge")
class GreedyMerge(GlobalOptimizer):
    """Legacy MTSM baseline: TopK-gap edge selection + greedy pairwise merge (Algorithms 1-2)."""

    def __init__(self, selection: str = "topk_gap", threshold: float | None = None, top_k: int | None = None,
                 components: list[str] | None = None):
        # 输入：selection、threshold、top_k、components；输出：None；初始化实例及校验参数；继承：GreedyMerge(GlobalOptimizer)。
        self.selection = selection
        self.threshold = threshold
        self.top_k = top_k
        self.components = components if components is not None else ["name_score", "value_score"]

    def optimize(self, graph: EdgeTable, ctx) -> Matching:
        # 输入：graph、ctx；输出：Matching（互斥且跨表的语义组）；继承：GreedyMerge(GlobalOptimizer)，重写父类方法。
        ranked = graph.sorted()
        edges, info = select_edges(ranked, self.selection, self.threshold, self.top_k, tuple(self.components))
        groups = greedy_groups(edges[["col_i", "col_j", "table_i", "table_j"]].itertuples(index=False, name=None))
        info["n_edges"] = len(edges)
        return Matching([frozenset(g) for g in groups], info)
