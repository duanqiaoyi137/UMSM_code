"""Evaluation of a multi-table matching against the ground truth.

cluster_*  the metric of the original paper (Sec. 4.1.2, evaluate.py of Matching-main), kept
           unchanged for comparability. It does not penalise splitting one true group into
           several predicted groups (each piece gets full precision and their recalls add up).
pair_*     precision / recall / F1 over the set of column pairs placed in the same group.
bcubed_*   B-cubed precision / recall / F1 over the columns occurring in a predicted or a
           ground-truth group (unlisted columns are singletons).
"""
from __future__ import annotations

from itertools import combinations

from .types import EdgeTable, Matching


def _f1(p: float, r: float) -> float:
    # 输入：p、r；输出：F1 浮点数；继承：无（模块函数）。
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def legacy_cluster_precision_recall(pred: list[frozenset[str]], gt: list[frozenset[str]]) -> tuple[float, float]:
    # 输入：pred、gt；输出：(旧版整组 precision, recall)；继承：无（模块函数）。
    precision_list, recall_list = [], []
    for a in pred:
        best, best_size = None, 0
        for t in gt:
            inter = len(a & t)
            if inter > best_size:
                best, best_size = t, inter
        if best:
            precision_list.append(best_size / len(a))
            recall_list.append(best_size / len(best))
    p = sum(precision_list) / len(pred) if pred else 0.0
    r = sum(recall_list) / len(gt) if gt else 0.0
    return p, r


def _pairs(groups: list[frozenset[str]]) -> set[frozenset[str]]:
    # 输入：groups；输出：组内全部无向列对集合；继承：无（模块函数）。
    return {frozenset(p) for g in groups for p in combinations(sorted(g), 2)}


def pairwise_precision_recall(pred: list[frozenset[str]], gt: list[frozenset[str]]) -> tuple[float, float]:
    # 输入：pred、gt；输出：(Pair precision, recall)；继承：无（模块函数）。
    pp, gp = _pairs(pred), _pairs(gt)
    tp = len(pp & gp)
    return (tp / len(pp) if pp else 0.0), (tp / len(gp) if gp else 0.0)


def bcubed_precision_recall(pred: list[frozenset[str]], gt: list[frozenset[str]]) -> tuple[float, float]:
    # 输入：pred、gt；输出：(B-cubed precision, recall)；继承：无（模块函数）。
    pred_of = {c: g for g in pred for c in g}
    gt_of = {c: g for g in gt for c in g}
    items = set(pred_of) | set(gt_of)
    if not items:
        return 0.0, 0.0
    p = r = 0.0
    for c in items:
        pg = pred_of.get(c, frozenset([c]))
        tg = gt_of.get(c, frozenset([c]))
        inter = len(pg & tg)
        p += inter / len(pg)
        r += inter / len(tg)
    return p / len(items), r / len(items)


def evaluate(matching: Matching, groundtruth: list[frozenset[str]]) -> dict:
    # 输入：matching、groundtruth；输出：完整匹配指标字典；继承：无（模块函数）。
    pred = [g for g in matching.groups if len(g) >= 2]
    gt = [g for g in groundtruth if len(g) >= 2]
    out = {"n_pred_groups": len(pred), "n_gt_groups": len(gt),
           "n_pred_columns": sum(len(g) for g in pred), "n_gt_columns": sum(len(g) for g in gt)}
    for name, fn in [("cluster", legacy_cluster_precision_recall), ("pair", pairwise_precision_recall),
                     ("bcubed", bcubed_precision_recall)]:
        p, r = fn(pred, gt)
        out[f"{name}_precision"], out[f"{name}_recall"], out[f"{name}_f1"] = p, r, _f1(p, r)
    out["exact_group_matches"] = len(set(pred) & set(gt))
    return out


METRIC_KEYS = [f"{m}_{k}" for m in ("cluster", "pair", "bcubed") for k in ("precision", "recall", "f1")]


def validate_matching(matching: Matching, graph: EdgeTable, where: str) -> None:
    # 输入：matching、graph、where；输出：None；分组非法时抛异常；继承：无（模块函数）。
    table_of = {c.id: c.table for c in graph.columns}
    seen = set()
    for g in matching.groups:
        unknown = [c for c in g if c not in table_of]
        if unknown:
            raise ValueError(f"{where}: optimizer returned unknown columns {unknown[:3]}")
        tables = [table_of[c] for c in g]
        if len(set(tables)) != len(tables):
            raise ValueError(f"{where}: group {sorted(g)} has two columns of the same table")
        if seen & g:
            raise ValueError(f"{where}: groups are not disjoint ({sorted(seen & g)})")
        seen |= g
