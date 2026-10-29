"""Cheap, Q-independent blocking; union directional Top-L followed by full closure."""
from __future__ import annotations

from itertools import combinations
import time
import numpy as np
import pandas as pd

from .global_optimizer.ilp import build_candidate_graph, get_components
from .embedding import DEFAULT_MODEL, get_encoder


def blocking_scores(source, ctx, spec):
    # 输入：source、ctx、spec；输出：含 col_i、col_j、w 的 blocking 边表；继承：无（模块函数）。
    """Headers and optional bounded OBSERVED values; never call a base matcher."""
    encoder = get_encoder(ctx, model=spec.get("model", DEFAULT_MODEL))
    names = encoder.encode_map([c.name for c in source.columns], namespace="names")
    eta = float(spec.get("eta", 1.))
    if not np.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("blocking eta must be in [0,1]")
    observed = {}
    if eta < 1:
        cap = spec.get("max_observed", 32)
        if not isinstance(cap, int) or cap < 1:
            raise ValueError("max_observed must be positive")
        for t in source.tables:
            for c in t.columns:
                values = t.df[c.name].dropna().astype(str).drop_duplicates().iloc[:cap].tolist()
                if values:
                    vectors = encoder.encode_map(values, namespace="blocking_values")
                    observed[c.id] = np.mean([vectors[v] for v in values], axis=0)
    def cosine(a, b):
        # 输入：a、b；输出：余弦相似度 float；继承：无（blocking_scores 内部函数）。
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.clip(np.dot(a, b) / denom, -1, 1)) if denom else 0.
    rows = []
    for ta, tb in combinations(source.tables, 2):
        for a in ta.columns:
            for b in tb.columns:
                score = (cosine(names[a.name], names[b.name]) + 1) / 2
                if a.id in observed and b.id in observed:
                    score = eta * score + (1 - eta) * (cosine(observed[a.id], observed[b.id]) + 1) / 2
                rows.append({"col_i": a.id, "col_j": b.id, "w": score})
    return pd.DataFrame(rows, columns=["col_i", "col_j", "w"])


def plan_candidates(source, ctx, spec):
    # 输入：source、ctx、spec；输出：(closure 列对集合, blocking 诊断)；继承：无（模块函数）。
    """Raw tables -> (allowed closure pairs, diagnostics); no input data is modified."""
    started = time.perf_counter()
    mode = spec.get("mode", "name_top_l")
    columns = source.columns
    n_possible = sum(len(a.columns) * len(b.columns) for a, b in combinations(source.tables, 2))
    if mode not in ("all", "name_top_l", "observed_top_l"):
        raise ValueError("candidates.mode must be name_top_l, observed_top_l (alias), or all")
    top_l = None if mode == "all" else spec.get("top_l", 10)
    if top_l is not None and (not isinstance(top_l, int) or top_l < 1):
        raise ValueError("candidates.top_l must be a positive integer or null")
    if mode == "all":
        df = pd.DataFrame([{"col_i": a.id, "col_j": b.id, "w": 1.}
                           for a, b in combinations(columns, 2) if a.table != b.table],
                          columns=["col_i", "col_j", "w"])
    else:
        df = blocking_scores(source, ctx, spec)
    cutoff = float(spec.get("min_score", 0.0))
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("candidates.min_score must be in [0, 1]")
    candidate = build_candidate_graph(df[df.w >= cutoff], 0.0, top_l, positive_only=False)
    components = get_components([c.id for c in columns], candidate)
    owner = {c.id: c.table for c in columns}
    closure = {frozenset((a, b)) for component in components for a, b in combinations(component, 2)
               if owner[a] != owner[b]}
    covered = {c for comp in components for c in comp}
    all_components = components + [[c.id] for c in columns if c.id not in covered]
    sizes = [len(comp) for comp in all_components]
    pairs = {frozenset((a, b)) for a, b in zip(candidate.col_i, candidate.col_j)}
    truth = {frozenset((a, b)) for group in (source.groundtruth or []) for a, b in combinations(group, 2)
             if a in owner and b in owner and owner[a] != owner[b]}
    info = {"mode": mode, "top_l": top_l, "min_score": cutoff,
            "n_possible_pairs": n_possible, "n_candidate_pairs": len(candidate),
            "n_evaluated_pairs": len(closure), "n_components": len(components),
            "max_component_columns": max(sizes, default=0),
            "n_components_with_singletons": len(all_components), "component_sizes": sizes,
            "mean_component_columns": float(np.mean(sizes)) if sizes else 0.,
            "recall_at_l": len(truth & pairs) / len(truth) if truth else None,
            "component_recall": len(truth & closure) / len(truth) if truth else None,
            "blocking_time_s": time.perf_counter() - started, "components": all_components,
            "blocking_edges": [{"col_i": r.col_i, "col_j": r.col_j, "b": float(r.w),
                                "candidate": frozenset((r.col_i, r.col_j)) in pairs}
                               for r in candidate.itertuples(index=False)]}
    return None if mode == "all" else closure, info


def compact_info(info):
    # 输入：info；输出：移除大数组的 blocking 诊断字典；继承：无（模块函数）。
    return {k: v for k, v in info.items() if k not in ("blocking_edges", "components")}
