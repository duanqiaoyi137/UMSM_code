"""Candidate distributions: generator-local temperatures, learned mixture, then truncation.

Generators return compatible raw scores, never probabilities. Fitting is separate from
inference; pseudo-masks are created only from the explicitly supplied calibration split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from ..registry import UNCERTAINTY_MODELS
from ..calibration import check_inference_split, load_artifact
from .base import CellDistribution, DistributionModel, object_frame
from .distributional import ColumnMarginal, KnnTuple


def compatible_scores(values, kind):
    # 输入：values、kind；输出：方向统一后的 NumPy 兼容分数数组；继承：无（模块函数）。
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("generator scores must be finite")
    if kind in ("similarity", "confidence"):
        return values
    if kind in ("distance", "rank"):
        return -values
    if kind in ("frequency", "density"):
        if (values <= 0).any():
            raise ValueError("frequency/density scores must be positive")
        return np.log(values)
    raise ValueError(f"unknown score kind: {kind}")


# Extension point: factory(params) -> generator implementing generate(frame, table, iset, ctx).
# Output: {(row_position, column): [(candidate, direction-normalized raw score), ...]}.
GENERATORS = {}


class VoteGenerator:
    def __init__(self, model):
        # 输入：model；输出：None；初始化实例及校验参数；继承：VoteGenerator(object)。
        self.model = model

    def generate(self, frame, table, iset, ctx):
        # 输入：frame、table、iset、ctx；输出：缺失位置到候选原始分数的字典；继承：VoteGenerator(object)。
        raw = self.model.cell_distributions(frame, table, iset, ctx)
        # Frequency and neighbour similarity votes are positive evidence counts.
        return {key: [(v, float(np.log(w))) for v, w in q if w > 0] for key, q in raw.items()}


GENERATORS.update({"column_marginal": lambda p: VoteGenerator(ColumnMarginal(**p)),
                   "knn_tuple": lambda p: VoteGenerator(KnnTuple(**p))})
DEFAULT_GENERATORS = [{"name": "knn_tuple", "params": {"k": 10}}, {"name": "column_marginal"}]


def generate_candidates(specs, frame, table, iset, ctx):
    # 输入：specs、frame、table、iset、ctx；输出：按生成器排列的候选分数字典列表；继承：无（模块函数）。
    return [GENERATORS[s["name"]](s.get("params", {})).generate(frame, table, iset, ctx) for s in specs]


def build_qx(generators, temperatures, weights):
    # 输入：generators、temperatures、weights；输出：(混合分布, 可用生成器权重)；继承：无（模块函数）。
    """Union supports; unavailable generators receive zero weight, others renormalize."""
    if len(generators) != len(temperatures) or len(generators) != len(weights):
        raise ValueError("generator/temperature/weight lengths differ")
    t, alpha = np.asarray(temperatures, float), np.asarray(weights, float)
    if (t <= 0).any() or not np.isfinite(t).all() or (alpha < 0).any() or not np.isfinite(alpha).all():
        raise ValueError("temperatures must be positive; mixture weights finite and nonnegative")
    active = np.array([bool(q) for q in generators])
    alpha = alpha * active
    if not active.any():
        return [], []
    if alpha.sum() <= 0:
        raise ValueError("active generators must have positive total mixture weight")
    alpha /= alpha.sum()
    merged = {}
    for q, temperature, weight in zip(generators, t, alpha):
        if not q:
            continue
        if len({v for v, _ in q}) != len(q):
            raise ValueError("each generator must aggregate duplicate candidates")
        scores = np.array([s for _, s in q], float)
        if not np.isfinite(scores).all():
            raise ValueError("generator scores must be finite")
        for (v, _), p in zip(q, softmax(scores / temperature)):
            merged[v] = merged.get(v, 0.) + float(weight * p)
    return list(merged.items()), alpha.tolist()


@UNCERTAINTY_MODELS.register("calibrated_mixture")
class CalibratedMixture(DistributionModel):
    """Frozen temperature-calibrated generator mixture; fitted only by explicit training phases."""

    def __init__(self, calibration: str | dict, generators=None, rho=.95, k_max=10):
        # 输入：calibration、generators、rho、k_max；输出：None；初始化实例及校验参数；继承：CalibratedMixture(DistributionModel)，重写父类方法。
        super().__init__(rho, k_max)
        self.calibration = calibration
        self.generators = DEFAULT_GENERATORS if generators is None else generators
        if not self.generators:
            raise ValueError("at least one candidate generator is required")

    def parameters(self, ctx, iset=None):
        # 输入：ctx、iset；输出：冻结的 Q 温度、权重和生成器配置；继承：CalibratedMixture(DistributionModel)。
        artifact = (load_artifact(self.calibration, ctx) if iset is None
                    else check_inference_split(self.calibration, iset, ctx))
        q = artifact["qx"]
        if q["generators"] != self.generators:
            raise ValueError("qx generator specification differs from fitted calibration")
        return q

    def cell_distributions(self, frame, table, iset, ctx):
        # 输入：frame、table、iset、ctx；输出：缺失位置到候选分布的字典；继承：CalibratedMixture(DistributionModel)，重写父类方法。
        p = self.parameters(ctx, iset)
        proposals = generate_candidates(self.generators, frame, table, iset, ctx)
        out = {}
        for key in set().union(*(g.keys() for g in proposals)):
            local = [g.get(key, []) for g in proposals]
            q, alpha = build_qx(local, p["temperatures"], p["weights"])
            out[key] = CellDistribution(q, {"active_weights": alpha, "temperatures": p["temperatures"],
                                            "generator_candidate_counts": [len(g) for g in local]})
        return out

    def estimate(self, iset, ctx):
        # 输入：iset、ctx；输出：UncertainData（不修改原表）；继承：CalibratedMixture(DistributionModel)，重写父类方法。
        result = super().estimate(iset, ctx)
        result.info["fitted_qx"] = self.parameters(ctx)
        return result


def pseudo_mask_examples(sets, specs, ctx, max_cells=500, seed=42):
    # 输入：sets、specs、ctx、max_cells、seed；输出：隐藏观测值及无泄漏候选的诊断样本列表；继承：无（模块函数）。
    """One hidden cell per example. Neither frame, Table nor IntegrationSet exposes its y.

    Max cells applies over the whole split; existing natural missing cells are preserved.
    """
    from dataclasses import replace
    rng = np.random.default_rng(seed)
    if max_cells < 1:
        raise ValueError("max_cells must be positive")
    blocks = [(si, ti, int(t.df.notna().to_numpy().sum())) for si, s in enumerate(sets) for ti, t in enumerate(s.tables)]
    ends = np.cumsum([n for _, _, n in blocks])
    total = int(ends[-1]) if len(ends) else 0
    chosen = rng.choice(total, min(max_cells, total), replace=False)
    # Keep only selected positions, rather than a Python tuple per observed dataset cell.
    positions = {}
    for bi in set(np.searchsorted(ends, chosen, side="right")):
        si, ti, _ = blocks[bi]
        start = int(ends[bi - 1]) if bi else 0
        observed = np.flatnonzero(sets[si].tables[ti].df.notna().to_numpy().ravel())
        for ix in chosen[(chosen >= start) & (chosen < ends[bi])]:
            r, c = np.unravel_index(observed[ix - start], sets[si].tables[ti].df.shape)
            positions[int(ix)] = si, ti, int(r), int(c)
    examples = []
    for ix in chosen:
        si, ti, r, ci = positions[int(ix)]
        source, table = sets[si], sets[si].tables[ti]
        frame = object_frame(table.df)
        truth, col = frame.iat[r, ci], frame.columns[ci]
        frame.iat[r, ci] = np.nan
        hidden_table = replace(table, df=frame, path=None)
        hidden_tables = [replace(t, path=None) for t in source.tables]
        hidden_tables[ti] = hidden_table
        hidden_set = replace(source, tables=hidden_tables, groundtruth=None)
        proposals = generate_candidates(specs, frame, hidden_table, hidden_set, ctx)
        examples.append({"generators": [p.get((r, col), []) for p in proposals], "truth": truth,
                         "set": source.name, "table": table.name, "row": r, "column": str(col)})
    return examples


def qx_diagnostics(examples, temperatures, weights, rho=.95, k_max=10):
    # 输入：examples、temperatures、weights、rho、k_max；输出：覆盖率、NLL、top1、熵等指标字典；继承：无（模块函数）。
    from .base import truncate_and_normalize
    rows = []
    for ex in examples:
        q, _ = build_qx(ex["generators"], temperatures, weights)
        kept, stats = truncate_and_normalize(q, rho, k_max)
        before, after = dict(q).get(ex["truth"], 0.), dict(kept).get(ex["truth"], 0.)
        rows.append({"union_covered": before > 0, "recall_at_k": after > 0,
                     "nll_union": -np.log(max(before, 1e-12)), "nll_retained": -np.log(max(after, 1e-12)),
                     "top1_accuracy": bool(kept and kept[0][0] == ex["truth"]), **stats})
    df = pd.DataFrame(rows)
    return {"n_masks": len(rows), "nll_floor": 1e-12, "rho": rho, "k_max": k_max,
            **({c: float(df[c].mean()) for c in df} if rows else {})}


def fit_mixture(examples, n_generators):
    # 输入：examples、n_generators；输出：学得的温度、混合权重及拟合诊断；继承：无（模块函数）。
    """Joint masked-value NLL. Uncovered examples contribute constant floored loss.

    Coverage is reported separately, so this floor cannot conceal generation failures.
    """
    if not examples or not any(any(ex["truth"] in dict(q) for q in ex["generators"]) for ex in examples):
        raise ValueError("no pseudo-masked truth covered; cannot fit qx calibration")
    def unpack(x):
        # 输入：x；输出：(正温度数组, softmax 混合权重)；继承：无（fit_mixture 内部函数）。
        return np.exp(x[:n_generators]), softmax(x[n_generators:])
    def loss(x):
        # 输入：x；输出：当前参数的负对数似然目标 float；继承：无（fit_mixture 内部函数）。
        temperatures, weights = unpack(x)
        return np.mean([-np.log(max(dict(build_qx(ex["generators"], temperatures, weights)[0])
                                           .get(ex["truth"], 0.), 1e-12)) for ex in examples])
    initial = np.zeros(2 * n_generators)
    result = minimize(loss, initial, method="L-BFGS-B", bounds=[(-4., 4.)] * n_generators + [(-12., 12.)] * n_generators)
    if not result.success:
        raise RuntimeError(f"qx calibration failed: {result.message}")
    temperatures, weights = unpack(result.x)
    return {"temperatures": temperatures.tolist(), "weights": weights.tolist(),
            "fit_nll_initial": float(loss(initial)), "fit_nll": float(result.fun), "fit_success": True,
            "log_temperature_bounds": [-4., 4.], "mixture_logit_bounds": [-12., 12.]}
