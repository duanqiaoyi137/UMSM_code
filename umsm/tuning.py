"""Validation-set threshold search shared by experimental baselines.

The calibrated logit optimizer has no theta lookup. Raw ILP, greedy and Hungarian
baselines can still tune their explicit threshold through ablation.py.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .context import RunContext
from .evaluation import METRIC_KEYS, evaluate, validate_matching


def default_grid(weights: np.ndarray, min_quantile: float = 0.5) -> np.ndarray:
    # 输入：weights、min_quantile；输出：验证阈值网格数组；继承：无（模块函数）。
    w = weights[np.isfinite(weights)]
    if len(w) == 0:
        return np.array([0.5])
    lo, hi = float(np.quantile(w, min_quantile)), float(w.max())
    fixed = np.arange(np.floor(lo * 100) / 100, hi + 0.01, 0.01)
    quant = np.quantile(w[w >= lo], np.linspace(0, 1, 101))
    # Search rounds to four decimals. A 1e-6 sentinel can round back to max(w),
    # losing the empty-matching option for baselines that accept w >= threshold.
    above = np.ceil(hi * 10000) / 10000 + 0.0001
    return np.unique(np.round(np.concatenate([fixed, quant, [above]]), 4))


def best_theta(scores: pd.Series) -> float:
    # 输入：scores；输出：验证指标最优的阈值；继承：无（模块函数）。
    """Middle of the thetas reaching the best score (ties are frequent: the ILP is piecewise constant)."""
    top = scores[scores >= scores.max() - 1e-12].index.to_numpy()
    return float(top[len(top) // 2])


def coarse_grid(grid: np.ndarray, step: float) -> np.ndarray:
    # 输入：grid、step；输出：粗搜索网格数组；继承：无（模块函数）。
    """Thetas from max(grid) down to min(grid) spaced by step (descending)."""
    hi, lo = float(grid.max()), float(grid.min())
    return np.round(np.arange(hi, lo - 1e-9, -step), 4)


def search_threshold(optimizer, param: str, cases: list, thetas: np.ndarray, objective: str = "pair_f1",
                     search: str = "coarse_fine", coarse_step: float = 0.05, fine_radius: float = 0.05,
                     patience: int = 2, tolerance: float = 0.05, logger=None) -> pd.DataFrame:
    # 输入：optimizer、param、cases 等配置参数（详见签名）；输出：以阈值为索引的指标 DataFrame；继承：无（模块函数）。
    """Metrics (mean over cases) of every evaluated value of optimizer.<param>, indexed by that value.

    cases: [(EdgeTable, ground truth)]. See the module docstring for the coarse_fine search.
    """
    if search not in ("coarse_fine", "full"):
        raise ValueError("search must be coarse_fine or full")
    if not cases:
        raise ValueError("threshold search needs at least one tuning case")
    quiet = RunContext(logger=logging.getLogger("umsm.tuning.quiet"))
    quiet.logger.setLevel(logging.WARNING)
    rows: dict[float, dict] = {}
    t0 = time.perf_counter()

    def run(theta: float) -> float:
        # 输入：候选阈值 theta；输出：独立验证样例上的目标均值；继承：无（内部函数）。
        theta = round(float(theta), 4)
        if theta not in rows:
            setattr(optimizer, param, theta)
            per_case = []
            for edges, gt in cases:
                m = optimizer.optimize(edges, quiet)
                validate_matching(m, edges, "threshold search")
                per_case.append({**evaluate(m, gt), "n_groups": len(m.groups)})
            df = pd.DataFrame(per_case)
            rows[theta] = {"theta": theta, **{c: float(df[c].mean()) for c in METRIC_KEYS},
                           "mean_groups": float(df["n_groups"].mean())}
            if logger and len(rows) % 25 == 0:
                logger.info(f"  {len(rows)} {param} values ({time.perf_counter() - t0:.1f}s)")
        return rows[theta][objective]

    if search == "full":
        for theta in thetas:
            run(theta)
    else:
        best, below = -np.inf, 0
        for theta in coarse_grid(thetas, coarse_step):
            score = run(theta)
            best = max(best, score)
            below = below + 1 if score < best - tolerance else 0
            if below >= patience:
                break
        coarse_best = best_theta(pd.Series({t: r[objective] for t, r in rows.items()}).sort_index())
        if logger:
            logger.info(f"  coarse search: {len(rows)} {param} values, best {coarse_best:.4f} "
                        f"({objective} {best:.4f}, {time.perf_counter() - t0:.1f}s)")
        for theta in thetas[np.abs(thetas - coarse_best) <= fine_radius + 1e-9]:
            run(theta)
    if logger:
        logger.info(f"  {len(rows)} {param} values evaluated ({time.perf_counter() - t0:.1f}s)")
    return pd.DataFrame(sorted(rows.values(), key=lambda r: r["theta"])).set_index("theta")
