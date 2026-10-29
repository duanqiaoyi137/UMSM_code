"""Frozen calibration artifacts and validation-only fitting for qx / matcher scores."""
from __future__ import annotations

import hashlib
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from .registry import MATCHERS


def set_fingerprint(iset):
    # 输入：iset；输出：数据内容 SHA256 字符串；继承：无（模块函数）。
    digest = hashlib.sha256()
    for t in sorted(iset.tables, key=lambda t: t.name):
        digest.update(t.name.encode())
        digest.update(repr(list(t.df.columns)).encode())
        digest.update(pd.util.hash_pandas_object(t.df, index=True).values.tobytes())
    return digest.hexdigest()


def load_artifact(source, ctx=None):
    # 输入：source、ctx；输出：验证格式后的冻结校准字典；继承：无（模块函数）。
    """Accept a frozen file or in-memory calibration; training needs no temporary model file."""
    if isinstance(source, dict):
        result = source
    else:
        path = ctx.resolve(source) if ctx is not None else source
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
    if result.get("version") != "qx-global-v2.1" or result.get("split_role") not in ("train", "validation"):
        raise ValueError("expected a V2.1 train/validation calibration artifact")
    return result


def check_inference_split(path, iset, ctx):
    # 输入：path、iset、ctx；输出：通过数据隔离校验的校准字典；继承：无（模块函数）。
    artifact = load_artifact(path, ctx)
    if not ctx.resources.get("calibration_fitting", False):
        if set_fingerprint(iset) in {s["fingerprint"] for s in artifact["fit_sets"]}:
            raise ValueError("test tables overlap the calibration split; choose held-out variants/sets")
    return artifact


def check_matcher(artifact, matcher):
    # 输入：artifact、matcher；输出：None；不兼容时抛异常；继承：无（模块函数）。
    """Compare constructed parameters so omitted constructor defaults are handled once."""
    fitted = artifact["matcher"]
    if artifact.get("matcher_scoring_version", 1) != matcher.scoring_version:
        raise ValueError("matcher scoring implementation changed; fit a new calibration artifact")
    original = MATCHERS.build(fitted["name"], fitted.get("params", {}))
    if original.registry_name != matcher.registry_name or original.params() != matcher.params():
        raise ValueError("matcher differs from fitted probability calibration; fit a new artifact")


def calibration_snapshot(comp, ctx):
    # 输入：comp、ctx；输出：组件校准内容及 SHA256 快照；继承：无（模块函数）。
    """Return frozen contents and hashes for embedding in the single result file."""
    found = {}
    for name, component in comp.items():
        base = getattr(component, "_base", component)
        path = getattr(base, "calibration", None)
        if path is None:
            continue
        if isinstance(path, dict):
            raw = json.dumps(path, sort_keys=True).encode()
            path = None  # in-memory training/programmatic use has no source file
        else:
            with open(ctx.resolve(path), "rb") as f:
                raw = f.read()
        found[name] = {"path": path, "sha256": hashlib.sha256(raw).hexdigest(),
                       "artifact": json.loads(raw)}
    return found


def match_labels(samples, groundtruth):
    # 输入：samples、groundtruth；输出：逐样本 0/1 标注数组；继承：无（模块函数）。
    if groundtruth is None:
        raise ValueError("matcher calibration requires validation matching labels")
    positives = {frozenset(pair) for group in groundtruth for pair in combinations(group, 2)}
    return np.array([int(frozenset((a, b)) in positives) for a, b in
                     zip(samples.df.col_i, samples.df.col_j)])


def fit_platt(scores, labels, weights=None):
    # 输入：scores、labels、weights；输出：单调 Platt 参数与拟合诊断字典；继承：无（模块函数）。
    s, y = np.asarray(scores, float), np.asarray(labels, float)
    if s.ndim != 1 or s.shape != y.shape or not np.isfinite(s).all() or set(y) != {0., 1.}:
        raise ValueError("Platt fitting needs finite scores and both positive/negative labels")
    w = np.ones(len(s)) if weights is None else np.asarray(weights, float)
    if w.shape != s.shape or (w < 0).any() or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("invalid calibration weights")
    w /= w.sum()
    def loss(x):
        # 输入：x；输出：当前参数的负对数似然目标 float；继承：无（fit_platt 内部函数）。
        z = x[0] * s + x[1]
        return float(np.dot(w, np.logaddexp(0., z) - y * z) + 1e-6 * x[0] ** 2)
    result = minimize(loss, [1., 0.], method="L-BFGS-B", bounds=[(0., 1000.), (-1000., 1000.)])
    if not result.success:
        raise RuntimeError(f"Platt calibration failed: {result.message}")
    return {"method": "platt", "slope": float(result.x[0]), "intercept": float(result.x[1]),
            "nll": float(result.fun), "n_samples": len(s), "fit_success": True,
            "weighting": "equal edge mass; equal MC draws within edge", "l2": 1e-6}


def calibrate_scores(scores, params):
    # 输入：scores、params；输出：逐样本校准概率数组；继承：无（模块函数）。
    if params and params.get('method')=='identity':
        values = np.asarray(scores,float)
        if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
            raise ValueError('Identity calibration requires finite probabilities in [0,1]')
        return values
    if not params or params.get("method") != "platt":
        raise ValueError("a fitted Platt calibrator is required; raw scores are not probabilities")
    return expit(params["slope"] * np.asarray(scores, float) + params["intercept"])


def probability_diagnostics(edges, groundtruth):
    # 输入：edges、groundtruth；输出：风险惩罚前的 Brier/NLL 字典；继承：无（模块函数）。
    """Evaluate mean calibrated probabilities per edge, before the risk penalty."""
    if edges.meta.get("potential") != "calibrated_logit" or edges.df.empty:
        return {}
    positives = {frozenset(p) for group in groundtruth for p in combinations(group, 2)}
    labels = np.array([int(frozenset((a, b)) in positives) for a, b in
                       zip(edges.df.col_i, edges.df.col_j)])
    p = np.clip(edges.df.mu.to_numpy(float), 1e-12, 1 - 1e-12)
    return {"calibration_brier": float(np.mean((p - labels) ** 2)),
            "calibration_nll": float(-np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p)))}


