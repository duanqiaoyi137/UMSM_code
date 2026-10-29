"""S-BERT name + quantile-value matcher (the column discrepancy of MTSM, Sec. 2.2).

Faithful port of matching/get_cosine.py and matching/get_k_weight_emd.py of Matching-main:
    D_name   = 1 - minmax(cos(name_i, name_j))
    D_values = minmax( sum_p w_p * (1 - cos(q_i^p, q_j^p)) ),  w = softmax-like exp(-|Var_i^p - Var_j^p|)
Output score = 1 - (w_n * D_name + w_v * D_values) / (w_n + w_v), i.e. higher = more similar;
with the default weights the ranking equals the original ascending D = D_name + D_values ranking.
Every column state c_i^(a) of module 1 gets its own quantile embedding; the min-max
normalisation runs over all evaluated (pair, a, b), so complete data reproduces the original.

Note: cell embeddings are L2-normalised, so the original "sort by distance to the zero vector"
orders non-zero embeddings by floating-point noise (~1e-7), which makes the result depend on how
the texts were batched during encoding. `norm_decimals` rounds the norms before the stable sort
(non-zero embeddings keep the row order, zero embeddings come first); None = original behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import hashlib

from ..embedding import DEFAULT_MODEL, get_encoder
from ..registry import MATCHERS
from ..types import ZERO_EMBEDDING, ScoreSamples, UncertainData
from .base import BaseMatcher, cross_table_pairs


def quantile_features(emb: np.ndarray, k: int, norm_decimals: int | None = 4) -> tuple[np.ndarray, np.ndarray]:
    # 输入：emb、k、norm_decimals；输出：(采样位置向量, 范数特征数组)；继承：无（模块函数）。
    n, d = emb.shape
    if n == 0:
        return np.zeros((k, d)), np.zeros(k)
    emb = emb.astype(np.float64)
    norms = np.linalg.norm(emb, axis=1)
    if norm_decimals is not None:
        norms = np.round(norms, norm_decimals)
    order = np.argsort(norms, kind="stable")
    s = emb[order]
    q_emb, q_var = np.zeros((k, d)), np.zeros(k)
    half = 1 / (2 * k)
    for i in range(1, k + 1):
        q = i / k
        q_emb[i - 1] = s[int(q * (n - 1))]
        lo, hi = max(0, int((q - half) * n)), min(n, int((q + half) * n))
        win = s[lo:hi]
        if len(win):
            q_var[i - 1] = np.linalg.norm(win - win.mean(axis=0), axis=1).mean()
    return q_emb, q_var


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # 输入：a、b；输出：逐行余弦相似度数组；继承：无（模块函数）。
    """Row-wise cosine along the last axis with broadcasting; 0 where a norm is 0."""
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    denom = na * nb
    return np.where(denom > 0, (a * b).sum(-1) / np.where(denom > 0, denom, 1), 0.0)


def _minmax(x: np.ndarray) -> np.ndarray:
    # 输入：x；输出：最小最大归一化数组；继承：无（模块函数）。
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


@MATCHERS.register("sbert_quantile")
class SBertQuantileMatcher(BaseMatcher):
    """S-BERT column-name cosine + quantile-embedding value discrepancy (original MTSM matcher)."""

    def __init__(self, quantiles: int = 5, model: str = DEFAULT_MODEL, device: str = "auto",
                 batch_size: int = 64, max_length: int | None = None, name_weight: float = 1.0,
                 value_weight: float = 1.0, norm_decimals: int | None = 4,
                 normalization: str = "legacy_minmax"):
        # 输入：quantiles、model、device 等配置参数（详见签名）；输出：None；初始化实例及校验参数；继承：SBertQuantileMatcher(BaseMatcher)。
        if normalization not in ("legacy_minmax", "fixed"):
            raise ValueError("normalization must be legacy_minmax or fixed")
        if name_weight < 0 or value_weight < 0 or name_weight + value_weight <= 0:
            raise ValueError("matcher weights must be nonnegative with positive total")
        if quantiles < 1:
            raise ValueError("quantiles must be positive")
        self.normalization = normalization
        self.quantiles = quantiles
        self.norm_decimals = norm_decimals
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.name_weight = name_weight
        self.value_weight = value_weight


    def score(self, data: UncertainData, ctx) -> ScoreSamples:
        # 输入：data、ctx；输出：ScoreSamples（对齐采样的原始分数）；继承：SBertQuantileMatcher(BaseMatcher)，重写父类方法。
        enc = get_encoder(ctx, self.model, self.device, self.batch_size, self.max_length)
        cols = data.columns

        with ctx.profiler.stage("matcher.encode"):
            states = {c: data.states(c) for c in cols}
            texts = [c.name.strip() for c in cols]
            texts += [str(v) for sts in states.values() for s in sts for v in s.values if v is not ZERO_EMBEDDING]
            vec = enc.encode_map(texts, data.source.dataset)
            zero = np.zeros(enc.dim, np.float32)
            name_vec = {c: vec[c.name.strip()] for c in cols}
            feats = {}
            for c, sts in states.items():
                qs, vs = [], []
                for s in sts:
                    digest = hashlib.sha256(repr(s.values).encode()).hexdigest()
                    def representation():
                        # 输入：当前列实现及编码映射；输出：分位表示与离散度；继承：无（score 内部函数）。
                        emb = np.stack([zero if v is ZERO_EMBEDDING else vec[str(v)] for v in s.values]) if s.values \
                            else np.zeros((0,enc.dim),np.float32)
                        return quantile_features(emb,self.quantiles,self.norm_decimals)
                    key = ('quantile_column',self.model,self.device,self.max_length,self.quantiles,self.norm_decimals,digest)
                    q, v = ctx.shared(key,representation)
                    qs.append(q)
                    vs.append(v)
                feats[c] = (np.stack(qs), np.stack(vs))

        with ctx.profiler.stage("matcher.pairs"):
            parts = []
            for ci, cj in cross_table_pairs(data):
                (qa, va), (qb, vb) = feats[ci], feats[cj]
                a_idx, b_idx, prob = data.pair_indices(ci, cj)
                w = np.exp(-np.abs(va[a_idx] - vb[b_idx]))
                w = w / w.sum(-1, keepdims=True)
                d = (w * np.maximum(0.0, 1.0 - _cos(qa[a_idx], qb[b_idx]))).sum(-1)
                parts.append((ci, cj, a_idx, b_idx, prob, d,
                              float(_cos(name_vec[ci][None], name_vec[cj][None])[0])))
            sizes = [len(p[2]) for p in parts]
            if not parts:
                return ScoreSamples.from_records([], cols, meta={"matcher": self.registry_name, **data.score_meta()})
            value_d = np.concatenate([p[5] for p in parts])
            name_cos = np.repeat([p[6] for p in parts], sizes)
            if self.normalization == "legacy_minmax":
                d_name = 1.0 - _minmax(name_cos)
                d_value = _minmax(value_d)
            else:
                # Cosines have a fixed [-1, 1] range and discrepancies [0, 2].
                # Scores of an unchanged pair then cannot move just because another
                # column acquired more states or a different imputation strategy.
                d_name = (1.0 - np.clip(name_cos, -1.0, 1.0)) / 2.0
                d_value = np.clip(value_d, 0.0, 2.0) / 2.0
            d = (self.name_weight * d_name + self.value_weight * d_value) / (self.name_weight + self.value_weight)
            df = pd.DataFrame({
                "col_i": np.repeat([p[0].id for p in parts], sizes),
                "col_j": np.repeat([p[1].id for p in parts], sizes),
                "table_i": np.repeat([p[0].table for p in parts], sizes),
                "table_j": np.repeat([p[1].table for p in parts], sizes),
                "state_i": np.concatenate([p[2] for p in parts]),
                "state_j": np.concatenate([p[3] for p in parts]),
                "prob": np.concatenate([p[4] for p in parts]),
                "score": 1.0 - d,
                "name_score": 1.0 - d_name,
                "value_score": 1.0 - d_value,
                "name_cosine": name_cos,
                "value_discrepancy_raw": value_d,
            })
        return ScoreSamples(df, cols, meta={"matcher": self.registry_name, **data.score_meta()})
