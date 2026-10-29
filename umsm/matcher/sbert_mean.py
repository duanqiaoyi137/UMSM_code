"""Frequency-weighted mean cell embeddings, invariant to input row permutations.

This is an experimental column representation, not a reproduction of MTSM/ALITE.
Repeated values retain their empirical mass; all distinct texts reuse the encoder cache.
"""
from collections import Counter
import hashlib

import numpy as np

from ..embedding import DEFAULT_MODEL, get_encoder
from ..registry import MATCHERS
from ..types import ZERO_EMBEDDING, ScoreSamples
from .base import BaseMatcher, cross_table_pairs, sample_record
from .sbert_quantile import _cos


def frequency_mean(values, vectors, dim):
    # 输入：values、vectors、dim；输出：保留频率且与行序无关的均值向量；继承：无（模块函数）。
    counts = Counter(str(v) for v in values if v is not ZERO_EMBEDDING)
    if not counts:
        return np.zeros(dim, dtype=np.float64)
    # Canonical summation order avoids floating-point changes under row permutations.
    keys = sorted(counts)
    weights = np.array([counts[k] for k in keys], dtype=np.float64)
    return weights @ np.stack([vectors[k] for k in keys]).astype(np.float64) / weights.sum()


@MATCHERS.register("sbert_mean")
class SBertMeanMatcher(BaseMatcher):
    """Frequency-preserving mean cell embeddings with optional header similarity."""

    def __init__(self, model=DEFAULT_MODEL, device="auto", batch_size=64, max_length=None,
                 name_weight=1.0, value_weight=1.0):
        # 输入：model、device、batch_size、max_length、name_weight、value_weight；输出：None；初始化实例及校验参数；继承：SBertMeanMatcher(BaseMatcher)。
        if not np.isfinite([name_weight, value_weight]).all() or min(name_weight, value_weight) < 0 or name_weight + value_weight <= 0:
            raise ValueError("weights must be finite, nonnegative, and have positive sum")
        self.model, self.device, self.batch_size, self.max_length = model, device, batch_size, max_length
        self.name_weight, self.value_weight = name_weight, value_weight

    def score(self, data, ctx):
        # 输入：data、ctx；输出：ScoreSamples（对齐采样的原始分数）；继承：SBertMeanMatcher(BaseMatcher)，重写父类方法。
        states = {c: data.states(c) for c in data.columns}
        texts = [c.name.strip() for c in states]
        texts += [str(v) for ss in states.values() for s in ss for v in s.values if v is not ZERO_EMBEDDING]
        enc = get_encoder(ctx, self.model, self.device, self.batch_size, self.max_length)
        vec = enc.encode_map(texts, data.source.dataset)
        dim = len(next(iter(vec.values()))) if vec else 1
        def representation(state):
            # 输入：一个列实现；输出：同内容复用的保频均值表示；继承：无（score 内部函数）。
            content = sorted(Counter(str(v) for v in state.values if v is not ZERO_EMBEDDING).items())
            digest = hashlib.sha256(repr(content).encode()).hexdigest()
            key = ('mean_column',self.model,self.device,self.max_length,digest)
            return ctx.shared(key,lambda:frequency_mean(state.values,vec,dim))
        means = {c: np.stack([representation(s) for s in ss]) for c, ss in states.items()}
        records = []
        for ci, cj in cross_table_pairs(data):
            ai, bj, prob = data.pair_indices(ci, cj)
            name = float(np.clip(_cos(vec[ci.name.strip()][None], vec[cj.name.strip()][None])[0], -1, 1))
            value = np.clip(_cos(means[ci][ai], means[cj][bj]), -1, 1)
            scores = .5 + .5 * (self.name_weight * name + self.value_weight * value) / (self.name_weight + self.value_weight)
            for r, (a, b, p, s) in enumerate(zip(ai, bj, prob, scores)):
                records.append(sample_record(ci, cj, int(a), int(b), p, s,
                                             name_score=(1 + name) / 2, value_score=(1 + value[r]) / 2))
        return ScoreSamples.from_records(records, data.columns, {"matcher": self.registry_name, **data.score_meta()})
