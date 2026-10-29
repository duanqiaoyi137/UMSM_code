"""ALITE column embeddings (Khatiwada et al., "Integrating Data Lake Tables", PVLDB 16(4), 2022).

Follows the align phase of https://github.com/northeastern-datalab/alite (codes/align_utilities/
align_phase.py, embedding types "bert_serialized" / "roberta_serialized"): a column is
represented by its values only (no header) -- distinct cell strings (at most `max_values`),
parenthesised parts removed, lower-cased and split into alphanumeric tokens, the `max_tokens`
tokens with the largest TF-IDF mass joined into one text and embedded with the pooler output
of a pre-trained BERT-style model. The match score of two columns is their cosine similarity
(clipped at 0). ALITE's holistic integration-id clustering is available as the global
optimizer `alite_cluster`.

Note: roberta-base has no pre-trained pooler weights (they are randomly initialised), so the
default model is bert-base-uncased; set pooling=mean for sentence-transformer models.
"""
from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..embedding import get_encoder
from ..registry import MATCHERS
from ..types import ZERO_EMBEDDING, ScoreSamples, UncertainData
from .base import BaseMatcher, cross_table_pairs


def alite_tokens(values: list, max_values: int, rng: np.random.Generator) -> list[str]:
    # 输入：values、max_values、rng；输出：ALITE 清洗后的词元列表；继承：无（模块函数）。
    distinct = sorted({str(v) for v in values if v is not ZERO_EMBEDDING})
    if len(distinct) > max_values:
        distinct = sorted(rng.choice(distinct, size=max_values, replace=False).tolist())
    text = re.sub(r"\([^)]*\)", "", " ".join(distinct))
    return sorted(set(re.sub(r"[^a-z0-9]+", " ", text.lower()).split()))


def select_by_tfidf(tokens: list[str], n: int) -> list[str]:
    # 输入：tokens、n；输出：TF-IDF 选择的词元列表；继承：无（模块函数）。
    try:
        scores = np.asarray(TfidfVectorizer().fit_transform(tokens).sum(axis=1)).ravel()
    except ValueError:
        return tokens[:n]
    order = np.argsort(-scores, kind="stable")
    return [tokens[i] for i in order[:n]]


@MATCHERS.register("alite")
class ALITEMatcher(BaseMatcher):
    """ALITE value-only column embeddings (TF-IDF token selection + BERT pooler) and cosine similarity."""

    def __init__(self, model: str = "bert-base-uncased", pooling: str = "pooler", max_tokens: int = 512,
                 max_values: int = 10000, device: str = "auto", batch_size: int = 16, max_length: int | None = 512):
        # 输入：model、pooling、max_tokens、max_values、device、batch_size、max_length；输出：None；初始化实例及校验参数；继承：ALITEMatcher(BaseMatcher)。
        self.model = model
        self.pooling = pooling
        self.max_tokens = max_tokens
        self.max_values = max_values
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    def column_text(self, values: list, rng: np.random.Generator) -> str:
        # 输入：values、rng；输出：ALITE 列序列化文本；继承：ALITEMatcher(BaseMatcher)。
        return " ".join(select_by_tfidf(alite_tokens(values, self.max_values, rng), self.max_tokens))

    def score(self, data: UncertainData, ctx) -> ScoreSamples:
        # 输入：data、ctx；输出：ScoreSamples（对齐采样的原始分数）；继承：ALITEMatcher(BaseMatcher)，重写父类方法。
        cols = data.columns
        with ctx.profiler.stage("matcher.encode"):
            texts = {}
            for c in cols:
                rng = np.random.default_rng([ctx.seed, zlib.crc32(c.id.encode("utf-8"))])
                texts[c] = [self.column_text(s.values, rng) for s in data.states(c)]
            enc = get_encoder(ctx, self.model, self.device, self.batch_size, self.max_length, pooling=self.pooling)
            vec = enc.encode_map([t for ts in texts.values() for t in ts if t], f"{data.source.dataset}/alite")
            dim = enc.dim or 1
            emb = {c: np.stack([vec[t] if t else np.zeros(dim, np.float32) for t in ts]).astype(np.float64)
                   for c, ts in texts.items()}

        with ctx.profiler.stage("matcher.pairs"):
            parts = []
            for ci, cj in cross_table_pairs(data):
                a_idx, b_idx, prob = data.pair_indices(ci, cj)
                s = np.clip(np.einsum("ij,ij->i", emb[ci][a_idx], emb[cj][b_idx]), 0.0, 1.0)
                parts.append((ci, cj, a_idx, b_idx, prob, s))
            sizes = [len(p[2]) for p in parts]
            df = pd.DataFrame({
                "col_i": np.repeat([p[0].id for p in parts], sizes),
                "col_j": np.repeat([p[1].id for p in parts], sizes),
                "table_i": np.repeat([p[0].table for p in parts], sizes),
                "table_j": np.repeat([p[1].table for p in parts], sizes),
                "state_i": np.concatenate([p[2] for p in parts]) if parts else [],
                "state_j": np.concatenate([p[3] for p in parts]) if parts else [],
                "prob": np.concatenate([p[4] for p in parts]) if parts else [],
                "score": np.concatenate([p[5] for p in parts]) if parts else [],
            })
        return ScoreSamples(df, cols, meta={"matcher": self.registry_name, **data.score_meta()})
