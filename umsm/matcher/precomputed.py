"""Scores produced outside this framework (e.g. Magneto, ALITE, an LLM matcher run in its own env).

Expected file per integration set: CSV with columns col_i, col_j, score [, is_match, ...], where
col_* are column ids "<table file>.<column>". Pairs missing from the file get score 0.
External scores describe a single data state, so the uncertainty model must be deterministic.
"""
from __future__ import annotations

import os

import pandas as pd

from ..registry import MATCHERS
from ..types import ScoreSamples, UncertainData
from .base import BaseMatcher, cross_table_pairs, sample_record


@MATCHERS.register("precomputed")
class PrecomputedScores(BaseMatcher):
    """Load pairwise scores from CSV files written by an external matcher."""

    def __init__(self, template: str, name: str = "external"):
        # template placeholders: {set} = integration set name, {slug} = file-system safe set name
        # 输入：template、name；输出：None；初始化实例及校验参数；继承：PrecomputedScores(BaseMatcher)。
        self.template = template
        self.name = name

    def score(self, data: UncertainData, ctx) -> ScoreSamples:
        # 输入：data、ctx；输出：ScoreSamples（对齐采样的原始分数）；继承：PrecomputedScores(BaseMatcher)，重写父类方法。
        if not data.is_deterministic:
            raise ValueError("precomputed scores need a deterministic uncertainty model (one data state per column)")
        iset = data.source
        path = ctx.resolve(self.template.format(set=iset.name, slug=iset.slug))
        if not os.path.exists(path):
            raise FileNotFoundError(f"precomputed scores {path} not found")
        ext = pd.read_csv(path, encoding="utf-8")
        extra_cols = [c for c in ext.columns if c not in ("col_i", "col_j", "table_i", "table_j", "score")]
        lookup = {tuple(sorted((row["col_i"], row["col_j"]))): row for row in ext.to_dict("records")}
        records, n_hit = [], 0
        for ci, cj in cross_table_pairs(data):
            row = lookup.get(tuple(sorted((ci.id, cj.id))))
            n_hit += row is not None
            extra = {c: (row[c] if row is not None else (False if c == "is_match" else None)) for c in extra_cols}
            records.append(sample_record(ci, cj, 0, 0, 1.0, row["score"] if row is not None else 0.0, **extra))
        if n_hit == 0 and records:
            raise ValueError(f"{path}: no column pair matched the loaded tables (check column ids)")
        return ScoreSamples.from_records(records, data.columns, meta={"matcher": f"precomputed.{self.name}"})
