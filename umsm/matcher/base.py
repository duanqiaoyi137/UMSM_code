"""Module 2: sampled columns -> finite raw scores; module 3 calibrates probabilities."""
from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations
from typing import Iterator

from ..types import ColumnRef, ColumnState, ScoreSamples, UncertainData


class BaseMatcher(ABC):
    """Matcher-agnostic local evidence.

    score() evaluates every cross-table column pair under the data states of module 1
    (UncertainData.states / UncertainData.table_realizations) and returns aligned r-th score
    samples with empirical prob=1/R. For column-local matchers a certain side broadcasts
    and a certain-certain pair is evaluated once. Table-context matchers retain all draws
    whenever either table varies. Cartesian products are confined to exact toy references.

    The interface is batched because some matchers normalise over all pairs (sbert_quantile)
    or work on whole tables (COMA, Cupid, ...). Simple matchers can subclass PairwiseMatcher.
    """

    registry_name = "?"
    stochastic = False  # True if the scores depend on the run seed beyond the data states
    scoring_version = 1
    sampling_scope = "column_pair"

    def expected_sample_counts(self, data):
        # 输入：data；输出：closure 列对到应有采样数的字典；继承：BaseMatcher(ABC)。
        """Table-context matchers must retain every draw, including complete-column pairs."""
        counts = {c: data.state_count(c) for c in data.columns}
        if self.sampling_scope == "table_pair":
            table_counts = {t.name: max((counts[c] for c in t.columns), default=1) for t in data.tables}
            counts = {c: table_counts[c.table] for c in data.columns}
        return {frozenset((a.id, b.id)): max(counts[a], counts[b]) for a, b in cross_table_pairs(data)}

    @abstractmethod
    def score(self, data: UncertainData, ctx) -> ScoreSamples:
        # 输入：data、ctx；输出：ScoreSamples（对齐采样的原始分数）；继承：BaseMatcher(ABC)。
        ...

    def params(self) -> dict:
        # 输入：实例状态；输出：组件构造参数字典（不包含内部缓存）；继承：BaseMatcher(ABC)。
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}



def cross_table_pairs(data) -> Iterator[tuple[ColumnRef, ColumnRef]]:
    # 输入：data；输出：允许评分的跨表列对迭代器；继承：无（模块函数）。
    """All column pairs of different tables, ordered by table then column (legacy order)."""
    for ta, tb in combinations(data.tables, 2):
        for ca in ta.columns:
            for cb in tb.columns:
                if data.allows(ca, cb):
                    yield ca, cb


def sample_record(ci: ColumnRef, cj: ColumnRef, a: int, b: int, prob: float, score: float, **extra) -> dict:
    # 输入：ci、cj、a、b、prob、score、**extra；输出：单个列对、状态及分数的记录字典；继承：无（模块函数）。
    return {"col_i": ci.id, "col_j": cj.id, "table_i": ci.table, "table_j": cj.table,
            "state_i": a, "state_j": b, "prob": float(prob), "score": float(score), **extra}


class PairwiseMatcher(BaseMatcher):
    """Helper base: implement similarity(ci, state_i, cj, state_j, ctx) -> finite raw score."""

    def score(self, data: UncertainData, ctx) -> ScoreSamples:
        # 输入：data、ctx；输出：ScoreSamples（对齐采样的原始分数）；继承：PairwiseMatcher(BaseMatcher)，重写父类方法。
        records = []
        for ci, cj in cross_table_pairs(data):
            sa, sb = data.states(ci), data.states(cj)
            a_idx, b_idx, prob = data.pair_indices(ci, cj)
            for a, b, p in zip(a_idx, b_idx, prob):
                s = self.similarity(ci, sa[a], cj, sb[b], ctx)
                records.append(sample_record(ci, cj, int(a), int(b), p, s))
        return ScoreSamples.from_records(records, data.columns, meta={"matcher": self.registry_name, **data.score_meta()})

    @abstractmethod
    def similarity(self, ci: ColumnRef, si: ColumnState, cj: ColumnRef, sj: ColumnState, ctx) -> float:
        # 输入：ci、si、cj、sj、ctx；输出：一对列状态的有限原始分数；继承：PairwiseMatcher(BaseMatcher)。
        ...
