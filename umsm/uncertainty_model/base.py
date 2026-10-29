"""Module 1: Missing-Value Uncertainty Model   (T, X^mis) -> Q = {Q_x : x in X^mis}."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..types import Candidate, IntegrationSet, UncertainData, UncertainTable, is_missing


class UncertaintyModel(ABC):
    """Turns incomplete tables into candidate distributions for the missing cells.

    Deterministic models are the special case of degenerate distributions (p = 1): they write
    the value into UncertainTable.frame. Distributional models leave the cell missing and attach
    Q_x in UncertainTable.distributions.
    """

    registry_name = "?"
    stochastic = False  # True if Q itself depends on the run seed

    @abstractmethod
    def estimate(self, iset: IntegrationSet, ctx) -> UncertainData:
        # 输入：iset、ctx；输出：UncertainData（不修改原表）；继承：UncertaintyModel(ABC)。
        ...

    def params(self) -> dict:
        # 输入：实例状态；输出：组件构造参数字典（不包含内部缓存）；继承：UncertaintyModel(ABC)。
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


def object_frame(df: pd.DataFrame) -> pd.DataFrame:
    # 输入：df；输出：object 类型的独立 DataFrame；继承：无（模块函数）。
    return df.astype(object).where(df.notna(), np.nan)


@dataclass
class CellDistribution:
    """Candidate weights and optional per-cell diagnostics, passed together."""
    candidates: list[Candidate]
    info: dict = field(default_factory=dict)


class DeterministicModel(UncertaintyModel):
    """Helper base: implement fill_table(frame, table, iset, ctx) -> filled frame."""

    def estimate(self, iset: IntegrationSet, ctx) -> UncertainData:
        # 输入：iset、ctx；输出：UncertainData（不修改原表）；继承：DeterministicModel(UncertaintyModel)，重写父类方法。
        tables = []
        for t in iset.tables:
            frame = object_frame(t.df)
            missing = frame.isna().to_numpy()
            filled = self.fill_table(frame, t, iset, ctx)
            tables.append(UncertainTable(t.name, filled, filled=missing & ~filled.isna().to_numpy()))
        return UncertainData(iset, tables, self.registry_name, info=self.params())

    @abstractmethod
    def fill_table(self, frame: pd.DataFrame, table, iset: IntegrationSet, ctx) -> pd.DataFrame:
        # 输入：frame、table、iset、ctx；输出：确定性补全后的 DataFrame；继承：DeterministicModel(UncertaintyModel)。
        ...


class DistributionModel(UncertaintyModel):
    """Helper base: implement cell_distributions(frame, table, iset, ctx) -> {(row, column): [(v, p), ...]}.

    Rows are positions (0..n-1). Only missing cells may receive a distribution; probabilities
    are normalised here and zero-probability candidates dropped.
    """

    def __init__(self, rho: float = 0.95, k_max: int = 10):
        # 输入：rho、k_max；输出：None；初始化实例及校验参数；继承：DistributionModel(UncertaintyModel)。
        if not np.isfinite(rho) or not 0 < rho <= 1:
            raise ValueError("rho must be in (0, 1]")
        if not isinstance(k_max, int) or k_max < 1:
            raise ValueError("k_max must be a positive integer")
        self.rho, self.k_max = rho, k_max

    def prepare_distribution(self, q):
        # 输入：q；输出：(规范化候选分布, 截断诊断)；继承：DistributionModel(UncertaintyModel)。
        cap = self.k_max
        legacy_cap = getattr(self, "top_k", None)
        if legacy_cap is not None:
            cap = min(cap, legacy_cap)
        candidates = q.candidates if isinstance(q, CellDistribution) else q
        retained, info = truncate_and_normalize(candidates, self.rho, cap)
        if isinstance(q, CellDistribution):
            info.update(q.info)
        return retained, info

    def estimate(self, iset: IntegrationSet, ctx) -> UncertainData:
        # 输入：iset、ctx；输出：UncertainData（不修改原表）；继承：DistributionModel(UncertaintyModel)，重写父类方法。
        tables = []
        for t in iset.tables:
            frame = object_frame(t.df)
            raw = self.cell_distributions(frame, t, iset, ctx)
            dists, info, prepared = {}, {}, {}
            for (r, c), q in raw.items():
                if not isinstance(r, (int, np.integer)) or not 0 <= r < len(frame):
                    raise ValueError(f"{self.registry_name}: invalid row position {r} of {t.name}")
                if not pd.isna(frame.iat[r, frame.columns.get_loc(c)]):
                    raise ValueError(f"{self.registry_name}: cell ({r}, {c}) of {t.name} is observed")
                key = id(q)  # identical column marginals/fallbacks are prepared only once
                if key not in prepared:
                    prepared[key] = self.prepare_distribution(q)
                q, stats = prepared[key]
                info[(int(r), c)] = stats
                if q:
                    dists[(r, c)] = q
            tables.append(UncertainTable(t.name, frame, distributions=dists, distribution_info=info))
        return UncertainData(iset, tables, self.registry_name, info=self.params())

    @abstractmethod
    def cell_distributions(self, frame: pd.DataFrame, table, iset: IntegrationSet, ctx
                           ) -> dict[tuple[int, str], list[Candidate] | CellDistribution]:
        # 输入：frame、table、iset、ctx；输出：缺失位置到候选分布的字典；继承：DistributionModel(UncertaintyModel)。
        ...


def normalise(q: list[Candidate]) -> list[Candidate]:
    # 输入：q；输出：归一化的候选概率列表；继承：无（模块函数）。
    q = [(v, float(p)) for v, p in q]
    if any(not np.isfinite(p) or p < 0 for _, p in q):
        raise ValueError("candidate weights must be finite and nonnegative")
    q = [(v, p) for v, p in q if p > 0]
    total = sum(p for _, p in q)
    if not np.isfinite(total):
        raise ValueError("candidate weight total must be finite")
    return [(v, p / total) for v, p in q] if total > 0 else []


def truncate_and_normalize(q: list[Candidate], rho: float = 0.95, k_max: int = 10):
    # 输入：q、rho、k_max；输出：(稀疏归一化分布, 保留质量等诊断)；继承：无（模块函数）。
    """Smallest descending prefix reaching rho, capped by k_max; record pre-renormalization mass."""
    if not np.isfinite(rho) or not 0 < rho <= 1 or not isinstance(k_max, int) or k_max < 1:
        raise ValueError("require 0 < rho <= 1 and a positive integer k_max")
    merged = {}
    for v, p in normalise(q):
        if is_missing(v):
            raise ValueError("a candidate value must not itself be missing")
        merged[v] = merged.get(v, 0.0) + p
    full = sorted(merged.items(), key=lambda x: -x[1])
    n = int(np.searchsorted(np.cumsum([p for _, p in full]), rho - 1e-12, side="left")) + 1
    kept = full[:min(n, k_max)]
    mass = float(sum(p for _, p in kept))
    retained = [(v, p / mass) for v, p in kept] if mass else []
    return retained, {"candidate_count_raw": len(full), "candidate_count_kept": len(kept),
                      "retained_mass": mass, "mass_target_met": mass >= rho - 1e-12,
                      "entropy": float(-sum(p * np.log(p) for _, p in retained))}


def top_k(weights: dict, k: int | None) -> list[Candidate]:
    # 输入：weights、k；输出：按权重保留的归一化候选列表；继承：无（模块函数）。
    """Highest-weight candidates; ties keep insertion order."""
    items = sorted(weights.items(), key=lambda kv: -kv[1])
    return items[:k] if k else items
