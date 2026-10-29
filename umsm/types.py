"""Data objects exchanged between the four modules.

    IntegrationSet
      --(1) Missing-Value Uncertainty Model -->  UncertainData   candidate distributions Q_x = {(v, p)}
      --(2) Base Matcher                    -->  ScoreSamples    paired s_ij^(r), empirical weight 1/R
      --(3) Uncertainty Propagation         -->  EdgeTable       mu_ij, sigma_ij, w_ij   (G = (V, E, W, tau))
      --(4) Global Multi-Table Optimizer    -->  Matching        Omega = {C_1, ..., C_q}

Column identifiers follow the ground-truth files of the benchmark: "<table file>.<column>",
e.g. "openrice.csv.Taste".
"""
from __future__ import annotations

import itertools
import json
import re
import zlib
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True, order=True)
class ColumnRef:
    table: str
    name: str

    @property
    def id(self) -> str:
        # 输入：实例状态；输出：table.column 标识；继承：ColumnRef(object)。
        return f"{self.table}.{self.name}"

    def __str__(self) -> str:
        # 输入：实例状态；输出：列标识字符串；继承：ColumnRef(object)。
        return self.id


@dataclass
class Table:
    name: str
    df: pd.DataFrame
    path: str | None = None

    @property
    def columns(self) -> list[ColumnRef]:
        # 输入：实例状态；输出：list[ColumnRef]；继承：Table(object)。
        return [ColumnRef(self.name, str(c)) for c in self.df.columns]


@dataclass
class IntegrationSet:
    """A group of tables that must be matched jointly, with its ground truth."""

    dataset: str
    name: str
    tables: list[Table]
    groundtruth: list[frozenset[str]] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def slug(self) -> str:
        # 输入：实例状态；输出：路径安全的集合名称；继承：IntegrationSet(object)。
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name)

    @property
    def columns(self) -> list[ColumnRef]:
        # 输入：实例状态；输出：list[ColumnRef]；继承：IntegrationSet(object)。
        return [c for t in self.tables for c in t.columns]

    def stats(self) -> dict:
        # 输入：实例状态；输出：该对象的统计信息字典；继承：IntegrationSet(object)。
        n_cells = sum(t.df.size for t in self.tables)
        n_missing = sum(int(t.df.isna().values.sum()) for t in self.tables)
        return {
            "n_tables": len(self.tables),
            "n_columns": sum(t.df.shape[1] for t in self.tables),
            "n_rows": sum(t.df.shape[0] for t in self.tables),
            "n_cells": n_cells,
            "n_missing_cells": n_missing,
            "missing_rate": n_missing / n_cells if n_cells else 0.0,
            "n_gt_groups": len(self.groundtruth) if self.groundtruth is not None else None,
        }


class _ZeroEmbedding:
    """Placeholder cell whose embedding is the all-zero vector (legacy MTSM behaviour)."""

    _instance = None

    def __new__(cls):
        # 输入：占位符类 cls；输出：单例 ZERO_EMBEDDING 对象；继承：_ZeroEmbedding(object)。
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        # 输入：实例状态；输出：零向量占位符字符串；继承：_ZeroEmbedding(object)。
        return "ZERO_EMBEDDING"


ZERO_EMBEDDING = _ZeroEmbedding()


def is_missing(v) -> bool:
    # 输入：v；输出：bool；继承：无（模块函数）。
    if v is ZERO_EMBEDDING:
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------------------------
# Module 1 output
# --------------------------------------------------------------------------------------------
Candidate = tuple[object, float]


@dataclass(frozen=True)
class RealizationSpec:
    """Fixed-R, conditionally factorized Monte Carlo; exact is for unit references only."""

    n_samples: int = 32
    max_exact_states: int = 8
    mode: str = "mc"

    def __post_init__(self):
        # 输入：实例状态；输出：None；初始化或校验数据对象；继承：RealizationSpec(object)。
        if not isinstance(self.n_samples, int) or self.n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        if not isinstance(self.max_exact_states, int) or self.max_exact_states < 1:
            raise ValueError("max_exact_states must be a positive integer")
        if self.mode not in ("mc", "exact"):
            raise ValueError("realization.mode must be mc or exact (unit-test reference only)")
        if self.mode == "mc" and self.n_samples < 2:
            raise ValueError("Monte Carlo requires n_samples >= 2 for sample variance")


@dataclass
class ColumnState:
    """One realization c_i^(a) of a column: its values in row order and its probability p_i^a."""

    values: list
    prob: float


def categorical_guide(q):
    # 输入：K 个候选及概率；输出：O(K) 构建的 CDF 和导引表，抽样期望 O(1)；继承：无（模块函数）。
    cdf=np.cumsum([p for _,p in q],dtype=float)
    cdf/=cdf[-1]
    guide=np.zeros(len(q),dtype=int)
    cursor=0
    for bucket in range(len(q)):
        while cdf[cursor]<=bucket/len(q):
            cursor+=1
        guide[bucket]=cursor
    return cdf,guide


def categorical_index(cdf,guide,u):
    # 输入：CDF、导引表、一个均匀随机数；输出：与 NumPy choice 一致的类别下标；继承：无（模块函数）。
    index=int(guide[int(u*len(guide))])
    while u>=cdf[index]:
        index+=1
    return index


@dataclass
class UncertainTable:
    """Output of the Missing-Value Uncertainty Model for one table.

    frame:          object-dtype copy of the table holding observed values and deterministic
                    fills (a degenerate distribution with p = 1); NaN = no evidence or uncertain.
    distributions:  Q_x for uncertain cells, {(row position, column): [(value, prob), ...]};
                    the corresponding frame cell stays NaN.
    filled:         boolean mask of the cells filled deterministically.
    """

    name: str
    frame: pd.DataFrame
    distributions: dict[tuple[int, str], list[Candidate]] = field(default_factory=dict)
    filled: np.ndarray | None = None
    distribution_info: dict = field(default_factory=dict)

    @property
    def columns(self) -> list[ColumnRef]:
        # 输入：实例状态；输出：list[ColumnRef]；继承：UncertainTable(object)。
        return [ColumnRef(self.name, str(c)) for c in self.frame.columns]

    def column_values(self, column: str) -> list:
        # 输入：column；输出：确定观测值与确定填充值列表；继承：UncertainTable(object)。
        """Deterministic evidence of a column (observed values + deterministic fills)."""
        return [v for v in self.frame[column].tolist() if not is_missing(v)]

    def uncertain_cells(self, column: str) -> list[tuple[int, list[Candidate]]]:
        # 输入：column；输出：按行排序的缺失位置及分布；继承：UncertainTable(object)。
        return sorted((r, q) for (r, c), q in self.distributions.items() if c == column)

    def column_states(self, column: str, spec: RealizationSpec, seed: int) -> list[ColumnState]:
        # 输入：column、spec、seed；输出：复用的列状态列表（MC 或小规模 exact）；继承：UncertainTable(object)。
        base = self.frame[column].tolist()
        cells = self.uncertain_cells(column)
        if not cells:
            return [ColumnState([v for v in base if not is_missing(v)], 1.0)]

        def realize(choice: dict[int, object]) -> list:
            # 输入：choice；输出：list；继承：无（column_states 内部函数）。
            out = []
            for r, v in enumerate(base):
                if r in choice:
                    out.append(choice[r])
                elif not is_missing(v):
                    out.append(v)
            return out

        rows = [r for r, _ in cells]
        if spec.mode == "exact":
            n_comb = 1
            for _, q in cells:
                n_comb *= len(q)
            if n_comb > spec.max_exact_states:
                raise ValueError("exact reference exceeds max_exact_states; use Monte Carlo")
            states = []
            for combo in itertools.product(*[q for _, q in cells]):
                prob = float(np.prod([p for _, p in combo]))
                states.append(ColumnState(realize({r: v for r, (v, _) in zip(rows, combo)}), prob))
            return states
        rng = np.random.default_rng([seed, zlib.crc32(f"{self.name}.{column}".encode("utf-8"))])
        draws = [rng.choice(len(q), size=spec.n_samples, p=[p for _, p in q]) for _, q in cells]
        return [ColumnState(realize({r: q[d[s]][0] for (r, q), d in zip(cells, draws)}), 1.0 / spec.n_samples)
                for s in range(spec.n_samples)]

    def table_realizations(self, spec: RealizationSpec, seed: int) -> list[tuple[pd.DataFrame, float]]:
        # 输入：spec、seed；输出：对齐的表状态及概率列表；继承：UncertainTable(object)。
        """Fixed-R table draws; exact enumeration is restricted to explicit toy references."""
        if not self.distributions:
            return [(self.external_frame(), 1.0)]
        rng = np.random.default_rng([seed, zlib.crc32(self.name.encode("utf-8"))])
        cells = sorted(self.distributions.items())
        col_pos = {c: k for k, c in enumerate(self.frame.columns)}
        n_comb = 1
        for _, q in cells:
            n_comb *= len(q)
            if n_comb > spec.max_exact_states:
                break
        if spec.mode == "exact":
            if n_comb > spec.max_exact_states:
                raise ValueError("exact reference exceeds max_exact_states; use Monte Carlo")
            out = []
            for combo in itertools.product(*[q for _, q in cells]):
                f = self.external_frame()
                for ((r, c), _), (v, _) in zip(cells, combo):
                    f.iat[r, col_pos[c]] = v
                out.append((f, float(np.prod([p for _, p in combo]))))
            return out
        draws = [rng.choice(len(q), size=spec.n_samples, p=[p for _, p in q]) for _, q in cells]
        out = []
        for s in range(spec.n_samples):
            f = self.external_frame()
            for ((r, c), q), d in zip(cells, draws):
                f.iat[r, col_pos[c]] = q[d[s]][0]
            out.append((f, 1.0 / spec.n_samples))
        return out

    def external_frame(self) -> pd.DataFrame:
        # 输入：实例状态；输出：将占位符转回 NaN 的 DataFrame；继承：UncertainTable(object)。
        """Frame for external tools: placeholders are turned back into NaN."""
        return self.frame.map(lambda v: np.nan if v is ZERO_EMBEDDING else v)


@dataclass
class UncertainData:
    """Output of module 1 for one integration set: Q = {Q_x : x in X^mis}."""

    source: IntegrationSet
    tables: list[UncertainTable]
    model: str
    info: dict = field(default_factory=dict)
    realization: RealizationSpec = field(default_factory=RealizationSpec)
    seed: int = 0
    _states: dict = field(default_factory=dict, repr=False)
    candidate_pairs: set[frozenset[str]] | None = None

    @property
    def columns(self) -> list[ColumnRef]:
        # 输入：实例状态；输出：list[ColumnRef]；继承：UncertainData(object)。
        return [c for t in self.tables for c in t.columns]

    def table(self, name: str) -> UncertainTable:
        # 输入：name；输出：指定名称的 UncertainTable；继承：UncertainData(object)。
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(name)

    def configure(self, realization: RealizationSpec, seed: int) -> "UncertainData":
        # 输入：realization、seed；输出：配置后并清空状态缓存的自身实例；继承：UncertainData(object)。
        self.realization, self.seed, self._states = realization, seed, {}
        return self

    def states(self, col: ColumnRef) -> list[ColumnState]:
        # 输入：col；输出：缓存的列状态列表；继承：UncertainData(object)。
        """Data states c_i^(a) with probabilities p_i^a (one state with p = 1 for certain columns)."""
        if col not in self._states:
            self._states[col] = self.table(col.table).column_states(col.name, self.realization, self.seed)
        return self._states[col]

    def state_count(self, col: ColumnRef) -> int:
        # 输入：列引用；输出：该列的实现数量，不构造任何实现；继承：UncertainData(object)。
        if col in self._states:
            return len(self._states[col])
        if self.realization.mode != 'mc':
            return len(self.states(col))
        return self.realization.n_samples if self.table(col.table).uncertain_cells(col.name) else 1

    def iter_realizations(self):
        # 输入：当前 Q、R 和种子；输出：逐轮一致的列实现字典，仅保留当前一轮；继承：UncertainData(object)。
        if self.realization.mode != 'mc':
            raise ValueError('Streaming inference requires Monte Carlo realizations')
        columns, streams = self.columns, {}
        for col in columns:
            table = self.table(col.table)
            cells = table.uncertain_cells(col.name)
            local = []
            for index, (row, q) in enumerate(cells):
                rng = np.random.default_rng([self.seed, zlib.crc32(col.id.encode('utf-8'))])
                # Each earlier cell consumed R uniform draws in column_states().
                # Jumping preserves existing MC draws without allocating R values per cell.
                rng.bit_generator.advance(index * self.realization.n_samples)
                cdf,guide=categorical_guide(q)
                local.append((row,q,cdf,guide,rng))
            streams[col] = (table.frame[col.name].tolist(), local)
        rounds = max((self.state_count(c) for c in columns), default=1)
        certain = {}
        for r in range(rounds):
            states = {}
            for col, (base, cells) in streams.items():
                if not cells and col in certain:
                    states[col] = certain[col]
                    continue
                choices = {row:q[categorical_index(cdf,guide,rng.random())][0] for row,q,cdf,guide,rng in cells}
                values = [choices[i] if i in choices else v for i,v in enumerate(base)
                          if i in choices or not is_missing(v)]
                states[col] = ColumnState(values,1.)
                if not cells:
                    certain[col] = states[col]
            yield r, states

    def allows(self, ci: ColumnRef, cj: ColumnRef) -> bool:
        # 输入：ci、cj；输出：bool；继承：UncertainData(object)。
        return self.candidate_pairs is None or frozenset((ci.id, cj.id)) in self.candidate_pairs

    def pair_indices(self, ci: ColumnRef, cj: ColumnRef):
        # 输入：ci、cj；输出：(两侧对齐状态下标, 样本概率数组)；继承：UncertainData(object)。
        """Aligned r-th draws, broadcasting a certain side. Never R x R in production."""
        sa, sb = self.states(ci), self.states(cj)
        if self.realization.mode == "exact":
            a, b = np.meshgrid(np.arange(len(sa)), np.arange(len(sb)), indexing="ij")
            return a.ravel(), b.ravel(), np.outer([s.prob for s in sa], [s.prob for s in sb]).ravel()
        n = max(len(sa), len(sb))
        a, b = np.arange(n) % len(sa), np.arange(n) % len(sb)
        return a, b, np.full(n, 1.0 / n)

    def score_meta(self) -> dict:
        # 输入：实例状态；输出：采样方式和原始缺失列元信息；继承：UncertainData(object)。
        incomplete = [c.id for t in self.source.tables for c in t.columns if t.df[c.name].isna().any()]
        return {"estimator": self.realization.mode, "score_space": "raw", "incomplete_columns": incomplete,
                "n_samples_requested": self.realization.n_samples}

    def sampling_records(self) -> list[dict]:
        # 输入：实例状态；输出：逐列缺失率和采样种子记录；继承：UncertainData(object)。
        records = []
        for t in self.source.tables:
            for c in t.columns:
                n = int(t.df[c.name].isna().sum())
                records.append({"column": c.id, "table": c.table, "missing_count": n,
                                "missing_rate": n / len(t.df) if len(t.df) else 0.0,
                                "R": self.state_count(c), "run_seed": self.seed,
                                "sampling_seed": f"{self.seed},{zlib.crc32(c.id.encode('utf-8'))}",
                                "mode": self.realization.mode})
        return records

    def table_realizations(self, name: str) -> list[tuple[pd.DataFrame, float]]:
        # 输入：name；输出：对齐的表状态及概率列表；继承：UncertainData(object)。
        t = self.table(name)
        if self.realization.mode == "exact":
            return t.table_realizations(self.realization, self.seed)
        states = {c: self.states(c) for c in t.columns}
        n = max((len(s) for s in states.values()), default=1)
        out = []
        for r in range(n):
            frame = t.external_frame()
            for ci, col in enumerate(t.columns):
                rows = [i for i, v in enumerate(t.frame[col.name])
                        if not is_missing(v) or (i, col.name) in t.distributions]
                values = states[col][r % len(states[col])].values
                for i, v in zip(rows, values):
                    frame.iat[i, ci] = np.nan if v is ZERO_EMBEDDING else v
            out.append((frame, 1.0 / n))
        return out

    @property
    def is_deterministic(self) -> bool:
        # 输入：实例状态；输出：bool；继承：UncertainData(object)。
        return all(len(q) == 1 for t in self.tables for q in t.distributions.values())

    def stats(self) -> dict:
        # 输入：实例状态；输出：该对象的统计信息字典；继承：UncertainData(object)。
        n_filled = sum(int(t.filled.sum()) for t in self.tables if t.filled is not None)
        dists = [q for t in self.tables for q in t.distributions.values()]
        n_left = sum(int(t.frame.map(is_missing).values.sum()) for t in self.tables) - len(dists)
        unc_cols = {(t.name, c) for t in self.tables for (_, c), q in t.distributions.items() if len(q) > 1}
        info = [s for t in self.tables for s in t.distribution_info.values()]
        return {"n_filled_cells": n_filled, "n_cells_without_evidence": n_left,
                "n_distribution_cells": len(dists),
                "n_uncertain_cells": sum(len(q) > 1 for q in dists),
                "mean_candidates": float(np.mean([len(q) for q in dists])) if dists else 0.0,
                "n_uncertain_columns": len(unc_cols),
                "mean_retained_mass": float(np.mean([s["retained_mass"] for s in info])) if info else None,
                "n_mass_targets_missed": sum(not s["mass_target_met"] for s in info)}


# --------------------------------------------------------------------------------------------
# Pairwise tables (module 2 and module 3 outputs)
# --------------------------------------------------------------------------------------------
class _PairFrame:
    REQUIRED: list[str] = ["col_i", "col_j", "table_i", "table_j"]

    def __init__(self, df: pd.DataFrame, columns: Iterable[ColumnRef], meta: dict | None = None):
        # 输入：df、columns、meta；输出：None；初始化实例及校验参数；继承：_PairFrame(object)。
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{type(self).__name__} is missing columns {missing}")
        if (df["table_i"] == df["table_j"]).any():
            raise ValueError(f"{type(self).__name__} must not contain same-table pairs")
        self.df = df.reset_index(drop=True)
        self.columns = list(columns)
        self.meta = meta or {}
        table_of = {c.id: c.table for c in self.columns}
        if len(table_of) != len(self.columns):
            raise ValueError("column ids must be unique")
        for side in ("i", "j"):
            owners = self.df[f"col_{side}"].map(table_of)
            if (owners.isna() | (owners != self.df[f"table_{side}"])).any():
                raise ValueError(f"{type(self).__name__}: unknown column or incorrect table_{side}")

    @classmethod
    def from_records(cls, records: list[dict], columns: Iterable[ColumnRef], meta: dict | None = None):
        # 输入：records、columns、meta；输出：对应的 ScoreSamples 或 EdgeTable 实例；继承：_PairFrame(object)。
        df = pd.DataFrame.from_records(records)
        if df.empty:
            df = pd.DataFrame(columns=cls.REQUIRED)
        return cls(df, columns, meta)

    def to_csv(self, path: str) -> None:
        # 输入：path；输出：None；写入 CSV；继承：_PairFrame(object)。
        self.df.to_csv(path, index=False, encoding="utf-8")

    def __len__(self) -> int:
        # 输入：实例状态；输出：记录行数 int；继承：_PairFrame(object)。
        return len(self.df)


class ScoreSamples(_PairFrame):
    """Module 2 output: one row per cross-table pair (i, j) and evaluated pair of data states (a, b).

    Required: col_i, col_j, table_i, table_j, state_i, state_j, prob, finite score.
    score_space=raw supports arbitrary matcher scales; legacy samples default to [0,1].
    Production MC aligns the r-th states with prob=1/R; exact toy references use product weights.
    sample_id identifies the draw even when a certain side broadcasts its single state.
    Optional numeric columns (e.g. name_score, value_score, is_match) are propagated as expectations.
    """

    REQUIRED = ["col_i", "col_j", "table_i", "table_j", "state_i", "state_j", "prob", "score"]

    def __init__(self, df, columns, meta=None):
        # 输入：df、columns、meta；输出：None；初始化实例及校验参数；继承：ScoreSamples(_PairFrame)，重写父类方法。
        super().__init__(df, columns, meta)
        if self.df.empty:
            return
        p = self.df["prob"].to_numpy(float)
        s = self.df["score"].to_numpy(float)
        if not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("ScoreSamples probabilities must be finite and nonnegative")
        raw = self.meta.get("score_space") == "raw"
        if not np.isfinite(s).all() or (not raw and ((s < -1e-8).any() or (s > 1 + 1e-8).any())):
            raise ValueError("ScoreSamples scores must be finite (legacy bounded scores must be in [0, 1])")
        if (self.df.groupby(["col_i", "col_j"], sort=False)["prob"].sum() <= 0).any():
            raise ValueError("ScoreSamples must have positive probability mass for every pair")
        self.df["score"] = s if raw else np.clip(s, 0.0, 1.0)
        if self.meta.get("estimator") == "mc" and "sample_id" not in self.df:
            self.df["sample_id"] = np.maximum(self.df.state_i.to_numpy(int), self.df.state_j.to_numpy(int))

    @property
    def n_pairs(self) -> int:
        # 输入：实例状态；输出：不同跨表列对数 int；继承：ScoreSamples(_PairFrame)。
        return int(self.df.groupby(["col_i", "col_j"], sort=False).ngroups) if len(self.df) else 0


class EdgeTable(_PairFrame):
    """Module 3 output, i.e. the weighted multipartite graph G = (V, E, W, tau) of module 4.

    V = self.columns (tau = ColumnRef.table), E = rows, W = column w.
    Required: col_i, col_j, table_i, table_j, mu, sigma, w.
    """

    REQUIRED = ["col_i", "col_j", "table_i", "table_j", "mu", "sigma", "w"]

    def __init__(self, df, columns, meta=None):
        # 输入：df、columns、meta；输出：None；初始化实例及校验参数；继承：EdgeTable(_PairFrame)，重写父类方法。
        super().__init__(df, columns, meta)
        if not self.df.empty:
            if not np.isfinite(self.df[["mu", "sigma", "w"]].to_numpy(float)).all():
                raise ValueError("EdgeTable mu, sigma and w must be finite")
            if (self.df["sigma"] < 0).any():
                raise ValueError("EdgeTable sigma must be nonnegative")

    def sorted(self, by: str = "w") -> pd.DataFrame:
        # 输入：by；输出：稳定降序排列的边 DataFrame；继承：EdgeTable(_PairFrame)。
        """Rows by descending weight; ties keep the original pair order (stable)."""
        return self.df.sort_values(by, ascending=False, kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Module 4 output
# --------------------------------------------------------------------------------------------
@dataclass
class Matching:
    """Final multi-table result: disjoint semantic groups of column ids.

    info:       small JSON-serialisable statistics of the optimizer.
    artifacts:  optional in-memory DataFrames (e.g. ILP solutions for debugging).
                Runners keep the selected diagnostics inside one HTML result.
    """

    groups: list[frozenset[str]]
    info: dict = field(default_factory=dict)
    artifacts: dict[str, pd.DataFrame] = field(default_factory=dict)

    def partition(self, columns) -> list[frozenset[str]]:
        # 输入：全部列引用；输出：含未匹配单例、覆盖所有列的论文分区 Ω；继承：Matching(object)。
        universe = {c.id for c in columns}
        used = {c for group in self.groups for c in group}
        if not used <= universe:
            raise ValueError('Matching contains columns outside the input')
        return self.groups + [frozenset((c,)) for c in sorted(universe-used)]

    def as_legacy_dict(self) -> dict[int, set[str]]:
        # 输入：实例状态；输出：组编号到列集合的字典；继承：Matching(object)。
        return {i + 1: set(g) for i, g in enumerate(self.groups)}

    def to_json(self, path: str) -> None:
        # 输入：path；输出：None；写入分组 JSON；继承：Matching(object)。
        payload = {"groups": [sorted(g) for g in self.groups], "info": self.info}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
