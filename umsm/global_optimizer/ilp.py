"""Shared partition ILP: closure, same-table exclusion, transitivity, solving and groups.

Maximize sum(a_ij*y_ij), with binary y, same-table y=0 and all three triangle
inequalities per triple. Singletons are legal; reported groups have size >= 2.
PartitionILP owns closure, constraints, solving and artifacts. Each optimizer only
chooses components and supplies explicit edge potentials a (current logit evidence or the explicit raw-score baseline).
Triangle mode 'lazy' adds violated triples until the solution is transitive.
"""
from __future__ import annotations

import time
from itertools import combinations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from ..types import EdgeTable, Matching
from .base import GlobalOptimizer


# --------------------------------------------------------------------------------------------
# 1. blocking
# --------------------------------------------------------------------------------------------
def build_candidate_graph(edges: pd.DataFrame, theta: float, top_l: int | None,
                          positive_only: bool = True) -> pd.DataFrame:
    # 输入：edges、theta、top_l、positive_only；输出：候选边 DataFrame；继承：无（模块函数）。
    """Sparse candidate graph G_c: union over columns of their top_l cross-table edges by w."""
    cand = edges[edges["w"] - theta > 0] if positive_only else edges
    if top_l is None or cand.empty:
        return cand
    both = pd.concat([
        pd.DataFrame({"u": cand["col_i"], "w": cand["w"], "eid": cand.index}),
        pd.DataFrame({"u": cand["col_j"], "w": cand["w"], "eid": cand.index}),
    ], ignore_index=True)
    both = both.sort_values("w", ascending=False, kind="mergesort")
    keep = both.groupby("u", sort=False).head(top_l)["eid"].unique()
    return cand.loc[sorted(keep)]


def get_components(column_ids: list[str], cand: pd.DataFrame) -> list[list[str]]:
    # 输入：column_ids、cand；输出：至少两列的连通分量列表；继承：无（模块函数）。
    """Connected components of G_c with >= 2 columns, columns in the order of column_ids."""
    parent = {c: c for c in column_ids}

    def find(x):
        # 输入：x；输出：并查集代表节点；继承：无（get_components 内部函数）。
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(cand["col_i"], cand["col_j"]):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    comps: dict[str, list[str]] = {}
    for c in column_ids:
        comps.setdefault(find(c), []).append(c)
    return [m for m in comps.values() if len(m) >= 2]


# --------------------------------------------------------------------------------------------
# 2. one component
# --------------------------------------------------------------------------------------------
@dataclass
class ComponentProblem:
    columns: list[str]
    tables: list[str]
    I: np.ndarray            # pair index -> i   (i < j)
    J: np.ndarray            # pair index -> j
    pos: np.ndarray          # (n, n) -> pair index
    w: np.ndarray            # NaN for same-table pairs
    a: np.ndarray            # objective potential, 0 for same-table pairs
    same_table: np.ndarray   # fixed-0 variables
    n_closure_pairs: int = 0
    stats: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        # 输入：实例状态；输出：分量列数 int；继承：ComponentProblem(object)。
        return len(self.columns)


def complete_component(columns: list[str], table_of: dict[str, str], w_lookup: dict[frozenset, float],
                       candidate_pairs: set[frozenset], theta: float, missing_w: float | None,
                       a_lookup: dict[frozenset, float] | None = None) -> ComponentProblem:
    # 输入：columns、table_of、w_lookup、candidate_pairs、theta、missing_w、a_lookup；输出：含所有列对变量和 signed potential 的 ComponentProblem；继承：无（模块函数）。
    """Component closure: a variable for every pair, the score of every cross-table pair."""
    n = len(columns)
    I, J = np.triu_indices(n, 1)
    pos = np.full((n, n), -1, dtype=np.int64)
    pos[I, J] = pos[J, I] = np.arange(len(I))
    tables = [table_of[c] for c in columns]
    same = np.array([tables[i] == tables[j] for i, j in zip(I, J)], dtype=bool)
    w = np.full(len(I), np.nan)
    n_closure = 0
    for k, (i, j) in enumerate(zip(I, J)):
        if same[k]:
            continue
        key = frozenset((columns[i], columns[j]))
        if key not in w_lookup and missing_w is None:
            raise ValueError(f"component closure needs score for {columns[i]} / {columns[j]}; "
                             "supply every cross-table pair or explicitly configure missing_w")
        w[k] = w_lookup.get(key, missing_w)
        n_closure += key not in candidate_pairs
    a = np.where(same, 0.0, w - theta)
    if a_lookup is not None:
        for k, (i, j) in enumerate(zip(I, J)):
            if not same[k]:
                a[k] = a_lookup[frozenset((columns[i], columns[j]))]
    return ComponentProblem(columns, tables, I, J, pos, w, a, same, n_closure)


def pair_lookup(frame: pd.DataFrame, value: str) -> dict[frozenset, float]:
    # 输入：frame、value；输出：无向列对到平均边值的字典；继承：无（模块函数）。
    """One undirected value per pair; average duplicate/directional observations."""
    values = {}
    for a, b, score in frame[["col_i", "col_j", value]].itertuples(index=False, name=None):
        values.setdefault(frozenset((a, b)), []).append(float(score))
    return {pair: float(np.mean(scores)) for pair, scores in values.items()}


def triangle_matrix(triples: np.ndarray, pos: np.ndarray, n_vars: int) -> sparse.csr_matrix:
    # 输入：triples、pos、n_vars；输出：三角约束的 CSR 稀疏矩阵；继承：无（模块函数）。
    """Three rows per triple i < j < k (see module docstring)."""
    if len(triples) == 0:
        return sparse.csr_matrix((0, n_vars))
    i, j, k = triples[:, 0], triples[:, 1], triples[:, 2]
    ij, jk, ik = pos[i, j], pos[j, k], pos[i, k]
    t = len(triples)
    rows = np.repeat(np.arange(3 * t), 3)
    cols = np.stack([ij, jk, ik, ij, ik, jk, ik, jk, ij], axis=1).reshape(-1)
    vals = np.tile([1.0, 1.0, -1.0], 3 * t)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(3 * t, n_vars))


def all_triples(n: int) -> np.ndarray:
    # 输入：n；输出：全部三元组下标数组；继承：无（模块函数）。
    if n < 3:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(list(combinations(range(n), 3)), dtype=np.int64)


def violated_triples(y: np.ndarray, prob: ComponentProblem) -> np.ndarray:
    # 输入：y、prob；输出：违反传递性的三元组数组；继承：无（模块函数）。
    """Triples (i < j < k) with two pairs at 1 and the third at 0."""
    Y = np.zeros((prob.n, prob.n), dtype=bool)
    on = y > 0.5
    Y[prob.I[on], prob.J[on]] = Y[prob.J[on], prob.I[on]] = True
    found = set()
    for j in range(prob.n):
        nb = np.flatnonzero(Y[j])
        if len(nb) < 2:
            continue
        sub = Y[np.ix_(nb, nb)]
        a, b = np.triu_indices(len(nb), 1)
        bad = ~sub[a, b]
        for x, z in zip(nb[a[bad]], nb[b[bad]]):
            found.add(tuple(sorted((int(x), j, int(z)))))
    return np.array(sorted(found), dtype=np.int64).reshape(-1, 3)


def build_ilp(prob: ComponentProblem, triples: np.ndarray) -> dict:
    # 输入：prob、triples；输出：SciPy MILP 的目标、边界和线性约束；继承：无（模块函数）。
    """c (minimisation form), bounds (fixed 0 for same-table pairs) and triangle constraints."""
    n_vars = len(prob.a)
    ub = np.where(prob.same_table, 0.0, 1.0)
    A = triangle_matrix(triples, prob.pos, n_vars)
    return {"c": -prob.a, "bounds": Bounds(np.zeros(n_vars), ub), "A": A,
            "integrality": np.ones(n_vars)}


def solve_ilp(model: dict, time_limit: float | None, mip_rel_gap: float) -> tuple[np.ndarray, int, str]:
    # 输入：model、time_limit、mip_rel_gap；输出：(二元解数组, 求解状态码, 求解消息)；继承：无（模块函数）。
    options = {"disp": False, "mip_rel_gap": mip_rel_gap}
    if time_limit is not None:
        options["time_limit"] = time_limit
    A = model["A"]
    cons = LinearConstraint(A, -np.inf, 1.0) if A.shape[0] else None
    res = milp(c=model["c"], integrality=model["integrality"], bounds=model["bounds"],
               constraints=cons, options=options)
    if res.x is None:
        raise RuntimeError(f"ILP solver failed: status={res.status} {res.message}")
    return np.round(res.x).astype(int), int(res.status), str(res.message)


def recover_groups(prob: ComponentProblem, y: np.ndarray) -> list[frozenset[str]]:
    # 输入：prob、y；输出：从二元解恢复的列语义组；继承：无（模块函数）。
    """Connected components of the solution graph y_ij = 1, with the debug checks of V1 Sec. 6."""
    parent = list(range(prob.n))

    def find(x):
        # 输入：x；输出：并查集代表节点；继承：无（recover_groups 内部函数）。
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k in np.flatnonzero(y == 1):
        ri, rj = find(prob.I[k]), find(prob.J[k])
        if ri != rj:
            parent[rj] = ri
    members: dict[int, list[int]] = {}
    for v in range(prob.n):
        members.setdefault(find(v), []).append(v)
    groups = []
    for m in members.values():
        if len(m) < 2:
            continue
        tabs = [prob.tables[v] for v in m]
        if len(set(tabs)) != len(tabs):
            raise RuntimeError(f"ILP group {[prob.columns[v] for v in m]} has two columns of one table")
        for x in range(len(m)):
            for z in range(x + 1, len(m)):
                if y[prob.pos[m[x], m[z]]] != 1:
                    raise RuntimeError(f"ILP group {[prob.columns[v] for v in m]} is not a clique (transitivity)")
        groups.append(frozenset(prob.columns[v] for v in m))
    return groups


def merge_results(per_component: list[list[frozenset[str]]]) -> list[frozenset[str]]:
    # 输入：per_component；输出：合并后的 list[frozenset[str]]；继承：无（模块函数）。
    return [g for groups in per_component for g in groups]


# --------------------------------------------------------------------------------------------
# 3. optimizer
# --------------------------------------------------------------------------------------------
class PartitionILP(GlobalOptimizer):
    """Shared solver; subclasses provide components and objective potentials."""

    def __init__(self, triangles: str = "auto", full_max_columns: int = 40,
                 time_limit: float | None = None, mip_rel_gap: float = 0.0, max_lazy_rounds: int = 200,
                 save_solution: bool = True, require_optimal: bool = True):
        # 输入：triangles、full_max_columns、time_limit、mip_rel_gap、max_lazy_rounds、save_solution、require_optimal；输出：None；初始化实例及校验参数；继承：PartitionILP(GlobalOptimizer)。
        if triangles not in ("auto", "full", "lazy"):
            raise ValueError("triangles must be 'auto', 'full' or 'lazy'")
        self.triangles = triangles
        self.full_max_columns = full_max_columns
        self.time_limit = time_limit
        self.mip_rel_gap = mip_rel_gap
        self.max_lazy_rounds = max_lazy_rounds
        self.save_solution = save_solution
        self.require_optimal = require_optimal

    def _solve_component(self, prob: ComponentProblem) -> np.ndarray:
        # 输入：prob；输出：二元解 y 数组；诊断写入 prob.stats；继承：PartitionILP(GlobalOptimizer)。
        mode = self.triangles
        if mode == "auto":
            mode = "full" if prob.n <= self.full_max_columns else "lazy"
        t0 = time.perf_counter()
        if mode == "full":
            triples = all_triples(prob.n)
            y, status, msg = solve_ilp(build_ilp(prob, triples), self.time_limit, self.mip_rel_gap)
            rounds = 1
        else:
            triples = np.zeros((0, 3), dtype=np.int64)
            rounds = 0
            while True:
                rounds += 1
                y, status, msg = solve_ilp(build_ilp(prob, triples), self.time_limit, self.mip_rel_gap)
                new = violated_triples(y, prob)
                if len(new) == 0:
                    break
                if rounds >= self.max_lazy_rounds:
                    raise RuntimeError(f"lazy triangle separation did not converge in {rounds} rounds")
                triples = np.unique(np.concatenate([triples, new]), axis=0)
        if self.require_optimal and status != 0:
            raise RuntimeError(f"ILP did not reach optimality: status={status} {msg}; "
                               "require_optimal=false explicitly permits feasible approximate results")
        prob.stats = {"n_columns": prob.n, "n_tables": len(set(prob.tables)), "n_variables": len(prob.a),
                      "n_fixed_zero": int(prob.same_table.sum()), "n_closure_pairs": prob.n_closure_pairs,
                      "triangle_mode": mode, "n_triangle_constraints": 3 * len(triples), "solver_rounds": rounds,
                      "solver_time_s": time.perf_counter() - t0, "objective": float(prob.a @ y),
                      "status": status, "message": msg}
        return y

    def _optimize_components(self, graph, ctx, comps, cand_pairs, w_lookup, *, info,
                             a_lookup=None, theta=0., missing_w=None):
        # 输入：graph、ctx、comps 等配置参数（详见签名）；输出：全部分量合并后的 Matching；继承：PartitionILP(GlobalOptimizer)。
        """Solve supplied components without changing the meaning of w or a."""
        column_ids = [c.id for c in graph.columns]
        table_of = {c.id: c.table for c in graph.columns}

        per_comp, comp_rows, sol_rows = [], [], []
        for cid, cols in enumerate(comps):
            prob = complete_component(cols, table_of, w_lookup, cand_pairs, theta, missing_w, a_lookup)
            y = self._solve_component(prob)
            groups = recover_groups(prob, y)
            per_comp.append(groups)
            comp_rows.append({"component": cid, **prob.stats, "n_groups": len(groups)})
            if self.save_solution:
                for k, (i, j) in enumerate(zip(prob.I, prob.J)):
                    sol_rows.append({"component": cid, "col_i": cols[i], "col_j": cols[j],
                                     "same_table": bool(prob.same_table[k]),
                                     "candidate": frozenset((cols[i], cols[j])) in cand_pairs,
                                     "w": prob.w[k], "a": prob.a[k], "y": int(y[k])})

        comp_df = pd.DataFrame(comp_rows)
        sizes = {}
        for c in graph.columns:
            sizes[c.table] = sizes.get(c.table, 0) + 1
        expected_pairs = (len(column_ids) ** 2 - sum(n ** 2 for n in sizes.values())) // 2
        info = {**info, "n_candidate_edges": len(cand_pairs),
                "scores_complete": len(w_lookup) == expected_pairs,
                "optimality_required": self.require_optimal,
                "mip_rel_gap": self.mip_rel_gap,
                "n_components": len(comps),
                "max_component_columns": int(comp_df["n_columns"].max()) if len(comp_df) else 0,
                "n_variables": int(comp_df["n_variables"].sum()) if len(comp_df) else 0,
                "n_triangle_constraints": int(comp_df["n_triangle_constraints"].sum()) if len(comp_df) else 0,
                "n_closure_pairs": int(comp_df["n_closure_pairs"].sum()) if len(comp_df) else 0,
                "solver_time_s": float(comp_df["solver_time_s"].sum()) if len(comp_df) else 0.0,
                "objective": float(comp_df["objective"].sum()) if len(comp_df) else 0.0,
                "all_optimal": bool((comp_df["status"] == 0).all()) if len(comp_df) else True}
        info["global_exact"] = (info["scores_complete"] and info["blocking_exact"] and info["all_optimal"]
                                and self.mip_rel_gap == 0 and missing_w is None)
        info["optimal_within_blocking"] = info["all_optimal"] and self.mip_rel_gap == 0 and missing_w is None
        if getattr(ctx, "logger", None):
            ctx.logger.info(f"  ILP: {info['n_components']} component(s), max {info['max_component_columns']} columns, "
                            f"{info['n_variables']} variables, {info['n_triangle_constraints']} triangle constraints, "
                            f"objective {info['objective']:.4f}, solver {info['solver_time_s']:.3f}s")
        artifacts = {"ilp_components": comp_df}
        if self.save_solution:
            artifacts["ilp_solution"] = pd.DataFrame(sol_rows)
        return Matching(merge_results(per_comp), info, artifacts)
