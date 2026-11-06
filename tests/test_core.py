"""Fast checks of the framework (no language model needed).

    python tests/test_core.py        or        python -m pytest tests

Global Optimization V1 tests follow Sec. 9-11 of Global_Optimization.docx (Toy 1, Toy 2, T3-T5).
"""
import itertools
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from umsm import OPTIMIZERS, UNCERTAINTY_MODELS
from umsm.context import RunContext  # noqa: E402
from umsm.data import load_dataset, read_groundtruth, write_groundtruth  # noqa: E402
from umsm.evaluation import evaluate, legacy_cluster_precision_recall  # noqa: E402
from umsm.global_optimizer.base import gap_cut  # noqa: E402
from umsm.global_optimizer.greedy_merge import greedy_groups  # noqa: E402
from umsm.matcher.base import PairwiseMatcher, sample_record  # noqa: E402
from umsm.types import ZERO_EMBEDDING, ColumnRef, EdgeTable, IntegrationSet, Matching, ScoreSamples, Table


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def _toy_set():
    t1 = Table("t1.csv", pd.DataFrame({"Ambience": [5, 10, None], "Dress": ["a", "b", "a"]}))
    t2 = Table("t2.csv", pd.DataFrame({"Atmosphere": [10, 10, 5], "Meals": ["x", None, "y"]}))
    t3 = Table("t3.csv", pd.DataFrame({"Env": [None, 5.0, 10.0]}))
    gt = [frozenset({"t1.csv.Ambience", "t2.csv.Atmosphere", "t3.csv.Env"})]
    return IntegrationSet("Toy", "Toy", [t1, t2, t3], gt)


def _graph(cols: list[ColumnRef], w: dict[tuple[str, str], float], **extra) -> EdgeTable:
    """EdgeTable over all cross-table pairs of cols; unspecified pairs get w = 0."""
    recs = []
    for a, b in itertools.combinations(cols, 2):
        if a.table == b.table:
            continue
        v = w.get((a.id, b.id), w.get((b.id, a.id), 0.0))
        recs.append({"col_i": a.id, "col_j": b.id, "table_i": a.table, "table_j": b.table,
                     "mu": v, "sigma": 0.0, "w": v, **{k: f(a, b) for k, f in extra.items()}})
    return EdgeTable(pd.DataFrame(recs), cols)


def _ilp(graph, **params):
    return OPTIMIZERS.build("ilp", {"theta": 0.5, "top_l": None, **params}).optimize(graph, RunContext())


def _y(matching, a, b):
    sol = matching.artifacts["ilp_solution"]
    row = sol[((sol.col_i == a) & (sol.col_j == b)) | ((sol.col_i == b) & (sol.col_j == a))]
    return int(row["y"].iloc[0]) if len(row) else 0


ABCD = [ColumnRef("T1", "A"), ColumnRef("T2", "B"), ColumnRef("T3", "C"), ColumnRef("T4", "D")]
TOY1 = {("T1.A", "T2.B"): 0.95, ("T1.A", "T3.C"): 0.87, ("T2.B", "T3.C"): 0.90,
        ("T3.C", "T4.D"): 0.70, ("T1.A", "T4.D"): 0.10, ("T2.B", "T4.D"): 0.15}


# ------------------------------------------------------------------------------------------
# Global Optimization V1 (Global_Optimization.docx Sec. 9-11)
# ------------------------------------------------------------------------------------------
def test_ilp_toy1_over_merge():
    for mode in ("full", "lazy"):
        m = _ilp(_graph(ABCD, TOY1), triangles=mode)
        assert m.groups == [frozenset({"T1.A", "T2.B", "T3.C"})], (mode, m.groups)
        assert [_y(m, *p) for p in [("T1.A", "T2.B"), ("T1.A", "T3.C"), ("T2.B", "T3.C")]] == [1, 1, 1]
        assert [_y(m, *p) for p in [("T3.C", "T4.D"), ("T1.A", "T4.D"), ("T2.B", "T4.D")]] == [0, 0, 0]
        assert abs(m.info["objective"] - 1.22) < 1e-9


def test_greedy_over_merges_toy1():
    m = OPTIMIZERS.build("greedy_merge", {"selection": "threshold", "threshold": 0.5}).optimize(
        _graph(ABCD, TOY1), RunContext())
    assert m.groups == [frozenset({"T1.A", "T2.B", "T3.C", "T4.D"})]


def test_ilp_toy2_same_table_conflict():
    cols = [ColumnRef("T1", "A1"), ColumnRef("T1", "A2"), ColumnRef("T2", "B")]
    m = _ilp(_graph(cols, {("T1.A1", "T2.B"): 0.90, ("T1.A2", "T2.B"): 0.88}))
    assert m.groups == [frozenset({"T1.A1", "T2.B"})]
    assert (_y(m, "T1.A1", "T1.A2"), _y(m, "T1.A1", "T2.B"), _y(m, "T1.A2", "T2.B")) == (0, 1, 0)
    sol = m.artifacts["ilp_solution"]
    assert bool(sol[(sol.col_i == "T1.A1") & (sol.col_j == "T1.A2")]["same_table"].iloc[0])
    # Sec. 10.2: swapped scores -> the solver really optimizes
    m = _ilp(_graph(cols, {("T1.A1", "T2.B"): 0.82, ("T1.A2", "T2.B"): 0.93}))
    assert m.groups == [frozenset({"T1.A2", "T2.B"})]


def test_ilp_t3_all_negative():
    m = _ilp(_graph(ABCD, {k: 0.2 for k in TOY1}), candidate_positive_only=False)
    assert m.groups == [] and all(m.artifacts["ilp_solution"]["y"] == 0)


def test_ilp_t4_all_positive():
    m = _ilp(_graph(ABCD, {k: 0.9 for k in TOY1}))
    assert m.groups == [frozenset(c.id for c in ABCD)]
    assert all(m.artifacts["ilp_solution"]["y"] == 1)


def test_ilp_t5_candidate_chain_closure():
    cols = ABCD[:3]
    # top-1 candidates: A->B, B->A, C->B  =>  candidate graph A - B - C, A-C not a candidate
    w = {("T1.A", "T2.B"): 0.90, ("T2.B", "T3.C"): 0.85, ("T1.A", "T3.C"): 0.60}
    m = _ilp(_graph(cols, w), top_l=1)
    sol = m.artifacts["ilp_solution"]
    ac = sol[(sol.col_i == "T1.A") & (sol.col_j == "T3.C")]
    assert len(ac) == 1 and not bool(ac["candidate"].iloc[0]) and abs(ac["w"].iloc[0] - 0.60) < 1e-12
    assert m.info["n_closure_pairs"] == 1 and m.groups == [frozenset({"T1.A", "T2.B", "T3.C"})]
    # with a negative closure pair, transitivity forbids the chain: best is {A, B} (0.40 > 0.40 + 0.35 - 0.40)
    w[("T1.A", "T3.C")] = 0.10
    m = _ilp(_graph(cols, w), top_l=1)
    assert m.groups == [frozenset({"T1.A", "T2.B"})]


def _brute_force(cols, a):
    """Best objective over all partitions with at most one column per table and group."""
    n, best = len(cols), -np.inf

    def partitions(items):
        if not items:
            yield []
            return
        first, rest = items[0], items[1:]
        for p in partitions(rest):
            yield [[first]] + p
            for k in range(len(p)):
                yield p[:k] + [[first] + p[k]] + p[k + 1:]

    for part in partitions(list(range(n))):
        if any(len({cols[v].table for v in g}) != len(g) for g in part):
            continue
        best = max(best, sum(a.get((min(x, z), max(x, z)), 0.0) for g in part for x, z in itertools.combinations(g, 2)))
    return best


def test_ilp_matches_brute_force_on_random_instances():
    rng = np.random.default_rng(0)
    for trial in range(15):
        tabs = rng.integers(0, 3, size=6)
        cols = [ColumnRef(f"T{t}", f"c{k}") for k, t in enumerate(tabs)]
        w = {}
        for x, z in itertools.combinations(range(6), 2):
            if cols[x].table != cols[z].table:
                w[(cols[x].id, cols[z].id)] = float(rng.uniform(0, 1))
        a = {(x, z): w[(cols[x].id, cols[z].id)] - 0.5 for x, z in itertools.combinations(range(6), 2)
             if (cols[x].id, cols[z].id) in w}
        best = _brute_force(cols, a)
        for mode in ("full", "lazy"):
            m = _ilp(_graph(cols, w), triangles=mode)
            assert abs(m.info["objective"] - max(best, 0.0)) < 1e-9, (trial, mode, m.info["objective"], best)


# ------------------------------------------------------------------------------------------
# Modules 1 and 3 (Four_Module.docx Sec. 9)
# ------------------------------------------------------------------------------------------


def test_uncertainty_models():
    iset = _toy_set()
    ctx = RunContext()
    d = UNCERTAINTY_MODELS.build("observed_only").estimate(iset, ctx)
    assert d.table("t1.csv").column_values("Ambience") == [5, 10] and d.is_deterministic
    d = UNCERTAINTY_MODELS.build("zero_embedding").estimate(iset, ctx)
    vals = d.table("t3.csv").column_values("Env")
    assert vals[0] is ZERO_EMBEDDING and len(vals) == 3 and d.stats()["n_filled_cells"] == 3
    d = UNCERTAINTY_MODELS.build("mode").estimate(iset, ctx)
    assert d.table("t2.csv").frame.loc[1, "Meals"] in ("x", "y") and d.stats()["n_cells_without_evidence"] == 0
    d = UNCERTAINTY_MODELS.build("column_marginal").estimate(iset, ctx)
    assert sorted(d.table("t1.csv").distributions[(2, "Ambience")]) == [(5, 0.5), (10, 0.5)]
    assert len(d.states(ColumnRef("t1.csv", "Ambience"))) == 32 and not d.is_deterministic
    d = UNCERTAINTY_MODELS.build("knn_tuple", {"k": 1}).estimate(iset, ctx)
    # row 2 of t1 has Dress = 'a' like row 0 (Ambience = 5) -> nearest neighbour votes 5
    assert d.table("t1.csv").distributions[(2, "Ambience")] == [(5, 1.0)]
    assert iset.tables[0].df["Ambience"].isna().sum() == 1  # original tables are never modified


class _ValueOverlap(PairwiseMatcher):
    """Jaccard overlap of the string values of two column states."""

    registry_name = "value_overlap"

    def similarity(self, ci, si, cj, sj, ctx):
        a, b = {str(float(v)) if isinstance(v, (int, float)) else str(v) for v in si.values}, \
               {str(float(v)) if isinstance(v, (int, float)) else str(v) for v in sj.values}
        return len(a & b) / len(a | b) if a | b else 0.0


# ------------------------------------------------------------------------------------------
# baselines, evaluation and data
# ------------------------------------------------------------------------------------------
def test_legacy_metric_does_not_penalise_split():
    gt = [frozenset("abcd")]
    p, r = legacy_cluster_precision_recall([frozenset("ab"), frozenset("cd")], gt)
    assert (p, r) == (1.0, 1.0)
    m = evaluate(Matching([frozenset("ab"), frozenset("cd")]), gt)
    assert m["pair_recall"] == 2 / 6 and m["pair_precision"] == 1.0
    assert m["bcubed_recall"] < 1.0


def test_evaluate_perfect():
    gt = [frozenset("abc"), frozenset("de")]
    m = evaluate(Matching(list(gt)), gt)
    assert all(m[f"{k}_f1"] == 1.0 for k in ("cluster", "pair", "bcubed"))


def test_gap_cut_matches_algorithm1():
    sims = 1 - np.array([0.1, 0.2, 0.8, 0.9])
    assert gap_cut(sims) + 1 == 2


def test_greedy_merge_example_2_2():
    edges = [("T3.Atm", "T1.Amb", "T3", "T1"), ("T4.Env", "T3.Atm", "T4", "T3"), ("T1.Amb", "T2.Overall", "T1", "T2")]
    assert greedy_groups(edges) == [{"T3.Atm", "T1.Amb", "T4.Env", "T2.Overall"}]


def test_greedy_merge_respects_same_table():
    edges = [("A.x", "B.y", "A", "B"), ("B.y", "A.z", "B", "A")]
    assert greedy_groups(edges) == [{"A.x", "B.y"}]


def test_local_merge_removes_same_table_conflicts():
    cols = [ColumnRef("A", "x"), ColumnRef("B", "y"), ColumnRef("C", "w"), ColumnRef("A", "z")]
    w = {("A.x", "B.y"): 0.9, ("B.y", "C.w"): 0.8, ("C.w", "A.z"): 0.7}
    g = _graph(cols, w, is_match=lambda a, b: float((a.id, b.id) in w or (b.id, a.id) in w))
    m = OPTIMIZERS.build("local_merge", {"local_decision": "is_match"}).optimize(g, RunContext())
    assert m.groups == [frozenset({"A.x", "B.y", "C.w"})]


def test_loader_and_groundtruth_roundtrip():
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "Toy", "original"))
        for t in _toy_set().tables:
            t.df.to_csv(os.path.join(root, "Toy", "original", t.name), index=False)
        write_groundtruth(_toy_set().groundtruth, os.path.join(root, "Toy", "groundtruth.txt"))
        sets = load_dataset("Toy", root)
        assert len(sets) == 1 and [t.name for t in sets[0].tables] == ["t1.csv", "t2.csv", "t3.csv"]
        assert sets[0].groundtruth == _toy_set().groundtruth
        assert [len(t.df) for t in sets[0].tables] == [len(t.df) for t in _toy_set().tables]
        assert sets[0].stats()["n_missing_cells"] == _toy_set().stats()["n_missing_cells"]
        assert read_groundtruth(os.path.join(root, "Toy", "groundtruth.txt")) == _toy_set().groundtruth
        os.makedirs(os.path.join(root, "Multi", "s1", "original"))
        for t in _toy_set().tables:
            t.df.to_csv(os.path.join(root, "Multi", "s1", "original", t.name), index=False)
        write_groundtruth(_toy_set().groundtruth, os.path.join(root, "Multi", "s1", "groundtruth.txt"))
        sets = load_dataset("Multi", root)
        assert [s.name for s in sets] == ["Multi/s1"] and sets[0].groundtruth


def test_pair_tables_reject_same_table_pairs():
    c = [ColumnRef("A", "x"), ColumnRef("A", "y")]
    try:
        ScoreSamples.from_records([sample_record(c[0], c[1], 0, 0, 1.0, 1.0)], c)
    except ValueError:
        return
    raise AssertionError("same-table pair accepted")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
