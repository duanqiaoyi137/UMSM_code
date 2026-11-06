"""Sampling theory: exact toy references, paired Monte Carlo, sparse closure and logs.

Run: python tests/test_sampling.py. No model downloads, API requests or temp files.
"""
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from umsm import OPTIMIZERS, PROPAGATIONS
from umsm.candidates import plan_candidates
from umsm.config import load_config
from umsm.context import RunContext
from umsm.experiment import compute_edges
from umsm.matcher.base import PairwiseMatcher, sample_record
from umsm.types import (ColumnRef, IntegrationSet, RealizationSpec, ScoreSamples, Table,
                        UncertainData, UncertainTable)
from umsm.uncertainty_model.base import DistributionModel, truncate_and_normalize


class Lookup(PairwiseMatcher):
    def __init__(self, scores):
        self._scores = scores
        self.calls = 0

    def similarity(self, ci, si, cj, sj, ctx):
        self.calls += 1
        return self._scores[(si.values[0], sj.values[0])] if si.values and sj.values else .8


def toy(distributions, r=10000, mode="mc", seed=2027):
    tables, uncertain = [], []
    for name, q in zip(("A", "B"), distributions):
        frame = pd.DataFrame({"x": [np.nan if q else 0]}, dtype=object)
        tables.append(Table(name, frame))
        uncertain.append(UncertainTable(name, frame.copy(), distributions={(0, "x"): q} if q else {}))
    return UncertainData(IntegrationSet("Toy", "Toy", tables), uncertain, "toy").configure(
        RealizationSpec(n_samples=r, mode=mode), seed)


def propagate(data, lookup, alpha=.5):
    samples = lookup.score(data, RunContext())
    edge = PROPAGATIONS.build("mean_std", {"alpha": alpha}).propagate(samples, RunContext()).df.iloc[0]
    return samples, edge


def test_toy1_exact_and_monte_carlo():
    q = [(5, .3), (10, .7)]
    scores = {(0, 5): .65, (0, 10): .90}
    _, exact = propagate(toy([None, q], mode="exact"), Lookup(scores))
    assert np.isclose(exact.mu, .825) and np.isclose(exact.variance, .013125)
    assert np.isclose(exact.sigma, np.sqrt(.013125))
    assert np.isclose(exact.w, .825 - .5 * np.sqrt(.013125))
    mc, edge = propagate(toy([None, q]), Lookup(scores))
    assert len(mc) == 10000 and edge.estimator == "mc"
    assert abs(edge.mu - exact.mu) < .01 and abs(edge.sigma - exact.sigma) < .01


def test_toy2_exact_and_monte_carlo():
    distributions = [[("a1", .6), ("a2", .4)], [("b1", .7), ("b2", .3)]]
    scores = {("a1", "b1"): .9, ("a1", "b2"): .6, ("a2", "b1"): .5, ("a2", "b2"): .2}
    exact_samples, exact = propagate(toy(distributions, mode="exact"), Lookup(scores))
    assert len(exact_samples) == 4
    assert np.isclose(exact.mu, .65) and np.isclose(exact.variance, .0573)
    assert np.isclose(exact.w, .65 - .5 * np.sqrt(.0573))
    lookup = Lookup(scores)
    samples, mc = propagate(toy(distributions), lookup)
    assert len(samples) == lookup.calls == 10000  # R calls, never R**2
    assert (samples.df.state_i == samples.df.state_j).all()
    assert abs(mc.mu - .65) < .02 and abs(mc.sigma - np.sqrt(.0573)) < .02


def test_unbiased_variance_and_clipped_weight():
    a, b = ColumnRef("A", "x"), ColumnRef("B", "x")
    samples = ScoreSamples.from_records([sample_record(a, b, i, i, .5, s) for i, s in enumerate([0., 1.])],
                                        [a, b], meta={"estimator": "mc"})
    e = PROPAGATIONS.build("mean_std", {"alpha": 1.}).propagate(samples, RunContext()).df.iloc[0]
    assert np.isclose(e.variance, .5)  # 1/(R-1), not .25 from 1/R
    assert e.w == 0. and np.isclose(e.sigma, np.sqrt(.5))


def test_probability_mass_truncation_and_cap():
    q, info = truncate_and_normalize([("a", .6), ("b", .3), ("c", .1)], rho=.85, k_max=10)
    assert len(q) == 2 and np.isclose(info["retained_mass"], .9)
    assert np.isclose(sum(p for _, p in q), 1.) and np.isclose(q[0][1], 2/3)
    q, info = truncate_and_normalize([("a", .6), ("b", .3), ("c", .1)], rho=.99, k_max=1)
    assert q == [("a", 1.)] and not info["mass_target_met"] and info["candidate_count_raw"] == 3


def test_many_missing_cells_never_enumerate_joint_worlds():
    frame = pd.DataFrame({"x": [np.nan] * 20}, dtype=object)
    table = UncertainTable("T", frame, distributions={(r, "x"): [(i, .2) for i in range(5)] for r in range(20)})
    with patch("umsm.types.itertools.product", side_effect=AssertionError("joint enumeration forbidden")):
        states = table.column_states("x", RealizationSpec(n_samples=16), 2027)
    assert len(states) == 16 and all(len(s.values) == 20 and s.prob == 1/16 for s in states)


def test_reproducible_column_reuse_and_table_alignment():
    data = toy([[(1, .2), (2, .8)], [(3, .6), (4, .4)]], r=32)
    col = data.columns[0]
    assert data.states(col) is data.states(col)
    again = toy([[(1, .2), (2, .8)], [(3, .6), (4, .4)]], r=32)
    assert data.states(col) == again.states(col)
    frames = data.table_realizations("A")
    assert [f.iat[0, 0] for f, _ in frames] == [s.values[0] for s in data.states(col)]
    assert data.sampling_records()[0]["R"] == 32


def test_complete_pair_is_scored_once():
    data = toy([None, None], r=64)
    lookup = Lookup({(0, 0): .7})
    samples, edge = propagate(data, lookup)
    assert len(samples) == lookup.calls == 1
    assert edge.sigma == 0. and edge.w == .7 and edge.edge_type == "complete_complete"


def test_empty_candidate_graph_has_valid_empty_outputs():
    data = toy([None, None], r=16)
    matcher = Lookup({(0, 0): .7})
    with patch('umsm.candidates.blocking_scores', return_value=pd.DataFrame(
            [{'col_i': 'A.x', 'col_j': 'B.x', 'w': 0.}])):
        data.candidate_pairs, _ = plan_candidates(data.source, RunContext(), {"min_score": 1., "top_l": 1})
    samples = matcher.score(data, RunContext())
    edges = PROPAGATIONS.build("mean_std").propagate(samples, RunContext())
    assert len(samples) == len(edges) == 0 and "variance" in edges.df and "n_samples" in edges.df
    m = OPTIMIZERS.build("ilp", {"theta": .5}).optimize(edges, RunContext())
    assert m.groups == []


def test_vector_matchers_score_only_aligned_candidate_draws():
    from test_matchers import FakeEncoder
    from umsm import MATCHERS
    from umsm.matcher import alite, magneto, sbert_quantile

    data = toy([[('a1', .6), ('a2', .4)], [('b1', .7), ('b2', .3)]], r=16)
    frame = pd.DataFrame({"other": ["unrelated"]})
    data.source.tables.append(Table("C", frame))
    data.tables.append(UncertainTable("C", frame))
    data.candidate_pairs = {frozenset(("A.x", "B.x"))}
    for name, module in (("sbert_quantile", sbert_quantile), ("alite", alite), ("magneto", magneto)):
        with patch.object(module, "get_encoder", return_value=FakeEncoder()):
            scores = MATCHERS.build(name).score(data, RunContext())
        assert scores.n_pairs == 1 and len(scores) == 16, name
        assert (scores.df.state_i == scores.df.state_j).all(), name


def test_candidate_screening_closes_components_before_R_scores():
    class NameMatcher(PairwiseMatcher):
        def __init__(self):
            self.calls = 0

        def similarity(self, ci, si, cj, sj, ctx):
            self.calls += 1
            if ci.name != cj.name:
                return 0.
            return {frozenset("AB"): .9, frozenset("BC"): .8, frozenset("AC"): .7}[frozenset((ci.table, cj.table))]

    source = IntegrationSet("Toy", "Toy", [Table(t, pd.DataFrame({"x": [np.nan], "y": [np.nan]}, dtype=object))
                                           for t in ("A", "B", "C")])
    tables = [UncertainTable(t.name, t.df.copy(), distributions={(0, c): [(1, .5), (2, .5)] for c in ("x", "y")})
              for t in source.tables]
    data = UncertainData(source, tables, "toy").configure(RealizationSpec(n_samples=16), 2027)
    matcher = NameMatcher()
    blocking = pd.DataFrame([{'col_i': a.id, 'col_j': b.id, 'w':
                              ({frozenset('AB'): .9, frozenset('BC'): .8, frozenset('AC'): .7}[frozenset((a.table, b.table))]
                               if a.name == b.name else 0.)}
                             for a, b in __import__('itertools').combinations(data.columns, 2) if a.table != b.table])
    with patch('umsm.candidates.blocking_scores', return_value=blocking):
        data.candidate_pairs, info = plan_candidates(data.source, RunContext(), {"mode": "name_top_l", "top_l": 1})
    assert info["n_possible_pairs"] == 12 and info["n_evaluated_pairs"] == 6
    assert info["n_candidate_pairs"] == 4, info  # two chains; closure supplies the other two pairs
    samples = matcher.score(data, RunContext())
    assert samples.n_pairs == 6 and len(samples) == 6 * 16
    assert matcher.calls == 6 * 16  # the cheap blocking pass never calls the base matcher
    edges = PROPAGATIONS.build("mean_std").propagate(samples, RunContext())
    m = OPTIMIZERS.build("ilp", {"theta": .5, "top_l": None}).optimize(edges, RunContext())
    assert set(m.groups) == {frozenset(f"{t}.x" for t in "ABC"), frozenset(f"{t}.y" for t in "ABC")}


def test_four_modules_mc_end_to_end_and_exact_rejected_in_runner():
    class Q(DistributionModel):
        def cell_distributions(self, frame, table, iset, ctx):
            return {(0, "x"): [(5, .3), (10, .7)]} if table.name == "B" else {}

    data = toy([None, [(5, .3), (10, .7)]], r=32)
    matcher = Lookup({(0, 5): .65, (0, 10): .90})
    cfg = load_config()
    cfg["matcher"]["cache_scores"] = False
    cfg["candidates"] = {"mode": "all"}
    components = {"uncertainty_model": Q(), "matcher": matcher,
                  "uncertainty_propagation": PROPAGATIONS.build("mean_std")}
    estimated, samples, edges, hit = compute_edges(data.source, components, RealizationSpec(), cfg, RunContext())
    assert not hit and len(samples) == 32 and (edges.df.w.between(0, 1)).all()
    assert np.isclose(sum(p for _, p in estimated.tables[1].distributions[(0, "x")]), 1.)
    result = OPTIMIZERS.build("ilp", {"theta": .5, "top_l": None}).optimize(edges, RunContext())
    assert result.groups == [frozenset(["A.x", "B.x"])]
    try:
        compute_edges(data.source, components, RealizationSpec(mode="exact"), cfg, RunContext())
    except ValueError as e:
        assert "unit-test" in str(e)
    else:
        raise AssertionError("real experiments must never enumerate joint worlds")


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")
