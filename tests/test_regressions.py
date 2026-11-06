"""Experiment-correctness regressions; no downloaded model or paid API required."""
import copy
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from umsm import MATCHERS, OPTIMIZERS, PROPAGATIONS, UNCERTAINTY_MODELS
from umsm.config import load_config
from umsm.context import RunContext
from umsm.embedding import TextEncoder
from umsm.score_cache import score_cache_key as _score_cache_key
from umsm.pipeline import compute_edges
from umsm.score_cache import candidate_cache_key
from umsm.global_optimizer.local_merge import hungarian_per_table_pair
from umsm.matcher import sbert_quantile as sbert_mod
from umsm.types import (ColumnRef, EdgeTable, IntegrationSet, RealizationSpec, ScoreSamples,
                        Table, UncertainTable)
from umsm.uncertainty_model.base import DistributionModel, normalise
from umsm.matcher.base import PairwiseMatcher
from umsm.tuning import default_grid


def _raises(error, fn):
    try:
        fn()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def _matrix_graph(matrix):
    cols = [ColumnRef("A", "x"), ColumnRef("A", "y"), ColumnRef("B", "x"), ColumnRef("B", "y")]
    rows = [{"col_i": a.id, "col_j": b.id, "table_i": a.table, "table_j": b.table,
             "mu": matrix[i][j], "sigma": 0., "w": matrix[i][j]}
            for i, a in enumerate(cols[:2]) for j, b in enumerate(cols[2:])]
    return EdgeTable.from_records(rows, cols)


def test_hungarian_allows_unmatched_and_uses_signed_evidence():
    g = _matrix_graph([[.9, .49], [.49, 0.]])
    kept = hungarian_per_table_pair(g.df, .5)
    assert list(zip(kept.col_i, kept.col_j)) == [("A.x", "B.x")]
    g = _matrix_graph([[1., .8], [.8, 0.]])
    kept = hungarian_per_table_pair(g.df, .7)
    assert list(zip(kept.col_i, kept.col_j)) == [("A.x", "B.x")]


def test_local_logit_uses_same_assignment_objective_as_ilp():
    from scipy.special import logit
    g = _matrix_graph([[.99, .8], [.8, .51]])
    g.df["a"] = logit(g.df.w.to_numpy())
    g.meta["potential"] = "calibrated_logit"
    # Monotone logit transformation preserves ranking but can change optimal assignments.
    probability = OPTIMIZERS.build("local_merge").optimize(g, RunContext())
    evidence = OPTIMIZERS.build("local_merge", {"weight": "a", "min_weight": 0.}).optimize(g, RunContext())
    global_result = OPTIMIZERS.build("ilp_logit").optimize(g, RunContext())
    assert set(probability.groups) != set(evidence.groups)
    assert set(evidence.groups) == set(global_result.groups) == {frozenset(("A.x", "B.x")),
                                                                frozenset(("A.y", "B.y"))}
    bad = EdgeTable(g.df.drop(columns="a"), g.columns)
    _raises(ValueError, lambda: OPTIMIZERS.build("local_merge", {"weight": "a"}).optimize(bad, RunContext()))


def test_score_cache_tracks_memory_data_Q_and_external_scores():
    cfg = load_config()
    iset = IntegrationSet("Toy", "Toy", [Table("a.csv", pd.DataFrame({"x": [1., np.nan]})),
                                         Table("b.csv", pd.DataFrame({"y": [1., 2.]}))])
    model = UNCERTAINTY_MODELS.build("column_marginal")
    matcher = MATCHERS.build("sbert_quantile")
    spec = RealizationSpec()
    d = model.estimate(iset, RunContext()).configure(spec, 42)
    raw_key = _score_cache_key(cfg, iset, model, matcher, spec)
    key = _score_cache_key(cfg, iset, model, matcher, spec, d)
    d.tables[0].distributions[(1, "x")] = [(9., 1.)]
    assert _score_cache_key(cfg, iset, model, matcher, spec, d) != key
    iset.tables[1].df.iloc[0, 0] = 9.
    assert _score_cache_key(cfg, iset, model, matcher, spec) != raw_key
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "scores.csv")
        ext = MATCHERS.build("precomputed", {"template": path})
        with open(path, "w") as f:
            f.write("col_i,col_j,score\na.csv.x,b.csv.y,0.8\n")
        before = _score_cache_key(cfg, iset, model, ext, spec)
        with open(path, "w") as f:
            f.write("col_i,col_j,score\na.csv.x,b.csv.y,0.2\n")
        assert before != _score_cache_key(cfg, iset, model, ext, spec)


def test_embedding_cache_separates_models_and_truncation():
    a = TextEncoder("first/model", "cache", max_length=32)
    b = TextEncoder("second/model", "cache", max_length=32)
    c = TextEncoder("first/model", "cache", max_length=64)
    assert len({e._cache_file("Toy") for e in (a, b, c)}) == 3


def test_invalid_module_outputs_fail_early():
    for q in [[("x", np.nan)], [("x", np.inf)], [("x", -.1)]]:
        _raises(ValueError, lambda: normalise(q))
    for kwargs in ({"n_samples": 0}, {"max_exact_states": 0}):
        _raises(ValueError, lambda: RealizationSpec(**kwargs))
    cols = [ColumnRef("A", "x"), ColumnRef("B", "y")]
    row = {"col_i": "A.x", "col_j": "B.y", "table_i": "A", "table_j": "B",
           "state_i": 0, "state_j": 0, "prob": 1., "score": .7}
    for update in ({"prob": 0.}, {"prob": -1.}, {"score": np.nan}, {"score": 2.}):
        _raises(ValueError, lambda: ScoreSamples.from_records([{**row, **update}], cols))


def test_table_realizations_preserve_exact_probabilities():
    t = UncertainTable("A", pd.DataFrame({"x": [np.nan]}, dtype=object),
                       distributions={(0, "x"): [(1, .2), (9, .8)]})
    reals = t.table_realizations(RealizationSpec(mode="exact"), 1)
    assert [(f.iat[0, 0], p) for f, p in reals] == [(1, .2), (9, .8)]
    t.distributions[(0, "x")] = [(1, 1.)]
    assert len(t.table_realizations(RealizationSpec(mode="exact"), 1)) == 1


def test_valentine_exact_states_use_independent_joint_probabilities():
    from types import SimpleNamespace
    from umsm.matcher.valentine_matchers import ValentineMatcher
    from umsm.types import UncertainData

    source = IntegrationSet("Toy", "Toy", [Table("A", pd.DataFrame({"x": [np.nan]}, dtype=object)),
                                           Table("B", pd.DataFrame({"x": [np.nan]}, dtype=object))])
    tables = [UncertainTable("A", source.tables[0].df, distributions={(0, "x"): [(0, .2), (1, .8)]}),
              UncertainTable("B", source.tables[1].df, distributions={(0, "x"): [(0, .3), (1, .7)]})]
    data = UncertainData(source, tables, "test").configure(RealizationSpec(mode="exact"), 1)
    pair = SimpleNamespace(source_table="A", source_column="x", target_table="B", target_column="x")

    class Result:
        def __init__(self, score):
            self.score = score

        def items(self):
            return [(pair, self.score)]

        def __iter__(self):
            return iter([pair])

    def match(frames, algorithm, **kwargs):
        return Result(float(frames[0].iat[0, 0] == frames[1].iat[0, 0]))

    with patch("valentine.valentine_match", side_effect=match), patch.object(ValentineMatcher, "_build"):
        scores = ValentineMatcher(local_selection="all").score(data, RunContext()).df
    assert len(scores) == 4
    assert np.isclose((scores.prob * scores.score).sum(), .2 * .3 + .8 * .7)


def test_threshold_grid_can_return_no_matches_after_rounding():
    grid = default_grid(np.array([1., 1.]))
    assert grid.max() > 1.


def test_ilp_closure_requires_real_evidence_and_flags_approximation():
    cols = [ColumnRef("A", "x"), ColumnRef("B", "x"), ColumnRef("C", "x")]
    rows = [{"col_i": a.id, "col_j": b.id, "table_i": a.table, "table_j": b.table,
             "mu": .9, "sigma": 0., "w": .9} for a, b in zip(cols, cols[1:])]
    graph = EdgeTable.from_records(rows, cols)
    _raises(ValueError, lambda: OPTIMIZERS.build("ilp", {"theta": .5}).optimize(graph, RunContext()))
    graph = _matrix_graph([[.9, .1], [.1, .9]])
    sparse = EdgeTable(graph.df[graph.df.w > .5], graph.columns)
    assert not OPTIMIZERS.build("ilp", {"theta": .5, "top_l": None}).optimize(
        sparse, RunContext()).info["global_exact"]
    with patch("umsm.global_optimizer.ilp.solve_ilp", return_value=(np.array([1]), 1, "time limit")):
        _raises(RuntimeError, lambda: OPTIMIZERS.build("ilp", {"theta": .5}).optimize(graph, RunContext()))
        result = OPTIMIZERS.build("ilp", {"theta": .5, "require_optimal": False}).optimize(graph, RunContext())
        assert not result.info["all_optimal"] and not result.info["global_exact"]


def test_stochastic_Q_is_shared_across_propagation_controls():
    class RandomQ(DistributionModel):
        def cell_distributions(self, frame, table, iset, ctx):
            return {(int(r), c): [("a", float(ctx.rng.uniform(.1, .9))), ("b", 1.)]
                    for c in frame.columns for r in np.flatnonzero(frame[c].isna())}

    class Overlap(PairwiseMatcher):
        def similarity(self, ci, si, cj, sj, ctx):
            return float(si.values == sj.values)

    cfg = load_config()
    cfg["matcher"]["cache_scores"] = False
    cfg["candidates"] = {"mode": "all"}
    source = IntegrationSet("Toy", "Toy", [Table("A", pd.DataFrame({"x": [None, "a"]})),
                                           Table("B", pd.DataFrame({"x": ["a", "b"]}))])
    comp = {"uncertainty_model": RandomQ(), "matcher": Overlap(),
            "uncertainty_propagation": PROPAGATIONS.build("expected")}
    ctx = RunContext(seed=7)
    before = compute_edges(source, comp, RealizationSpec(), cfg, ctx)
    ctx.rng.random(100)  # mimic other strategies consuming the stream
    comp["uncertainty_propagation"] = PROPAGATIONS.build("mean_std")
    after = compute_edges(source, comp, RealizationSpec(), cfg, ctx)
    assert before[0].tables[0].distributions == after[0].tables[0].distributions
    pd.testing.assert_frame_equal(before[1].df, after[1].df)


def test_blocking_cache_is_independent_of_crossed_methods():
    import copy
    cfg = load_config()
    source = IntegrationSet('Toy', 'Toy', [Table('A', pd.DataFrame({'x': ['a', None]})),
                                         Table('B', pd.DataFrame({'y': ['b', 'a']}))])
    ctx = RunContext()
    before = candidate_cache_key(cfg, source, ctx)
    changed = copy.deepcopy(cfg)
    changed['uncertainty_model'] = {'name': 'knn_tuple', 'params': {'k': 3}, 'realization': {'n_samples': 64}}
    changed['matcher'] = {'name': 'llm', 'params': {}}
    changed['global_optimizer'] = {'name': 'greedy_merge', 'params': {}}
    assert candidate_cache_key(changed, source, ctx) == before
    changed['candidates']['top_l'] = 1
    assert candidate_cache_key(changed, source, ctx) != before
    source.tables[0].df.loc[0, 'x'] = 'different'
    assert candidate_cache_key(cfg, source, ctx) != before


def test_sbert_fixed_scale_keeps_unchanged_pair_scores():
    class Encoder:
        dim = 3

        def encode_map(self, texts, namespace):
            result = {}
            for t in texts:
                v = np.array([sum(map(ord, t)) % 5 + 1., len(t) + 1., 1.])
                result[t] = v / np.linalg.norm(v)
            return result

    iset = IntegrationSet("Toy", "Toy", [Table("a", pd.DataFrame({"x": ["red", "blue"]})),
                                         Table("b", pd.DataFrame({"y": ["red", "green"]}))])
    ctx = RunContext()
    matcher = MATCHERS.build("sbert_quantile", {"normalization": "fixed"})
    data = UNCERTAINTY_MODELS.build("observed_only").estimate(iset, ctx)
    with patch.object(sbert_mod, "get_encoder", return_value=Encoder()):
        before = matcher.score(data, ctx).df.iloc[0].score
        other = copy.deepcopy(iset)
        other.tables.append(Table("c", pd.DataFrame({"z": ["completely new", "data"]})))
        after = matcher.score(UNCERTAINTY_MODELS.build("observed_only").estimate(other, ctx), ctx).df.iloc[0].score
    assert abs(before - after) < 1e-12


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")
