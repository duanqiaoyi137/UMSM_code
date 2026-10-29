"""Frequency representation, table-context sampling, and isolated calibration transfer."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from umsm.presets import paper_config
from test_matchers import FakeEncoder
from umsm import MATCHERS, PROPAGATIONS, UNCERTAINTY_MODELS
from umsm.calibration import check_inference_split, check_matcher, set_fingerprint
from umsm.context import RunContext
from umsm.matcher.sbert_mean import frequency_mean
from umsm.matcher.valentine_matchers import ValentineMatcher
from umsm.pipeline import compute_edges
from umsm.training import fit_pipeline
from umsm.types import IntegrationSet, RealizationSpec, Table


def source(name="toy"):
    return IntegrationSet("Toy", name, [Table("A", pd.DataFrame({"city": ["a", "b", None], "age": [1, 2, 3]})),
                                       Table("B", pd.DataFrame({"city": ["a", "b", "c"], "age": [1, 2, 3]}))],
                          [frozenset(("A.city", "B.city")), frozenset(("A.age", "B.age"))])


def test_frequency_mean_preserves_counts_and_is_order_invariant():
    vectors = {"a": np.array([1., 0.]), "b": np.array([0., 1.])}
    mean = frequency_mean(["a", "a", "b"], vectors, 2)
    assert np.array_equal(mean, frequency_mean(["b", "a", "a"], vectors, 2))
    assert not np.allclose(mean, frequency_mean(["a", "b", "b"], vectors, 2))
    assert np.allclose(mean, [2/3, 1/3])
    assert np.array_equal(frequency_mean([], vectors, 2), np.zeros(2))


def test_mean_scores_are_row_invariant_and_use_aligned_draws():
    original = source()
    shuffled = copy.deepcopy(original)
    for table in shuffled.tables:
        table.df = table.df.iloc[::-1].reset_index(drop=True)
    ctx = RunContext()
    with patch("umsm.matcher.sbert_mean.get_encoder", return_value=FakeEncoder()):
        matcher = MATCHERS.build("sbert_mean")
        a = matcher.score(UNCERTAINTY_MODELS.build("observed_only").estimate(original, ctx), ctx)
        b = matcher.score(UNCERTAINTY_MODELS.build("observed_only").estimate(shuffled, ctx), ctx)
        np.testing.assert_allclose(a.df.score, b.df.score, atol=1e-15)
        q = UNCERTAINTY_MODELS.build("column_marginal").estimate(original, ctx).configure(
            RealizationSpec(n_samples=8), 42)
        sampled = matcher.score(q, ctx)
    assert len(sampled) == 2 * 8 + 2
    assert sampled.df.groupby(["col_i", "col_j"]).size().to_dict() == {
        ("A.city", "B.city"): 8, ("A.city", "B.age"): 8, ("A.age", "B.city"): 1, ("A.age", "B.age"): 1}


def test_table_context_preserves_complete_pair_uncertainty():
    s = source()
    data = UNCERTAINTY_MODELS.build("column_marginal").estimate(s, RunContext()).configure(
        RealizationSpec(n_samples=16), 7)
    pair = SimpleNamespace(source_table="A", source_column="age", target_table="B", target_column="age")

    class Result:
        def __init__(self, score):
            self.score = score

        def items(self):
            return [(pair, self.score)]

        def __iter__(self):
            return iter([pair])

    def match(frames, algorithm, **kwargs):
        return Result(.9 if frames[0].loc[2, "city"] == "a" else .1)

    matcher = ValentineMatcher(algorithm="coma_instance", local_selection="all")
    with patch("valentine.valentine_match", side_effect=match), patch.object(matcher, "_build"):
        samples = matcher.score(data, RunContext())
    expected = matcher.expected_sample_counts(data)
    assert set(expected.values()) == {16}
    assert len(samples) == 4 * 16
    edge_samples = samples.df[(samples.df.col_i == "A.age") & (samples.df.col_j == "B.age")]
    assert edge_samples.sample_id.tolist() == list(range(16))
    assert edge_samples.score.nunique() == 2 and np.allclose(edge_samples.prob, 1/16)
    edges = PROPAGATIONS.build("mean_std").propagate(samples, RunContext()).df
    assert edges.loc[(edges.col_i == "A.age") & (edges.col_j == "B.age"), "sigma"].iloc[0] > 0
    try:
        check_matcher({"matcher": {"name": "valentine", "params": {"algorithm": "coma_instance"}}}, matcher)
    except ValueError as e:
        assert "implementation changed" in str(e)
    else:
        raise AssertionError("old table-context calibration must be rejected")


def test_real_coma_accepts_dense_scores_and_unsampled_rows():
    data = UNCERTAINTY_MODELS.build("observed_only").estimate(source(), RunContext())
    matcher = ValentineMatcher(algorithm="coma_instance", instance_sample_size=None, local_selection="all",
        algorithm_params={"max_n": 0, "delta": 1., "threshold": 0., "use_instances": True, "use_schema": True})
    # Keep the smoke test offline even on a machine without downloaded NLTK stopwords.
    with patch("valentine.algorithms.coma.similarity.tfidf._english_stopwords", return_value=frozenset()):
        samples = matcher.score(data, RunContext())
    assert samples.n_pairs == 4 and np.isfinite(samples.df.score).all()




def test_paired_report_comparison_aligns_methods_and_calibration_metrics():
    from compare import compare_same_methods
    from umsm.report import write_report
    with tempfile.TemporaryDirectory() as folder:
        paths = [Path(folder) / name for name in ("baseline.html", "method.html")]
        for path, f1, brier in zip(paths, (.4, .5), (.1, .08)):
            write_report(path, {"run_id": path.stem, "config": {}, "rows": [{
                "missing": "Q_risk", "optimizer": "global", "variant": "original", "set": "Toy/test",
                "pair_f1": f1, "calibration_brier": brier, "calibration_nll": brier * 2}]})
        result = compare_same_methods(paths[1], paths[0]).iloc[0]
        assert np.isclose(result.delta_pp, 10.) and np.isclose(result.calibration_brier_delta, -.02)




def test_platt_transfer_does_not_refit_Q_and_guards_both_sources():
    cfg = paper_config("calibration/unused.json")
    cfg["uncertainty_model"]["realization"]["n_samples"] = 4
    old, new = source("old_fit"), source("new_fit")
    new.tables[0].df.loc[0, "city"] = "d"
    artifact = {"version": "qx-global-v2.1", "split_role": "validation", "matcher": cfg["matcher"],
        "fit_sets": [{"fingerprint": set_fingerprint(old), "name": old.name}],
        "qx": {"generators": [{"name": "knn_tuple", "params": {"k": 10}}, {"name": "column_marginal"}],
               "temperatures": [1., 1.], "weights": [.7, .3]}}
    (ROOT / "cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "transfer.json")
        cfg["paths"]["cache_dir"] = folder
        with patch("umsm.training.load_sets", return_value=[new]), \
             patch("umsm.training._fit_qx", side_effect=AssertionError("Q must stay frozen")), \
             patch("umsm.matcher.sbert_mean.get_encoder", return_value=FakeEncoder()):
            fitted = fit_pipeline(cfg, str(ROOT), path, qx_from=artifact)
        assert fitted["qx"] == artifact["qx"] and fitted["platt"]["fit_success"]
        assert len(fitted["fit_sets"]) == 2 and len(fitted["platt_fit_sets"]) == 1
        for s in (old, new):
            try:
                check_inference_split(path, s, RunContext())
            except ValueError:
                pass
            else:
                raise AssertionError("both Q-fit and Platt-fit sources must be held out")


def test_magneto_context_draws_and_legacy_calibration_boundary():
    from umsm.matcher.magneto import MagnetoMatcher
    from umsm.matcher.base import cross_table_pairs, sample_record
    from umsm.config import load_config
    from umsm.score_cache import score_cache_key
    from umsm.types import ScoreSamples

    class ContextMatcher(MagnetoMatcher):
        def _score_once(self, data, ctx, llm_text_override=None):
            self._n_llm_prompts = self._n_llm_failures = 0
            value = data.states(data.columns[0])[0].values[-1]
            score = .2 if value == 'a' else .8
            rows = [sample_record(a, b, 0, 0, 1., score) for a, b in cross_table_pairs(data)]
            return ScoreSamples.from_records(rows, data.columns, meta=data.score_meta())

    iset = source('context')
    ctx = RunContext(project_root=str(ROOT), seed=42)
    spec = RealizationSpec(n_samples=16)
    model = UNCERTAINTY_MODELS.build('column_marginal')
    data = model.estimate(iset, ctx).configure(spec, ctx.seed)
    matcher = ContextMatcher(reranker='none')
    samples = matcher.score(data, ctx)
    complete = samples.df[(samples.df.col_i == 'A.age') & (samples.df.col_j == 'B.age')]
    assert len(complete) == 16 and complete.score.nunique() == 2
    assert complete.sample_id.nunique() == 16 and np.isclose(complete.prob.sum(), 1)
    assert all(len(g) == 16 for _, g in samples.df.groupby(['col_i', 'col_j']))

    artifact = {'version': 'qx-global-v2.1', 'split_role': 'validation', 'fit_sets': [],
                'matcher': {'name': 'magneto', 'params': matcher.params()},
                'platt': {'method': 'platt', 'slope': 1., 'intercept': 0.}}
    cfg = load_config(None)
    cfg['matcher'] = {'name': 'magneto', 'params': {'reranker': 'none'}, 'cache_scores': False}
    cfg['candidates'] = {'mode': 'all'}
    comp = {'uncertainty_model': model, 'matcher': matcher,
            'uncertainty_propagation': PROPAGATIONS.build('calibrated_logit', {'calibration': artifact})}
    try:
        compute_edges(iset, comp, spec, cfg, ctx)
    except ValueError as exc:
        assert 'table-context MC' in str(exc)
    else:
        raise AssertionError('old context calibration cannot score new Q draws')
    artifact['context_sampling_version'] = 2
    cfg['experiment']['save_score_samples'] = True  # explicit raw-draw diagnostic
    _, new_samples, _, _ = compute_edges(iset, comp, spec, cfg, ctx)
    assert len(new_samples.df) == 64

    artifact.pop('context_sampling_version')
    observed = UNCERTAINTY_MODELS.build('observed_only')
    comp['uncertainty_model'] = observed
    observed_data, _, _, _ = compute_edges(iset, comp, spec, cfg, ctx)
    old_observed_key = score_cache_key(cfg, iset, observed, matcher, spec, observed_data)
    new_mc_key = score_cache_key(cfg, iset, model, matcher, spec, data)
    with patch.object(MagnetoMatcher, 'context_sampling_version', None):
        assert score_cache_key(cfg, iset, observed, matcher, spec, observed_data) == old_observed_key
        assert score_cache_key(cfg, iset, model, matcher, spec, data) != new_mc_key


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} tests passed")
