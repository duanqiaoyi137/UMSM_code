"""Document examples, leakage protection and complete four-module V2.1 flow (offline)."""
import json
import os
import sys
import tempfile
from itertools import combinations
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from umsm import MATCHERS, OPTIMIZERS, PROPAGATIONS, UNCERTAINTY_MODELS
from umsm.calibration import (calibrate_scores, check_inference_split, fit_platt,
                              set_fingerprint)
from umsm.training import fit_pipeline
from umsm.candidates import plan_candidates
from umsm.config import load_config
from umsm.context import RunContext
from umsm.experiment import build_components, compute_edges
from umsm.matcher.base import PairwiseMatcher
from umsm.types import ColumnRef, EdgeTable, IntegrationSet, ScoreSamples, Table
from umsm.uncertainty_model.base import truncate_and_normalize
from umsm.uncertainty_model.mixture import (GENERATORS, build_qx, compatible_scores, fit_mixture,
                                           pseudo_mask_examples, qx_diagnostics)


def temporary():
    os.makedirs(os.path.join(ROOT, "cache", "test_v2"), exist_ok=True)
    return tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "cache", "test_v2"))


def write_artifact(directory, **kwargs):
    record = {"version": "qx-global-v2.1", "split_role": "validation", "fit_sets": [],
              "qx": {"generators": [{"name": "knn_tuple", "params": {"k": 10}}, {"name": "column_marginal"}],
                     "temperatures": [1., 1.], "weights": [.7, .3]},
              "platt": {"method": "platt", "slope": 8., "intercept": -4.}, **kwargs}
    path = os.path.join(directory, "calibration.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f)
    return path


def test_document_qx_example_mixture_before_truncation():
    raw = [[(v, np.log(p)) for v, p in zip("ABC", probs)] for probs in ([.6, .3, .1], [.2, .7, .1])]
    q, active = build_qx(raw, [1, 1], [.7, .3])
    np.testing.assert_allclose([p for _, p in q], [.48, .42, .10])
    kept, stats = truncate_and_normalize(q, .90, 10)
    assert [v for v, _ in kept] == ["A", "B"]
    np.testing.assert_allclose([p for _, p in kept], [8 / 15, 7 / 15])
    assert np.isclose(stats["retained_mass"], .9)


def test_union_zero_for_unproposed_and_unavailable_renormalization():
    q, weights = build_qx([[('A', 0)], [('B', 0)], []], [1, 1, 1], [.2, .3, .5])
    assert weights == [.4, .6, 0.] and dict(q) == {'A': .4, 'B': .6}
    assert build_qx([[], []], [1, 1], [.5, .5]) == ([], [])
    np.testing.assert_array_equal(compatible_scores([1, 3], 'distance'), [-1, -3])
    np.testing.assert_allclose(compatible_scores([1, 3], 'frequency'), [0, np.log(3)])


def test_joint_temperature_weight_fit_and_separate_coverage():
    examples = [{"truth": "a", "generators": [[('a', 3.), ('b', 0.)], [('b', 2.), ('a', 0.)]]}] * 15
    examples += [{"truth": "absent", "generators": [[('a', 1.)], [('b', 1.)]]}]
    fit = fit_mixture(examples, 2)
    assert fit['fit_nll'] < fit['fit_nll_initial']
    assert np.isclose(sum(fit['weights']), 1) and min(fit['temperatures']) > 0
    diag = qx_diagnostics(examples, fit['temperatures'], fit['weights'], .95, 1)
    assert diag['union_covered'] == 15 / 16 and diag['recall_at_k'] <= diag['union_covered']
    assert diag['nll_floor'] == 1e-12


def test_pseudomasking_never_exposes_the_hidden_value():
    class Spy:
        def generate(self, frame, table, iset, ctx):
            pd.testing.assert_frame_equal(frame, table.df)
            pd.testing.assert_frame_equal(frame, iset.tables[0].df)
            assert table.path is None and iset.groundtruth is None
            assert frame.isna().to_numpy().sum() == 1
            return {}
    source = IntegrationSet('Toy', 'train', [Table('A', pd.DataFrame({'x': ['unique1', 'unique2']}))])
    with patch.dict(GENERATORS, {'spy': lambda p: Spy()}):
        examples = pseudo_mask_examples([source], [{'name': 'spy'}], RunContext(), 2)
    assert len(examples) == 2 and source.tables[0].df.notna().all().all()


def sample_scores(values):
    return ScoreSamples(pd.DataFrame([{'col_i': 'A.x', 'col_j': 'B.x', 'table_i': 'A', 'table_j': 'B',
                                      'state_i': i, 'state_j': i, 'prob': 1 / len(values), 'score': v}
                                     for i, v in enumerate(values)]),
                        [ColumnRef('A', 'x'), ColumnRef('B', 'x')], {'estimator': 'mc'})


def test_calibrate_draws_before_unbiased_moments_and_logit():
    with temporary() as d:
        path = write_artifact(d)
        s = sample_scores([.1, .6, .9])
        e = PROPAGATIONS.build('calibrated_logit', {'calibration': path, 'risk_lambda': .25}).propagate(s, RunContext())
        p = expit(8 * np.array([.1, .6, .9]) - 4)
        row = e.df.iloc[0]
        assert np.isclose(row.mu, p.mean()) and np.isclose(row.sigma, p.std(ddof=1))
        assert not np.isclose(row.mu, expit(8 * .533333333 - 4))
        assert np.isclose(row.a, logit(row.mu - .25 * row.sigma))
        assert 'calibrated_probability' in s.df


def test_complete_sigma_zero_risk_clipping_and_prior():
    with temporary() as d:
        path = write_artifact(d)
        s = sample_scores([.8])
        row = PROPAGATIONS.build('calibrated_logit', {'calibration': path, 'prior': .7}).propagate(s, RunContext()).df.iloc[0]
        assert row.sigma == 0 and np.isclose(row.a, logit(expit(2.4)) - logit(.7))
        row = PROPAGATIONS.build('calibrated_logit', {'calibration': path, 'risk_lambda': 100}).propagate(
            sample_scores([0., 1.]), RunContext()).df.iloc[0]
        assert row.p_tilde == 1e-6 and np.isfinite(row.a)


def test_document_global_example_keeps_negative_closure_evidence():
    cols = [ColumnRef(t, 'x') for t in 'ABC']
    rows = []
    for (a, b), p in zip(combinations(cols, 2), [.88, .18, .80]):
        rows.append(dict(col_i=a.id, col_j=b.id, table_i=a.table, table_j=b.table,
                         mu=p, sigma=0., w=p, a=logit(p)))
    edge = EdgeTable(pd.DataFrame(rows), cols, {'potential': 'calibrated_logit'})
    for triangles in ['full', 'lazy']:
        result = OPTIMIZERS.build('ilp_logit', {'triangles': triangles}).optimize(edge, RunContext())
        assert result.groups == [frozenset(['A.x', 'B.x'])]
        assert np.isclose(result.info['objective'], logit(.88))
        assert result.artifacts['ilp_solution'].query("col_i == 'A.x' and col_j == 'C.x'").a.iloc[0] < 0
    try:
        OPTIMIZERS.build('ilp_logit').optimize(EdgeTable(pd.DataFrame(rows[:2]), cols,
                                                       {'potential': 'calibrated_logit'}), RunContext())
    except ValueError as exc:
        assert 'closure' in str(exc)
    else:
        raise AssertionError('missing negative closure must not become zero evidence')


def test_blocker_independent_of_q_and_expensive_matcher_and_reports_recall():
    from test_matchers import FakeEncoder
    source = IntegrationSet('Toy', 'test', [Table(t, pd.DataFrame({'x': [None], 'y': [None]})) for t in 'ABC'],
                            [frozenset(f'{t}.x' for t in 'ABC'), frozenset(f'{t}.y' for t in 'ABC')])
    data = UNCERTAINTY_MODELS.build('observed_only').estimate(source, RunContext())
    with patch('umsm.candidates.get_encoder', return_value=FakeEncoder()):
        pairs, info = plan_candidates(source, RunContext(), {'top_l': 1, 'eta': .5})
    assert data.candidate_pairs is None and len(pairs) == 6  # planning does not mutate Q or tables
    assert info['n_evaluated_pairs'] == 6 and info['component_recall'] == 1
    assert 4 / 6 <= info['recall_at_l'] <= 1 and info['component_sizes'] == [3, 3], info


@MATCHERS.register('v2_overlap_test')
class Overlap(PairwiseMatcher):
    def similarity(self, ci, si, cj, sj, ctx):
        a, b = set(map(str, si.values)), set(map(str, sj.values))
        return len(a & b) / len(a | b) if a | b else 0.


def test_explicit_fit_frozen_artifact_four_modules_and_leakage_guard():
    source = IntegrationSet('Toy', 'train', [Table('A', pd.DataFrame({'x': ['a', 'a', 'b', None], 'z': ['q'] * 4})),
                                           Table('B', pd.DataFrame({'x': ['a', 'a', 'b', 'b'], 'z': ['q'] * 4}))],
                            [frozenset(['A.x', 'B.x']), frozenset(['A.z', 'B.z'])])
    cfg = load_config()
    cfg['matcher'] = {'name': 'v2_overlap_test', 'params': {}, 'cache_scores': False}
    cfg['global_optimizer'] = {'name': 'ilp_logit', 'params': {}}
    cfg['candidates'] = {'mode': 'all'}
    with temporary() as d:
        cfg['paths']['cache_dir'] = d
        path = os.path.join(d, 'fitted.json')
        with patch('umsm.training.load_sets', return_value=[source]):
            artifact = fit_pipeline(cfg, ROOT, path, max_cells=12)
        assert artifact['qx']['diagnostics']['heldout_masks']['n_masks'] > 0 and artifact['platt']['fit_success']
        assert artifact['fit_sets'][0]['fingerprint'] == set_fingerprint(source)
        try:
            check_inference_split(path, source, RunContext())
        except ValueError as exc:
            assert 'overlap' in str(exc)
        else:
            raise AssertionError('training tables accepted as test')
        test = IntegrationSet('Toy', 'test', [Table(t.name, t.df.copy()) for t in source.tables], source.groundtruth)
        test.tables[1].df.loc[3, 'x'] = 'c'
        cfg['uncertainty_model'] = {'name': 'calibrated_mixture', 'params': {'calibration': path}}
        cfg['uncertainty_propagation'] = {'name': 'calibrated_logit', 'params': {'calibration': path}}
        comp, spec = build_components(cfg)
        data, samples, edges, _ = compute_edges(test, comp, spec, cfg, RunContext())
        result = comp['global_optimizer'].optimize(edges, RunContext())
        assert len(samples) == 2 * 32 + 2 and np.isfinite(edges.df.a).all()
        assert result.info['all_optimal'] and data.info['fitted_qx']['weights'] == artifact['qx']['weights']


def test_platt_fitting_reduces_nll_and_rejects_missing_classes():
    fit = fit_platt([.1, .2, .8, .9], [0, 0, 1, 1])
    p = calibrate_scores([.1, .9], fit)
    assert p[0] < .1 and p[1] > .9
    try:
        fit_platt([.1, .2], [0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError('must not fit on one class')


def test_raw_cosine_or_reranker_scores_are_not_clipped_to_probabilities():
    with temporary() as d:
        path = write_artifact(d)
        bounded = sample_scores([.1, .9])
        raw = bounded.df.copy()
        raw['score'] = [-2., 3.]
        samples = ScoreSamples(raw, bounded.columns, {'estimator': 'mc', 'score_space': 'raw'})
        edges = PROPAGATIONS.build('calibrated_logit', {'calibration': path}).propagate(samples, RunContext())
        np.testing.assert_array_equal(samples.df.score, [-2., 3.])
        assert np.isclose(edges.df.mu.iloc[0], expit(8 * np.array([-2., 3.]) - 4).mean())


def test_global_potentials_and_closure_counts_do_not_depend_on_artifact_saving():
    cols = [ColumnRef(t, 'x') for t in 'ABC']
    rows = [dict(col_i=a.id, col_j=b.id, table_i=a.table, table_j=b.table,
                 mu=p, sigma=0., w=p, a=logit(p))
            for (a, b), p in zip(combinations(cols, 2), [.88, .18, .80])]
    blocking = {'mode': 'name_top_l', 'components': [[c.id for c in cols]],
                'blocking_edges': [{'col_i': r['col_i'], 'col_j': r['col_j'], 'candidate': i != 1}
                                   for i, r in enumerate(rows)]}
    graph = EdgeTable(pd.DataFrame(rows), cols, {'potential': 'calibrated_logit', 'candidates': blocking})
    original = graph.df.copy(deep=True)
    for save in [False, True]:
        result = OPTIMIZERS.build('ilp_logit', {'save_solution': save}).optimize(graph, RunContext())
        assert result.groups == [frozenset(['A.x', 'B.x'])]
        assert result.info['n_candidate_edges'] == 2 and result.info['n_closure_pairs'] == 1
        assert result.artifacts['ilp_components'].iloc[0].n_closure_pairs == 1
        if save:
            ac = result.artifacts['ilp_solution'].query("col_i == 'A.x' and col_j == 'C.x'").iloc[0]
            assert ac.w == .18 and np.isclose(ac.a, logit(.18)) and not ac.candidate
    pd.testing.assert_frame_equal(original, graph.df)


def test_failed_training_leaves_existing_artifact_untouched():
    source = IntegrationSet('Toy', 'train', [Table('A', pd.DataFrame({'x': ['a', 'a', 'b']}))])
    with temporary() as d:
        path = write_artifact(d)
        with open(path, 'rb') as f:
            original = f.read()
        with patch('umsm.training.load_sets', return_value=[source]), \
                patch('umsm.training._fit_matcher', side_effect=RuntimeError('fit failed')):
            try:
                fit_pipeline(load_config(), ROOT, path, max_cells=3)
            except RuntimeError as exc:
                assert str(exc) == 'fit failed'
            else:
                raise AssertionError('failed fitting must not publish partial calibration')
        with open(path, 'rb') as f:
            assert f.read() == original
        assert not os.path.exists(path + '.tmp')


def test_in_memory_calibration_preserves_leakage_guard_and_cell_diagnostics():
    source = IntegrationSet('Toy', 'train', [Table('A', pd.DataFrame({'x': ['a', 'b', None]}))])
    with temporary() as d:
        path = write_artifact(d, fit_sets=[{'fingerprint': set_fingerprint(source)}])
        with open(path, encoding='utf-8') as f:
            artifact = json.load(f)
        model = UNCERTAINTY_MODELS.build('calibrated_mixture', {'calibration': artifact})
        try:
            model.estimate(source, RunContext())
        except ValueError as exc:
            assert 'overlap' in str(exc)
        else:
            raise AssertionError('in-memory parameters must not bypass split checks')
        ctx = RunContext()
        ctx.resources['calibration_fitting'] = True
        data = model.estimate(source, ctx)
        info = data.tables[0].distribution_info[(2, 'x')]
        assert info['active_weights'] == [.7, .3] and info['temperatures'] == [1., 1.]
        top = UNCERTAINTY_MODELS.build('argmax', {'model': 'calibrated_mixture',
                                                'model_params': {'calibration': artifact}}).estimate(source, ctx)
        assert top.tables[0].frame.iloc[2, 0] == data.tables[0].distributions[(2, 'x')][0][0]


def test_magneto_calibration_and_four_modules_share_frozen_parameters_offline():
    from test_matchers import FakeEncoder
    source = IntegrationSet('Toy', 'train', [
        Table('A', pd.DataFrame({'city': ['a', 'a', 'b', None], 'age': [1, 1, None, 2]})),
        Table('B', pd.DataFrame({'city': ['a', None, 'b', 'b'], 'age': [1, 1, 2, None]}))],
        [frozenset(['A.city', 'B.city']), frozenset(['A.age', 'B.age'])])
    from umsm.presets import paper_config
    cfg = paper_config('calibration/unused.json')
    cfg['matcher'] = load_config(os.path.join(ROOT,'configs','magneto_deepseek.yaml'))['matcher']
    cfg['matcher']['params']['llm_backend'] = 'mock'
    cfg['matcher']['cache_scores'] = False
    cfg['uncertainty_model']['realization']['n_samples'] = 4
    with temporary() as d, patch('umsm.matcher.magneto.get_encoder', return_value=FakeEncoder()):
        path = os.path.join(d, 'magneto.json')
        cfg['paths']['cache_dir'] = d
        with patch('umsm.training.load_sets', return_value=[source]):
            artifact = fit_pipeline(cfg, ROOT, path, max_cells=12)
        assert artifact['matcher']['name'] == 'magneto' and artifact['platt']['fit_success']
        test = IntegrationSet('Toy', 'test', [Table(t.name, t.df.copy()) for t in source.tables], source.groundtruth)
        test.tables[1].df.loc[3, 'city'] = 'c'
        cfg['uncertainty_model']['params']['calibration'] = path
        cfg['uncertainty_propagation']['params']['calibration'] = path
        components, spec = build_components(cfg)
        _, samples, edges, _ = compute_edges(test, components, spec, cfg,
                                             RunContext(project_root=ROOT, cache_dir=d))
        assert len(samples) == 4 * 4 and edges.meta['potential'] == 'calibrated_logit'
        assert np.isfinite(edges.df.a).all() and edges.df.llm_scored.eq(1).all()
        result = components['global_optimizer'].optimize(edges, RunContext())
        assert result.info['all_optimal']


if __name__ == '__main__':
    tests = [f for n, f in sorted(globals().items()) if n.startswith('test_')]
    for fn in tests:
        fn()
        print('ok ', fn.__name__)
    print(f'{len(tests)} tests passed')
