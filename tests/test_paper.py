"""Paper Algorithm 1, frozen calibration, bounded score storage and the unified CLI."""
import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from umsm import MATCHERS,UNCERTAINTY_MODELS
from umsm.calibration import calibrate_scores
from umsm.cli import main
from umsm.context import RunContext
from umsm.matcher.base import BaseMatcher,cross_table_pairs,sample_record
from umsm.online import OnlineMoments
from umsm.pipeline import build_components,compute_edges
from umsm.presets import paper_config
from umsm.score_cache import score_cache_key
from umsm.types import ColumnRef,IntegrationSet,Matching,RealizationSpec,ScoreSamples,Table,UncertainTable
from umsm.uncertainty_propagation.base import moments


@MATCHERS.register('paper_fixture')
class ContextFixture(BaseMatcher):
    sampling_scope='table_pair'

    def score(self,data,ctx):
        self._calls=getattr(self,'_calls',0)+1
        value=data.states(data.columns[0])[0].values[-1]
        score=.8 if value=='a' else .2
        rows=[sample_record(a,b,0,0,1.,score) for a,b in cross_table_pairs(data)]
        return ScoreSamples.from_records(rows,data.columns,data.score_meta())


def fixture(root):
    source=IntegrationSet('Toy','test',[Table('A',pd.DataFrame({'v':['a','b',None,None],'fixed':[1,2,3,4]})),
        Table('B',pd.DataFrame({'v':['a',None,'b',None],'fixed':[1,2,3,4]}))])
    artifact={'version':'qx-global-v2.1','split_role':'validation','fit_sets':[],
        'matcher':{'name':'paper_fixture','params':{}},'platt':{'method':'platt','slope':4.,'intercept':-1.}}
    cfg=paper_config(artifact)
    cfg['dataset']['name']='Toy'
    cfg['matcher']={'name':'paper_fixture','params':{},'cache_scores':True}
    cfg['uncertainty_model']={'name':'column_marginal','params':{'rho':1.},'realization':{'mode':'mc','n_samples':11}}
    cfg['paths']['cache_dir']=str(root/'cache')
    cfg['experiment']['output_dir']=str(root/'results')
    return source,cfg,artifact


def reference(source,cfg,artifact):
    model=UNCERTAINTY_MODELS.build('column_marginal',{'rho':1.})
    data=model.estimate(source,RunContext()).configure(RealizationSpec(n_samples=11),42)
    a,b=data.columns[0],data.columns[2]
    all_states={c:data.states(c) for c in data.columns}
    records=[]
    for r in range(11):
        score=.8 if all_states[a][r].values[-1]=='a' else .2
        for ci,cj in cross_table_pairs(data):
            records.append(sample_record(ci,cj,r%len(all_states[ci]),r%len(all_states[cj]),1/11,score,sample_id=r))
    raw=ScoreSamples.from_records(records,data.columns,data.score_meta())
    calibrated=raw.df.copy(); calibrated['score']=calibrate_scores(calibrated.score,artifact['platt'])
    return data,raw,moments(ScoreSamples(calibrated,data.columns,data.score_meta()))


def test_streamed_draws_exactly_match_vectorized_draws_without_R_materialization():
    with tempfile.TemporaryDirectory() as folder:
        source,cfg,artifact=fixture(Path(folder)); data,_,_=reference(source,cfg,artifact)
        expected={c:[s.values for s in data.states(c)] for c in data.columns}
        data._states={}; actual={c:[] for c in data.columns}
        with patch.object(UncertainTable,'column_states',side_effect=AssertionError('No batch sampling')):
            for r,states in data.iter_realizations():
                for c,state in states.items(): actual[c].append(state.values)
        assert not data._states
        for c in data.columns:
            assert actual[c]==[expected[c][r%len(expected[c])] for r in range(11)]


def test_online_welford_matches_calibrate_then_unbiased_variance_and_context_draws():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); source,cfg,artifact=fixture(root); _,raw,expected=reference(source,cfg,artifact)
        comp,spec=build_components(cfg)
        with patch.object(UncertainTable,'column_states',side_effect=AssertionError('No R states in inference')):
            data,samples,edges,hit=compute_edges(source,comp,spec,cfg,RunContext(project_root=str(root),cache_dir=str(root/'cache')))
        assert not hit and not data._states and samples.raw is None and len(samples)==44
        assert comp['matcher']._calls==11
        actual=edges.df.set_index(['col_i','col_j']); expected=expected.set_index(['col_i','col_j']).reindex(actual.index)
        np.testing.assert_allclose(actual[['mu','sigma']],expected[['mu','sigma']],atol=1e-14)
        assert actual.loc[('A.fixed','B.fixed'),'sigma']>0
        try: samples.df
        except ValueError: pass
        else: raise AssertionError('Raw R score tables must not be retained implicitly')


def test_summary_cache_reuses_mu_sigma_when_only_lambda_prior_change():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); source,cfg,artifact=fixture(root); comp,spec=build_components(cfg)
        ctx=RunContext(project_root=str(root),cache_dir=str(root/'cache'))
        _,_,original,_=compute_edges(source,comp,spec,cfg,ctx)
        comp['uncertainty_propagation'].risk_lambda=0.; comp['uncertainty_propagation'].prior=.2
        with patch.object(ContextFixture,'score',side_effect=AssertionError('No rescoring')):
            _,receipt,changed,hit=compute_edges(source,comp,spec,cfg,ctx)
        assert hit and receipt.raw is None
        np.testing.assert_array_equal(changed.df[['mu','sigma']],original.df[['mu','sigma']])
        np.testing.assert_allclose(changed.df.p_tilde,changed.df.mu)


def test_round_cache_can_recalibrate_without_reasking_a_matcher():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); source,cfg,artifact=fixture(root); cfg['paths']['raw_score_cache']=True
        comp,spec=build_components(cfg); ctx=RunContext(project_root=str(root),cache_dir=str(root/'cache'))
        compute_edges(source,comp,spec,cfg,ctx)
        assert len(list((root/'cache/score_rounds').rglob('*.pkl')))==11
        artifact['platt']['slope']=2.
        comp,spec=build_components(cfg)
        with patch.object(ContextFixture,'score',side_effect=AssertionError('Only replay raw round files')):
            _,_,edges,hit=compute_edges(source,comp,spec,cfg,ctx)
        _,_,expected=reference(source,cfg,artifact)
        np.testing.assert_allclose(edges.df.mu,expected.set_index(['col_i','col_j']).reindex(pd.MultiIndex.from_frame(edges.df[['col_i','col_j']])).mu)
        assert hit


def test_legacy_score_cache_is_preserved_and_imported_without_rematching():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); source,cfg,artifact=fixture(root); data,raw,expected=reference(source,cfg,artifact)
        comp,spec=build_components(cfg); ctx=RunContext(project_root=str(root),cache_dir=str(root/'cache'))
        key=score_cache_key({**cfg,'_project_root':str(root)},source,comp['uncertainty_model'],comp['matcher'],spec,data)
        path=root/'cache/scores'/(key+'.pkl'); path.parent.mkdir(parents=True)
        pd.to_pickle({'df':raw.df,'meta':raw.meta},path); before=path.read_bytes()
        with patch.object(ContextFixture,'score',side_effect=AssertionError('Legacy cache must be reused')):
            _,samples,edges,hit=compute_edges(source,comp,spec,cfg,ctx)
        assert hit and samples.meta['legacy_cache_import'] and path.read_bytes()==before
        assert samples.raw is None
        np.testing.assert_allclose(edges.df.mu,expected.set_index(['col_i','col_j']).reindex(pd.MultiIndex.from_frame(edges.df[['col_i','col_j']])).mu)


def test_online_updates_reject_duplicate_missing_and_nonuniform_draws():
    with tempfile.TemporaryDirectory() as folder:
        source,cfg,artifact=fixture(Path(folder)); data,raw,_=reference(source,cfg,artifact)
        expected={frozenset((a,b)):11 for a,b in zip(raw.df.col_i,raw.df.col_j)}
        accumulator=OnlineMoments(data.columns,expected,data.score_meta())
        try: accumulator.finish()
        except ValueError: pass
        else: raise AssertionError('Incomplete closure must fail')
        try: accumulator.update(raw.df,artifact['platt'])
        except ValueError: pass
        else: raise AssertionError('Duplicate pairs inside one round must fail')
        frame=raw.df[raw.df.sample_id==0].copy(); frame['prob']=.5
        try: accumulator.update(frame,artifact['platt'])
        except ValueError: pass
        else: raise AssertionError('Nonuniform weights must fail')


def test_identity_calibration_is_a_validated_probability_identity():
    np.testing.assert_array_equal(calibrate_scores([0.,.4,1.],{'method':'identity'}),[0.,.4,1.])
    for values in [[-.01],[1.1],[np.nan],[np.inf]]:
        try: calibrate_scores(values,{'method':'identity'})
        except ValueError: pass
        else: raise AssertionError('Identity must not silently clip arbitrary raw scores')


def test_unified_cli_module_overrides_propagate_to_every_ablation_without_loading_models():
    with redirect_stdout(io.StringIO()):
        cfg=main(['--check-config','--matcher','mean','-s','modules.uncertainty_model.name=column_marginal',
                  '-s','modules.uncertainty_model.params.rho=0.9','-s','modules.uncertainty_propagation.params.prior=0.2'])
    assert cfg['matcher']['name']=='sbert_mean' and cfg['research']['matcher']=='mean'
    assert cfg['uncertainty_model']['params']=={'rho':.9}
    assert cfg['ablation']['missing']['top1']['uncertainty_model']['params']['model']=='column_marginal'
    assert cfg['research']['prior']==.2 and cfg['ablation']['optimizers']['greedy']['params']['threshold']==.2
    assert all(s['uncertainty_propagation']['params']['prior']==.2 for s in cfg['ablation']['missing'].values())


def test_full_semantic_partition_contains_unmatched_singletons():
    columns=[ColumnRef('A','x'),ColumnRef('B','x'),ColumnRef('C','z')]
    matching=Matching([frozenset(('A.x','B.x'))])
    assert matching.partition(columns)==[frozenset(('A.x','B.x')),frozenset(('C.z',))]


def test_paper_pair_scope_prevents_other_columns_from_inducing_uncertainty():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); source,cfg,artifact=fixture(root)
        cfg['matcher']={'name':'column_pair','params':{'matcher':'paper_fixture','matcher_params':{}},'cache_scores':False}
        artifact['matcher']=copy.deepcopy(cfg['matcher'])
        comp,spec=build_components(cfg)
        _,samples,edges,_=compute_edges(source,comp,spec,cfg,RunContext(project_root=str(root),cache_dir=str(root/'cache')))
        complete=edges.df[(edges.df.col_i=='A.fixed')&(edges.df.col_j=='B.fixed')].iloc[0]
        assert complete.sigma==0 and complete.n_samples==1 and len(samples)==34
    with redirect_stdout(io.StringIO()):
        coma=main(['--check-config','--matcher','coma'])
    assert coma['research']['evidence_scope']=='pair' and coma['matcher']['name']=='column_pair'
    assert coma['matcher']['params']['matcher']=='valentine'


def test_identity_fit_explicitly_skips_score_calibration():
    from umsm.training import fit_pipeline
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); source,cfg,artifact=fixture(root); cfg['calibration']={'method':'identity'}
        qx={'generators':[{'name':'knn_tuple','params':{'k':10}},{'name':'column_marginal'}],
            'temperatures':[1.,1.],'weights':[.7,.3]}
        with patch('umsm.training.load_sets',return_value=[source]),patch('umsm.training._fit_qx',return_value=qx), \
             patch('umsm.training._fit_matcher',side_effect=AssertionError('Identity does not fit Platt')):
            fitted=fit_pipeline(cfg,str(root),str(root/'identity.json'))
        assert fitted['platt']['method']=='identity'


def test_guided_categorical_draws_preserve_numpy_inverse_cdf_for_unequal_probabilities():
    from umsm.types import categorical_guide,categorical_index
    q=[('a',.01),('b',.09),('c',.3),('d',.3),('e',.3)]
    cdf,guide=categorical_guide(q)
    left=np.random.default_rng(7).choice(len(q),size=1000,p=[p for _,p in q])
    right=[categorical_index(cdf,guide,u) for u in np.random.default_rng(7).random(1000)]
    np.testing.assert_array_equal(left,right)


def test_qx_transfer_is_matcher_independent_and_never_reuses_old_platt():
    from umsm.training import fit_pipeline
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); source,cfg,artifact=fixture(root)
        qx={'generators':[{'name':'knn_tuple','params':{'k':10}},{'name':'column_marginal'}],
            'temperatures':[.8,1.2],'weights':[.7,.3]}
        old={'version':'qx-global-v2.1','split_role':'validation','fit_sets':[{'fingerprint':'old-source','name':'old'}],
             'qx':qx,'matcher':{'name':'sbert_mean','params':{}},'platt':{'method':'platt','slope':9.,'intercept':-9.}}
        new={'method':'platt','slope':2.,'intercept':-1.}
        with patch('umsm.training.load_sets',return_value=[source]), \
             patch('umsm.training._fit_qx',side_effect=AssertionError('Do not refit shared Q')), \
             patch('umsm.training._fit_matcher',return_value=(new,[])) as fit:
            fitted=fit_pipeline(cfg,str(root),str(root/'transfer.json'),qx_from=old)
        assert fit.call_count==1 and fitted['qx']==qx and fitted['platt']==new
        assert fitted['matcher']['name']=='paper_fixture' and len(fitted['fit_sets'])==2


if __name__=='__main__':
    for name,fn in sorted(list(globals().items())):
        if name.startswith('test_'):
            fn(); print('ok',name)
