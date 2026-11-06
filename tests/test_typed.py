"""Typed candidates, masked fitting, probability scoring and old-protocol isolation; offline."""
import copy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from umsm import UNCERTAINTY_MODELS, MATCHERS
from umsm.context import RunContext
from umsm.types import Table,IntegrationSet,RealizationSpec
from umsm.uncertainty_model.mixture import (GENERATORS,CalibratedMixture,pseudo_mask_examples)
from umsm.uncertainty_model.typed import (ColumnProfile,profile_column,profiles,TypedGenerator,
    DEFAULT_ROUTES,fit_typed,typed_diagnostics,calibrated_q,crps,gower_distances)
from umsm.pipeline import compute_edges,build_components
from umsm.presets import paper_config
from umsm.training import fit_pipeline
from test_matchers import FakeEncoder


def toy():
    frame=pd.DataFrame({'group':['a']*10+['b']*10,'x':np.arange(20,dtype=float),
        'price':[f'${100+i*2:,.2f} kg' for i in range(20)],
        'text':[f'model-{i//2}' for i in range(20)],'id':[f'http://item/{i}' for i in range(20)]})
    frame.loc[[2,12],'price']=None; frame.loc[[3,13],'group']=None; frame.loc[[4,14],'text']=None
    return IntegrationSet('Toy','train',[Table('A',frame)])


def test_profiles_preserve_numeric_date_formats_and_explicit_identifiers():
    p=profile_column(pd.Series(['$1,200.50 kg','$1,201.25 kg',None],name='mass'))
    assert p.kind=='numeric' and p.number('$1,200.50 kg')==1200.5
    assert p.render(1250.25)=='$1,250.25 kg'
    d=profile_column(pd.Series(['2024/01/01','2024/01/05',None],name='day'))
    assert d.kind=='date' and d.render(d.number('2024/01/05'))=='2024/01/05'
    assert profile_column(pd.Series([1.,2.,3.],name='id'),'A',{'overrides':{'A.id':'identifier'}}).kind=='identifier'
    assert profile_column(pd.Series(['url-a','url-b','url-c'])).kind=='identifier'
    assert profile_column(pd.Series([1,1,1,2,2,2,2,1])).kind=='ordered_numeric'
    assert profile_column(pd.Series([None,None],name='number'),'A',{'overrides':{'number':'numeric'}}).kind=='numeric'


def test_gower_uses_numeric_closeness_and_excludes_target():
    frame=pd.DataFrame({'x':[100.,101.,10000.],'target':[None,'near','far']})
    prof=profiles(frame,'A',{'overrides':{'target':'categorical'}})
    distances=gower_distances(frame,prof,'target')
    assert distances[0,1]<distances[0,2]
    table=Table('A',frame); source=IntegrationSet('Toy','train',[table])
    q=TypedGenerator('gower_knn',k=1).generate(frame,table,source,RunContext(),prof)
    assert q[(0,'target')][0][0]=='near'
    assert np.isinf(gower_distances(frame[['target']],{'target':prof['target']},'target')[0,1])


def test_typed_pseudomasks_hide_truth_from_every_input_and_stratify():
    class Spy:
        def generate(self,frame,table,source,ctx):
            pd.testing.assert_frame_equal(frame,table.df)
            pd.testing.assert_frame_equal(frame,source.tables[0].df)
            assert table.path is None and source.groundtruth is None
            assert frame.isna().to_numpy().sum()==7
            return {}
    routes={k:['typed_spy'] for k in ('categorical','numeric','text')}; routes['identifier']=[]
    source=toy()
    with patch.dict(GENERATORS,{'typed_spy':lambda params:Spy()}):
        examples=pseudo_mask_examples([source],routes,RunContext(),12)
    assert len(examples)==12 and len({e['kind'] for e in examples})>=3
    assert sum(source.tables[0].df.isna().sum())==6


def test_continuous_quantiles_keep_mass_and_no_arbitrary_frequency_prefix():
    profile=ColumnProfile('numeric',100,1.,scale=50.,as_string=False)
    ex={'kind':'numeric','profile':asdict(profile),'generators':[[(float(i),0.) for i in range(100)]]}
    params={'by_type':{'numeric':{'temperatures':[1.],'weights':[1.],'dispersion':1.,'fit_status':'test'}}}
    q=calibrated_q(ex,params,10)
    assert len(q.candidates)==10 and min(v for v,w in q.candidates)>0 and max(v for v,w in q.candidates)>90
    artifact={'version':'qx-global-v2.1','split_role':'validation','fit_sets':[],
        'qx':{'generators':{'numeric':['quantile_marginal']},'routing':'by_type','typed_version':1,'type_options':{},**params}}
    model=CalibratedMixture(artifact,routing='by_type',generators={'numeric':['quantile_marginal']},rho=.5)
    kept,stats=model.prepare_distribution(q)
    assert len(kept)==10 and abs(sum(w for v,w in kept)-1)<1e-12 and stats['numeric_quantile_mass_kept']==1.
    assert crps(np.array([0.,2.]),np.array([.5,.5]),1.)==.5


def test_fit_uses_numeric_crps_and_diagnostics_use_coverage_not_exact_nll():
    profile=asdict(ColumnProfile('numeric',4,1.,scale=2.,as_string=False))
    examples=[{'kind':'numeric','profile':profile,'truth':y,'numeric_truth':y,
        'generators':[[(0.,0.),(2.,0.)],[(20.,0.),(22.,0.)]],
        'marginal_baseline':[(0.,0.),(2.,0.)]} for y in (.8,1.,1.2,1.)]
    params=fit_typed(examples,{'numeric':['quantile_marginal','gower_knn']})
    fitted=params['by_type']['numeric']
    assert fitted['fit_status']=='fitted_crps' and fitted['weights'][0]>.9
    assert fitted['fit_crps']<fitted['fit_crps_initial']
    row=next(x for x in typed_diagnostics(examples,params)['by_type'] if x['column_type']=='numeric')
    assert row['n_evaluated']==4 and 'nll_retained' not in row
    assert sum(row['pit_histogram'])==4 and 0<=row['coverage_80']<=1


def test_date_crps_is_translation_invariant_and_fits_at_atomic_knots():
    from umsm.uncertainty_model.typed import fit_continuous
    values=np.array([-4.,0.,3.,9.]); weights=np.array([.1,.2,.3,.4]); truth=2.
    exact=weights@np.abs(values-truth)-.5*np.sum(weights[:,None]*weights[None,:]*np.abs(values[:,None]-values[None,:]))
    assert np.isclose(crps(values,weights,truth),exact,atol=1e-12)
    assert np.isclose(crps(values+1e12,weights,truth+1e12),exact,atol=1e-12)
    def examples(origin):
        profile=asdict(ColumnProfile('date',4,1.,scale=2.,date_format='%Y-%m-%d'))
        def day(offset): return (pd.Timestamp(origin)+pd.Timedelta(days=offset)).strftime('%Y-%m-%d')
        return [{'kind':'date','profile':profile,'truth':day(y),'numeric_truth':ColumnProfile(**profile).number(day(y)),
            'generators':[[(day(0),0.),(day(2),0.)],[(day(8),0.),(day(10),0.)]]} for y in (0,1,2,1,0,2)]
    fits=[fit_continuous(examples(origin),2) for origin in ('2001-01-01','2101-01-01')]
    for fitted in fits:
        assert fitted['optimizer_success'] and fitted['optimizer']=='Powell'
        assert np.isfinite(fitted['fit_crps']) and fitted['fit_crps']<=fitted['fit_crps_initial']
        assert .25<=fitted['dispersion']<=4 and np.isclose(sum(fitted['weights']),1.)
    assert np.isclose(fits[0]['fit_crps'],fits[1]['fit_crps'],atol=1e-10)
    np.testing.assert_allclose(fits[0]['weights'],fits[1]['weights'],atol=1e-10)


def test_cluster_nb_mice_and_embedding_proposals_are_observed_only_and_repeatable():
    source=toy(); table=source.tables[0]; frame=table.df; ctx=RunContext(seed=7)
    prof=profiles(frame,'A')
    for name,target in [('cluster_marginal','group'),('nb_classifier','group'),('mice_posterior','price')]:
        gen=TypedGenerator(name,n_draws=6,max_iter=2,quantiles=4)
        first=gen.generate(frame,table,source,ctx,prof,{(3,'group'),(2,'price')})
        assert first and any(c==target for r,c in first)
        assert first==gen.generate(frame,table,source,ctx,prof,{(3,'group'),(2,'price')})
        assert all(np.isfinite(s) for q in first.values() for v,s in q)
    class Encoder(FakeEncoder):
        def encode_map(self,texts,namespace='default'):
            assert all('text:' not in t for t in texts)
            return super().encode_map(texts,namespace)
    with patch('umsm.embedding.get_encoder',return_value=Encoder()):
        q=TypedGenerator('row_embedding_knn').generate(frame,table,source,ctx,prof,{(4,'text')})
    assert q[(4,'text')] and all(v in set(frame.text.dropna()) for v,s in q[(4,'text')])


def test_statistical_baselines_preserve_observed_cells_formats_and_skip_identifiers():
    source=toy(); source.tables[0].df.loc[1,'id']=None
    for method in ('mean_mode','median_mode','knn','mice','missforest'):
        model=UNCERTAINTY_MODELS.build('statistical',{'method':method,'n_estimators':3,'max_iter':2})
        result=model.estimate(source,RunContext(seed=2)).tables[0].frame
        original=source.tables[0].df
        for c in original:
            assert result.loc[original[c].notna(),c].tolist()==original.loc[original[c].notna(),c].tolist()
        assert pd.isna(result.loc[1,'id']) and isinstance(result.loc[2,'price'],str)
        assert result.loc[2,'price'].startswith('$') and result.loc[2,'price'].endswith(' kg')


@MATCHERS.register('typed_test_matcher')
class ToyMatcher:
    """Offline matcher that exercises the real score/calibration/online pipeline."""
    sampling_scope='column_pair'; scoring_version=1; stochastic=False
    def params(self): return {}
    def expected_sample_counts(self,data):
        from umsm.matcher.base import BaseMatcher
        return BaseMatcher.expected_sample_counts(self,data)
    def score(self,data,ctx):
        from umsm.matcher.base import PairwiseMatcher
        return PairwiseMatcher.score(self,data,ctx)
    def similarity(self,ci,si,cj,sj,ctx): return .8 if ci.name==cj.name else .2


def test_frozen_typed_fit_runs_online_pipeline_argmax_and_rejects_changed_routing():
    a=pd.DataFrame({'number':list(range(20)),'category':['a']*10+['b']*10})
    a.loc[2,'number']=np.nan; a.loc[3,'category']=None
    source=IntegrationSet('Toy','fit',[Table('A',a.copy()),Table('B',a.copy())],
        [frozenset(('A.number','B.number')),frozenset(('A.category','B.category'))])
    cfg=paper_config('unused'); cfg['matcher']={'name':'typed_test_matcher','params':{},'cache_scores':False}
    routes={'categorical':['column_marginal','gower_knn'],'numeric':['quantile_marginal','gower_knn'],'identifier':[]}
    cfg['uncertainty_model']['params'].update(routing='by_type',generators=routes)
    cfg['uncertainty_model']['realization']['n_samples']=4
    with tempfile.TemporaryDirectory(dir=ROOT/'cache') as folder:
        path=str(Path(folder)/'calibration.json')
        with patch('umsm.training.load_sets',return_value=[source]):
            artifact=fit_pipeline(cfg,str(ROOT),path,max_cells=12)
        assert artifact['qx']['routing']=='by_type' and artifact['platt']['fit_success']
        cfg['uncertainty_model']['params']['calibration']=path
        cfg['uncertainty_propagation']['params']['calibration']=path
        heldout=copy.deepcopy(source); heldout.tables[0].df.loc[0,'number']=100.
        comp,real=build_components(cfg); ctx=RunContext(project_root=str(ROOT),cache_dir=folder)
        data,samples,graph,hit=compute_edges(heldout,comp,real,cfg,ctx)
        assert np.isfinite(graph.df[['mu','sigma','a']]).all().all()
        assert len(graph.df)==4 and data.tables[0].distributions
        model=UNCERTAINTY_MODELS.build('argmax',{'model':'calibrated_mixture','model_params':cfg['uncertainty_model']['params']})
        assert pd.notna(model.estimate(heldout,ctx).tables[0].frame.loc[2,'number'])
        wrong=CalibratedMixture(path,routing='by_type',generators=routes,type_options={'parse_fraction':.9})
        try: wrong.estimate(heldout,ctx)
        except ValueError as exc: assert 'options differ' in str(exc)
        else: raise AssertionError('Changed typing must require a new artifact')


def test_type_strata_keep_all_edge_summary_and_bootstrap_separate():
    from umsm.isolated_metrics import quality_rows,paired_bootstrap
    from umsm.isolation import isolation_summary
    frame=pd.DataFrame({'col_i':['A.x','A.y'],'col_j':['B.x','B.y'],'label':[1,0],
        'mu':[.8,.2],'p_tilde':[.8,.2],'sigma':[0.,0.],'missing_bin':['0-20%']*2,
        'column_type_pair':['numeric / numeric','categorical / categorical']})
    rows=quality_rows(frame,{'missing':'Q_mean','optimizer':'none','prior':.5})
    summary=isolation_summary(rows)
    assert any(r.get('column_type_pair')=='numeric / numeric' for r in summary)
    blocks=[{'set':'a','frames':{'observed_only':frame,'Q_mean':frame}}]
    ci=paired_bootstrap(blocks,{'unit':'task','resamples':0,'type_strata':True})
    assert any(r.get('column_type_pair')=='numeric / numeric' for r in ci)
    assert all(r['difference']==0 for r in ci if r['difference'] is not None)


def test_typed_research_profile_fit_qcheck_prior_and_isolated_reports():
    from umsm.config import load_config
    from umsm.data.preparation import publish_task
    from umsm.research import research_config,run_research
    with tempfile.TemporaryDirectory(dir=ROOT/'cache') as folder:
        root=Path(folder)
        for offset,role in enumerate(('fit','validation','test')):
            frame=pd.DataFrame({'number':np.arange(20,dtype=float)+offset*100,'category':['a']*10+['b']*10})
            frame.loc[2,'number']=np.nan; frame.loc[3,'category']=None
            publish_task(root/'data/Toy'/role,{'A.csv':frame,'B.csv':frame.copy()},
                [frozenset(('A.csv.number','B.csv.number')),frozenset(('A.csv.category','B.csv.category'))],{'role':role})
        cfg=load_config(str(ROOT/'configs/typed_gower.yaml'))
        cfg['modules']['matcher']={'name':'typed_test_matcher','params':{}}
        cfg['research'].update(name='typed_offline',matcher='typed_test',fold='fold',n_samples=4,max_cells=12,
            roles={r:{'name':'Toy','sets':[r]} for r in ('fit','validation','test')},plots=False,
            bootstrap={'unit':'task','resamples':0},prior_grid=[.2,.5],
            missing=['observed_only','top1','Q_mean','Q_risk','stat_knn'])
        cfg=research_config(cfg,root)
        profile=run_research(cfg,'profile',root)
        assert profile['analysis_tables']['Natural-missing columns'][0]['kind'] in ('numeric','categorical')
        run_research(cfg,'fit',root)
        checked=run_research(cfg,'qcheck',root)
        assert checked['analysis_tables']['Q diagnostics by column type']
        prior=run_research(cfg,'validate-prior',root)
        pairwise=run_research(cfg,'pairwise',root,prior['result_file'])
        assert 'stat_knn' in {r['missing'] for r in pairwise['rows']}
        assert pairwise['analysis_tables']['Pairwise decisions by column type']
        assert any(r.get('column_type_pair')=='numeric / numeric'
            for r in pairwise['analysis_tables']['Paired pairwise-decision cluster bootstrap'])


if __name__=='__main__':
    tests=[fn for name,fn in sorted(globals().items()) if name.startswith('test_')]
    for fn in tests:
        fn(); print('ok',fn.__name__)
    print(f'{len(tests)} tests passed')
