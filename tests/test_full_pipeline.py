"""Offline tests for aligned multiple imputation, factorial attribution and retrieval-aware calibration."""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from test_isolation import fixture
from test_matchers import FakeEncoder,_people_set
from umsm.calibration import calibrate_scores,fit_retrieval_calibrators
from umsm.context import RunContext
from umsm.factorial_analysis import factorial_4x2
from umsm.online import OnlineMoments
from umsm.types import ColumnRef,IntegrationSet,RealizationSpec,Table
from umsm.uncertainty_model.multiple_mice import MultipleMICE


def test_mice_draws_preserve_table_alignment_and_observed_cells():
    frame=pd.DataFrame({'x':np.arange(24,dtype=float),'y':np.arange(24,dtype=float)*2})
    frame.loc[[2,4,6],'x']=np.nan; frame.loc[[3,5,7],'y']=np.nan
    source=IntegrationSet('Toy','Toy/data',[Table('A.csv',frame),Table('B.csv',frame.copy())],[])
    model=MultipleMICE(m=5,max_iter=2,type_options={'overrides':{'x':'numeric','y':'numeric'}})
    data=model.estimate(source,RunContext(seed=42)).configure(RealizationSpec(n_samples=5),42)
    draws=list(data.iter_realizations()); assert len(draws)==5
    tables=data.table_realizations('A.csv'); assert len(tables)==5
    for r,states in draws:
        observed=tables[r][0]
        assert np.allclose(pd.to_numeric(observed.loc[frame.x.notna(),'x']),frame.loc[frame.x.notna(),'x'])
        assert list(states[ColumnRef('A.csv','x')].values)==observed.x.tolist()
        assert list(states[ColumnRef('A.csv','y')].values)==observed.y.tolist()
    again=model.estimate(source,RunContext(seed=42)).configure(RealizationSpec(n_samples=5),42)
    assert [x[0].x.tolist() for x in data.table_realizations('A.csv')]==[x[0].x.tolist() for x in again.table_realizations('A.csv')]


def test_mice_posterior_completes_predictors_without_filling_hidden_target():
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from umsm.uncertainty_model.typed import TypedGenerator, profiles
    frame=pd.DataFrame({'target':[1.,2.,np.nan,4.,5.,6.], 'feature':[2.,np.nan,6.,8.,np.nan,12.]})
    prof=profiles(frame,options={'overrides':{'target':'numeric','feature':'numeric'}})
    seen=[]; original_fit=IterativeImputer.fit
    def checked_fit(model,matrix,*args,**kwargs):
        seen.append(np.asarray(matrix).copy())
        assert np.isnan(matrix[2,0]) and np.isfinite(matrix[:,1:]).all()
        return original_fit(model,matrix,*args,**kwargs)
    with patch.object(IterativeImputer,'fit',checked_fit):
        result=TypedGenerator('mice_posterior',predictor_missing='median',n_draws=4,max_iter=2)._mice(
            frame,prof,'target',prof['target'],np.array([2]),RunContext(seed=42))
    assert len(seen)==1 and result[(2,'target')]
    assert np.isfinite([score for _,score in result[(2,'target')]]).all()


def test_profile_supports_natural_split_labels_and_constructed_split_details():
    from umsm.research import run_research
    with fixture() as (root,cfg):
        cfg['research']['profile_role']='test'
        path=root/'data/Toy/test/provenance.json'
        manifest=json.loads(path.read_text(encoding='utf-8'))
        for split,expected in [('test',None),({'actual_sibling_ratio':.3},.3)]:
            manifest['protocol']['split']=split
            path.write_text(json.dumps(manifest),encoding='utf-8')
            report=run_research(copy.deepcopy(cfg),'profile',root)
            assert report['analysis_tables']['Integration sets'][0]['sibling_ratio']==expected


def test_multiple_mice_averages_raw_scores_before_nonlinear_calibration():
    a,b=ColumnRef('A','x'),ColumnRef('B','x'); pair=frozenset((a.id,b.id))
    tracker=OnlineMoments([a,b],{pair:2},{'aggregation':'raw_mean_then_calibrate'})
    params={'method':'platt','slope':4.,'intercept':-1.}
    for raw in (0.,1.):
        tracker.update(pd.DataFrame([{'col_i':a.id,'col_j':b.id,'prob':.5,'score':raw}]),params)
    result=tracker.finish().iloc[0]
    assert np.isclose(result.mu,calibrate_scores([.5],params)[0])
    assert result.sigma==0. and not np.isclose(result.mu,calibrate_scores([0.,1.],params).mean())


def test_factorial_4x2_cluster_bootstrap_and_holm():
    levels=['observed_only','stat_mice','mice_multiple','Q_risk']; rows=[]
    for i in range(12):
        for j,level in enumerate(levels):
            for opt in ('local','global'):
                rows.append({'set':f'S/{i}','draw':0,'k':6,'missing':level,'optimizer':opt,
                             'pair_f1':.4+j*.03+(.02+j*.01 if opt=='global' else 0.)})
    cases,summary=factorial_4x2(rows,levels,{'resamples':100,'min_clusters':10})
    values={r['effect']:r for r in summary}
    assert len(cases)==12 and np.isclose(values['interaction:Q_risk']['difference'],.03)
    assert np.isclose(values['optimizer_main']['difference'],.035)
    assert all(0<=r['p_unadjusted']<=r['p_holm']<=1 for r in summary)
    _,small=factorial_4x2(rows[:9*8],levels,{'resamples':100,'min_clusters':10})
    assert not any(r['eligible_n'] or r['supported'] for r in small)


def test_retrieval_feature_changes_probability_and_is_required():
    scores=[0.]*10+[.2]*10+[.7]*10; labels=[0]*9+[1]+[0]*8+[1]*2+[0]*2+[1]*8
    flags=[0]*10+[1]*20; candidates=fit_retrieval_calibrators(scores,labels,flags)
    assert set(candidates)=={'platt_single','platt_retrieved','isotonic_retrieved'}
    for name in ('platt_retrieved','isotonic_retrieved'):
        p=calibrate_scores([0.,0.],candidates[name],[0,1]); assert p[0]!=p[1]
        try: calibrate_scores([0.],candidates[name])
        except ValueError: pass
        else: raise AssertionError('Missing retrieval indicators must not be treated as zero')
    try: fit_retrieval_calibrators([.1,.5],[0,1],[1,1])
    except ValueError as exc: assert 'both retrieved and pruned' in str(exc)
    else: raise AssertionError('A retrieval effect cannot be estimated without both fit groups')


def test_magneto_retrieval_indicator_is_binary_under_bidirection():
    from umsm import MATCHERS,UNCERTAINTY_MODELS
    ctx=RunContext(seed=9); source=_people_set()
    data=UNCERTAINTY_MODELS.build('observed_only').estimate(source,ctx).configure(RealizationSpec(n_samples=2),9)
    with patch('umsm.matcher.magneto.get_encoder',return_value=FakeEncoder()):
        result=MATCHERS.build('magneto',{'bidirectional':True,'topk':1,'embedding_threshold':-1.}).score(data,ctx).df
    assert set(result.retrieved.unique())<={0.,1.} and set(result.retrieved.unique())=={0.,1.}
    assert (result[result.retrieved==0].score==0).all()


def test_real_4x2_pipeline_executes_with_shared_frozen_calibration():
    from umsm.presets import ablation_specs
    from umsm.research import run_research
    with fixture() as (root,cfg):
        cfg['research'].update(multiple_mice=True,multiple_mice_m=5,statistical_baselines=['mice'],
            factorial_levels=['observed_only','stat_mice','mice_multiple','Q_risk'])
        cfg['ablation']=ablation_specs(cfg)
        result=run_research(copy.deepcopy(cfg),'factorial',root)
        assert len(result['rows'])==8
        assert result['analysis_tables']['Factorial effects by task']
        assert result['analysis_tables']['Factorial cluster CI and Holm tests']


def test_validation_offset_grid_is_frozen_for_each_factorial_pipeline():
    from umsm.presets import ablation_specs
    from umsm.research import run_research
    with fixture() as (root,cfg):
        path=root/'calibration/frozen.json'; artifact=json.loads(path.read_text()); artifact['platt']['fit_positive_prior']=.2
        path.write_text(json.dumps(artifact))
        cfg['research'].update(prior_policy='fit_prevalence',lambda_optimizer='pairwise',lambda_grid=[0.,.1],
            multiple_mice=True,multiple_mice_m=5,statistical_baselines=['mice'],
            factorial_levels=['observed_only','stat_mice','mice_multiple','Q_risk'],
            missing=['observed_only','stat_mice','mice_multiple','Q_risk'],optimizers=['local','global'],
            decision_offset_grid=[-.5,0.,.5])
        cfg['ablation']=ablation_specs(cfg)
        initial=run_research(copy.deepcopy(cfg),'validate-lambda',root)
        selected=run_research(copy.deepcopy(cfg),'validate-offsets',root,initial['result_file'])
        result=run_research(copy.deepcopy(cfg),'factorial',root,selected['result_file'])
        offsets=selected['selection']['decision_offsets']
        assert selected['selection']['held_fixed']==.2 and len(result['rows'])==8
        assert all(r['decision_offset']==offsets[r['missing']][r['optimizer']] for r in result['rows'])
        assert selected['selection']['strongest_validation_baseline']['missing']!='Q_risk'
        from combine_factorial import combine as combine_factorial
        from combine_pipeline import combine as combine_pipeline
        from umsm.report import read_report
        pooled=combine_factorial([result['result_file']],root/'factorial.html',root)
        assert not any(r['eligible_n'] for r in read_report(pooled)['rows'])
        main=run_research(copy.deepcopy(cfg),'missing',root,selected['result_file'])
        table=combine_pipeline([main['result_file']],root/'main.html',root)
        comparisons=read_report(table)['analysis_tables']['Q risk + ILP versus paired pipelines']
        assert any(r['baseline']=='strongest validation-selected pipeline' for r in comparisons)


def test_relaxing_same_table_exclusion_retains_transitivity_but_can_conflict():
    from umsm.study_eval import no_same_table_constraint,triangle_counts
    from umsm.types import EdgeTable
    a,b,c=ColumnRef('A','x'),ColumnRef('A','y'),ColumnRef('B','z')
    frame=pd.DataFrame([{'col_i':x.id,'col_j':c.id,'table_i':x.table,'table_j':c.table,
                         'a':2.,'w':.9,'mu':.9,'sigma':0.} for x in (a,b)])
    graph=EdgeTable(frame,[a,b,c],{'potential':'calibrated_logit'})
    result=no_same_table_constraint(graph,RunContext(),{'triangles':'full'})
    assert result.info['same_table_conflicts']==1
    from umsm.research_analysis import pair_set
    assert triangle_counts(graph.columns,pair_set(result.groups))['open_wedges']==0


def test_aligned_donor_selection_uses_observed_anchors_and_skips_incompatible_sources():
    from umsm.data.study import choose_compatible_donor
    base=pd.DataFrame({'brand':['a','b']*6,'price':np.arange(12,dtype=float)})
    good=base.copy(); good.loc[good.brand=='a','price']=np.nan
    bad=pd.DataFrame({'unrelated':['c','d']*6,'price':np.arange(12,dtype=float)})
    donors=[{'task':'D/bad','table':Table('bad.csv',bad),'band':'high'},
            {'task':'D/good','table':Table('good.csv',good),'band':'high'}]
    donor,result,mask,audit=choose_compatible_donor(base,donors,'high',0,{},42)
    assert donor['task']=='D/good' and audit['unaligned_rows']==0
    assert result.price.isna().tolist()==(base.brand=='a').tolist() and mask[:,1].sum()==6


def test_declared_source_site_overlap_is_rejected_when_required():
    from umsm.research import ensure_disjoint
    left=IntegrationSet('L','L/1',[Table('L.csv',pd.DataFrame({'v':[1,2]}))],[],{'source_sites':['shop.example']})
    right=IntegrationSet('R','R/1',[Table('R.csv',pd.DataFrame({'v':[3,4]}))],[],{'source_sites':['shop.example']})
    ensure_disjoint([left],[right])
    try: ensure_disjoint([left],[right],by_site=True)
    except ValueError as exc: assert 'websites' in str(exc)
    else: raise AssertionError('Same recorded site must not be treated as independent roles')


def test_magneto_fit_validation_and_selected_cache_work_offline():
    from umsm.presets import ablation_specs
    from umsm.research import role_config,run_research
    from umsm.training import fit_pipeline
    with fixture() as (root,cfg), patch('umsm.matcher.magneto.get_encoder',return_value=FakeEncoder()):
        cfg['matcher']={'name':'magneto','params':{'reranker':'llm','bidirectional':True,'topk':1,
            'embedding_threshold':-1.,'llm_backend':'mock','llm_workers':1},'cache_scores':True}
        cfg['uncertainty_model']={'name':'observed_only','params':{},'realization':{'mode':'mc','n_samples':4}}
        cfg['calibration']={'method':'retrieval_candidates'}; cfg['paths']['raw_score_cache']=True
        cfg['research'].update(matcher='magneto',prior_policy='fit_prevalence',lambda_optimizer='pairwise',
                               lambda_grid=[0.,.1],missing=['observed_only'],optimizers=['local','global'])
        cfg['ablation']=ablation_specs(cfg); fitcfg,_=role_config(cfg,'fit',root)
        path=root/'calibration/new_magneto.json'; artifact=fit_pipeline(fitcfg,str(root),str(path),max_cells=20)
        assert set(artifact['calibration_candidates'])=={'platt_single','platt_retrieved','isotonic_retrieved'}
        cfg['research']['calibration']='calibration/new_magneto.json'
        selected=run_research(copy.deepcopy(cfg),'select-calibrator',root)
        cfg['research']['calibration']=str(selected['calibration_file'])
        for section in ('uncertainty_propagation',): cfg[section]['params']['calibration']=cfg['research']['calibration']
        cfg['ablation']=ablation_specs(cfg)
        validated=run_research(copy.deepcopy(cfg),'validate-lambda',root)
        result=run_research(copy.deepcopy(cfg),'global',root,validated['result_file'])
        assert len(result['rows'])==2 and result['calibration_sha256']


if __name__=='__main__':
    tests=[f for name,f in sorted(globals().items()) if name.startswith('test_')]
    for test in tests: test(); print('ok',test.__name__,flush=True)
    print(f'{len(tests)} tests passed')
