"""Offline mathematical fixtures only; no public datasets, model downloads or APIs."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from test_matchers import FakeEncoder
from test_research import toy
from umsm.calibration import set_fingerprint
from umsm.config import load_config
from umsm.data.preparation import publish_task
from umsm.global_optimizer.local_merge import LocalMerge
from umsm.global_optimizer.logit import LogitILP
from umsm.isolated_metrics import (decision_metrics, factorial_effects, paired_bootstrap,
                                    pairwise_decisions, quality_rows)
from umsm.research import research_config,run_research,select_parameter
from umsm.report import read_report
from umsm.types import ColumnRef,EdgeTable


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        for role in ('fit','validation','test'):
            source=toy(role)
            for table in source.tables:
                table.name=role+'_'+table.name
            source.groundtruth=[frozenset(t.name+'.'+c for t in source.tables) for c in ('city','kind')]
            publish_task(root/'data/Toy'/role,{t.name:t.df for t in source.tables},source.groundtruth,{'role':role})
        cfg=load_config(str(ROOT/'configs/research/isolate_alaska.yaml'))
        cfg['dataset']['name']='Toy'
        cfg['research'].update(name='isolated_fixture',fold='fold',calibration='calibration/frozen.json',n_samples=4,
            roles={r:{'name':'Toy','sets':[r]} for r in ('fit','validation','test')},plots=False,
            bootstrap={'unit':'task','resamples':30,'seed':7},prior_grid=[.3,.5,.7],frontier_priors=[.3,.5,.7])
        cfg=research_config(cfg,root)
        from umsm.research import role_config
        _,sources=role_config(cfg,'fit',root)
        artifact={'version':'qx-global-v2.1','split_role':'validation','matcher':cfg['matcher'],
            'matcher_scoring_version':1,'fit_sets':[{'dataset':'Toy','name':'Toy/fit','variant':'original','fingerprint':set_fingerprint(sources[0])}],
            'qx':{'generators':[{'name':'knn_tuple','params':{'k':10}},{'name':'column_marginal'}],
                  'temperatures':[1.,1.],'weights':[.7,.3]},
            'platt':{'method':'platt','slope':4.,'intercept':-2.}}
        (root/'calibration').mkdir()
        (root/'calibration/frozen.json').write_text(json.dumps(artifact),encoding='utf-8')
        with patch('umsm.matcher.sbert_mean.get_encoder',return_value=FakeEncoder()):
            yield root,cfg


def test_pairwise_keeps_nontransitive_edges_without_groups_or_bfs():
    columns=[ColumnRef(t,'x') for t in ('A','B','C')]
    df=pd.DataFrame([{'col_i':a.id,'col_j':b.id,'table_i':a.table,'table_j':b.table,
                       'mu':p,'sigma':0.,'w':p,'p_tilde':p,'a':np.log(p/(1-p))}
                       for a,b,p in ((columns[0],columns[1],.9),(columns[1],columns[2],.9),(columns[0],columns[2],.1))])
    from umsm.types import IntegrationSet,Table
    source=IntegrationSet('Toy','triangle',[Table(t,pd.DataFrame({'x':['v']})) for t in ('A','B','C')],
                          [frozenset(c.id for c in columns)])
    metrics,record,rows,frame=pairwise_decisions(EdgeTable(df,columns),source)
    assert metrics['TP']==2 and metrics['FN']==1 and metrics['pair_f1']==.8
    assert ['A.x','C.x'] not in record['accepted_pairs'] and 'groups' not in record
    assert len(rows)==3 and len(frame)==3


def test_pairwise_counts_truth_outside_closure_as_false_negative():
    source=toy(tables=2)
    columns=source.columns
    a,b=columns[0],columns[2]
    df=pd.DataFrame([{'col_i':a.id,'col_j':b.id,'table_i':a.table,'table_j':b.table,
                      'mu':.9,'sigma':0.,'w':.9,'p_tilde':.9,'a':2.}])
    metrics,_,_,_=pairwise_decisions(EdgeTable(df,columns),source)
    assert metrics['TP']==1 and metrics['FN']==1


def bootstrap_frames():
    baseline=pd.DataFrame({'col_i':['A.x','A.x','A.y','A.y'],'col_j':['B.x','B.y','B.x','B.y'],
        'table_i':['A']*4,'table_j':['B']*4,'missing_bin':['50-80%']*4,'label':[0,1,0,1],
        'mu':[.4,.6,.6,.4],'p_tilde':[.4,.6,.6,.4]})
    method=baseline.copy(); method['mu']=[.1,.9,.2,.8]; method['p_tilde']=method.mu-.05
    return baseline,method


def test_cluster_bootstrap_is_paired_and_refuses_single_task_iid_interval():
    base,method=bootstrap_frames()
    blocks=[{'set':'task1','frames':{'observed_only':base,'top1':base,'Q_mean':method,'Q_risk':method}}]
    rows=paired_bootstrap(blocks,{'unit':'task','resamples':100,'seed':7})
    result=next(r for r in rows if r['contrast']=='Q_mean-observed' and r['missing_bin']=='all' and r['metric']=='brier')
    assert np.isclose(result['difference'],-.235) and result['ci_low'] is None and result['n_clusters']==1
    column=paired_bootstrap(blocks,{'unit':'column','resamples':40,'seed':7})
    assert any(r['valid_resamples']>0 for r in column)
    duplicate=copy.deepcopy(blocks[0]); duplicate['set']='task2'
    task=paired_bootstrap(blocks+[duplicate],{'unit':'task','resamples':40,'seed':7})
    result=next(r for r in task if r['contrast']=='Q_mean-observed' and r['missing_bin']=='all' and r['metric']=='brier')
    assert np.isclose(result['ci_low'],-.235) and np.isclose(result['ci_high'],-.235)
    grouped=paired_bootstrap(blocks+[duplicate],{'unit':'task','resamples':40,'task_groups':{'task1':'shared','task2':'shared'}})
    assert all(r['ci_low'] is None for r in grouped)
    single_class=copy.deepcopy(duplicate)
    for frame in single_class['frames'].values():
        frame['label']=0
    insufficient=paired_bootstrap(blocks+[single_class],{'unit':'task','resamples':40})
    auc=next(r for r in insufficient if r['contrast']=='Q_mean-observed' and r['missing_bin']=='all' and r['metric']=='auc')
    assert auc['n_valid_task_clusters']==1 and auc['ci_low'] is None


def test_factorial_interaction_uses_difference_of_differences():
    rows=[{'set':'task','draw':0,'k':3,'missing':m,'optimizer':o,'pair_f1':f}
          for m,o,f in [('observed_only','local',.5),('observed_only','global',.6),('Q_mean','local',.7),('Q_mean','global',.9)]]
    effects,ci=factorial_effects(rows)
    assert np.isclose(effects[0]['interaction'],.1) and ci[0]['ci_low'] is None


def test_edges_and_pairwise_never_call_partition_optimizers():
    with fixture() as (root,cfg),patch.object(LogitILP,'optimize',side_effect=AssertionError('ILP forbidden')), \
         patch.object(LocalMerge,'optimize',side_effect=AssertionError('BFS forbidden')):
        prior=run_research(cfg,'validate-prior',root)
        assert prior['selection']['decision_protocol']=='pairwise' and prior['selection']['base_strategy']=='observed_only'
        edge=run_research(cfg,'edges',root,prior['result_file'])
        assert not edge['predictions'] and all(r['optimizer']=='none' for r in edge['rows'])
        assert {r['score_space'] for r in edge['rows']}=={'mu','p_tilde'}
        assert all('mean_sigma' in r and 'mean_abs_risk_shift' in r for r in edge['rows'])
        pair=run_research(cfg,'pairwise',root,prior['result_file'])
        assert len(pair['rows'])==5 and all('matching' not in r for r in pair['predictions'])
        assert len({r['prior'] for r in pair['rows']})==1
        assert not pair['protocol']['diagnostic_test_fit']
        from analyze_research import analyze
        overview=analyze([edge['result_file'],pair['result_file']],root/'overview.html')
        assert overview['analysis_tables'] and not overview['figures']


def test_frontier_reuses_one_matrix_and_diagnostic_cannot_be_used_for_selection():
    with fixture() as (root,cfg):
        from umsm.isolation import compute_edges as original
        with patch('umsm.isolation.compute_edges',wraps=original) as compute:
            frontier=run_research(cfg,'frontier',root)
        assert compute.call_count==1 and len(frontier['rows'])==6
        assert {r['prior'] for r in frontier['rows']}=={.3,.5,.7}
        assert len(frontier['analysis_tables']['All prior points and Pareto membership'])==6
        from umsm.isolation import isolation_plots
        assert '<svg' in isolation_plots(frontier)[0]
        from analyze_research import analyze
        overview=analyze([frontier['result_file']],root/'frontier_overview.html')
        assert {r['prior'] for r in overview['analysis_tables']['Paired descriptive means']}=={.3,.5,.7}
        before=(root/'calibration/frozen.json').read_bytes()
        diagnostic=run_research(cfg,'diagnostic-calibration',root)
        assert (root/'calibration/frozen.json').read_bytes()==before
        assert diagnostic['kind']=='research_diagnostic' and diagnostic['protocol']['diagnostic_test_fit']
        assert all(r['oracle'] and not r['claim_eligible'] for r in diagnostic['rows'])
        assert {r['calibration_condition'] for r in diagnostic['rows']}=={'frozen','test_fitted'}
        assert 'TEST-FITTED CALIBRATION DIAGNOSTIC' in Path(diagnostic['result_file']).read_text(encoding='utf-8')
        try:
            select_parameter(diagnostic['result_file'],cfg,root)
        except ValueError as error:
            assert 'Diagnostic/test-fitted' in str(error)
        else:
            raise AssertionError('In-sample diagnostic cannot choose a formal prior')
        try:
            analyze([diagnostic['result_file']],root/'forbidden_overview.html')
        except ValueError:
            pass
        else:
            raise AssertionError('Formal summaries must reject diagnostic reports')
        factorial=run_research(cfg,'factorial',root)
        assert len(factorial['rows'])==4 and factorial['analysis_tables']['2x2 interaction by task']


def test_frozen_selection_ignores_runtime_cache_flags_but_rejects_changed_scores():
    with fixture() as (root,cfg):
        prior=run_research(cfg,'validate-prior',root)
        cold=copy.deepcopy(cfg); cold['matcher']['cache_scores']=False
        assert select_parameter(prior['result_file'],cold,root)['value']==prior['selection']['value']
        changed=copy.deepcopy(cfg); changed['matcher']['params']['name_weight']=0.
        try: select_parameter(prior['result_file'],changed,root)
        except ValueError: pass
        else: raise AssertionError('Changed score semantics must not reuse a validation choice')


if __name__=='__main__':
    tests=[f for n,f in sorted(globals().items()) if n.startswith('test_')]
    for test in tests:
        test()
        print('ok',test.__name__)
    print(len(tests),'tests passed')
