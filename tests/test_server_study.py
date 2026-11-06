"""Preregistered prevalence, isolated lambda, transplant lineage, split semantics and handoff."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from umsm.calibration import fit_platt
from umsm.context import RunContext
from umsm.data.preparation import publish_task
from umsm.data.study import split_siblings,prepare_study
from umsm.data.transplant import transplant_mask
from umsm.research import apply_fit_prior,run_research,source_clusters
from umsm.study_eval import outer_union_metrics,triangle_counts
from umsm.types import ColumnRef,EdgeTable,IntegrationSet,Matching,Table
from test_isolation import fixture


def test_fit_prior_counts_edges_not_monte_carlo_rows():
    labels=[1]*10+[0,0]; scores=[.8]*10+[.1,.2]; weights=[.1]*10+[1.,1.]
    artifact=fit_platt(scores,labels,weights)
    assert np.isclose(artifact['fit_positive_prior'],1/3)
    assert not np.isclose(artifact['fit_positive_prior'],np.mean(labels))


def test_lambda_validation_is_pairwise_freezes_fit_prevalence_and_can_choose_zero():
    with fixture() as (root,cfg):
        path=root/'calibration/frozen.json'; artifact=json.loads(path.read_text()); artifact['platt']['fit_positive_prior']=.2
        path.write_text(json.dumps(artifact))
        cfg['research'].update(prior_policy='fit_prevalence',lambda_optimizer='pairwise',lambda_grid=[0.,.1,.5])
        with patch('umsm.global_optimizer.logit.LogitILP.optimize',side_effect=AssertionError('ILP in imputation validation')):
            result=run_research(copy.deepcopy(cfg),'validate-lambda',root)
            tested=run_research(copy.deepcopy(cfg),'pairwise',root,result['result_file'])
        assert result['selection']['axis']=='lambda' and result['selection']['held_fixed']==.2
        assert {r['prior'] for r in result['rows']}=={.2}
        assert {r['risk_lambda'] for r in result['rows']}=={0.,.1,.5}
        assert all(r['prior']==.2 for r in tested['rows'])


def test_aligned_transplant_copies_real_conditional_pattern_without_rescaling():
    donor=pd.DataFrame({'brand':['a','a','b','b'],'price':[None,None,3.,4.]})
    base=pd.DataFrame({'brand':['a','b','a','b'],'price':[1.,2.,3.,4.]})
    result,mask,audit=transplant_mask(base,donor,42,'aligned',anchors={'brand':'brand'})
    assert result.price.isna().tolist()==[True,False,True,False]
    assert not mask[:,0].any() and audit['unaligned_rows']==0
    assert audit['cells']==[[0,1],[2,1]]
    pd.testing.assert_frame_equal(base,pd.DataFrame({'brand':['a','b','a','b'],'price':[1.,2.,3.,4.]}))
    try: transplant_mask(base.rename(columns={'brand':'unrelated'}),donor,42,'aligned')
    except ValueError as exc: assert 'anchors' in str(exc)
    else: raise AssertionError('Unrelated domains cannot be labelled row-aligned')


def test_sibling_split_has_stable_groundtruth_real_overlap_and_auditable_derivation():
    frame=pd.DataFrame({'price':np.arange(60,dtype=float)+100,'brand':['a','b']*30,'size':np.arange(60,dtype=float)})
    frames,groups,audit=split_siblings(frame,'base.csv',6,20,.5,[.4,.9],.3,42)
    assert len(frames)==6 and any('__sale' in s for s in audit['derivations'])
    assert all(abs(x-.5)<.051 for x in audit['adjacent_entity_overlap'])
    for group in groups:
        owners=[c.rsplit('.',1)[0] for c in group]
        assert len(set(owners))==len(owners)
    for table,columns in audit['columns'].items():
        for column,meta in columns.items():
            if '__sale' in meta['semantic']:
                formula=audit['derivations'][meta['semantic']]
                expected=frame[meta['base_column']].iloc[audit['source_rows'][table]].to_numpy()*formula['scale']
                from umsm.uncertainty_model.typed import ColumnProfile
                renderer=ColumnProfile(**formula['rendering'])
                expected=[renderer.number(renderer.render(v)) for v in expected]
                np.testing.assert_allclose(frames[table][column],expected)


def test_C2_C3_publish_oracle_and_keep_shared_donor_dependence():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        frame=pd.DataFrame({'brand':['a','b']*30,'price':np.arange(60,dtype=float)+100})
        groups=[frozenset(('A.csv.brand','B.csv.brand')),frozenset(('A.csv.price','B.csv.price'))]
        publish_task(root/'data/Base/test',{'A.csv':frame,'B.csv':frame.copy()},groups,{'origin':'base'})
        donor=frame.copy(); donor.loc[donor.brand=='a','price']=None
        publish_task(root/'data/Donor/natural',{'D.csv':donor,'E.csv':donor.copy()},[],{'origin':'donor'})
        spec={'bases':[{'dataset':'Base','sets':['test']}],'donors':[{'dataset':'Donor','sets':['natural']}],
            'rows':[10],'seeds':[42],'tables':[3],'sibling_ratios':[.3],'mixed_formats':[False],
            'donor_bands':['low'],'alignments':['random','aligned'],'c2_name':'C2','c3_name':'C3'}
        result=prepare_study(spec,root)
        assert result['prepared'] and {r['construction'] for r in result['prepared']}=={'C2','C3'}
        for dataset in ('C2','C3'):
            for directory in (root/'data'/dataset).iterdir():
                manifest=json.loads((directory/'provenance.json').read_text())
                assert manifest['protocol']['dependency_fingerprints'] and (directory/'oracle').is_dir()
        from umsm.research import attach_preparation
        from umsm.data.loader import load_dataset
        sources=attach_preparation(load_dataset('C2',str(root/'data')))
        assert len(set(source_clusters(sources).values()))==1


def test_outer_union_and_triangle_diagnostics_distinguish_schema_errors():
    source=IntegrationSet('Toy','test',[Table('A',pd.DataFrame({'x':[1.,2.],'y':[10.,20.]})),
        Table('B',pd.DataFrame({'x':[3.,4.],'y':[30.,40.]}))],
        [frozenset(('A.x','B.x')),frozenset(('A.y','B.y'))])
    good=outer_union_metrics(source,Matching(source.groundtruth,{}))
    bad=outer_union_metrics(source,Matching([frozenset(('A.x','B.y')),frozenset(('A.y','B.x'))],{}))
    assert good['attribute_cell_accuracy']==1 and good['mean_numeric_query_relative_error']==0
    assert bad['attribute_cell_accuracy']<1 and bad['mean_numeric_query_relative_error']>0
    cols=[ColumnRef(t,'x') for t in ('A','B','C')]
    result=triangle_counts(cols,{frozenset(('A.x','B.x')),frozenset(('B.x','C.x'))})
    assert result['open_wedges']==1


def test_new_comparison_optimizers_are_explicit_about_table_constraints():
    from umsm import OPTIMIZERS
    from umsm.evaluation import validate_matching
    columns=[ColumnRef(t,c) for t in ('A','B','C') for c in ('x','y')]
    frame=pd.DataFrame([{'col_i':a.id,'col_j':b.id,'table_i':a.table,'table_j':b.table,'a':1.,'w':.8,'mu':.8,'sigma':0.}
        for i,a in enumerate(columns) for b in columns[i+1:] if a.table!=b.table])
    graph=EdgeTable(frame,columns)
    pivot=OPTIMIZERS.build('kwik_cluster').optimize(graph,RunContext())
    validate_matching(pivot,graph,'pivot')
    raw=OPTIMIZERS.build('threshold_components').optimize(graph,RunContext())
    assert raw.info['diagnostic_unconstrained'] and not raw.info['problem_feasible']


def test_resource_worker_and_constraint_downstream_phases_use_existing_interfaces():
    from umsm.research import role_config
    from umsm.study_eval import measured_worker,run_extended
    with fixture() as (root,cfg):
        work,sources=role_config(cfg,'test',root)
        job={'root':str(root),'config':work,'set':sources[0].name,'R':4,'online':True}
        result=measured_worker(job)
        assert result['n_edges']>0 and result['time_s']>0 and result['peak_rss_mb']>=result['baseline_rss_mb']
        assert result['score_cache_hit'] is False and result['edges']
        constraints=run_extended(copy.deepcopy(cfg),'constraints',root)
        full=next(r for r in constraints['rows'] if r['optimizer']=='full')
        lazy=next(r for r in constraints['rows'] if r['optimizer']=='lazy')
        assert abs(full['objective']-lazy['objective'])<1e-8 and full['open_wedges']==lazy['open_wedges']==0
        downstream=run_extended(copy.deepcopy(cfg),'downstream',root)
        assert all(r['attribute_cell_accuracy']==1. for r in downstream['rows'] if r['optimizer']=='oracle_schema')


def test_benchmark_compares_equal_R_and_preserves_edge_references():
    from umsm.study_eval import measured_worker,run_extended
    with fixture() as (root,cfg):
        cfg['research']['sweeps']={'R':[2,4],'tables':[2]}
        actual_run=subprocess.run
        def worker(command,**kwargs):
            # Python's Linux platform probe also uses subprocess.run; only fake our JSON worker.
            if 'input' not in kwargs: return actual_run(command,**kwargs)
            result=measured_worker(json.loads(kwargs['input']))
            return SimpleNamespace(returncode=0,stdout=json.dumps(result),stderr='')
        with patch('umsm.study_eval.subprocess.run',side_effect=worker):
            result=run_extended(copy.deepcopy(cfg),'benchmark',root)
        assert len(result['rows'])==4
        assert all(r['online_batch_max_mu_difference']<1e-10 for r in result['rows'])
        assert all('edges' not in r for r in result['rows'])


def test_resource_worker_entrypoint_is_a_real_fresh_process_without_model_download():
    from umsm.research import role_config
    from umsm.pipeline import build_components
    with fixture() as (root,cfg):
        work,sources=role_config(cfg,'test',root)
        work['matcher']={'name':'valentine','params':{'algorithm':'jaccard_levenshtein','local_selection':'all'},'cache_scores':False}
        artifact=json.loads((root/'calibration/frozen.json').read_text())
        artifact['matcher']=work['matcher']; artifact['matcher_scoring_version']=2
        path=root/'calibration/worker.json'; path.write_text(json.dumps(artifact))
        work['uncertainty_model']['params']['calibration']=str(path)
        work['uncertainty_propagation']['params']['calibration']=str(path)
        job={'root':str(root),'config':work,'set':sources[0].name,'R':2,'online':True}
        completed=subprocess.run([sys.executable,'-m','umsm.study_eval','--worker'],input=json.dumps(job),
            text=True,capture_output=True,cwd=ROOT)
        assert completed.returncode==0,completed.stderr
        result=json.loads(completed.stdout)
        assert result['n_edges']>0 and result['R']==2


def test_suite_reuses_verified_phase_receipts_without_rerunning():
    from umsm.study import run_suite
    with fixture() as (root,cfg):
        artifact=json.loads((root/'calibration/frozen.json').read_text()); artifact['platt']['fit_positive_prior']=.2
        path=root/'calibration/server/suite_test_mean_fold.json'; path.parent.mkdir(); path.write_text(json.dumps(artifact))
        raw=copy.deepcopy(cfg); raw.pop('ablation',None)
        raw['research'].update(name='suite_test',prior_policy='fit_prevalence',lambda_optimizer='pairwise',lambda_grid=[0.,.1])
        raw['study']={'state':'results/suite_state.html','matchers':['mean'],
            'phases':['qcheck','validate-lambda','audit','edges'],'jobs':[{'id':'pilot'}]}
        first=run_suite(raw,root)
        with patch('umsm.research.run_research',side_effect=AssertionError('Completed phase was repeated')):
            second=run_suite(raw,root)
        assert first['result_file']==second['result_file']


def test_bundle_manifest_excludes_credentials_and_venv():
    from umsm.study import bundle_study
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary); (root/'data/Toy').mkdir(parents=True)
        (root/'data/Toy/provenance.json').write_text('{}'); (root/'.env').write_text('TEST_SECRET=local-only')
        (root/'.umsm').mkdir(); (root/'.umsm/private.txt').write_text('not portable')
        result=bundle_study({'study':{'bundle':{'datasets':['Toy']}}},root,'dist/server.zip')
        with zipfile.ZipFile(result['bundle']) as archive:
            assert 'UMSM/data/Toy/provenance.json' in archive.namelist()
            assert not any('/.env' in name or '/.umsm/' in name for name in archive.namelist())
            manifest=json.loads(archive.read('UMSM/DATA_MANIFEST.json'))
            assert len(manifest['files'])==1 and archive.testzip() is None


def test_multiple_siblings_keep_original_only_and_sibling_only_sources():
    frame=pd.DataFrame({'price':np.arange(60,dtype=float)+100,'brand':['a','b']*30})
    _,_,audit=split_siblings(frame,'base.csv',6,20,.5,[.4,.9],.5,42)
    members={s:{t for t,columns in audit['columns'].items() if any(m['semantic']==s for m in columns.values())}
             for s in audit['derivations']}
    for semantic,spec in audit['derivations'].items():
        base=str(spec['base_column'])
        if semantic==base: continue
        assert members[semantic]&members[base]
        assert members[semantic]-members[base]
        assert members[base]-members[semantic]


def test_resource_figures_and_unmatched_suite_pattern_are_explicit():
    from umsm.study_eval import render_study_plots
    from umsm.study import run_suite
    rows=[{'set':'Toy','k':2,'axis':'resources','R':r,'online':mode,
        'time_s':float(r),'peak_rss_minus_baseline_mb':10.,'edge_pairs_per_second':3.}
        for r in (2,4) for mode in (True,False)]
    svg=render_study_plots({'rows':rows})
    assert len(svg)==1 and '<svg' in svg[0] and 'online' in svg[0] and 'batch' in svg[0]
    with tempfile.TemporaryDirectory() as folder:
        try: run_suite({'study':{'jobs':[{'id':'pilot'}]}},Path(folder),'typo-*')
        except ValueError as exc: assert 'Unknown study job pattern' in str(exc)
        else: raise AssertionError('Misspelt job pattern must not look successful')


def test_server_matcher_alias_does_not_inherit_paper_mean_override():
    from umsm.config import load_config
    from umsm.research import research_config
    raw=load_config(str(ROOT/'configs/server.yaml'))
    assert 'matcher' not in raw['modules']
    for label,expected in [('distribution','column_pair'),('quantile','sbert_quantile')]:
        raw['research']['matcher']=label
        cfg=research_config(raw,ROOT)
        assert cfg['matcher']['name']==expected
        if label=='distribution': assert cfg['matcher']['params']['matcher_params']['algorithm']=='distribution_based'
        else: assert cfg['matcher']['params']['device']=='auto'


if __name__=='__main__':
    tests=[f for name,f in sorted(globals().items()) if name.startswith('test_')]
    for f in tests: f(); print('ok',f.__name__)
    print(f'{len(tests)} tests passed')
