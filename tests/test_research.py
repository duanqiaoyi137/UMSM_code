"""Public-data adapters, masking mechanisms, paired protocols and research diagnostics."""
from dataclasses import replace
import copy
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from prepare_alaska import read_archive
from prepare_gdc import read_source
from prepare_variants import prepare as prepare_variants
from test_matchers import FakeEncoder
from umsm.config import load_config
from umsm.context import RunContext
from umsm.data.masking import mask_frame
from umsm.data.preparation import partition_annotations, publish_task
from umsm.global_optimizer.local_merge import LocalMerge
from umsm.global_optimizer.logit import LogitILP
from umsm.report import read_report
from umsm.research import ensure_disjoint, research_config, run_research, select_parameter, subset_cases, task_signature
from umsm.research_analysis import local_conflicts, partition_objective, probability_metrics, uncertainty_diagnostics
from umsm.types import ColumnRef, EdgeTable, IntegrationSet, Table


def toy(name='test', tables=3):
    frames = [Table(f't{i}.csv',pd.DataFrame({'city':['a','b','a','b',None,'a','b','a'],
                                           'kind':['x','y','x','y','x','x','y','x']})) for i in range(tables)]
    return IntegrationSet('Toy',name,frames,[frozenset(f't{i}.csv.city' for i in range(tables)),
                                           frozenset(f't{i}.csv.kind' for i in range(tables))])


def test_annotations_exclude_multitarget_and_resolve_alias_without_merging_targets():
    frames={'a.csv':pd.DataFrame({'one':['x',None],'alias':['x','x'],'ambiguous':['y','y']}),
            'b.csv':pd.DataFrame({'city':['x','x'],'other':['y','y']})}
    mappings=[('a.csv','one','city'),('a.csv','alias','city'),('a.csv','ambiguous','city'),
              ('a.csv','ambiguous','other'),('b.csv','city','city'),('b.csv','other','other'),('a.csv','absent','city')]
    result,gold,audit=partition_annotations(frames,mappings,drop_absent=True)
    assert list(result['a.csv'])==['alias']
    assert frozenset(('a.csv.alias','b.csv.city')) in gold
    assert frozenset(('a.csv.ambiguous','b.csv.other')) not in gold
    assert {a['reason'] for a in audit}=={'source_column_has_multiple_targets','same_table_alias','annotation_absent_from_source'}


def test_alaska_stream_reader_uses_schema_gold_and_preserves_absence():
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'camera.tar.gz'
        with tarfile.open(path,'w:gz') as archive:
            entries={'camera/camera_specs/site/1.json':json.dumps({'raw':'x'}),
                     'camera/camera_specs/site/2.json':json.dumps({'other':'y'}),
                     'camera/camera_ground_truths/camera_schema_matching_gt.csv':'source_attribute_id,target_attribute_name\nsite//raw,city\n'}
            for name,value in entries.items():
                raw=value.encode(); member=tarfile.TarInfo(name); member.size=len(raw)
                archive.addfile(member,io.BytesIO(raw))
        frames,mappings=read_archive(path,'camera')
    assert frames['site.csv'].raw.isna().sum()==1
    assert mappings==[('site.csv','raw','city')]


def test_gdc_official_column_cleanup_and_local_input():
    frame=pd.DataFrame({'Age in Months at Time of Tissue Procurement \t':[10,20]})
    with patch('prepare_gdc.fetch_file',return_value=Path('cached.xlsx')),patch('pandas.read_excel',return_value=frame):
        result,_=read_source('McDermott.csv',{'Dataset URL':'https://example.org/a.xlsx','Sheet Name':'clinical'},Path('.'))
    assert result.columns.tolist()==['Age in Months at Time of Tissue Procurement']


def test_masking_preserves_original_nulls_and_is_reproducible():
    frame=pd.DataFrame({'driver':np.arange(3000),'value':np.arange(3000),'other':['a']*3000})
    frame.loc[:9,'value']=np.nan
    original=frame.copy()
    for mode in ('mcar','mar','mnar'):
        result,mask,audit=mask_frame(frame,mode,.4,42)
        again,mask2,_=mask_frame(frame,mode,.4,42)
        pd.testing.assert_frame_equal(result,again)
        assert np.array_equal(mask,mask2) and not (mask&frame.isna().to_numpy()).any()
        assert result.loc[:9,'value'].isna().all()
        if mode=='mar':
            assert not mask[:,0].any() and audit['mar_driver']=='driver'
        if mode in ('mar','mnar'):
            assert mask[2000:,1].mean()>mask[10:1000,1].mean()+.15
    pd.testing.assert_frame_equal(frame,original)


def test_transplant_keeps_sampled_natural_row_column_pattern():
    frame=pd.DataFrame(np.ones((20,3)))
    donor=pd.DataFrame([[1,None,3],[None,2,None],[1,2,3]])
    result,mask,audit=mask_frame(frame,'transplant',.4,7,donor)
    expected=donor.isna().to_numpy()[np.ix_(audit['donor_rows'],audit['donor_columns'])]
    assert np.array_equal(mask,expected) and np.array_equal(result.isna(),expected)


def test_two_table_objectives_agree_and_local_conflicts_are_pre_bfs():
    columns=[ColumnRef('A','x'),ColumnRef('A','y'),ColumnRef('B','x'),ColumnRef('B','y')]
    rows=[{'col_i':a.id,'col_j':b.id,'table_i':a.table,'table_j':b.table,'w':p,'a':np.log(p/(1-p))}
          for (a,b),p in zip([(columns[0],columns[2]),(columns[0],columns[3]),(columns[1],columns[2]),(columns[1],columns[3])],[.9,.8,.7,.1])]
    edges=EdgeTable(pd.DataFrame(rows).assign(mu=lambda d:d.w,sigma=0.),columns,{'potential':'calibrated_logit'})
    local=LocalMerge(weight='a',min_weight=0.).optimize(edges,RunContext())
    global_=LogitILP().optimize(edges,RunContext())
    assert np.isclose(partition_objective(local,edges)['partition_objective'],partition_objective(global_,edges)['partition_objective'])
    extra=ColumnRef('C','x')
    for index,row in enumerate(rows):
        row['w']=.95 if index==0 else .1
        row['a']=float(np.log(row['w']/(1-row['w'])))
    rows += [{'col_i':columns[2].id,'col_j':extra.id,'table_i':'B','table_j':'C','w':.9,'a':2.},
             {'col_i':columns[1].id,'col_j':extra.id,'table_i':'A','table_j':'C','w':.9,'a':2.}]
    conflict=local_conflicts(EdgeTable(pd.DataFrame(rows).assign(mu=lambda d:d.w,sigma=0.),columns+[extra]))
    assert conflict['inconsistent_triangles_before_bfs']>0
    assert conflict['same_table_pairs_in_local_components']>0


def test_subsets_are_unique_paired_and_keep_unmatched_groundtruth_columns():
    source=toy(tables=5)
    cases=list(subset_cases(source,[2,3,'all',7],20,12))
    assert len(cases)==10+10+1
    assert [(s.name,t) for s,t in cases]==[(s.name,t) for s,t in subset_cases(source,[2,3,'all',7],20,12)]
    for subset,meta in cases:
        allowed={c.id for c in subset.columns}
        assert len(subset.tables)==meta['k'] and all(g<=allowed for g in subset.groundtruth)


def test_uncertainty_diagnostics_handles_constant_sigma_single_class_and_perfect_scores():
    frame=pd.DataFrame({'label':[0,1,0,1],'mu':[.1,.9,.2,.8],'sigma':[0.,0.,0.,0.]})
    metrics=probability_metrics(frame)
    assert metrics['auc']==1. and metrics['n_positive']==2
    result=uncertainty_diagnostics(frame)
    assert result['regression']['status']=='not_identifiable' and result['aurc']==0.
    assert probability_metrics(frame.iloc[[0,2]])['auc'] is None


def test_publish_masks_and_oracle_then_verify_frozen_inputs():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        source=toy()
        publish_task(root/'data/Toy/test',{t.name:t.df for t in source.tables},source.groundtruth,{'test':True})
        args=SimpleNamespace(dataset='Toy',sets=['test'],name='Masked',modes=['mcar'],rates=[.4],rows=[5],seeds=[42],
                             dense_max_missing=1.,complete_rows=False,min_rows=2,donor_dataset=None,donor_sets=None)
        manifest=prepare_variants(args,root)[0]
        case=next((root/'data/Masked').iterdir())
        for table in source.tables:
            a=pd.read_csv(case/'original'/table.name); b=pd.read_csv(case/'oracle'/table.name)
            assert (a.isna() | b.notna()).to_numpy().all()
        assert any(k.startswith('oracle/') for k in manifest['files'])
        assert prepare_variants(args,root)[0]==manifest


def test_research_matcher_configs_do_not_share_incompatible_params():
    for dataset in ('alaska','sotab','gdc','madi'):
        for label in ('mean','name','value','coma','jaccard','distribution','magneto'):
            cfg=load_config(str(ROOT/'configs/research'/f'{dataset}.yaml'))
            cfg['research']['matcher']=label
            from umsm.pipeline import build_components
            build_components(research_config(cfg,ROOT))


def test_research_fit_run_select_and_single_report_offline():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary)
        for index,role in enumerate(('fit','validation','test')):
            source=toy(role)
            # Different source filenames also ensure the populations do not share a source table.
            for table in source.tables:
                table.name=role+'_'+table.name
                table.df.loc[0,'city']=f'{role}_{index}'
            source.groundtruth=[frozenset(t.name+'.'+c for t in source.tables) for c in ('city','kind')]
            publish_task(root/'data/Toy'/role,{t.name:t.df for t in source.tables},source.groundtruth,{'role':role})
        cfg=load_config(str(ROOT/'configs/research/alaska.yaml'))
        cfg['dataset']['name']='Toy'
        cfg['research'].update(name='offline',fold='fold',calibration='calibration/test.json',
            roles={r:[r] for r in ('fit','validation','test')},max_cells=20,n_samples=4,
            optimizers=['local','global'],plots=False,subsets={'k':[2,3],'repeats':2},missing=['observed_only','Q_mean','Q_risk'])
        cfg=research_config(cfg,root)
        with patch('umsm.matcher.sbert_mean.get_encoder',return_value=FakeEncoder()):
            run_research(cfg,'fit',root)
            validation=run_research(cfg,'validate-prior',root)
            selected=select_parameter(validation['result_file'],cfg,root)
            assert selected['axis']=='prior'
            result=run_research(cfg,'missing',root,validation['result_file'])
            subsets=run_research(cfg,'subsets',root)
        report=read_report(result['result_file'])
        assert len(report['rows'])==6 and report['analysis_tables']['Missingness strata']
        assert len(subsets['rows'])==6
        assert all(r.get('two_table_objective_agrees',True) for r in subsets['rows'])
        assert len(list((root/'results').rglob('*.html')))==3
        assert not list((root/'results').rglob('*.csv'))
        # Exercise the actual scientific plots and the no-API cross-report reader.
        from analyze_research import analyze
        analyzed=analyze([result['result_file'],subsets['result_file']],root/'comparison.html')
        assert analyzed['analysis_tables']['Paired ILP-local effects']
        assert '<svg' in (root/'comparison.html').read_text(encoding='utf-8')
        # Run saved pre-mask oracle and compute recovery only for positive denominators.
        args=SimpleNamespace(dataset='Toy',sets=['test'],name='Controlled',modes=['mcar'],rates=[.4],rows=[5],seeds=[42],
            dense_max_missing=1.,complete_rows=False,min_rows=2,donor_dataset=None,donor_sets=None)
        prepare_variants(args,root)
        controlled=copy.deepcopy(cfg)
        controlled['research']['roles']['test']={'name':'Controlled','sets':['*']}
        with patch('umsm.matcher.sbert_mean.get_encoder',return_value=FakeEncoder()):
            recovered=run_research(controlled,'missing',root)
        assert any(r['missing']=='oracle' for r in recovered['rows'])
        for row in recovered['rows']:
            if row['recovery_denominator'] is None or row['recovery_denominator']<=1e-12:
                assert row['recovery_ratio'] is None
        changed=copy.deepcopy(cfg)
        changed['candidates']={'mode':'name_top_l'}
        try:
            select_parameter(validation['result_file'],changed,root)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed candidate universe must invalidate selection')


def test_research_source_overlap_rejected_even_when_task_names_differ():
    a=toy('fit'); b=replace(a,name='test')
    assert task_signature(a)!=task_signature(replace(a,groundtruth=[]))
    with tempfile.TemporaryDirectory() as temporary:
        a.meta['table_dir']=str(Path(temporary)/'original')
        b.meta=a.meta
        try:
            ensure_disjoint([a],[b])
        except ValueError:
            pass
        else:
            raise AssertionError('Task renaming cannot remove source overlap')


def test_constant_sigma_coverage_is_tie_averaged_and_sites_grouping_is_explicit():
    frame=pd.DataFrame({'mu':[.9,.9,.9,.9],'sigma':[0.,0.,0.,0.],'label':[1,1,0,0]})
    diagnostics=uncertainty_diagnostics(frame)
    assert all(np.isclose(row['risk'],.5) for row in diagnostics['risk_coverage'])
    from prepare_sotab import group_records
    records=[{'source':str(i),'host':str(i//2),'topic':'Hotel','labels':{'a':'city','b':'country'}} for i in range(12)]
    groups=group_records(records,5,2,42,'distinct_sites')
    assert groups and all(len(g)==5 and len({r['host'] for r in g})==5 for g in groups)


if __name__=='__main__':
    tests=[f for n,f in sorted(globals().items()) if n.startswith('test_')]
    for test in tests:
        test()
        print('ok',test.__name__)
    print(len(tests),'tests passed')
