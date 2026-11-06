"""Pool natural-data pipeline reports with validation-frozen baseline and paired inference."""
import argparse
from datetime import datetime
import glob
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from umsm.report import read_report,write_report
from umsm.research import role_config,source_clusters,task_signature
from umsm.factorial_analysis import comparable_protocol


def combine(patterns,output,root,baseline_patterns=()):
    # Input: held-out missing-phase reports; output: source-cluster summaries, contrasts, CIs and Holm tests.
    root=Path(root); paths=sorted({Path(p).resolve() for q in patterns for p in glob.glob(q,recursive=True)})
    if not paths: raise FileNotFoundError('No missing-phase reports matched')
    latest={}
    for p in paths:
        if p.parent not in latest or p.stat().st_mtime_ns>latest[p.parent].stat().st_mtime_ns: latest[p.parent]=p
    groups={}
    for path in latest.values():
        report=read_report(path); cfg=report['config']; protocol=report.get('protocol',{})
        if protocol.get('role')!='test' or protocol.get('phase')!='missing': raise ValueError('Expected held-out missing-phase reports')
        baseline=report.get('selection',{}).get('strongest_validation_baseline')
        if not baseline: raise ValueError('Missing frozen strongest validation baseline')
        matcher=cfg['research']['matcher']; g=groups.setdefault(matcher,{'sources':{},'cases':[],'protocol':comparable_protocol(cfg)})
        if g['protocol']!=comparable_protocol(cfg): raise ValueError('Do not pool different matcher, MICE, Q or validation-grid settings')
        _,sources=role_config(cfg,'test',root)
        for source in sources:
            if source.name in g['sources']: raise ValueError(f'Duplicate test task {source.name}')
            g['sources'][source.name]=source
            rows=[r for r in report['rows'] if r['set']==source.name and not r.get('oracle')]
            g['cases'].append((source.name,rows,(baseline['missing'],baseline['optimizer']),baseline['validation_pair_f1'],
                               {r:cfg['research']['roles'][r] for r in ('fit','validation')}))
    additional={}
    baseline_paths=sorted({Path(p).resolve() for q in baseline_patterns for p in glob.glob(q,recursive=True)})
    baseline_latest={}
    for p in baseline_paths:
        if p.parent not in baseline_latest or p.stat().st_mtime_ns>baseline_latest[p.parent].stat().st_mtime_ns: baseline_latest[p.parent]=p
    if baseline_patterns and not baseline_latest: raise FileNotFoundError('No ALITE baseline reports matched')
    for path in baseline_latest.values():
        report=read_report(path); cfg=report['config']; protocol=report.get('protocol',{})
        if protocol.get('role')!='test' or protocol.get('phase')!='missing' or cfg['research']['matcher']!='alite_native':
            raise ValueError('Additional reports must be held-out ALITE evidence/cluster results')
        chosen=report.get('selection',{}).get('strongest_validation_baseline')
        if not chosen: raise ValueError('ALITE report lacks a frozen validation selection')
        _,sources=role_config(cfg,'test',root)
        for source in sources:
            if source.name in additional: raise ValueError('Duplicate ALITE baseline task')
            matching=[r for r in report['rows'] if r['set']==source.name and r['missing']=='observed_only' and r['optimizer']=='alite']
            if len(matching)!=1: raise ValueError('Missing ALITE pipeline case')
            additional[source.name]={'row':{**matching[0],'missing':'ALITE_native'},'validation':chosen['validation_pair_f1'],
                'roles':{r:cfg['research']['roles'][r] for r in ('fit','validation')},'signature':task_signature(source)}
    summaries=[]; contrasts=[]; rng=np.random.default_rng(2026)
    for matcher,g in sorted(groups.items()):
        clusters=source_clusters(list(g['sources'].values())); metrics=['pair_precision','pair_recall','pair_f1','bcubed_f1','ari','TP','FP','FN']
        per=[]; differences={}
        for task,rows,best,best_validation,roles in g['cases']:
            if baseline_patterns:
                if task not in additional: raise ValueError(f'ALITE baseline missing for {task}')
                extra=additional[task]
                if extra['signature']!=task_signature(g['sources'][task]) or extra['roles']!=roles:
                    raise ValueError(f'ALITE baseline uses different task or fit/validation roles: {task}')
                rows=rows+[extra['row']]
                if extra['validation']>best_validation+1e-12: best=('ALITE_native','alite')
            lookup={(r['missing'],r['optimizer']):r for r in rows}; ours=lookup.get(('Q_risk','global'))
            if ours is None or best not in lookup: raise ValueError(f'Incomplete full-pipeline case {task}')
            for (missing,opt),row in lookup.items():
                per.append({'task':task,'cluster':clusters[task],'missing':missing,'optimizer':opt,
                            **{m:row.get(m) for m in metrics},'time_s':row.get('compute_edges_s',0)+row.get('optimizer_time_s',0),
                            'rss_peak_mb_in_process':row.get('rss_peak_mb_in_process'),
                            'all_optimal':row.get('all_optimal')})
                if (missing,opt)!=('Q_risk','global'):
                    differences.setdefault(missing+' / '+opt,[]).append((clusters[task],ours['pair_f1']-row['pair_f1']))
            differences.setdefault('strongest validation-selected pipeline',[]).append((clusters[task],ours['pair_f1']-lookup[best]['pair_f1']))
        frame=pd.DataFrame(per)
        for (missing,opt),block in frame.groupby(['missing','optimizer'],sort=True):
            clustered=block.groupby('cluster',sort=True)[metrics+['time_s','rss_peak_mb_in_process']].mean(numeric_only=True)
            values={m:(float(clustered[m].mean()) if clustered[m].notna().any() else None) for m in clustered}
            summaries.append({'matcher':matcher,'missing':missing,'optimizer':opt,'n_tasks':block.task.nunique(),
                              'n_source_clusters':len(clustered),**values,
                              'nonoptimal_runs':int((block.all_optimal==False).sum())})
        family=[]
        for name,pairs in differences.items():
            d=pd.DataFrame(pairs,columns=['cluster','delta']).groupby('cluster',sort=True).delta.mean().to_numpy(float); n=len(d)
            boot=rng.choice(d,size=(2000,n),replace=True).mean(axis=1) if n>=2 else np.array([])
            lo,hi=np.quantile(boot,[.025,.975]) if len(boot) else (None,None)
            signs=np.asarray(list(product((-1.,1.),repeat=n))) if n<=16 else rng.choice((-1.,1.),size=(10000,n))
            extreme=np.sum(np.abs((signs*d).mean(axis=1))>=abs(d.mean())-1e-12)
            p=float(extreme/len(signs)) if n<=16 else float((extreme+1)/(len(signs)+1))
            family.append({'matcher':matcher,'baseline':name,'difference':float(d.mean()),'ci_low':lo,'ci_high':hi,
                           'n_source_clusters':n,'p_unadjusted':p,'eligible_n':n>=10})
        primary=next(i for i,r in enumerate(family) if r['baseline']=='strongest validation-selected pipeline')
        family[primary]['test_family']='RQ1 primary, one validation-selected baseline'
        family[primary]['p_holm']=family[primary]['p_unadjusted']
        order=sorted((i for i in range(len(family)) if i!=primary),key=lambda i:family[i]['p_unadjusted']); previous=0.
        for rank,i in enumerate(order):
            previous=max(previous,min(1.,(len(order)-rank)*family[i]['p_unadjusted'])); family[i]['p_holm']=previous
            family[i]['test_family']='prespecified secondary pipeline contrasts'
        for row in family:
            row['supported']=bool(row['eligible_n'] and row['ci_low'] is not None and row['ci_low']>0 and row['p_holm']<.05)
        contrasts.extend(family)
    output=Path(output); output=output if output.is_absolute() else root/output
    write_report(output,{'run_id':datetime.now().strftime('%Y%m%d-%H%M%S-%f'),'name':'Natural-data full pipeline comparison',
        'kind':'pipeline_pooled','rows':summaries,'analysis_tables':{'Pipeline metrics (equal source-cluster weights)':summaries,
        'Q risk + ILP versus paired pipelines':contrasts},'protocol':{'reports':[str(p) for p in latest.values()],
        'note':'Strongest baseline is selected on each fold validation set. RQ1 primary (one contrast) and all secondary pipeline contrasts are separate preregistered families. CIs are unadjusted paired cluster bootstrap; Holm adjusts two-sided sign-flip p-values within each family and matcher. RSS is process peak during each case, including already-loaded resources; cached runs are not cold-runtime comparisons. Clusters need not represent independent model training.',
        'additional_baseline_reports':[str(p) for p in baseline_latest.values()]}})
    print('Pooled pipeline report:',output,flush=True); return str(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--reports',nargs='+',required=True)
    p.add_argument('--baseline-reports',nargs='*',default=[])
    p.add_argument('--output',default='output/results/full_pipeline_main.html'); args=p.parse_args()
    combine(args.reports,args.output,Path(__file__).resolve().parent,args.baseline_reports)
