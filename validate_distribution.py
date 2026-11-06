"""Fit-only acceptance and timing of the exact strict Distribution execution path.

This command never fits Q, selects hyperparameters or uses test labels. A cached
Q fit is required to include actual completed column states in the acceptance.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd

from umsm.config import load_config,deep_merge
from umsm.context import RunContext
from umsm.execution import cpu_context,cpu_worker_init,effective_workers,resource_limits
from umsm.matcher.distribution_fast import PreparedColumn,TwoColumnDistribution,prepare_column


_HELPER=None


def initialize(params):
    # Input: frozen reference settings; output: worker-local exact helper and bounded BLAS threads.
    cpu_worker_init()
    global _HELPER
    _HELPER=TwoColumnDistribution(params)


def reference(pair):
    # Input: two observed/realized fit columns; output: original Valentine score, decision and runtime.
    from valentine import valentine_match
    from valentine.algorithms import DistributionBased
    a,b=pair;t=time.perf_counter()
    if not a.values or not b.values: raw={}
    else:
        raw=valentine_match([pd.DataFrame({a.name[2]:pd.Series(a.values,dtype=object)}),
                            pd.DataFrame({b.name[2]:pd.Series(b.values,dtype=object)})],
            DistributionBased(quantiles=_HELPER.q,threshold1=_HELPER.t1,threshold2=_HELPER.t2),
            df_names=[a.name[0],b.name[0]],instance_sample_size=None)
    return (float(next(iter(raw.values()))) if raw else 0.,bool(raw),time.perf_counter()-t)


def compare(block):
    # Input: ordered test column pairs; output: both scores, decision equality and measured timings.
    out=[]
    for a,b in block:
        expected,matched,seconds=reference((a,b));t=time.perf_counter();actual,flag=_HELPER.score(a,b)
        out.append((expected,actual,matched,flag,seconds,time.perf_counter()-t))
    return out


def fit_columns(root,cfg,job):
    # Input: suite/job and an existing verified Q cache; output: observed columns plus actual Q realizations from fit sources only.
    from umsm import UNCERTAINTY_MODELS
    from umsm.calibration import set_fingerprint
    from umsm.research import research_config,role_config
    from umsm.study_parallel import expand_jobs
    from umsm.types import UncertainData,UncertainTable,RealizationSpec
    definition,_=expand_jobs(cfg,job)[0]
    source=deep_merge(cfg,definition.get('overrides',{}));source['research']['matcher']='distribution'
    config=research_config(source,root);_,sets=role_config(config,'fit',root)
    params=copy.deepcopy(config['uncertainty_model']['params']);params.pop('calibration',None)
    files=['training.py','numerics.py','uncertainty_model/mixture.py','uncertainty_model/typed.py','uncertainty_model/torch_bayes.py']
    code={name:hashlib.sha256((root/'umsm'/name).read_bytes()).hexdigest() for name in files}
    inputs={'v':1,'parameters':params,'sets':[set_fingerprint(s) for s in sets],
        'max_cells':config['research'].get('max_cells',200),'seed':config['experiment']['seed'],
        'mask_seed':config['research'].get('mask_seed',42),'code':code}
    key=hashlib.sha256(json.dumps(inputs,sort_keys=True,default=str).encode()).hexdigest()
    cache=root/config['paths']['cache_dir']/'qx_fit'/(key+'.json')
    if not cache.exists(): raise FileNotFoundError(f'Fit-only Q cache required (no automatic refit): {cache}')
    payload=json.loads(cache.read_text(encoding='utf-8'))
    assert payload['key']==key and payload['sha256']==hashlib.sha256(json.dumps(payload['qx'],sort_keys=True,default=str).encode()).hexdigest()
    ctx=RunContext(seed=config['experiment']['seed'],project_root=str(root),cache_dir=str(root/'output/cache'))
    ctx.resources['calibration_fitting']=True;observed=[]
    for task in sets:
        data=UncertainData(task,[UncertainTable(t.name,t.df.copy()) for t in task.tables],'validation').configure(RealizationSpec(n_samples=32),ctx.seed)
        observed.extend(prepare_column(data,c,data.states(c)[0],None) for c in data.columns)
    # Q estimation is table-local. Use two whole original fitting tables, never truncated rows/context.
    def useful_missing(table):
        return sum(int(table.df[c].isna().sum()) for c in table.df
            if 1<table.df[c].nunique()<=max(2,len(table.df[c].dropna())*.5))
    candidates=[(task,t) for task in sets for t in task.tables if useful_missing(t)>0]
    chosen=sorted(candidates,key=lambda pair:(-useful_missing(pair[1])/max(1,len(pair[1].df.columns)),len(pair[1].df.columns)))[:2];completed=[]
    n_draw_columns=0
    artifact={'version':'qx-global-v2.1','split_role':'train','fit_sets':[],'qx':payload['qx']}
    model=UNCERTAINTY_MODELS.build('calibrated_mixture',{**params,'calibration':artifact})
    for task,table in chosen:
        print('Acceptance: generating unchanged Q states on whole fit table',table.name,flush=True)
        data=model.estimate(replace(task,tables=[table]),ctx).configure(RealizationSpec(n_samples=32),ctx.seed)
        n_draw_columns+=sum(data.state_count(c)>1 for c in data.columns)
        completed.extend(prepare_column(data,c,state,None) for c in data.columns for state in data.states(c))
    assert n_draw_columns>0,'Acceptance must contain genuinely incomplete, multiple-realization Q columns'
    return observed,completed,{'qx_cache_key':key,'protected_source_sha256':code,
        'fit_sets':[s.name for s in sets],'completed_fit_tables':[t.name for _,t in chosen],
        'Q_incomplete_columns':n_draw_columns,'Q_unique_realized_columns':len({c.digest for c in completed})}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c','--config',default='configs/full_pipeline_gpu.yaml')
    parser.add_argument('--study-part',default='alaska-camera');parser.add_argument('--pairs',type=int,default=5000)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--benchmark',type=int,default=1000)
    parser.add_argument('--output',default='output/cache/distribution_acceptance.json')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    observed,completed,provenance=fit_columns(root,load_config(args.config),args.study_part)
    assert completed,'Real Q completions are required for acceptance'
    params={};initialize(params);rng=random.Random(20261103);pairs=[];population=observed+completed
    for i in range(args.pairs):
        a=rng.choice(observed if i%2==0 else completed)
        options=[b for b in population if b.name[0]!=a.name[0]]
        # Include many potentially nonzero cases, not just trivial disjoint zeros.
        overlap=[b for b in options if not a.support.isdisjoint(b.support)]
        b=rng.choice(overlap if overlap and i%3 else options);pairs.append((a,b))
    workers=effective_workers(args.workers);blocks=[pairs[i:i+25] for i in range(0,len(pairs),25)];rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=cpu_context(),initializer=initialize,initargs=(params,)) as pool:
        for i,block in enumerate(pool.map(compare,blocks)):
            rows.extend(block)
            if (i+1)%10==0 or len(rows)==len(pairs): print('Reference acceptance:',len(rows),'/',len(pairs),flush=True)
    errors=[abs(a-b) for a,b,*_ in rows];zero_equal=all((a==0.)==(b==0.) for a,b,*_ in rows)
    decisions=all(a==b for _,_,a,b,*_ in rows);passed=max(errors,default=0.)<=1e-9 and zero_equal and decisions
    bench=rows[:args.benchmark]; old=sum(r[4] for r in bench);new=sum(r[5] for r in bench)
    report={**provenance,'passed':passed,'pairs':len(rows),'observed_columns':len(observed),'Q_realized_columns':len(completed),
        'max_absolute_error':max(errors,default=0.),'zero_nonzero_identical':zero_equal,'is_match_identical':decisions,
        'bitwise_score_fraction':sum(a==b for a,b,*_ in rows)/len(rows),
        'nonzero_reference_pairs':sum(r[0]!=0. for r in rows),'kernel_identity':_HELPER.identity,
        'benchmark':{'pairs':len(bench),'reference_sum_seconds':old,'fast_sum_seconds':new,'speedup':old/new},
        'environment':{'python':platform.python_version(),**resource_limits()},'workers':workers,
        'claim':'Numerical equivalence <=1e-9 and identical zero/nonzero/selection decisions on fit-only acceptance; not a universal bitwise proof'}
    path=root/args.output;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    from umsm.report import write_report
    write_report(path.with_suffix('.html'),{'name':'Distribution implementation acceptance','kind':'implementation_validation',
        'run_id':'distribution-acceptance','rows':[],'environment':report,'analysis_tables':{'Acceptance':[report]}})
    print(json.dumps(report,indent=2),flush=True)
    if not passed: raise SystemExit('FAILED: disable fast_distribution, use original matcher; do not reuse this version for changed scores')


if __name__=='__main__': main()
