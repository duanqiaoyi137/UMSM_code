"""CPU-only workers exchange prepared column arrays, never UncertainData/CUDA objects."""
import warnings
from collections import OrderedDict
import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from ..execution import cpu_context,cpu_worker_init,effective_workers

_PREPARED=None
_DISTRIBUTION=None
_PARAMS=None
_REALIZATIONS=None


def init_distribution(columns,params):
    # Input: immutable prepared column arrays/settings; output: one CPU worker's read-only store.
    cpu_worker_init()
    from .distribution_fast import TwoColumnDistribution
    global _PREPARED,_DISTRIBUTION
    _PREPARED=columns; _DISTRIBUTION=TwoColumnDistribution(params)


def distribution_group(jobs):
    # Input: ordered content-key/column-key triples; output: ordered exact raw scores.
    return [(key,*_DISTRIBUTION.score(_PREPARED[left],_PREPARED[right])) for key,left,right in jobs]


def distribution_pool(columns,params,workers):
    # Input: prepared arrays and worker budget; output: a spawn-based CPU pool safe after CUDA initialization.
    return ProcessPoolExecutor(max_workers=effective_workers(workers),mp_context=cpu_context(),
        initializer=init_distribution,initargs=(columns,params))


def frame_column(data,col,state):
    # Input: one aligned column state; output: its reference single-column frame without any unrelated columns.
    from ..types import ZERO_EMBEDDING,is_missing
    table=data.table(col.table)
    frame=table.frame[[col.name]].map(lambda v:np.nan if v is ZERO_EMBEDDING else v)
    rows=[i for i,v in enumerate(table.frame[col.name]) if not is_missing(v) or (i,col.name) in table.distributions]
    for row,value in zip(rows,state.values): frame.iat[row,0]=np.nan if value is ZERO_EMBEDDING else value
    return frame


def init_valentine(columns,params):
    # Input: prepared single-column frames/settings; output: CPU-only worker-local matcher.
    cpu_worker_init()
    from .valentine_matchers import ValentineMatcher
    global _PREPARED,_PARAMS
    _PREPARED=[(table,col,pd.DataFrame({col:pd.Series(values,index=index)})) for table,col,values,index in columns]
    _PARAMS=ValentineMatcher(**params)


def valentine_group(jobs):
    # Input: ordered two-column array references; output: ordered raw scores and local-selection indicators.
    from valentine import valentine_match
    from dateutil.parser import UnknownTimezoneWarning
    from .valentine_matchers import _key
    out=[]
    for i,j in jobs:
        ta,ca,fa=_PREPARED[i]; tb,cb,fb=_PREPARED[j]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=UnknownTimezoneWarning,module=r'dateutil\.parser\._parser')
            if _PARAMS.algorithm=='distribution_based' and _PARAMS._empty_instance_side(fa,fb): raw={}
            else: raw=valentine_match([fa,fb],_PARAMS._build(),df_names=[ta,tb],instance_sample_size=_PARAMS.instance_sample_size)
        selected=raw.one_to_one_hungarian(_PARAMS.local_threshold) if raw and _PARAMS.local_selection=='one_to_one' else raw
        key=tuple(sorted((ta+'.'+ca,tb+'.'+cb)))
        scores={}
        for pair,s in raw.items():
            name=_key(pair); scores[name]=max(scores.get(name,0.),float(s))
        out.append((scores.get(key,0.),key in {_key(pair) for pair in selected}))
    return out


def score_valentine_columns(data,base,ctx):
    # Input: aligned strict column evidence; output: deterministic serial/parallel reference-library scores.
    from .base import cross_table_pairs
    from ..types import ScoreSamples
    from ..execution import effective_workers
    pairs=list(cross_table_pairs(data)); workers=effective_workers(ctx.resources.get('workers',1),len(pairs))
    if workers<=1: return None
    columns=[]; indices={};digests=[]
    for col in data.columns:
        indices[col]=[]
        for state in data.states(col):
            frame=frame_column(data,col,state)
            indices[col].append(len(columns)); columns.append((col.table,col.name,frame[col.name].array,frame.index))
            digests.append(hashlib.sha256(pickle.dumps((col.table,col.name,frame),protocol=4)).hexdigest())
    cache=ctx.shared(('valentine_column_content_scores',repr(base.params())),OrderedDict);computed=0;total=0
    owner_i=[];owner_j=[];sizes=[];ai_all=[];bj_all=[];prob_all=[];scores=[];matches=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=cpu_context(),initializer=init_valentine,
                             initargs=(columns,base.params())) as pool:
        for start in range(0,len(pairs),128):
            batch=pairs[start:start+128]; jobs=[]; metadata=[]
            for ci,cj in batch:
                ai,bj,prob=data.pair_indices(ci,cj); metadata.append((ci,cj,ai,bj,prob))
                jobs.extend((indices[ci][int(a)],indices[cj][int(b)]) for a,b in zip(ai,bj))
            keys=[(digests[a],digests[b]) for a,b in jobs];unique={}
            found={key:cache[key] for key in keys if key in cache}
            for key,job in zip(keys,jobs):
                if key not in found:unique.setdefault(key,job)
            todo=list(unique.values());chunks=[todo[i:i+32] for i in range(0,len(todo),32)]
            raw=[x for block in pool.map(valentine_group,chunks) for x in block]
            for key,value in zip(unique,raw):
                found[key]=value;cache[key]=value
                if len(cache)>100000:cache.popitem(last=False)
            values=[found[key] for key in keys];position=0;computed+=len(unique);total+=len(jobs)
            for ci,cj,ai,bj,prob in metadata:
                owner_i.append(ci);owner_j.append(cj);sizes.append(len(ai));ai_all.append(ai);bj_all.append(bj);prob_all.append(prob)
                local=values[position:position+len(ai)];position+=len(ai)
                scores.extend(s for s,_ in local);matches.extend(m for _,m in local)
            if start==0 or start+len(batch)==len(pairs) or (start//128)%20==0:
                print(f'{base.algorithm} pairs: {start+len(batch)}/{len(pairs)}; workers={workers}; reuse={1-computed/max(1,total):.1%}',flush=True)
    if not pairs: return ScoreSamples.from_records([],data.columns,{**data.score_meta(),'evidence_scope':'pair'})
    repeat=lambda x:np.repeat(np.asarray(x,dtype=object),sizes)
    df=pd.DataFrame({'col_i':repeat([c.id for c in owner_i]),'col_j':repeat([c.id for c in owner_j]),
        'table_i':repeat([c.table for c in owner_i]),'table_j':repeat([c.table for c in owner_j]),
        'state_i':np.concatenate(ai_all),'state_j':np.concatenate(bj_all),'prob':np.concatenate(prob_all),
        'score':scores,'sample_id':np.concatenate([np.arange(n) for n in sizes]),'is_match':matches})
    return ScoreSamples(df,data.columns,{'matcher':'column_pair','evidence_scope':'pair',**data.score_meta(),
        'execution_backend':'valentine_parallel_columns_v1','workers':workers,'unique_pair_evaluations':computed})


def init_native(realizations,params):
    # Input: table realization arrays; output: reconstructed worker-local frames and serial reference matcher.
    cpu_worker_init()
    from .valentine_matchers import ValentineMatcher
    global _REALIZATIONS,_PARAMS
    _PARAMS=ValentineMatcher(**params)
    _REALIZATIONS={name:[(pd.DataFrame({c:pd.Series(a,index=index) for c,a in arrays},index=index),p)
        for arrays,index,p in frames] for name,frames in realizations.items()}


def native_table_pair(job):
    # Input: one table pair plus allowed column/count metadata; output: its original ordered sample records.
    from valentine import valentine_match
    from dateutil.parser import UnknownTimezoneWarning
    from .valentine_matchers import _key
    from .base import sample_record
    from ..types import ColumnRef
    ta,tb,active,exact=job; ra,rb=_REALIZATIONS[ta],_REALIZATIONS[tb]
    runs=[(a,b,pa*pb) for a,(_,pa) in enumerate(ra) for b,(_,pb) in enumerate(rb)] if exact else [
        (r%len(ra),r%len(rb),1./max(len(ra),len(rb))) for r in range(max(len(ra),len(rb)))]
    records=[]; empty=0
    for sample,(a,b,prob) in enumerate(runs):
        fa,fb=ra[a][0],rb[b][0]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=UnknownTimezoneWarning,module=r'dateutil\.parser\._parser')
            if _PARAMS.algorithm=='distribution_based' and _PARAMS._empty_instance_side(fa,fb): raw={};empty+=1
            else: raw=valentine_match([fa,fb],_PARAMS._build(),df_names=[ta,tb],instance_sample_size=_PARAMS.instance_sample_size)
        selected=raw.one_to_one_hungarian(_PARAMS.local_threshold) if raw and _PARAMS.local_selection=='one_to_one' else raw
        scores={}
        for pair,s in raw.items():
            name=_key(pair); scores[name]=max(scores.get(name,0.),float(s))
        local={_key(p) for p in selected}
        for ca,cb,na,nb in active:
            ci,cj=ColumnRef(ta,ca),ColumnRef(tb,cb); key=tuple(sorted((ci.id,cj.id)))
            records.append(sample_record(ci,cj,a%na,b%nb,prob,scores.get(key,0.),is_match=key in local,sample_id=sample))
    return records,empty


def score_native_tables(data,base,ctx):
    # Input: whole-table reference matcher/evidence; output: ordered independent table-pair results or serial fallback.
    from itertools import combinations
    from ..types import ScoreSamples
    jobs=[]
    for a,b in combinations(data.tables,2):
        active=[(ci.name,cj.name,len(data.states(ci)),len(data.states(cj)))
            for ci in a.columns for cj in b.columns if data.allows(ci,cj)]
        if active: jobs.append((a.name,b.name,active,data.realization.mode=='exact'))
    workers=effective_workers(ctx.resources.get('workers',1),len(jobs))
    if workers<=1: return None
    realizations={t.name:[([(c,f[c].array) for c in f],f.index,p) for f,p in data.table_realizations(t.name)] for t in data.tables}
    with ProcessPoolExecutor(max_workers=workers,mp_context=cpu_context(),initializer=init_native,initargs=(realizations,base.params())) as pool:
        result=list(pool.map(native_table_pair,jobs))
    return ScoreSamples.from_records([r for records,_ in result for r in records],data.columns,
        {'matcher':'valentine.'+base.algorithm,'empty_instance_runs':sum(e for _,e in result),**data.score_meta(),
         'execution_backend':'valentine_parallel_table_pairs_v1','workers':workers})
