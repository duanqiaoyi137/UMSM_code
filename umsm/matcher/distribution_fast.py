"""Exact two-column Valentine Distribution execution without per-call files/LPs.

All ranks, histograms and EMDs still use the installed Valentine's primitives.
Two-node cutoff/cluster decisions have the same unique binary optimum as its LP.
Only this strict two-column adapter uses the shortcut; native multi-table matching
keeps Valentine's original algorithm.
"""
from collections import Counter,OrderedDict
from dataclasses import dataclass
import hashlib
import inspect
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import sqlite3
import time
import uuid
import warnings

import numpy as np
import pandas as pd

from ..types import ZERO_EMBEDDING,ScoreSamples,is_missing
from .base import cross_table_pairs


@dataclass
class PreparedColumn:
    name: tuple
    values: tuple
    support: frozenset
    digest: str


def prepare_column(data,col,state,sample_size):
    # Input: one aligned column state; output: the reference adapter's sampled column values and identity.
    from valentine.data_sources.dataframe.dataframe_table import DataframeTable
    table=data.table(col.table)
    frame=table.frame[[col.name]].map(lambda v:np.nan if v is ZERO_EMBEDDING else v)
    positions=[i for i,v in enumerate(table.frame[col.name]) if not is_missing(v) or (i,col.name) in table.distributions]
    for row,value in zip(positions,state.values): frame.iat[row,0]=np.nan if value is ZERO_EMBEDDING else value
    from dateutil.parser import UnknownTimezoneWarning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=UnknownTimezoneWarning,module=r'dateutil\.parser\._parser')
        column=DataframeTable(frame,col.table,sample_size).get_instances_columns()[0]
    values=tuple(column.data)
    # Distribution uses counts/sets, never source row order after instance sampling.
    counts=sorted(Counter((type(v).__name__,repr(v)) for v in values).items())
    name=(col.table,col.table,col.name,f'{col.table}:{col.name}')
    digest=hashlib.sha256(pickle.dumps((name,counts),protocol=4)).hexdigest()
    return PreparedColumn(name,values,frozenset(values),digest)


class TwoColumnDistribution:
    def __init__(self,params):
        # Input: original DistributionBased settings; output: exact two-node execution helper.
        from valentine.algorithms import DistributionBased
        from valentine.algorithms.distribution_based import clustering_utils,column_model,emd_utils,quantile_histogram,discovery,distribution_based
        DistributionBased(**params)  # Keep the original argument validation.
        self.q=int(params.get('quantiles',256)); self.t1=float(params.get('threshold1',.15)); self.t2=float(params.get('threshold2',.15))
        if self.t1==0. or self.t2==0.:
            raise NotImplementedError('Zero-cutoff sentinel handling requires the original adapter')
        self.bloom=bool(params.get('use_bloom_filters',False))
        self.clustering,self.columns,self.emd,self.hist=clustering_utils,column_model,emd_utils,quantile_histogram
        if not hasattr(column_model,'_GLOBAL_RANKS_CACHE') or not hasattr(clustering_utils,'_compute_ranks'):
            raise AttributeError('Installed Valentine does not expose the verified in-memory primitives')
        code={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (clustering_utils,column_model,emd_utils,quantile_histogram,discovery,distribution_based)}
        code['convert_data_type']=hashlib.sha256(inspect.getsource(clustering_utils.convert_data_type).encode()).hexdigest()
        self.identity=hashlib.sha256(json.dumps({'version':1,'kernel':'cdf_v1','params':params,'library_code':code,
            'numpy':np.__version__,'pandas':pd.__version__,'POT':version('POT')},sort_keys=True).encode()).hexdigest()

    def key(self,left,right):
        # Input: two prepared columns; output: a persistent exact raw-score cache key.
        return hashlib.sha256((self.identity+left.digest+right.digest).encode()).hexdigest()

    def score(self,left,right):
        # Input: only two prepared columns; output: raw score and reference is_match flag.
        if not left.values or not right.values: return 0.,False
        if not self.bloom and left.support.isdisjoint(right.support): return 0.,False
        # The original disk store aliases these names. Keep the reference path for such cases.
        safe=self.clustering.make_filename_safe
        if safe(left.name[0])+'_'+safe(left.name[2])==safe(right.name[0])+'_'+safe(right.name[2]):
            raise NotImplementedError('Valentine sanitized column-name collision needs the reference adapter')
        namespace='umsm-dist-'+uuid.uuid4().hex
        self.columns._GLOBAL_RANKS_CACHE[namespace]=self.clustering._compute_ranks(set(left.values)|set(right.values))
        try:
            def column(prepared):
                result=self.columns.CorrelationClusteringColumn(prepared.name[2],prepared.name[3],list(prepared.values),
                    prepared.name[0],prepared.name[1],namespace)
                if result.size:
                    result.quantile_histogram=self.hist.QuantileHistogram(result.long_name,result.ranks,result.size,self.q)
                return result
            a,b=column(left),column(right)
            from .distribution_pair import quantile_emd_cdf,intersection_emd_cdf
            first=quantile_emd_cdf(a,b,self.q)
            if np.isfinite(first) and abs(first-self.t1)<1e-10:
                first=self.emd.quantile_emd(a,b,self.q)
            def accepted(distance,threshold):
                return distance<=self.clustering.compute_cutoff_threshold([{'e':distance,'c':1}],threshold)
            if not accepted(first,self.t1): return 0.,False
            # Phases 2 and final ranking sort the connected component by its complete column tuple.
            x,y=sorted((a,b),key=lambda c:c.long_name)
            intersect=intersection_emd_cdf(x,y,namespace,self.q,self.bloom)
            if np.isfinite(intersect) and abs(intersect-self.t2)<1e-10:
                intersect=self.emd.intersection_emd(x,y,namespace,self.q,self.bloom)
            if not accepted(intersect,self.t2): return 0.,False
            distance=quantile_emd_cdf(x,y,self.q)
            return float(1./(1.+float(round(distance,12)))),True
        finally:
            self.columns.clear_global_ranks_cache(namespace)


class PairScoreCache:
    def __init__(self,ctx):
        # Input: run context; output: bounded memory and optional transactional raw-pair cache.
        self.memory=OrderedDict(); self.db=None; self.pending=[]; self.hits=0
        if ctx.resources.get('pair_cache_enabled',True):
            path=Path(ctx.resolve(ctx.cache_dir))/'distribution_pairs_v1.sqlite'
            path.parent.mkdir(parents=True,exist_ok=True)
            self.db=sqlite3.connect(path,timeout=60.)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL NOT NULL, matched INTEGER NOT NULL) WITHOUT ROWID')
            self.db.commit()

    def lookup(self,keys):
        # Input: keys for one column pair's draws; output: cached scores, preserving no experimental labels.
        found={k:self.memory[k] for k in keys if k in self.memory}
        missing=list(set(keys)-found.keys())
        if missing and self.db:
            query='SELECT key,score,matched FROM scores WHERE key IN ('+','.join('?' for _ in missing)+')'
            found.update({k:(float(s),bool(m)) for k,s,m in self.db.execute(query,missing)})
        self.hits+=len(found)
        return found

    def put(self,key,value):
        # Input: completed raw score; output: in-memory entry plus buffered transactional checkpoint.
        self.memory[key]=value
        if len(self.memory)>100000: self.memory.popitem(last=False)
        if self.db:
            self.pending.append((key,float(value[0]),int(value[1])))
            if len(self.pending)>=4096: self.flush()

    def flush(self):
        # Input: buffered completed scores; output: committed restart-safe cache entries.
        if self.db and self.pending:
            self.db.executemany('INSERT OR REPLACE INTO scores VALUES (?,?,?)',self.pending)
            self.db.commit(); self.pending.clear()

    def close(self):
        # Input: finished/interrupted execution; output: persisted complete entries and released connection.
        self.flush()
        if self.db: self.db.close()


def score_distribution_pairs(data,base,ctx):
    # Input: fixed E*, aligned states and original matcher; output: every original draw/weight with exact fast raw scores.
    if data.realization.mode!='mc' or base.local_selection!='all': return None
    try:
        helper=TwoColumnDistribution(base.algorithm_params)
    except (ImportError,AttributeError,NotImplementedError):
        return None  # Older Valentine releases can still use their original complete adapter.
    started=time.perf_counter(); cache=PairScoreCache(ctx); prepared={}; pairs=list(cross_table_pairs(data))
    columns=data.columns; sizes=[]; owner_i=[]; owner_j=[]; state_i=[]; state_j=[]; probs=[]; values=[]; flags=[]
    from ..execution import effective_workers
    computed=0; last_log=started; pool=None
    workers=effective_workers(ctx.resources.get('workers',1),len(pairs))
    try:
        def prepare(col):
            if col not in prepared:
                prepared[col]=[prepare_column(data,col,s,base.instance_sample_size) for s in data.states(col)]
            return prepared[col]
        if workers>1:
            active={c for pair in pairs for c in pair}
            for col in columns:
                if col in active: prepare(col)
            from .parallel_pairs import distribution_pool
            pool=distribution_pool({p.digest:p for group in prepared.values() for p in group},base.algorithm_params,workers)
        for start in range(0,len(pairs),128):
            batch=pairs[start:start+128]; metadata=[]; found={}; pending={}
            for ci,cj in batch:
                left,right=prepare(ci),prepare(cj); ai,bj,prob=data.pair_indices(ci,cj)
                keys=[helper.key(left[int(a)],right[int(b)]) for a,b in zip(ai,bj)]
                found.update(cache.lookup(keys)); metadata.append((ci,cj,ai,bj,prob,keys))
                for key,a,b in zip(keys,ai,bj):
                    if key in found or key in pending: continue
                    a,b=left[int(a)],right[int(b)]
                    if not helper.bloom and (not a.values or not b.values or a.support.isdisjoint(b.support)):
                        found[key]=(0.,False); computed+=1; cache.put(key,found[key])
                    elif pool: pending[key]=(a,b)
                    else:
                        found[key]=helper.score(a,b);computed+=1;cache.put(key,found[key])
            if pending:
                from .parallel_pairs import distribution_group
                jobs=[(key,a.digest,b.digest) for key,(a,b) in pending.items()]
                chunks=[jobs[i:i+32] for i in range(0,len(jobs),32)]
                for result in pool.map(distribution_group,chunks):
                    for key,score,match in result:
                        found[key]=(score,match);computed+=1;cache.put(key,found[key])
            for ci,cj,ai,bj,prob,keys in metadata:
                result=[found[key] for key in keys]
                sizes.append(len(ai)); owner_i.append(ci); owner_j.append(cj)
                state_i.append(ai); state_j.append(bj); probs.append(prob)
                values.append(np.asarray([s for s,_ in result]));flags.append(np.asarray([m for _,m in result]))
            done=start+len(batch)
            now=time.perf_counter()
            if start==0 or done==len(pairs) or now-last_log>=10.:
                cache.flush(); elapsed=now-started; remaining=elapsed*(len(pairs)-done)/done
                reuse=1-computed/max(sum(sizes),1)
                print(f'Distribution pairs: {done}/{len(pairs)}; unique computed={computed}; reuse={reuse:.1%}; workers={workers}; elapsed={elapsed:.1f}s; ETA~{remaining:.1f}s',flush=True)
                last_log=now
    except NotImplementedError:
        # Filename aliases are rare; use the complete original adapter, never change its answer.
        return None
    finally:
        if pool: pool.shutdown(wait=True,cancel_futures=True)
        cache.close()
    if not pairs: return ScoreSamples.from_records([],columns,{'matcher':'column_pair','evidence_scope':'pair'})
    repeat=lambda vals:np.repeat(np.asarray(vals,dtype=object),sizes)
    frame=pd.DataFrame({'col_i':repeat([c.id for c in owner_i]),'col_j':repeat([c.id for c in owner_j]),
        'table_i':repeat([c.table for c in owner_i]),'table_j':repeat([c.table for c in owner_j]),
        'state_i':np.concatenate(state_i),'state_j':np.concatenate(state_j),'prob':np.concatenate(probs),
        'score':np.concatenate(values),'sample_id':np.concatenate([np.arange(n) for n in sizes]),
        'is_match':np.concatenate(flags)})
    return ScoreSamples(frame,columns,{'matcher':'column_pair','evidence_scope':'pair',**data.score_meta(),
        'execution_backend':'exact_distribution_two_column_v1','unique_pair_evaluations':computed,'pair_cache_hits':cache.hits})
