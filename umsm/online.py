"""Algorithm 1: one shared realization per round, calibrated Welford statistics.

Ordinary inference retains O(|E*|) scores/statistics, not R score tables. Optional
raw round caches live on disk; a legacy pickle import is explicitly marked because
loading that old format temporarily requires its full table in memory.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .calibration import calibrate_scores
from .score_cache import candidate_cache_key, score_cache_key
from .candidates import plan_candidates
from .types import ScoreSamples


class OnlineMoments:
    """Vectorized Welford updates; only count, mean, M2 and optional feature means."""

    def __init__(self, columns, expected, meta):
        # 输入：列、每边预期次数、采样元信息；输出：空的在线统计器；继承：OnlineMoments(object)。
        self.columns, self.expected, self.meta = columns, expected, meta
        owner = {c.id:c.table for c in columns}
        self.pairs = [tuple(sorted(pair)) for pair in expected]
        self.position = {frozenset(pair):i for i,pair in enumerate(self.pairs)}
        self.frame = pd.DataFrame([{'col_i':a,'col_j':b,'table_i':owner[a],'table_j':owner[b]} for a,b in self.pairs],
                                  columns=['col_i','col_j','table_i','table_j'])
        self.count = np.zeros(len(expected),dtype=np.int64)
        self.mean = np.zeros(len(expected)); self.m2 = np.zeros(len(expected))
        self.features = {}

    def update(self, frame, calibration):
        # 输入：一轮原始分数及固定校准参数；输出：None，原地更新 μ 和 S；继承：OnlineMoments(object)。
        keys = [frozenset((a,b)) for a,b in zip(frame.col_i,frame.col_j)]
        if len(keys)!=len(set(keys)) or not set(keys)<=self.position.keys():
            raise ValueError('A scoring round must contain each allowed pair at most once')
        idx = np.array([self.position[k] for k in keys],dtype=int)
        if not len(idx):
            return
        expected = np.array([self.expected[k] for k in keys])
        if not np.allclose(frame.prob,1/expected,rtol=0,atol=1e-12):
            raise ValueError('Every Monte Carlo edge must have equal 1/R weights')
        values = calibrate_scores(frame.score.to_numpy(float),calibration)
        self.count[idx] += 1
        delta = values-self.mean[idx]
        self.mean[idx] += delta/self.count[idx]
        self.m2[idx] += delta*(values-self.mean[idx])
        exclude = set(ScoreSamples.REQUIRED)|{'sample_id','calibrated_probability'}
        for name in frame.columns:
            if name in exclude or not (pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name])):
                continue
            feature = self.features.setdefault(name,np.zeros(len(self.pairs)))
            feature[idx] += (frame[name].to_numpy(float)-feature[idx])/self.count[idx]

    def finish(self):
        # 输入：已结束的统计器；输出：每边 μ、R−1 标准差及完整性检查后的统计表；继承：OnlineMoments(object)。
        target = np.array(list(self.expected.values()),dtype=int)
        if not np.array_equal(self.count,target):
            raise ValueError('Algorithm 1 requires every closure pair and all its aligned draws')
        variance = np.maximum(self.m2/np.maximum(self.count-1,1),0.)
        incomplete = set(self.meta.get('incomplete_columns',[]))
        frame = self.frame.assign(mu=self.mean,sigma=np.sqrt(variance),variance=variance,
            n_states=self.count,n_samples=self.count,estimator=np.where(self.count==1,'deterministic','mc'))
        frame['edge_type'] = [['complete_complete','complete_incomplete','incomplete_incomplete'][int(a in incomplete)+int(b in incomplete)] for a,b in self.pairs]
        for name,values in self.features.items():
            frame[name] = values
        return frame


class EvidenceSamples:
    """Small inference receipt; full samples are available only when explicitly retained."""

    def __init__(self, columns, meta, count, n_pairs, raw=None):
        # 输入：列、采样元信息、计数及可选调试原始表；输出：轻量评分凭据；继承：EvidenceSamples(object)。
        self.columns,self.meta = columns,meta
        self.count,self.n_pairs,self.raw = count,n_pairs,raw

    def __len__(self):
        # 输入：评分凭据；输出：实际评分样本数；继承：EvidenceSamples(object)。
        return self.count

    @property
    def df(self):
        # 输入：评分凭据；输出：显式保留的原始分数表，否则拒绝隐式重算；继承：EvidenceSamples(object)。
        if self.raw is None:
            raise ValueError('Online inference does not retain R score tables; enable experiment.save_score_samples for explicit raw diagnostics')
        return self.raw


def atomic_pickle(payload, path):
    # 输入：缓存对象与路径；输出：None，原子发布且并发写不共享临时文件；继承：无（模块函数）。
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as stream:
        temporary = Path(stream.name)
    try:
        pd.to_pickle(payload,temporary)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def estimate_online(data, model, matcher, propagation, artifact, cfg, ctx):
    # 输入：Q、matcher、冻结校准与配置；输出：轻量样本凭据、在线概率边、缓存标志；继承：无（模块函数）。
    if matcher.registry_name=='sbert_quantile' and matcher.normalization!='fixed':
        raise ValueError('Algorithm 1 needs realization-local scores: set sbert_quantile normalization=fixed; legacy_minmax requires explicit experiment.online=false')
    if getattr(matcher,'llm_max_variants',None):
        raise ValueError('Algorithm 1 evaluates all R realizations; set llm_max_variants=null')
    params = artifact['platt']
    enabled = cfg['matcher'].get('cache_scores',True)
    capture = cfg['experiment'].get('save_score_samples',False)
    cache_cfg={**cfg,'_project_root':ctx.project_root}
    legacy_key=score_cache_key(cache_cfg,data.source,model,matcher,data.realization,data)
    key=score_cache_key(cache_cfg,data.source,model,matcher,data.realization,data,canonical=True)
    calibration_key = hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:16]
    store = Path(ctx.cache_dir)/'evidence'/(key+'_'+calibration_key+'.pkl')
    legacy = Path(ctx.cache_dir)/'scores'/(legacy_key+'.pkl')
    if enabled and store.exists():
        cached = pd.read_pickle(store)
        if not capture or cached.get('raw') is not None:
            meta = {**cached['meta'],'evidence_cache_hit':True,'legacy_cache_import':False,
                    'matcher_sampling_scope':matcher.sampling_scope}
            receipt = EvidenceSamples(data.columns,meta,cached['count'],len(cached['moments']),cached.get('raw'))
            return receipt,propagation.from_moments(cached['moments'],data.columns,meta,params),True
    plan = lambda:plan_candidates(data.source,ctx,cfg.get('candidates',{}))
    pairs,info = ctx.shared(('candidates',candidate_cache_key(cfg,data.source,ctx)),plan) if enabled else plan()
    data.candidate_pairs = pairs
    expected = matcher.expected_sample_counts(data)
    retain = capture or max(expected.values(),default=0)<=1
    meta = {**data.score_meta(),'matcher':matcher.registry_name,'candidates':info,'online_statistics':True,
            'raw_samples_retained':capture,'legacy_cache_import':False,'matcher_sampling_scope':matcher.sampling_scope}
    accumulator = OnlineMoments(data.columns,expected,meta)
    saved = []
    hit = False
    if enabled and legacy.exists():
        # Compatibility import only: no matcher/API call, no change to old cache files.
        cached = pd.read_pickle(legacy)
        raw = ScoreSamples(cached['df'],data.columns,cached['meta'])
        if 'sample_id' not in raw.df:
            raw.df['sample_id'] = raw.df.groupby(['col_i','col_j'],sort=False).cumcount()
        meta.update(candidates=raw.meta.get('candidates',info),legacy_cache_import=True)
        if retain:
            saved.append(raw.df)
        for _,frame in raw.df.groupby('sample_id',sort=True):
            accumulator.update(frame,params)
        del raw,cached
        hit = True
    else:
        raw_dir = Path(ctx.cache_dir)/'score_rounds'/key
        persist_raw = enabled and cfg['paths'].get('raw_score_cache',False)
        counts = {c.id:data.state_count(c) for c in data.columns}
        ctx.reset_rng()
        calls = 0
        for r,states in data.iter_realizations():
            active = {pair for pair,n in expected.items() if r<n}
            if not active:
                continue
            cached_round = raw_dir/f'{r:06d}.pkl'
            if enabled and cached_round.exists():
                frame = pd.read_pickle(cached_round)
            else:
                snapshot = copy.copy(data)
                snapshot._states = {c:[state] for c,state in states.items()}
                snapshot.candidate_pairs = active
                with ctx.profiler.stage('matcher',set=data.source.name):
                    frame = matcher.score(snapshot,ctx).df.copy()
                frame['state_i'] = frame.col_i.map(lambda c:r%counts[c])
                frame['state_j'] = frame.col_j.map(lambda c:r%counts[c])
                frame['sample_id'] = r
                frame['prob'] = [1/expected[frozenset((a,b))] for a,b in zip(frame.col_i,frame.col_j)]
                if persist_raw:
                    atomic_pickle(frame,cached_round)
                calls += 1
            actual = {frozenset((a,b)) for a,b in zip(frame.col_i,frame.col_j)}
            if actual!=active:
                raise ValueError('A matcher omitted or added pairs in an Algorithm 1 round')
            with ctx.profiler.stage('uncertainty_propagation',set=data.source.name):
                accumulator.update(frame,params)
            if retain:
                saved.append(frame)
        hit = enabled and calls==0
    moments = accumulator.finish()
    raw = pd.concat(saved,ignore_index=True) if saved else (pd.DataFrame(columns=ScoreSamples.REQUIRED) if not expected else None)
    count = int(sum(expected.values()))
    if enabled:
        atomic_pickle({'moments':moments,'meta':meta,'raw':raw,'count':count},store)
    receipt = EvidenceSamples(data.columns,meta,count,len(expected),raw)
    return receipt,propagation.from_moments(moments,data.columns,meta,params),hit
