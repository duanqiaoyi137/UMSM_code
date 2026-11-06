"""Equivalence, raw-score restart cache and unchanged MC accounting for the exact shortcut."""
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from umsm.context import RunContext
from umsm.matcher.distribution_fast import PreparedColumn,TwoColumnDistribution
from umsm import MATCHERS
from umsm.types import IntegrationSet,Table,UncertainData,UncertainTable,RealizationSpec,ZERO_EMBEDDING


def test_two_column_shortcut_matches_native_random_thresholds_types_and_column_order():
    from valentine import valentine_match
    from valentine.algorithms import DistributionBased
    rng=np.random.default_rng(42)
    cases=[([],['x']),(['nan'],['nan']),(['x'],['x']*4),(['a','b'],['b','c']),
        (['1','1.0','001','2'],[1.,2.,3.]),(['2020-01-01','x'],['2020-01-01','y']),
        (['',None,'β'],['β','']),([0,0,1,2],[0,1,1,2]),(['inf','nan'],['inf','NaN'])]
    for _ in range(70):
        population=['a','b','c','d','1','1.0','2']
        cases.append((rng.choice(population,int(rng.integers(1,18))).tolist(),rng.choice(population,int(rng.integers(1,18))).tolist()))
    for index,(left,right) in enumerate(cases):
        params={'threshold1':[.01,.15,.5,1.][index%4],'threshold2':[1.,.5,.15,.01][index%4],
                'quantiles':[4,16,256][index%3]}
        # Lexical sort intentionally differs from input order on alternating cases.
        names=('Z.csv','A.csv') if index%2 else ('A.csv','Z.csv')
        frames=[pd.DataFrame({'x':left}),pd.DataFrame({'y':right})]
        # The reference wrapper guards an actually empty instance side before calling Valentine.
        if not frames[0].x.dropna().size or not frames[1].y.dropna().size: expected=(0.,False)
        else:
            raw=valentine_match(frames,DistributionBased(**params),df_names=list(names),instance_sample_size=None)
            expected=(float(next(iter(raw.values()))) if raw else 0.,bool(raw))
        columns=[]
        for name,c,values in zip(names,['x','y'],[frames[0].x.dropna().tolist(),frames[1].y.dropna().tolist()]):
            columns.append(PreparedColumn((name,name,c,name+':'+c),tuple(values),frozenset(values),'test'))
        actual=TwoColumnDistribution(params).score(*columns)
        np.testing.assert_allclose(actual[0],expected[0],atol=1e-12,rtol=0.)
        assert actual[1]==expected[1],(index,params,left,right,actual,expected)


def fixture():
    # Input: none; output: small mixed incomplete source with duplicate and distinct MC realizations.
    frames={'Z.csv':pd.DataFrame({'x':['a',None,'b',None],'n':[1.,2.,np.nan,4.]}),
        'A.csv':pd.DataFrame({'y':['a','b',None,'c'],'m':[1.,None,3.,4.]}),
        'B.csv':pd.DataFrame({'empty':[None]*4,'v':['a','b','a','b']})}
    source=IntegrationSet('Toy','Toy/fast',[Table(t,f) for t,f in frames.items()],[])
    tables=[]
    for name,frame in frames.items():
        q={}
        if name=='Z.csv': q={(1,'x'):[('a',1.)],(3,'x'):[('a',.4),('b',.6)],(2,'n'):[(2.,.5),(3.,.5)]}
        if name=='A.csv': q={(2,'y'):[('a',.5),('c',.5)],(1,'m'):[(2.,.5),(3.,.5)]}
        tables.append(UncertainTable(name,frame.astype(object).where(frame.notna(),np.nan),distributions=q))
    return UncertainData(source,tables,'test').configure(RealizationSpec(n_samples=5),42)


def test_adapter_shortcut_preserves_all_scores_draws_probabilities_and_instance_sampling():
    for sample_size in (None,0,2):
        data=fixture(); ctx=RunContext(seed=42); ctx.resources['pair_cache_enabled']=False
        params={'algorithm':'distribution_based','local_selection':'all','instance_sample_size':sample_size}
        reference=MATCHERS.build('column_pair',{'matcher':'valentine','matcher_params':params,'fast_distribution':False})
        fast=MATCHERS.build('column_pair',{'matcher':'valentine','matcher_params':params})
        expected=reference.score(data,ctx); actual=fast.score(data,ctx)
        assert reference.params()==fast.params() and reference.scoring_version==fast.scoring_version
        pd.testing.assert_frame_equal(actual.df,expected.df,check_dtype=False,atol=1e-12,rtol=0.)
        assert fast.expected_sample_counts(data)==reference.expected_sample_counts(data)


def test_partial_raw_cache_reuses_scores_and_keeps_every_mc_observation():
    with tempfile.TemporaryDirectory() as temporary:
        data=fixture(); ctx=RunContext(seed=42,cache_dir=temporary)
        fast=MATCHERS.build('column_pair',{'matcher':'valentine','matcher_params':{
            'algorithm':'distribution_based','local_selection':'all','instance_sample_size':None}})
        first=fast.score(data,ctx); second=fast.score(data,ctx)
        pd.testing.assert_frame_equal(first.df,second.df)
        assert first.meta['unique_pair_evaluations']>0 and second.meta['unique_pair_evaluations']==0
        assert len(second.df)>second.n_pairs
        assert Path(temporary,'distribution_pairs_v1.sqlite').exists()


def test_interrupted_scoring_commits_only_completed_pair_scores_for_resume():
    with tempfile.TemporaryDirectory() as temporary:
        ctx=RunContext(seed=42,cache_dir=temporary); data=fixture()
        matcher=MATCHERS.build('column_pair',{'matcher':'valentine','matcher_params':{
            'algorithm':'distribution_based','local_selection':'all','instance_sample_size':None}})
        original=TwoColumnDistribution.score; calls=[0]
        def interrupted(self,left,right):
            calls[0]+=1
            if calls[0]==6: raise KeyboardInterrupt()
            return original(self,left,right)
        try:
            with patch.object(TwoColumnDistribution,'score',interrupted): matcher.score(data,ctx)
        except KeyboardInterrupt: pass
        else: raise AssertionError('Interruption was not triggered')
        resumed=matcher.score(data,ctx)
        assert resumed.meta['pair_cache_hits']>=5
        assert resumed.meta['unique_pair_evaluations']>0
        reference=MATCHERS.build('column_pair',{'matcher':'valentine','matcher_params':{
            'algorithm':'distribution_based','local_selection':'all','instance_sample_size':None},'fast_distribution':False})
        pd.testing.assert_frame_equal(resumed.df,reference.score(data,ctx).df,check_dtype=False,atol=1e-12,rtol=0.)


if __name__=='__main__':
    tests=[v for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for test in tests: test(); print('ok',test.__name__,flush=True)
    print(len(tests),'tests passed')
