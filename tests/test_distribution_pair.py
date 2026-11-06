"""CDF transport kernel and deterministic worker equivalence; no API/model download."""
from pathlib import Path
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from test_distribution_fast import fixture
from umsm import MATCHERS,OPTIMIZERS,PROPAGATIONS
from umsm.context import RunContext
from umsm.online import OnlineMoments


def test_parallel_scores_moments_and_decisions_are_identical():
    requested=int(os.environ.get('UMSM_TEST_WORKERS','2'))
    spec={'matcher':'valentine','matcher_params':{'algorithm':'distribution_based','instance_sample_size':None,'local_selection':'all'}}
    matcher=MATCHERS.build('column_pair',spec);results=[];graphs=[]
    for workers in (1,requested):
        data=fixture();ctx=RunContext();ctx.resources.update(workers=workers,pair_cache_enabled=False)
        scores=matcher.score(data,ctx);results.append(scores.df)
        tracker=OnlineMoments(data.columns,matcher.expected_sample_counts(data),data.score_meta())
        for _,block in scores.df.groupby('sample_id',sort=True):
            tracker.update(block,{'method':'platt','slope':4.,'intercept':-2.})
        graphs.append(tracker.finish())
    pd.testing.assert_frame_equal(results[0],results[1],check_exact=True)
    pd.testing.assert_frame_equal(graphs[0],graphs[1],check_exact=True)
    from umsm.types import EdgeTable
    for name,params in [('local_merge',{'weight':'a','min_weight':0.}),('ilp_logit',{})]:
        answers=[]
        for frame in graphs:
            frame=frame.copy();frame['w']=frame.mu;frame['a']=np.log(frame.mu/(1-frame.mu))
            answer=OPTIMIZERS.build(name,params).optimize(EdgeTable(frame,fixture().columns,{'potential':'calibrated_logit'}),RunContext())
            answers.append({frozenset(g) for g in answer.groups})
        assert answers[0]==answers[1]


def test_reference_jaccard_strict_and_native_parallel_are_bitwise_equal():
    # Reference library algorithm with strings/empty sides, in both evidence scopes.
    params={'algorithm':'jaccard_levenshtein','instance_sample_size':None,'local_selection':'all'}
    data=fixture()
    for name,spec in [('column_pair',{'matcher':'valentine','matcher_params':params}),('valentine',params)]:
        frames=[]
        for workers in (1,2):
            ctx=RunContext();ctx.resources.update(workers=workers,pair_cache_enabled=False)
            frames.append(MATCHERS.build(name,spec).score(data,ctx).df)
        pd.testing.assert_frame_equal(*frames,check_exact=True)


def test_reference_coma_strict_and_native_parallel_are_bitwise_equal():
    from nltk.corpus import stopwords
    try: stopwords.words('english')
    except LookupError:
        print('skip COMA: install official NLTK stopwords before its acceptance; offline tests never download it')
        return False
    params={'algorithm':'coma_instance','instance_sample_size':None,'local_selection':'all'}
    data=fixture()
    for name,spec in [('column_pair',{'matcher':'valentine','matcher_params':params}),('valentine',params)]:
        frames=[]
        for workers in (1,2):
            ctx=RunContext();ctx.resources.update(workers=workers,pair_cache_enabled=False)
            frames.append(MATCHERS.build(name,spec).score(data,ctx).df)
        pd.testing.assert_frame_equal(*frames,check_exact=True)


def test_public_kernel_keeps_empty_and_constant_cases():
    from umsm.matcher.distribution_pair import distribution_pair_score
    assert distribution_pair_score([],['a'])==0.
    assert distribution_pair_score(['a']*5,['a']*3)==1.
    assert distribution_pair_score(['a']*5,['b']*3)==0.


if __name__=='__main__':
    passed=[]
    for name,test in sorted(list(globals().items())):
        if name.startswith('test_') and callable(test):
            outcome=test()
            if outcome is not False:passed.append(name);print('ok',name,flush=True)
    if os.environ.get('UMSM_TEST_WORKERS'):
        import json
        from umsm.execution import effective_workers
        output=Path(__file__).resolve().parents[1]/'cache/parallel_acceptance.json';output.parent.mkdir(exist_ok=True)
        complete='test_reference_coma_strict_and_native_parallel_are_bitwise_equal' in passed
        output.write_text(json.dumps({'passed':complete,'tests':passed,'requested_workers':int(os.environ['UMSM_TEST_WORKERS']),
            'effective_distribution_test_workers':effective_workers(int(os.environ['UMSM_TEST_WORKERS']),12),
            'raw_mu_sigma_and_local_ilp_decisions_bitwise_identical':True,'coma_and_jaccard_scopes_identical':complete},indent=2),encoding='utf-8')
