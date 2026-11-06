"""Numerical equivalence tests; set UMSM_REQUIRE_CUDA=1 for the server acceptance check."""
from pathlib import Path
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import BayesianRidge
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from umsm.numerics import knn_impute,quantile_pair_values
from umsm.uncertainty_model.torch_bayes import TorchBayesianRidge
from umsm.uncertainty_model.typed import gower_distances,profiles,candidate_examples
from umsm.context import RunContext
from test_matchers import FakeEncoder,_people_set
from umsm import MATCHERS,UNCERTAINTY_MODELS
from umsm.types import RealizationSpec


def devices():
    # Input: local CUDA availability; output: explicit Torch test devices; module function.
    if os.environ.get('UMSM_REQUIRE_CUDA')=='1': assert torch.cuda.is_available(),'Server test requires CUDA'
    return ['cpu']+(['cuda'] if torch.cuda.is_available() else [])


def test_bayesian_posterior_matches_sklearn_for_tall_wide_and_singular_designs():
    rng=np.random.default_rng(71)
    for n,p in [(35,7),(8,20),(12,12)]:
        x=rng.normal(size=(n,p)); x[:,-1]=x[:,0]; y=rng.normal(size=n); query=rng.normal(size=(5,p))
        reference=BayesianRidge().fit(x,y); mean,std=reference.predict(query,return_std=True)
        for device in devices():
            model=TorchBayesianRidge(device=device).fit(x,y); got_mean,got_std=model.predict(query,return_std=True)
            np.testing.assert_allclose(model.coef_,reference.coef_,rtol=1e-6,atol=1e-8)
            np.testing.assert_allclose(got_mean,mean,rtol=1e-6,atol=1e-8)
            np.testing.assert_allclose(got_std,std,rtol=2e-6,atol=1e-7)
            assert model.device_.startswith(device)


def test_iterative_mice_keeps_schedule_posterior_sampling_and_observed_cells():
    rng=np.random.default_rng(90); x=rng.normal(size=(35,4)); x[::4,0]=np.nan; x[1::5,2]=np.nan
    kwargs={'sample_posterior':True,'max_iter':3,'random_state':42}
    expected=IterativeImputer(**kwargs).fit_transform(x)
    for device in devices():
        model=IterativeImputer(estimator=TorchBayesianRidge(device=device),**kwargs)
        actual=model.fit_transform(x)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
        np.testing.assert_array_equal(actual[np.isfinite(x)],x[np.isfinite(x)])


def test_gower_full_and_query_rows_match_with_mixed_types_and_no_overlap():
    frame=pd.DataFrame({'x':[1.,2.,3.,np.nan], 'group':['a','a','b',None], 'target':[5.,np.nan,7.,None]})
    prof=profiles(frame,options={'overrides':{'x':'numeric','group':'categorical','target':'numeric'}})
    expected=gower_distances(frame,prof,'target')
    for device in devices():
        actual=gower_distances(frame,prof,'target',device,block_size=2)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        np.testing.assert_allclose(gower_distances(frame,prof,'target',device,rows=[1,3]),expected[[1,3]],atol=1e-12)


def test_distance_weighted_knn_matches_sklearn_including_missing_overlap():
    rng=np.random.default_rng(17); x=rng.normal(size=(23,5)); x[rng.random(x.shape)<.2]=np.nan; x[-1]=np.nan
    expected=KNNImputer(n_neighbors=4,weights='distance').fit_transform(x)
    for device in devices(): np.testing.assert_allclose(knn_impute(x,4,device,block_size=3),expected,rtol=1e-9,atol=1e-10)


def test_quantile_and_mean_matcher_gpu_kernels_preserve_scores():
    source=_people_set(); ctx=RunContext(seed=42)
    data=UNCERTAINTY_MODELS.build('observed_only').estimate(source,ctx).configure(RealizationSpec(n_samples=4),42)
    for name,path in [('sbert_quantile','umsm.matcher.sbert_quantile.get_encoder'),('sbert_mean','umsm.matcher.sbert_mean.get_encoder')]:
        with patch(path,return_value=FakeEncoder()):
            expected=MATCHERS.build(name,{'numerical_device':'cpu'}).score(data,ctx).df.score.to_numpy()
            for device in devices():
                actual=MATCHERS.build(name,{'numerical_device':device}).score(data,ctx).df.score.to_numpy()
                np.testing.assert_allclose(actual,expected,atol=2e-7)


def test_quantile_batched_kernel_matches_reference_and_zero_vectors():
    from umsm.matcher.sbert_quantile import _cos
    rng=np.random.default_rng(15); q=rng.normal(size=(9,5,6)); v=rng.random((9,5)); names=rng.normal(size=(3,6))
    q[0]=0.; names[0]=0.; left=[0,1,3,4,6]; right=[2,4,6,2,8]; nl=[0,1,1,2,2]; nr=[1,0,2,1,0]
    weights=np.exp(-np.abs(v[left]-v[right])); weights/=weights.sum(-1,keepdims=True)
    expected=(weights*np.maximum(0.,1.-_cos(q[left],q[right]))).sum(-1)
    for device in devices():
        d,n=quantile_pair_values(q,v,names,left,right,nl,nr,device,chunk_size=2)
        np.testing.assert_allclose(d,expected,atol=1e-12)
        np.testing.assert_allclose(n,_cos(names[nl],names[nr]),atol=1e-12)


def test_q_fit_cache_reuses_only_identical_fit_inputs_parameters_and_seed():
    from umsm.training import _fit_qx
    with tempfile.TemporaryDirectory() as temporary:
        ctx=RunContext(seed=9,project_root=temporary,cache_dir=str(Path(temporary)/'cache'))
        ctx.resources['calibration_fitting']=True; source=_people_set()
        with patch('umsm.training._compute_qx',return_value={'weights':[.4,.6]}) as compute:
            _fit_qx([source],{'calibration':'first.json'},ctx,10,42)
            _fit_qx([source],{'calibration':'second.json'},ctx,10,42)
            assert compute.call_count==1 and ctx.resources['qx_fit_cache']['hit']
            _fit_qx([source],{'rho':.8},ctx,10,42)
            assert compute.call_count==2
            _fit_qx([source],{'rho':.8},ctx,10,43)
            assert compute.call_count==3
        path=next((Path(temporary)/'cache/qx_fit').glob('*.json'))
        import json
        payload=json.loads(path.read_text()); payload['qx']['weights']=[1.,0.]
        path.write_text(json.dumps(payload))
        key=path.stem
        # Whichever request owns this cache must fail rather than reuse modified fit parameters.
        requests=[({'calibration':'third.json'},42),({'rho':.8},42),({'rho':.8},43)]
        failures=0
        for params,seed in requests:
            try: _fit_qx([source],params,ctx,10,seed)
            except ValueError as exc: assert 'checksum' in str(exc); failures+=1
        assert failures==1


if __name__=='__main__':
    tests=[v for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for test in tests: test(); print('ok',test.__name__,flush=True)
    print(f'{len(tests)} tests passed; devices={devices()}')
