"""Check float64 CUDA kernels on observed real tables; no LLM calls or scientific F1 experiment."""
import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge

from umsm.config import load_config
from umsm.pipeline import load_sets
from umsm.report import write_report
from umsm.uncertainty_model.typed import profiles,predictor_matrix,gower_distances,CONTINUOUS
from umsm.uncertainty_model.torch_bayes import TorchBayesianRidge


def check(root,dataset,set_name,rows,output):
    # Input: local observed dataset/task/row cap; output: real-input CPU/CUDA numerical errors and timings.
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for this check')
    cfg=load_config(); cfg['dataset'].update(name=dataset,sets=[set_name],root='data',variant='original')
    source=load_sets(cfg,str(root))[0]
    chosen=None
    for table in sorted(source.tables,key=lambda t:(-len(t.df.columns),t.name)):
        frame=table.df.iloc[:rows].reset_index(drop=True); prof=profiles(frame,table.name)
        for target,p in prof.items():
            if p.kind not in CONTINUOUS: continue
            y=np.array([p.number(v) for v in frame[target]])
            x=predictor_matrix(frame,prof,target); good=np.isfinite(y)
            if good.sum()>=5 and x.shape[1]: chosen=(table,frame,prof,target,p,x,y,good); break
        if chosen: break
    if not chosen: raise ValueError('No usable numeric/date regression in this observed task')
    table,frame,prof,target,p,x,y,good=chosen
    x=SimpleImputer(strategy='median').fit_transform(x); yy=(y[good]-np.median(y[good]))/p.scale
    # Warm CUDA startup without including it in one backend's kernel time.
    torch.linalg.svd(torch.ones((8,4),dtype=torch.float64,device='cuda'),full_matrices=False); torch.cuda.synchronize()
    measurements=[]
    def measured(label,function):
        torch.cuda.synchronize(); start=time.perf_counter(); result=function(); torch.cuda.synchronize()
        measurements.append({'operation':label,'seconds_including_transfers':time.perf_counter()-start}); return result
    cpu=measured('Gower CPU',lambda:gower_distances(frame,prof,target))
    gpu=measured('Gower CUDA',lambda:gower_distances(frame,prof,target,'cuda'))
    np.testing.assert_allclose(gpu,cpu,rtol=1e-9,atol=1e-10)
    ref=measured('BayesianRidge sklearn CPU',lambda:BayesianRidge().fit(x[good],yy))
    actual=measured('BayesianRidge Torch CUDA',lambda:TorchBayesianRidge(device='cuda').fit(x[good],yy))
    mean,std=ref.predict(x,return_std=True); got_mean,got_std=actual.predict(x,return_std=True)
    np.testing.assert_allclose(got_mean,mean,rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(got_std,std,rtol=2e-5,atol=2e-6)
    finite=np.isfinite(cpu)&np.isfinite(gpu)
    errors={'gower_max_abs':float(np.max(np.abs(cpu[finite]-gpu[finite]))) if finite.any() else 0.,
        'posterior_mean_max_abs':float(np.max(np.abs(mean-got_mean))),
        'posterior_std_max_abs':float(np.max(np.abs(std-got_std)))}
    destination=Path(output); destination=destination if destination.is_absolute() else root/destination
    write_report(destination,{'name':'Real-input GPU implementation check','kind':'gpu_check','run_id':'gpu-check','rows':measurements,
        'analysis_tables':{'Numerical differences':[errors],'Measured operations':measurements},
        'protocol':{'task':source.name,'table':table.name,'target':target,'rows':len(frame),'features':x.shape[1],
            'gpu':torch.cuda.get_device_name(0),'dtype':'float64','note':'One real-table implementation check; not an accuracy experiment or a guaranteed full-pipeline speedup.'}})
    for row in measurements: print(row,flush=True)
    print('Numerical errors:',errors); print('Report:',destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='Alaska_Partition'); parser.add_argument('--set',default='monitor_pool')
    parser.add_argument('--rows',type=int,default=300); parser.add_argument('--output',default='output/results/gpu_backend_check.html')
    args=parser.parse_args()
    if args.rows<5: parser.error('--rows must be >=5')
    check(Path(__file__).resolve().parent,args.dataset,args.set,args.rows,args.output)
