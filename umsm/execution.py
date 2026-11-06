"""Execution-only resource limits, safe CPU pools and Q-cache locking.

No scientific parameters or Q source files are changed by this module.
"""
import functools
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

from filelock import FileLock


def resource_limits():
    # Input: OS/cgroup quotas; output: effective CPU and memory limits rather than host totals.
    cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    for quota_path,period_path in (('/sys/fs/cgroup/cpu.max',None),
            ('/sys/fs/cgroup/cpu/cpu.cfs_quota_us','/sys/fs/cgroup/cpu/cpu.cfs_period_us')):
        try:
            fields=Path(quota_path).read_text().split()
            quota,period=(fields[0],fields[1]) if period_path is None else (fields[0],Path(period_path).read_text().strip())
            if quota!='max' and int(quota)>0: cpus=min(cpus,max(1,int(int(quota)/int(period))))
        except (OSError,ValueError,IndexError): pass
    memory=None
    for path in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            value=Path(path).read_text().strip()
            if value!='max' and int(value)<2**60: memory=int(value); break
        except (OSError,ValueError): pass
    return {'effective_cpus':max(1,cpus),'memory_limit_bytes':memory,'host_cpus':os.cpu_count()}


def effective_workers(requested,items=None):
    # Input: requested CPU processes/task count; output: positive quota-bounded worker count.
    if int(requested)<1: raise ValueError('workers must be positive')
    n=min(int(requested),resource_limits()['effective_cpus'])
    return min(n,max(1,int(items))) if items is not None else n


def cpu_worker_init():
    # Input: spawned CPU worker; output: one BLAS/OpenMP thread per worker.
    os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'
    from threadpoolctl import threadpool_limits
    global _THREAD_LIMIT
    _THREAD_LIMIT=threadpool_limits(limits=1)


def cpu_context():
    # Input: execution platform; output: spawn context safe after CUDA/threaded libraries were initialized.
    return mp.get_context('spawn')


def configure_context(ctx,cfg):
    # Input: run context/configuration; output: execution metadata shared with matcher/solver modules.
    ctx.resources['workers']=effective_workers(cfg.get('experiment',{}).get('workers',1))
    ctx.resources['pair_cache_enabled']=cfg.get('matcher',{}).get('cache_scores',True)
    ctx.resources['fast_distribution']=cfg.get('experiment',{}).get('fast_distribution',True)


def install_q_cache_lock():
    # Input: training module at runtime; output: locked Q-cache access without modifying its hash-protected source.
    from . import training
    if getattr(training._fit_qx,'_execution_lock',False): return
    original=training._fit_qx
    @functools.wraps(original)
    def locked(sets,params,ctx,max_cells,mask_seed):
        from .calibration import set_fingerprint
        clean=dict(params); clean.pop('calibration',None)
        key=hashlib.sha256(json.dumps({'sets':[set_fingerprint(s) for s in sets],
            'params':clean,'seed':ctx.seed,'max_cells':max_cells,'mask_seed':mask_seed},sort_keys=True,default=str).encode()).hexdigest()
        path=Path(ctx.resolve(ctx.cache_dir))/'locks'/('qx_'+key+'.lock'); path.parent.mkdir(parents=True,exist_ok=True)
        with FileLock(str(path)): return original(sets,params,ctx,max_cells,mask_seed)
    locked._execution_lock=True
    training._fit_qx=locked
