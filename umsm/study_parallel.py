"""Bounded process scheduling for independent experiment jobs with CPU budget division."""
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
import fnmatch
import hashlib
import itertools
from pathlib import Path
import contextlib
import traceback
import os
import signal

from .config import deep_merge
from .execution import resource_limits,cpu_context,cpu_worker_init


def expand_jobs(raw,part=None):
    # Input: suite templates/patterns; output: concrete independent (definition, matcher) jobs in stable order.
    plan=raw['study'];wanted=part.split(',') if part else None;out=[];names=[]
    def replace(value,point):
        if isinstance(value,dict): return {k:replace(v,point) for k,v in value.items()}
        if isinstance(value,list): return [replace(v,point) for v in value]
        if isinstance(value,str):
            for key,replacement in point.items(): value=value.replace('{'+key+'}',str(replacement))
        return value
    for template in plan.get('jobs',[]):
        matrix=template.get('matrix',{})
        for values in itertools.product(*matrix.values()):
            definition=replace(template,dict(zip(matrix,values)));definition.pop('matrix',None);names.append(definition['id'])
            if wanted and not any(fnmatch.fnmatchcase(definition['id'],p) for p in wanted): continue
            for matcher in definition.get('matchers',plan.get('matchers',['distribution','quantile'])):
                out.append((definition,matcher))
    if wanted and any(not any(fnmatch.fnmatchcase(name,p) for name in names) for p in wanted):
        raise ValueError('Unknown study job pattern')
    return out


def suite_worker(raw,definition,matcher,root,workers,log):
    # Input: CPU/GPU-independent suite job description; output: receipt/error with its own log.
    from .study import run_suite
    from .execution import install_q_cache_lock
    install_q_cache_lock()
    cfg=copy.deepcopy(raw); cfg['study']['parallel_jobs']=1
    definition=copy.deepcopy(definition);definition['matchers']=[matcher]
    cfg['study']['jobs']=[definition];cfg['experiment']['workers']=workers
    log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        try: return {'job':definition['id']+'/'+matcher,'result':run_suite(cfg,root),'log':str(log)}
        except Exception as exc:
            traceback.print_exc();return {'job':definition['id']+'/'+matcher,'error':str(exc),'log':str(log)}


def run_parallel_suite(raw,root,part=None):
    # Input: suite and requested --jobs/workers; output: quota-bounded independent jobs and shared locked receipts.
    jobs=expand_jobs(raw,part); limits=resource_limits(); count=min(int(raw['study'].get('parallel_jobs',1)),limits['effective_cpus'],max(1,len(jobs)))
    workers=min(int(raw['experiment'].get('workers',1)),max(1,limits['effective_cpus']//count))
    print(f'Scheduler: jobs={count}, scoring workers/job={workers}, effective CPUs={limits["effective_cpus"]}; logs in output/results/job_logs',flush=True)
    result=[]
    pool=ProcessPoolExecutor(max_workers=count,mp_context=cpu_context(),initializer=cpu_worker_init)
    pending=[]
    try:
        for definition,matcher in jobs:
            tag=hashlib.sha256((definition['id']+'/'+matcher+str(raw['study'].get('state'))).encode()).hexdigest()[:12]
            log=Path(root)/'output/results/job_logs'/(definition['id'].replace('/','_')+'_'+matcher+'_'+tag+'.log')
            pending.append(pool.submit(suite_worker,raw,definition,matcher,str(root),workers,str(log)))
        for future in as_completed(pending):
            item=future.result();result.append(item)
            print(('FAILED ' if item.get('error') else 'Completed ')+item['job']+'; log='+item['log'],flush=True)
    except KeyboardInterrupt:
        # Stop this queue's descendants only; never kill unrelated experiments or the resource tracker.
        import psutil
        for future in pending:future.cancel()
        for process in psutil.Process(os.getpid()).children(recursive=True):
            try:
                if any('spawn_main' in word for word in process.cmdline()):os.kill(process.pid,signal.SIGINT)
            except (ProcessLookupError,psutil.NoSuchProcess):pass
        print('Interrupted: cancelling queued jobs and waiting for atomic cache writes.',flush=True)
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
    failures=[r for r in result if r.get('error')]
    if failures: raise RuntimeError('Independent jobs failed: '+', '.join(r['job'] for r in failures)+'; inspect job logs and state')
    return {'result_file':str(Path(root)/raw['study'].get('state','output/results/server_study_state.html')),'jobs':result}
