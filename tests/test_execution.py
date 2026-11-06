"""Scheduling/state merge and atomic shared caches, without research experiments."""
from pathlib import Path
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from umsm.study_state import StudyState
from umsm.execution import cpu_context,resource_limits
from umsm.report import read_report


def record_worker(path,root,i):
    manager=StudyState(path,root);manager.select_job('job'+str(i)+'/distribution',['job'+str(i)+'/distribution'])
    manager.record('job'+str(i)+'/distribution/global',{'signature':str(i),'result_file':'unused','sha256':'unused'})


def test_shared_state_preserves_independent_receipts():
    with tempfile.TemporaryDirectory() as root:
        path=str(Path(root)/'state.html')
        with ProcessPoolExecutor(max_workers=2,mp_context=cpu_context()) as pool:
            list(pool.map(record_worker,[path]*8,[root]*8,range(8)))
        report=read_report(path)
        assert len(report['completed'])==len(report['rows'])==8
        assert len(list(Path(root,'state').glob('*.html')))==8


def test_resource_limits_and_matrix_expansion():
    from umsm.study_parallel import expand_jobs
    assert resource_limits()['effective_cpus']>=1
    cfg={'study':{'matchers':['distribution','coma'],'jobs':[{'id':'k{k}','matrix':{'k':[5,8,12]}}]}}
    assert len(expand_jobs(cfg,'k8,k12'))==4
    assert expand_jobs(cfg,'k5')[0][0]['id']=='k5'


def test_parallel_suite_uses_one_receipt_store_and_resumes():
    from test_isolation import fixture
    from umsm.study import run_suite
    with fixture() as (root,cfg):
        cfg['study']={'state':'results/parallel_state.html','parallel_jobs':2,'matchers':['distribution','coma'],
            'phases':['profile'],'jobs':[{'id':'parallel'}]}
        cfg['experiment']['workers']=2
        run_suite(cfg,root)
        before=read_report(root/'results/parallel_state.html')['completed']
        assert len(before)==2
        run_suite(cfg,root)
        assert read_report(root/'results/parallel_state.html')['completed']==before


if __name__=='__main__':
    for name,test in sorted(list(globals().items())):
        if name.startswith('test_') and callable(test): test();print('ok',name,flush=True)
