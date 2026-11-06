"""Locked phase receipts with per-job/matcher shards and a compatible aggregate HTML."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile

from filelock import FileLock
from .report import read_report,write_report


def atomic_report(path,payload):
    # Input: report destination/payload; output: atomically published HTML, never a partially written receipt.
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,suffix='.html'); os.close(fd)
    temporary=Path(name)
    try: write_report(temporary,payload); os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)


class StudyState:
    def __init__(self,path,root):
        # Input: aggregate path/project root; output: process-safe receipt manager and shard directory.
        self.path=Path(path); self.root=Path(root); self.shards=self.path.with_suffix('')
        self.job=''; self.prefixes=[]
        self.path.parent.mkdir(parents=True,exist_ok=True)

    def read(self):
        # Input: current state files; output: compatible state, including migration from a legacy aggregate.
        if self.path.exists(): return read_report(self.path)
        completed={}; rows=[]
        if self.shards.exists():
            for path in sorted(self.shards.glob('*.html')):
                report=read_report(path); completed.update(report.get('completed',{})); rows.extend(report.get('rows',[]))
        return {'completed':completed,'rows':rows}

    def select_job(self,job,prefixes):
        # Input: job ID/matcher and receipt prefixes; output: selected independent job journal.
        self.job=job; self.prefixes=prefixes

    def record(self,token,completed=None,error=None,status='completed'):
        # Input: one completed/failed phase; output: merged aggregate plus a task/matcher shard under a lock.
        with FileLock(str(self.path)+'.lock'):
            state=self.read(); receipts=state.setdefault('completed',{}); rows=state.setdefault('rows',[])
            if completed is not None: receipts[token]=completed
            entry={'job':token,'status':status}
            if error is not None: entry['reason']=str(error)
            if completed is not None: entry['result_file']=completed['result_file']
            state['rows']=[r for r in rows if r.get('job')!=token]+[entry]
            state.update(run_id='server-study',name='Server study progress',kind='study_state')
            state['analysis_tables']={'Progress':state['rows']}
            atomic_report(self.path,state)
            self.shards.mkdir(parents=True,exist_ok=True)
            keep=lambda key:any(key==prefix or key.startswith(prefix+'/') for prefix in self.prefixes)
            shard={'run_id':'server-study','name':self.job,'kind':'study_job_state',
                'completed':{k:v for k,v in receipts.items() if keep(k)},'rows':[r for r in state['rows'] if keep(r.get('job',''))]}
            shard['analysis_tables']={'Progress':shard['rows']}
            name=self.job.replace('/','__')+'_'+hashlib.sha256(self.job.encode()).hexdigest()[:8]+'.html'
            atomic_report(self.shards/name,shard)

    def phase_lock(self,token):
        # Input: globally shared phase token; output: lock also protecting shared calibration across jobs.
        path=self.root/'output/cache/study_locks'/(hashlib.sha256(token.encode()).hexdigest()+'.lock')
        path.parent.mkdir(parents=True,exist_ok=True)
        return FileLock(str(path))


def execute_phase(cfg,phase,root,selection,token,signature,artifact_path,store):
    # Input: phase identity/config; output: verified immutable receipt, computing it at most once concurrently.
    from .research import run_research
    artifact_path=Path(artifact_path)
    with store.phase_lock(token):
        previous=store.read().get('completed',{}).get(token)
        if previous is None and phase=='fit' and artifact_path.exists():
            # Import a verified older separate-matcher state; never adopt an untracked JSON by guessing.
            for path in (Path(root)/'output/results').glob('*state*.html'):
                candidate=read_report(path).get('completed',{}).get(token)
                if candidate and candidate.get('signature')==signature and Path(candidate['result_file']).resolve()==artifact_path.resolve():
                    previous=candidate;break
        if previous and previous['signature']==signature and Path(previous['result_file']).exists():
            path=Path(previous['result_file'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=previous['sha256']: raise ValueError('Completed artifact modified: '+token)
            if phase=='select-calibrator' and previous.get('report_file'):
                report=Path(previous['report_file'])
                if not report.exists() or hashlib.sha256(report.read_bytes()).hexdigest()!=previous.get('report_sha256'):
                    raise ValueError('Calibration selection report changed: '+token)
            print('Resume: reuse',token,flush=True); store.record(token,previous);return previous
        if phase=='fit' and artifact_path.exists():
            raise FileExistsError(f'Untracked/different calibration {artifact_path}; use a new protocol name')
        try:
            use=phase=='validate-offsets' or (phase not in ('profile','fit','qcheck','audit','select-calibrator') and not phase.startswith('validate-'))
            result=run_research(copy.deepcopy(cfg),phase,root,selection if use else None)
            path=artifact_path if phase=='fit' else Path(result['calibration_file'] if phase=='select-calibrator' else result['result_file'])
            receipt={'signature':signature,'result_file':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            if phase=='select-calibrator':
                report=Path(result['result_file']); receipt.update(report_file=str(report),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
            store.record(token,receipt);return receipt
        except Exception as exc:
            store.record(token,error=exc,status='failed');raise
