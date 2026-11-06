"""Cross-platform study preparation/scheduling and portable handoff, using run.py modules."""
import copy
from datetime import datetime
import hashlib
import json
import itertools
import fnmatch
from pathlib import Path
import zipfile

from .calibration import load_artifact
from .config import deep_merge
from .report import read_report,write_report


def audit_job(cfg, root):
    # Input: declared validation/test roles and gates; output: profile/claim-scope audit without tuning on test outcomes; module function.
    from .research import role_config,source_clusters,report_folder,task_signature
    from .uncertainty_model.typed import profiles
    root=Path(root); _,tests=role_config(cfg,'test',root); groups=source_clusters(tests)
    rules=cfg['research'].get('gates',{}); minimum=int(rules.get('min_sources',10)); rows=[]
    for source in tests:
        types={p.kind for table in source.tables for p in profiles(table.df,table.name,cfg['uncertainty_model']['params'].get('type_options',{})).values()}
        rows.append({'set':source.name,'source_cluster':groups[source.name],'k':len(source.tables),
            'ilp_subgroup':len(source.tables)>=rules.get('min_ilp_tables',6),'column_types':sorted(types)})
    candidates=([Path(cfg['research']['qcheck_report'])] if cfg['research'].get('qcheck_report') else
                list((report_folder(cfg,root)/'qcheck').glob('*.html'))); diagnostics=[]
    if candidates:
        report=read_report(max(candidates,key=lambda p:p.stat().st_mtime))
        for row in report.get('analysis_tables',{}).get('Q diagnostics by column type',[]):
            kind=row['column_type']; entry={**row,'eligible':False}
            if kind in ('numeric','date'):
                numeric=rules.get('numeric',{})
                entry['eligible']=bool(numeric and row.get('n_evaluated',0)>=rules.get('min_masks',30)
                    and numeric.get('coverage_low',.7)<=row.get('coverage_80',-1)<=numeric.get('coverage_high',.95)
                    and row.get('crps_improvement',-1)>=numeric.get('min_crps_improvement',0.))
            else:
                entry['eligible']=bool(row.get('n_masks',0)>=rules.get('min_masks',30)
                    and row.get('top1_accuracy',1.)<=rules.get('max_top1',.7)
                    and row.get('recall_at_k',0.)>=rules.get('min_coverage',.9))
            diagnostics.append(entry)
    allowed={r['column_type'] for r in diagnostics if r['eligible']}
    for row in rows:
        row['gated_column_types']=sorted(set(row['column_types'])&allowed)
        row['imputation_subgroup']=bool(row['gated_column_types'])
    summary=[]
    for field in ('ilp_subgroup','imputation_subgroup'):
        n=len({r['source_cluster'] for r in rows if r[field]})
        summary.append({'claim':field,'n_independent_source_clusters':n,'minimum':minimum,'enough_sources':n>=minimum,
            'scope':'validation-only type gates applied without test F1; outcomes for all tasks must still be reported'})
    run_id=datetime.now().strftime('%Y%m%d-%H%M%S-%f'); path=report_folder(cfg,root)/'audit'/(run_id+'.html')
    write_report(path,{'run_id':run_id,'name':'Preregistered eligibility and dependence audit','kind':'protocol_audit','rows':[],
        'config':cfg,'analysis_tables':{'Claim sample size':summary,'Test membership (not performance selection)':rows,
            'Validation-only Q type gates':diagnostics},
        'note':'Ten seeds or overlapping subsets do not make ten independent sources. Numeric columns use explicitly declared CRPS/interval gates, not exact-value categorical coverage. Column CIs for one Alaska task are conditional, not cross-task evidence.'})
    print('Audit:',path,flush=True)
    return {'result_file':str(path),'claims':summary}


def run_suite(raw_cfg, root, part=None):
    # Input: suite config/root and optional job names; output: resumable single receipt HTML plus regular per-phase reports; module function.
    from .research import research_config,run_research,role_config,task_signature,apply_fit_prior
    from .execution import install_q_cache_lock
    from .study_state import StudyState,execute_phase
    install_q_cache_lock()
    if int(raw_cfg['study'].get('parallel_jobs',1))>1:
        from .study_parallel import run_parallel_suite
        return run_parallel_suite(raw_cfg,root,part)
    root=Path(root); plan=raw_cfg['study']; state_path=root/plan.get('state','output/results/server_study_state.html')
    store=StudyState(state_path,root); state=store.read()
    completed=state.get('completed',{}); entries=state.get('rows',[])
    wanted=set(part.split(',')) if part else None
    definitions=[]
    def substitute(value, values):
        # Input: job template and declared matrix coordinates; output: recursively expanded configuration; nested function.
        if isinstance(value,dict): return {k:substitute(v,values) for k,v in value.items()}
        if isinstance(value,list): return [substitute(v,values) for v in value]
        if isinstance(value,str):
            for key,replacement in values.items(): value=value.replace('{'+key+'}',str(replacement))
        return value
    for template in plan.get('jobs',[]):
        matrix=template.get('matrix',{})
        for point in itertools.product(*matrix.values()):
            definition=substitute(template,dict(zip(matrix,point))); definition.pop('matrix',None); definitions.append(definition)
    if wanted and not all(any(fnmatch.fnmatchcase(d['id'],p) for d in definitions) for p in wanted):
        raise ValueError('Unknown study job pattern; check study.jobs IDs rather than silently running nothing')
    for definition in definitions:
        if wanted and not any(fnmatch.fnmatchcase(definition['id'],p) for p in wanted): continue
        for matcher in definition.get('matchers',plan.get('matchers',['distribution','quantile'])):
            source=deep_merge(raw_cfg,definition.get('overrides',{})); source['research']['matcher']=matcher
            source.setdefault('modules',{}).pop('matcher',None)
            name=source['research']['name']; group=source['research'].get('calibration_group',name)
            suffix=plan.get('protocol_suffix')
            if suffix:
                name=name+'_'+suffix; group=group+'_'+suffix
                source['research'].update(name=name,calibration_group=group)
            source['research']['calibration']=f'output/calibration/server/{group}_{matcher}_{source["research"].get("fold","default")}.json'
            cfg=research_config(source,root); frozen_selection=None
            store.select_job(definition['id']+'/'+matcher,[definition['id']+'/'+matcher,
                group+'/'+source['research'].get('fold','default')+'/'+matcher])
            if definition.get('allow_empty',False):
                try: role_config(cfg,'test',root)
                except (FileNotFoundError,ValueError) as exc:
                    message=str(exc)
                    if isinstance(exc,ValueError) and not message.startswith('Empty test role'): raise
                    store.record(definition['id']+'/'+matcher,error=message,status='unavailable_predeclared_data_subgroup')
                    print('Unavailable subgroup:',definition['id'],message,flush=True)
                    continue
            for phase in definition.get('phases',plan.get('phases',['profile','fit','qcheck','validate-lambda','audit','edges','pairwise','global','factorial'])):
                initialization=phase in ('profile','fit','qcheck','select-calibrator') or phase.startswith('validate-')
                token=(group+'/'+source['research'].get('fold','default') if initialization else definition['id'])+'/'+matcher+'/'+phase
                role='fit' if phase=='fit' else 'validation' if phase in ('profile','qcheck','select-calibrator') or phase.startswith('validate-') else 'test'
                _,tasks=role_config(cfg,role,root)
                signature_config=cfg
                if initialization:
                    signature_config={key:cfg[key] for key in ('uncertainty_model','matcher','candidates','calibration')}
                    signature_config.update(seed=cfg['experiment']['seed'],max_cells=cfg['research'].get('max_cells',200))
                    if phase.startswith('validate-'):
                        signature_config.update(prior_policy=cfg['research'].get('prior_policy'),lambda_grid=cfg['research'].get('lambda_grid'),
                                                prior=cfg['research'].get('prior'),prior_grid=cfg['research'].get('prior_grid'))
                    if phase=='validate-offsets':
                        signature_config.update(decision_offset_grid=cfg['research'].get('decision_offset_grid'),
                            missing=cfg['research'].get('missing'),optimizers=cfg['research'].get('optimizers'),
                            ablation=cfg['ablation'],lambda_selection_sha256=hashlib.sha256(Path(frozen_selection).read_bytes()).hexdigest() if frozen_selection else None)
                    if phase=='select-calibrator':
                        signature_config['selection_pool']=cfg['research'].get('calibrator_selection_pool',['platt_retrieved','isotonic_retrieved'])
                signature=hashlib.sha256(json.dumps({'config':signature_config,'role':[task_signature(s) for s in tasks]},sort_keys=True,default=str).encode()).hexdigest()
                artifact_path=root/cfg['research']['calibration']
                receipt=execute_phase(cfg,phase,root,frozen_selection,token,signature,artifact_path,store)
                if phase.startswith('validate-'): frozen_selection=receipt['result_file']
                if phase=='qcheck': cfg['research']['qcheck_report']=receipt['result_file']
                if phase=='select-calibrator':
                    source['research']['calibration']=receipt['result_file']; cfg=research_config(source,root)
    print('Study state:',state_path,flush=True)
    return {'result_file':str(state_path)}


def bundle_study(cfg, root, output):
    # Input: declared data/model directory allowlist; output: portable source/data ZIP with hashes and no credentials/venv; module function.
    from .release import package_source
    root=Path(root).resolve(); output=Path(output); output=output if output.is_absolute() else root/output
    source_zip=root/'output/dist/umsm_source.zip'
    if output.resolve()==source_zip.resolve(): raise ValueError('Use a different filename for the server bundle')
    package_source(root,source_zip); files=[]; spec=cfg.get('study',{}).get('bundle',{})
    for name in spec.get('datasets',[]):
        folder=(root/'data'/name).resolve()
        if not folder.is_relative_to(root/'data') or not folder.is_dir(): raise ValueError(f'Invalid/missing dataset: {name}')
        files.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.csv','.txt','.json')
                     and not any(part.startswith('.') for part in p.relative_to(folder).parts))
    for name in spec.get('models',[]):
        folder=(root/name).resolve()
        if not folder.is_relative_to(root/'models') or not folder.is_dir(): raise ValueError(f'Invalid/missing model: {name}')
        files.extend(p for p in folder.rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.relative_to(folder).parts))
    output.parent.mkdir(parents=True,exist_ok=True); manifest=[]
    with zipfile.ZipFile(source_zip) as source,zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for member in source.infolist(): archive.writestr(member,source.read(member))
        for path in sorted(set(files)):
            if path.name=='.env' or path.is_symlink(): raise ValueError('Credentials and symlinks are excluded')
            relative=path.relative_to(root).as_posix(); archive.write(path,'UMSM/'+relative)
            manifest.append({'path':relative,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        archive.writestr('UMSM/DATA_MANIFEST.json',json.dumps({'files':manifest,'source':'SOURCE_MANIFEST.json'},indent=2))
    print('Server bundle:',output,'data/model files:',len(manifest),flush=True)
    return {'bundle':str(output)}
