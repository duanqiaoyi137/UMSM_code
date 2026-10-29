"""Controlled research protocols on existing four-module interfaces, one report per run."""
import copy
from dataclasses import replace
from datetime import datetime
import fnmatch
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .ablation import strategy_config
from .calibration import calibration_snapshot, check_inference_split, load_artifact, set_fingerprint
from .context import RunContext
from .candidates import compact_info
from .data.loader import load_dataset
from .data.preparation import file_sha256, sha256
from .evaluation import evaluate, validate_matching
from .pipeline import build_components, compute_edges, load_sets
from .profiling import Profiler
from .embedding import encoder_stats
from .experiment import environment_info
from .llm import llm_stats
from .registry import MATCHERS, OPTIMIZERS
from .report import diagnostics as report_diagnostics, matching_record, read_report, write_report
from .research_analysis import (edge_records, local_conflicts, missing_strata, partition_objective,
                                probability_metrics, research_summary, uncertainty_diagnostics)
from .presets import ablation_specs, paper_config
from .config import deep_merge
from .training import fit_pipeline
from .uncertainty_model.mixture import pseudo_mask_examples, qx_diagnostics

PHASES = ('profile','fit','qcheck','global','missing','subsets','validate-lambda','validate-prior',
          'edges','pairwise','frontier','diagnostic-calibration','factorial')


def task_signature(source):
    # 输入：多表任务；输出：同时覆盖表内容、真值和准备协议的验证签名；继承：无（模块函数）。
    return sha256(json.dumps({'data':set_fingerprint(source),'groundtruth':sorted(sorted(g) for g in source.groundtruth or []),
                             'preparation':source.meta.get('preparation_sha256')},sort_keys=True).encode())


def attach_preparation(sets):
    # 输入：已加载任务；输出：原任务列表，校验准备哈希并附加协议；继承：无（模块函数）。
    checked = {}
    for source in sets:
        case = Path(source.meta['table_dir']).parent
        for parent in (case,case.parent):
            path = parent/'provenance.json'
            if not path.exists():
                continue
            if path not in checked:
                manifest = json.loads(path.read_text(encoding='utf-8'))
                if any(file_sha256(parent/f)!=h for f,h in manifest['files'].items()):
                    raise ValueError(f'Prepared input changed: {path}')
                checked[path] = manifest
            source.meta.update(preparation=checked[path]['protocol'],preparation_sha256=file_sha256(path))
            break
    return sets


def ensure_disjoint(left, right):
    # 输入：两个角色的多表任务列表；输出：None；任务/源表内容或派生来源重合时拒绝；继承：无（模块函数）。
    def identities(sets):
        # 输入：任务列表；输出：完整任务、各表及已记录原任务的内容指纹集合；继承：无（内部函数）。
        found = set()
        for source in sets:
            found.add(set_fingerprint(source))
            found.update(set_fingerprint(replace(source,tables=[t])) for t in source.tables)
            protocol = source.meta.get('preparation',{})
            if protocol.get('source_fingerprint'):
                found.add(protocol['source_fingerprint'])
        return found
    if identities(left)&identities(right):
        raise ValueError('Roles overlap in task/source-table contents or recorded pre-mask origin')


def calibration_sources(artifact, cfg, root):
    # 输入：冻结校准、数据根配置、项目路径；输出：内容一致的全部 Q/Platt 拟合源，用于子表隔离；继承：无（模块函数）。
    restored = []
    data_root = Path(cfg['dataset']['root'])
    data_root = data_root if data_root.is_absolute() else root/data_root
    for record in artifact['fit_sets']:
        candidates = load_dataset(record['dataset'],str(data_root),variant=record.get('variant'),
                                  sets=[record['name'].split('/')[-1]])
        matching = [s for s in candidates if set_fingerprint(s)==record['fingerprint']]
        if len(matching)!=1:
            raise ValueError('Calibration source changed; preserve frozen fitting inputs for research audits')
        restored.extend(attach_preparation(matching))
    return restored


def role_config(cfg, role, root):
    # 输入：研究配置、角色、项目根目录；输出：显式加载且非空的该角色配置与多表任务；继承：无（模块函数）。
    spec = cfg['research']['roles'][role]
    spec = {'sets':spec} if isinstance(spec,list) else spec
    out = copy.deepcopy(cfg)
    out['dataset'].update({k:v for k,v in spec.items() if k not in ('sets','exclude')})
    patterns = spec.get('sets')
    directory = Path(out['dataset']['root'])
    directory = directory if directory.is_absolute() else root/directory
    directory = directory/out['dataset']['name']
    selected = [p.name for p in directory.iterdir() if p.is_dir()
                and (patterns is None or any(fnmatch.fnmatchcase(p.name,q) for q in patterns))
                and not any(fnmatch.fnmatchcase(p.name,q) for q in spec.get('exclude',[]))]
    out['dataset']['sets'] = selected
    sets = attach_preparation(load_sets(out,str(root)))
    if not sets:
        raise ValueError(f'Empty {role} role: {spec}')
    out['dataset']['sets'] = [s.name.split('/')[-1] for s in sets]
    return out,sets


def matcher_spec(label, params, root):
    # 输入：实验 matcher 标签、覆盖参数、根目录；输出：四模块兼容 matcher 配置；继承：无（模块函数）。
    if label in ('mean','name','value'):
        weights = {'name_weight':0.,'value_weight':1.} if label=='value' else (
            {'name_weight':1.,'value_weight':0.} if label=='name' else {})
        result = {'name':'sbert_mean','params':weights}
    elif label in ('coma','jaccard','distribution'):
        algorithm = {'coma':'coma_instance','jaccard':'jaccard_levenshtein','distribution':'distribution_based'}[label]
        result = {'name':'valentine','params':{'algorithm':algorithm,'local_selection':'all','instance_sample_size':None}}
        if label=='coma':
            result['params']['algorithm_params'] = {'use_instances':True,'use_schema':False,'max_n':0,'delta':1.,'threshold':0.}
    elif label=='magneto':
        from .config import load_config
        result = copy.deepcopy(load_config(str(root/'configs/magneto_deepseek.yaml'))['matcher'])
    else:
        raise ValueError(f'Unknown research matcher: {label}')
    params = params or {}
    if set(params)&{'mean','name','value','coma','jaccard','distribution','magneto'}:
        params = params.get(label,{})
    result['params'].update(params)
    result['cache_scores'] = True
    return result


def research_config(cfg, root):
    # 输入：用户研究配置与根目录；输出：补齐四模块、策略和优化器的独立配置；继承：无（模块函数）。
    spec = cfg['research']
    label = spec.get('matcher','mean')
    artifact = spec['calibration'].format(matcher=label,fold=spec.get('fold','default'))
    result = paper_config(artifact,risk_lambda=spec.get('risk_lambda',.5),prior=spec.get('prior',.5))
    for name in ('dataset','paths','profiling','experiment'):
        result[name].update(cfg.get(name,{}))
    result['candidates'] = copy.deepcopy(cfg.get('candidates',{'mode':'all'}))
    custom_matcher = cfg.get('modules',{}).get('matcher')
    if label not in ('mean','name','value','coma','jaccard','distribution','magneto') and custom_matcher:
        result['matcher'] = copy.deepcopy(custom_matcher)
    else:
        result['matcher'] = matcher_spec(label,spec.get('matcher_params',{}),root)
    result['matcher']['cache_scores'] = spec.get('cache_scores',True)
    result['uncertainty_model']['realization']['n_samples'] = spec.get('n_samples',32)
    result['research'] = {**copy.deepcopy(spec),'matcher':label,'calibration':artifact}
    result['global_optimizer']['params'].update(spec.get('ilp',{}))
    for name,override in cfg.get('modules',{}).items():
        if name not in ('uncertainty_model','matcher','uncertainty_propagation','global_optimizer'):
            raise ValueError(f'Unknown module: {name}')
        base = result[name]
        if override.get('name',base['name'])!=base['name']:
            base = {**base,'params':{}}
        result[name] = deep_merge(base,override)
    scope = spec.get('evidence_scope','table')
    if scope not in ('pair','table'):
        raise ValueError('research.evidence_scope must be pair or table')
    if scope=='pair':
        from .registry import MATCHERS
        base_matcher = MATCHERS.build(result['matcher']['name'],result['matcher'].get('params',{}))
        if base_matcher.sampling_scope=='table_pair':
            result['matcher'] = {**result['matcher'],'name':'column_pair','params':{
                'matcher':base_matcher.registry_name,'matcher_params':copy.deepcopy(result['matcher'].get('params',{}))}}
    result['ablation'] = deep_merge(ablation_specs(result),cfg.get('ablation',{}))
    result['calibration'] = copy.deepcopy(cfg.get('calibration',{'method':'platt'}))
    result['research']['n_samples'] = result['uncertainty_model']['realization']['n_samples']
    result['research']['evidence_scope'] = scope
    if result['uncertainty_propagation']['name']=='calibrated_logit':
        result['research'].update({key:result['uncertainty_propagation']['params'][key] for key in ('prior','risk_lambda')})
    return result


def subset_cases(source, sizes, repeats, seed):
    # 输入：完整多表任务、k 列表、重复次数、独立子集种子；输出：(子任务,配对元信息) 迭代器；继承：无（模块函数）。
    from .data.masking import stable_seed
    if repeats<1:
        raise ValueError('subset repeats must be positive')
    n = len(source.tables)
    seen_k = set()
    for value in sizes:
        k = n if value=='all' else int(value)
        if k<2 or k>n or k in seen_k:
            continue
        seen_k.add(k)
        from math import comb
        count = min(repeats,comb(n,k))
        rng = np.random.default_rng(stable_seed(seed,source.name+':'+str(k)))
        choices = set()
        if comb(n,k)<=repeats:
            choices.update(combinations(range(n),k))
        else:
            while len(choices)<count:
                choices.add(tuple(sorted(rng.choice(n,size=k,replace=False).tolist())))
        for draw,indices in enumerate(sorted(choices)):
            tables = [source.tables[i] for i in indices]
            allowed = {c.id for t in tables for c in t.columns}
            gold = [frozenset(g&allowed) for g in source.groundtruth if g&allowed]
            yield replace(source,name=source.name+f'__k{k}_d{draw}',tables=tables,groundtruth=gold), {
                'set':source.name,'k':k,'draw':draw,'source_tables':[t.name for t in tables]}


def select_parameter(path, cfg, root):
    # 输入：验证报告路径或 latest、当前配置、根目录；输出：通过协议与校准哈希检查的冻结选择；继承：无（模块函数）。
    folder = report_folder(cfg,root)
    if path=='latest':
        pattern = 'validate-prior/*.html' if cfg['research'].get('prior_optimizer')=='pairwise' else 'validate-*/*.html'
        candidates = list(folder.glob(pattern))
        if not candidates:
            raise FileNotFoundError('Run a validate-lambda or validate-prior phase first')
        path = max(candidates,key=lambda p:p.stat().st_mtime)
    path = Path(path)
    if not path.is_absolute():
        path = root/path
    report = read_report(path)
    if report.get('protocol',{}).get('diagnostic_test_fit') or any(
            r.get('oracle') or r.get('test_label_fit') or r.get('claim_eligible') is False for r in report.get('rows',[])):
        raise ValueError('Diagnostic/test-fitted reports cannot select formal parameters')
    _,expected_sets = role_config(cfg,'validation',root)
    fitted_matcher=report.get('config',{}).get('matcher',{})
    previous=MATCHERS.build(fitted_matcher.get('name',''),fitted_matcher.get('params',{}))
    current=MATCHERS.build(cfg['matcher']['name'],cfg['matcher'].get('params',{}))
    same_matcher=previous.registry_name==current.registry_name and previous.params()==current.params()
    if (report.get('protocol',{}).get('role')!='validation' or not same_matcher
            or report.get('protocol',{}).get('fingerprints') != [task_signature(s) for s in expected_sets]
            or report.get('calibration_sha256') != file_sha256(root/cfg['research']['calibration'])
            or report['config']['candidates'] != cfg['candidates']
            or report['config']['uncertainty_model']['realization'] != cfg['uncertainty_model']['realization']):
        raise ValueError('Selection must use unchanged independent validation data, matcher, Q and calibration')
    selection = report['selection']
    if selection['axis']=='lambda' and report['config']['ablation']['missing']['Q_risk']['uncertainty_model']!=cfg['ablation']['missing']['Q_risk']['uncertainty_model']:
        raise ValueError('The Q model changed since lambda validation')
    if selection['axis']=='lambda' and cfg['research'].get('prior',.5)!=selection['held_fixed']:
        raise ValueError('Prior differs from lambda validation; do not silently combine separate sweeps')
    if selection['axis']=='prior':
        learned_protocol = selection.get('decision_protocol','global')
        if learned_protocol!=cfg['research'].get('prior_optimizer','global'):
            raise ValueError('Prior validation decision protocol changed; select one shared prior again')
        base = selection.get('base_strategy','Q_risk')
        current = cfg['ablation']['missing'][base]['uncertainty_propagation']['params']['risk_lambda']
        if current!=selection['held_fixed']:
            raise ValueError('The base strategy lambda differs from prior validation')
    return {**selection,'report':str(path),'report_sha256':file_sha256(path)}


def report_folder(cfg, root):
    # 输入：研究配置、根目录；输出：该研究/折/matcher 的报告目录 Path；继承：无（模块函数）。
    spec = cfg['research']
    return root/cfg['experiment']['output_dir']/spec.get('name','study')/spec.get('fold','default')/spec['matcher']


def run_research(cfg, phase, root, selection=None):
    # 输入：完整配置、阶段、根目录、可选冻结验证选择；输出：单 HTML 路径及完整结果；继承：无（模块函数）。
    spec = cfg['research']
    artifact_path = root/spec['calibration']
    if phase=='fit':
        if artifact_path.exists():
            raise FileExistsError(f'{artifact_path} exists; retain frozen calibration or choose a new path')
        fitcfg,fitsets = role_config(cfg,'fit',root)
        for role in ('validation','test'):
            _,other = role_config(cfg,role,root)
            ensure_disjoint(fitsets,other)
            if spec.get('qx_from'):
                ensure_disjoint(calibration_sources(load_artifact(str(root/spec['qx_from'])),cfg,root),other)
        artifact = fit_pipeline(fitcfg,str(root),str(artifact_path),max_cells=spec.get('max_cells',200),
                                qx_from=spec.get('qx_from'),mask_seed=spec.get('mask_seed',42),
                                split_role=spec.get('calibration_split_role','validation'))
        print('Calibration:',artifact_path)
        return {'calibration':artifact}
    role = 'validation' if phase.startswith('validate-') or phase=='qcheck' else 'test'
    workcfg,sets = role_config(cfg,role,root)
    _,validation_sets = role_config(cfg,'validation',root)
    if role=='test':
        ensure_disjoint(sets,validation_sets)
    _,fitsets = role_config(cfg,'fit',root)
    ensure_disjoint(sets,fitsets)
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    path = report_folder(cfg,root)/phase/(run_id+'.html')
    seed = int(cfg['experiment']['seed'])
    ctx = RunContext(seed=seed,project_root=str(root),cache_dir=str(root/cfg['paths']['cache_dir']),
                     embedding_cache=cfg['paths'].get('embedding_cache',True))
    if phase=='profile':
        profiles = [{'set':s.name,**s.stats()} for s in sets]
        columns = [{'set':s.name,'table':t.name,'column':c.name,'rows':len(t.df),
                    'missing_rate':float(t.df[c.name].isna().mean()),'observed':int(t.df[c.name].notna().sum()),
                    'distinct_observed':int(t.df[c.name].nunique())} for s in sets for t in s.tables for c in t.columns]
        payload={'run_id':run_id,'name':spec['name']+' data profile','kind':'data_profile','config':cfg,
                 'analysis_tables':{'Integration sets':profiles,'Natural-missing columns':columns}}
        write_report(path,payload)
        print(pd.DataFrame(profiles).to_string(index=False))
        print('Result:',path)
        return {'result_file':str(path),**payload}
    frozen = load_artifact(str(artifact_path))
    ensure_disjoint(sets,calibration_sources(frozen,cfg,root))
    if phase=='qcheck':
        for source in sets:
            check_inference_split(str(artifact_path),source,ctx)
        q = frozen['qx']
        examples = pseudo_mask_examples(sets,q['generators'],ctx,spec.get('max_cells',200),42)
        diagnostics = []
        for index in range(-1,len(q['generators'])):
            weights = q['weights'] if index<0 else [float(i==index) for i in range(len(q['generators']))]
            isolated = examples if index<0 else [{**ex,'generators':[g if i==index else [] for i,g in enumerate(ex['generators'])]} for ex in examples]
            diagnostics.append({'method':'mixture' if index<0 else q['generators'][index]['name'],
                                **qx_diagnostics(isolated,q['temperatures'],weights)})
        payload = {'run_id':run_id,'name':spec['name']+' Q diagnostics','config':cfg,'qx_diagnostics':diagnostics}
        write_report(path,payload)
        print('Result:',path)
        return {'result_file':str(path),**payload}
    if selection:
        chosen = select_parameter(selection,cfg,root)
        if chosen['axis']=='lambda':
            cfg['ablation']['missing']['Q_risk']['uncertainty_propagation']['params']['risk_lambda']=chosen['value']
        else:
            for strategy in cfg['ablation']['missing'].values():
                strategy['uncertainty_propagation']['params']['prior']=chosen['value']
            cfg['ablation']['optimizers']['greedy']['params']['threshold']=chosen['value']
    else:
        chosen = None
    from .isolation import ISOLATED_PHASES, run_isolation
    if phase in ISOLATED_PHASES or (phase=='validate-prior' and spec.get('prior_optimizer','global')=='pairwise'):
        return run_isolation(cfg,phase,sets,workcfg,ctx,frozen,chosen,path,run_id,root)
    strategies = copy.deepcopy(cfg['ablation']['missing'])
    if phase in ('global','subsets'):
        strategies = {n:strategies[n] for n in spec.get('global_missing',['observed_only'])}
    elif phase.startswith('validate-'):
        axis = phase.split('-')[1]
        parameter = 'risk_lambda' if axis=='lambda' else 'prior'
        grid = spec.get('lambda_grid',[0.,.1,.25,.5]) if axis=='lambda' else spec.get('prior_grid',[.4,.45,.5])
        base_strategy = 'Q_risk' if axis=='lambda' else spec.get('prior_strategy','observed_only')
        strategies = {}
        for value in grid:
            variant = copy.deepcopy(cfg['ablation']['missing'][base_strategy])
            variant['uncertainty_propagation']['params'][parameter]=value
            strategies[f'Q_{axis}_{value:g}']=variant
    else:
        strategies = {n:strategies[n] for n in spec.get('missing',list(strategies))}
    optimizer_specs = {n:cfg['ablation']['optimizers'][n] for n in spec.get('optimizers',['local','greedy','global'])}
    if phase.startswith('validate-'):
        optimizer_specs = {'global':cfg['ablation']['optimizers']['global']}
    rows, predictions, strata, risk, conditional, regressions, edge_details, stage_records = [],[],[],[],[],[],[],[]
    profiler = Profiler().start()
    ctx.profiler = profiler
    calibrations = {}
    try:
        for source in sets:
            subset = spec.get('subsets',{})
            cases = subset_cases(source,subset.get('k',[2,3,4,6,8,12,'all']),subset.get('repeats',20),subset.get('seed',2026)) if phase=='subsets' else [(source,{'set':source.name,'k':len(source.tables),'draw':0,'source_tables':[t.name for t in source.tables]})]
            for current,case in cases:
                prep = source.meta.get('preparation',{})
                tags = {**case,'variant':f'k{case["k"]}_d{case["draw"]}' if phase=='subsets' else workcfg['dataset'].get('variant','original'),
                        'mechanism':prep.get('mechanism','natural'),'rate':prep.get('rate'), 'row_cap':prep.get('row_cap')}
                start_case = len(rows)
                # Oracle reference uses pre-mask values, not truth-driven parameter selection.
                if prep.get('mechanism') and phase!='subsets':
                    oracle_set = load_dataset(source.dataset,str(root/cfg['dataset']['root']),variant='oracle',sets=[source.name.split('/')[-1]])[0]
                    active = [('oracle',cfg['ablation']['missing']['observed_only'],oracle_set)] + [(n,s,current) for n,s in strategies.items()]
                else:
                    active = [(n,s,current) for n,s in strategies.items()]
                for missing,strategy,iset in active:
                    local_cfg = strategy_config(cfg,strategy)
                    local_cfg['dataset'] = copy.deepcopy(workcfg['dataset'])
                    comp,realization = build_components(local_cfg)
                    calibrations[missing] = calibration_snapshot(comp,ctx)
                    before = len(profiler.records)
                    llm_before = llm_stats(ctx)
                    started = time.perf_counter()
                    data,samples,edges,hit = compute_edges(iset,comp,realization,local_cfg,ctx)
                    elapsed = time.perf_counter()-started
                    llm_after = llm_stats(ctx)
                    stages = profiler.records[before:]
                    stage_records.append({**tags,'missing':missing,'stages':stages,'cache_hit':hit})
                    blocking = compact_info(samples.meta.get('candidates',{}))
                    frame = edge_records(edges,iset)
                    edge_metrics = probability_metrics(frame)
                    conflicts = local_conflicts(edges) if spec.get('diagnostics',True) else {'conflict_rate':None}
                    if spec.get('diagnostics',True):
                        ud = uncertainty_diagnostics(frame,float(edges.meta['prior']))
                        risk.extend({**tags,'missing':missing,**r} for r in ud['risk_coverage'])
                        conditional.extend({**tags,'missing':missing,**r} for r in ud['conditional_errors'])
                        regressions.append({**tags,'missing':missing,**ud['regression'],'aurc':ud.get('aurc')})
                    detail = report_diagnostics(data,samples,edges,cfg)
                    if detail:
                        if cfg['experiment'].get('save_edges'):
                            detail['edges'] = frame
                        edge_details.append({**tags,'missing':missing,**detail})
                    for name,ospec in optimizer_specs.items():
                        optimizer = OPTIMIZERS.build(ospec['name'],ospec['params'])
                        started = time.perf_counter()
                        result = optimizer.optimize(edges,ctx)
                        optimizer_time = time.perf_counter()-started
                        validate_matching(result,edges,f'{missing}/{name}/{current.name}')
                        record = matching_record(result,iset.groundtruth,edges.columns)
                        rows.append({**tags,'missing':missing,'optimizer':name,'oracle':False,
                            **evaluate(result,iset.groundtruth),**{k:record[k] for k in ('TP','FP','FN')},
                            'mean_sigma':float(frame.sigma.mean()) if len(frame) else 0.,'edge_auc':edge_metrics['auc'],
                            'edge_brier':edge_metrics['brier'],'edge_nll':edge_metrics['nll'],**conflicts,
                            'compute_edges_s':elapsed,'optimizer_time_s':optimizer_time,'matcher_cache_hit':hit,
                            **{k:samples.meta.get(k,False) for k in ('online_statistics','legacy_cache_import','raw_samples_retained')},
                            'n_score_samples':len(samples),'n_edges':len(edges.df),'R':realization.n_samples,
                            'component_recall':blocking.get('component_recall'),'recall_at_l':blocking.get('recall_at_l'),
                            'llm_calls':llm_after.get('llm_calls',0)-llm_before.get('llm_calls',0),
                            'llm_cost_usd':llm_after.get('llm_cost_usd',0)-llm_before.get('llm_cost_usd',0),
                            'n_uncertain_edges':int((frame.sigma>1e-12).sum()),
                            'uncertainty_model_s':sum(r['time_s'] for r in stages if r['stage']=='uncertainty_model'),
                            'sampling_s':sum(r['time_s'] for r in stages if r['stage']=='sampling'),
                            'matcher_s':sum(r['time_s'] for r in stages if r['stage']=='matcher'),
                            'propagation_s':sum(r['time_s'] for r in stages if r['stage']=='uncertainty_propagation'),
                            **partition_objective(result,edges),
                            **{k:result.info.get(k) for k in ('all_optimal','global_exact','n_variables','n_triangle_constraints','max_component_columns','solver_time_s')}})
                        predictions.append({**tags,'missing':missing,'optimizer':name,'matching':record})
                        if spec.get('diagnostics',True):
                            strata.extend({**tags,'missing':missing,'optimizer':name,**r} for r in missing_strata(edge_records(edges,iset,result)))
                    method_rows = [r for r in rows[start_case:] if r['missing']==missing]
                    optimum = next((r for r in method_rows if r['optimizer']=='global'),None)
                    for row in method_rows:
                        row['objective_gap'] = (optimum['partition_objective']-row['partition_objective']) if optimum and row['pairs_outside_closure']==0 else None
                        if row['k']==2 and row['optimizer']=='local' and optimum:
                            row['two_table_objective_agrees'] = abs(row['objective_gap'])<1e-6
                case_rows = rows[start_case:]
                for row in case_rows:
                    obs = next((r for r in case_rows if r['missing']=='observed_only' and r['optimizer']==row['optimizer']),None)
                    oracle = next((r for r in case_rows if r['missing']=='oracle' and r['optimizer']==row['optimizer']),None)
                    denominator = oracle['pair_f1']-obs['pair_f1'] if oracle and obs else None
                    row['recovery_denominator'] = denominator
                    row['recovery_ratio'] = (row['pair_f1']-obs['pair_f1'])/denominator if denominator is not None and denominator>1e-12 else None
                print(f'{case["set"]} k={case["k"]} draw={case["draw"]}: completed {len(case_rows)} method decisions',flush=True)
    finally:
        resources = profiler.stop()
    selection_record = chosen
    if phase.startswith('validate-'):
        axis = phase.split('-')[1]
        if axis=='lambda' and not any(r['n_uncertain_edges'] for r in rows):
            raise ValueError('No uncertain validation edges; lambda is not identifiable')
        scores = pd.DataFrame(rows).groupby('missing').pair_f1.mean()
        best = scores.max()
        candidates = [float(name.rsplit('_',1)[1]) for name,value in scores.items() if value>=best-1e-12]
        value = min(candidates) if axis=='lambda' else max(candidates)
        selection_record = {'axis':axis,'value':value,'objective':'macro pair_f1 of global on validation only',
                            'score':float(best),'tie_rule':'smaller lambda/larger prior',
                            'base_strategy':base_strategy,
                            'held_fixed':spec.get('prior',.5) if axis=='lambda' else cfg['ablation']['missing'][base_strategy]['uncertainty_propagation']['params']['risk_lambda']}
    payload = {'run_id':run_id,'name':spec['name']+' '+phase,'kind':'research','config':cfg,
        'protocol':{'role':role,'phase':phase,'fingerprints':[task_signature(s) for s in sets],
                    'note':'Repeated overlapping source subsets are paired cases, not independent datasets; no test-fitted thresholds.'},
        'calibration':calibrations,'calibration_sha256':file_sha256(artifact_path),'selection':selection_record,
        'rows':rows,'summary':research_summary(rows),'predictions':predictions,
        'environment':environment_info(str(root)),
        'resources':{**resources,'embedding':encoder_stats(ctx),'llm':llm_stats(ctx)},
        'analysis_tables':{'Missingness strata':strata,'Uncertainty regressions':regressions,'Conditional errors':conditional,
                           'Risk coverage':risk,'Frozen validation selection':[selection_record] if selection_record else [],
                           'Runtime and ILP size':[{k:r.get(k) for k in ('set','k','draw','missing','optimizer','R','n_edges','n_score_samples',
                            'compute_edges_s','optimizer_time_s','matcher_cache_hit','matcher_s','sampling_s','uncertainty_model_s','propagation_s',
                            'n_variables','n_triangle_constraints','max_component_columns','all_optimal','llm_calls','llm_cost_usd')} for r in rows],
                           'Objective and recovery':[{k:r.get(k) for k in ('set','k','draw','missing','optimizer','pair_f1','partition_objective',
                            'objective_gap','pairs_outside_closure','two_table_objective_agrees','mechanism','rate','row_cap','recovery_denominator','recovery_ratio')} for r in rows]},
        'diagnostics':edge_details,'stage_records':stage_records}
    if spec.get('plots',True):
        payload['figures'] = True  # write_report renders only trusted numeric data, once.
    write_report(path,payload)
    print('Result:',path)
    return {'result_file':str(path),**payload}
