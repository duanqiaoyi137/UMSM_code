"""Three isolated evaluation layers, paired prior frontiers and explicit test-fit diagnostics."""
import copy
import time

import numpy as np
import pandas as pd
from scipy.special import logit

from .ablation import strategy_config
from .calibration import calibrate_scores, calibration_snapshot, fit_platt, match_labels
from .evaluation import evaluate, validate_matching
from .isolated_metrics import (factorial_effects, frontier_rows, paired_bootstrap,
                               paired_frame, pairwise_decisions, quality_rows, recovery_rows, recovery_cluster_ci,
                               risk_decision_changes, recall_at_precision)
from .data.loader import load_dataset
from .pipeline import build_components, compute_edges
from .registry import OPTIMIZERS
from .report import diagnostics, matching_record, write_report
from .research_analysis import edge_records
from .types import EdgeTable, ScoreSamples
from .uncertainty_propagation.base import moments

ISOLATED_PHASES = ('edges','pairwise','frontier','diagnostic-calibration','factorial')


def prior_graph(graph, prior):
    # 输入：固定概率边、共同 prior；输出：仅重算 signed a 的 EdgeTable，不重新评分；继承：无（模块函数）。
    if not np.isfinite(prior) or not 0<prior<1:
        raise ValueError('Prior must be finite and in (0,1)')
    frame = graph.df.copy()
    frame['a'] = logit(frame.p_tilde.to_numpy(float))-logit(prior)
    return EdgeTable(frame,graph.columns,{**graph.meta,'prior':prior})


def risk_graph(graph, value):
    # Input: fixed mu/sigma evidence and lambda; output: risk-adjusted graph with no matcher rerun; module function.
    if not np.isfinite(value) or value<0: raise ValueError('lambda must be finite and nonnegative')
    frame=graph.df.copy(); epsilon=graph.meta.get('epsilon',1e-6)
    frame['p_tilde']=(frame.mu-value*frame.sigma).clip(epsilon,1-epsilon)
    frame['w']=frame.p_tilde; frame['a']=logit(frame.p_tilde)-logit(graph.meta['prior'])
    return EdgeTable(frame,graph.columns,{**graph.meta,'risk_lambda':float(value)})


def diagnostic_graph(samples, columns, params, original):
    # 输入：已有原始样本、列、测试内 Platt 和正式图；输出：明确带测试拟合标记的诊断图；继承：无（模块函数）。
    frame = samples.df.copy()
    frame['score'] = calibrate_scores(frame.score,params,frame.get('retrieved'))
    out = moments(ScoreSamples(frame,columns,samples.meta))
    # This diagnostic uses observed_only and lambda=0, never changes/fits q_x.
    epsilon = original.meta.get('epsilon',1e-6)
    out['p_tilde'] = out.mu.clip(epsilon,1-epsilon)
    out['w'] = out.p_tilde
    out['a'] = logit(out.p_tilde)
    return EdgeTable(out,columns,{**original.meta,'prior':.5,'risk_lambda':0.,'diagnostic_test_fit':True,
                                  'potential':'calibrated_logit','calibration':params})


def isolation_summary(rows):
    # 输入：分层评估行；输出：按策略/决策器/概率空间/先验分别汇总的任务宏均值；继承：无（模块函数）。
    data = pd.DataFrame(rows)
    if data.empty:
        return []
    keys = [c for c in ('missing','optimizer','score_space','missing_bin','column_type_pair','prior','risk_lambda','calibration_condition') if c in data]
    metrics = [c for c in ('auc','average_precision','brier','nll','pair_precision','pair_recall','pair_f1') if c in data]
    data[metrics] = data[metrics].apply(pd.to_numeric,errors='coerce')
    result = data.groupby(keys,dropna=False,sort=False)[metrics].mean().reset_index()
    result['n_cases'] = data.groupby(keys,dropna=False,sort=False).size().to_numpy()
    return result.to_dict('records')


def validate_shared_graph(frames, names):
    # 输入：同任务各策略的概率边、参与策略名；输出：None；闭包/标签或共同 Q 实现不一致时拒绝；继承：无（模块函数）。
    available = [n for n in names if n in frames]
    if available:
        for name in available[1:]:
            paired_frame(frames[name],frames[available[0]])
    if {'Q_mean','Q_risk'}<=set(frames):
        left,right = paired_frame(frames['Q_mean'],frames['Q_risk'])
        if not np.allclose(left[['mu','sigma']],right[['mu','sigma']],rtol=0,atol=1e-12):
            raise ValueError('Risk ablation must share Q, draws and mu/sigma; change only lambda')


def run_isolation(cfg, phase, sets, workcfg, ctx, frozen, chosen, path, run_id, root):
    # 输入：已检查隔离的配置/数据/冻结校准/运行上下文；输出：唯一 HTML，严分概率、表对决策与最终分区；继承：无（模块函数）。
    from .research import task_signature, source_clusters
    spec = cfg['research']
    validation = phase in ('validate-prior','validate-lambda')
    lambda_validation=phase=='validate-lambda'
    diagnostic = phase=='diagnostic-calibration'
    if validation and spec.get('prior_strategy','observed_only')!='observed_only':
        raise ValueError('Isolated shared-prior selection is fitted on observed_only only')
    names = (['Q_risk'] if lambda_validation else ['observed_only'] if validation or phase in ('frontier','diagnostic-calibration') else
             spec.get('factorial_levels',['observed_only',spec.get('factorial_q','Q_mean')]) if phase=='factorial' else spec.get('missing',list(cfg['ablation']['missing'])))
    if phase=='factorial' and not spec.get('factorial_levels') and names[1] not in ('Q_mean','Q_risk'):
        raise ValueError('factorial_q must be Q_mean or Q_risk')
    bootstrap = copy.deepcopy(spec.get('bootstrap',{}))
    # Repeated masks of one original task must never become independent task clusters.
    groups = source_clusters(sets,bootstrap.get('task_groups',{}))
    bootstrap['task_groups'] = groups
    bootstrap['min_clusters']=spec.get('gates',{}).get('min_sources',10)
    bootstrap['type_strata']=bool(spec.get('type_strata',False))
    rows,predictions,per_pair,edge_rows,details,edge_blocks,decision_blocks,case_graphs = [],[],[],[],[],[],[],[]
    snapshots = {}
    risk_changes = []
    shared_platt = None
    for source in sets:
        frames,decisions = {},{}
        prep = source.meta.get('preparation',{})
        active = [(name,source) for name in names]
        if phase in ('edges','pairwise') and prep.get('mechanism') and spec.get('oracle_reference',False):
            data_root = root/cfg['dataset']['root']
            oracle = load_dataset(source.dataset,str(data_root),variant='oracle',sets=[source.name.split('/')[-1]])[0]
            if oracle.groundtruth!=source.groundtruth or {c.id for c in oracle.columns}!={c.id for c in source.columns}:
                raise ValueError('Pre-mask oracle must preserve columns and ground truth')
            active.append(('oracle',oracle))
        for name,inference_source in active:
            scfg = strategy_config(cfg,cfg['ablation']['missing']['observed_only' if name=='oracle' else name])
            scfg['dataset'] = copy.deepcopy(workcfg['dataset'])
            components,realization = build_components(scfg)
            snapshot = calibration_snapshot(components,ctx)
            fingerprint = snapshot['uncertainty_propagation']['sha256']
            if shared_platt is not None and shared_platt!=fingerprint:
                raise ValueError('Isolation requires one shared frozen Platt calibrator for all strategies')
            shared_platt = fingerprint
            snapshots[name] = snapshot
            started = time.perf_counter()
            data,samples,graph,hit = compute_edges(inference_source,components,realization,scfg,ctx)
            graph.meta.update(report_type_strata=bootstrap['type_strata'],
                              type_options=cfg['uncertainty_model'].get('params',{}).get('type_options',{}))
            compute_time = time.perf_counter()-started
            frame = edge_records(graph,source)
            frames[name] = frame
            tags = {'set':source.name,'variant':workcfg['dataset'].get('variant','original'),'missing':name,
                    'k':len(source.tables),'draw':0,'prior':graph.meta['prior'],'oracle':diagnostic,
                    'pre_mask_reference':name=='oracle','source_cluster':groups[source.name],
                    'mechanism':prep.get('mechanism','natural'),'rate':prep.get('rate'),'row_cap':prep.get('row_cap'),
                    **{key:prep.get(key) for key in ('construction','donor_band','alignment','requested_sibling_ratio','mixed_format','entity_overlap')},
                    'calibration_condition':'frozen','test_label_fit':False,'claim_eligible':not diagnostic,
                    'R':realization.n_samples,'matcher_cache_hit':hit,'compute_edges_s':compute_time,
                    **{k:samples.meta.get(k,False) for k in ('online_statistics','legacy_cache_import','raw_samples_retained')}}
            # Probability rows are computed once, independent of all optimizer loops.
            edge_rows.extend(quality_rows(frame,tags))
            detail = diagnostics(data,samples,graph,cfg)
            if detail:
                details.append({**tags,**detail})
            if phase=='edges':
                continue
            if phase=='pairwise':
                started = time.perf_counter()
                metrics,record,pair_rows,decisions[name] = pairwise_decisions(graph,source)
                rows.append({**tags,'optimizer':'pairwise','decision_time_s':time.perf_counter()-started,**metrics})
                predictions.append({**tags,'optimizer':'pairwise','pairwise_decisions':record})
                per_pair.extend({**tags,**r} for r in pair_rows)
                if spec.get('include_retriever_bp',False) and name=='observed_only':
                    from .magneto_study import retriever_bp
                    metrics,record=retriever_bp(graph,source)
                    rows.append({**tags,'optimizer':'magneto_retriever_bp',**metrics})
                    predictions.append({**tags,'optimizer':'magneto_retriever_bp','pairwise_decisions':record})
                continue
            case_graphs.append({'source':source,'graph':graph,'samples':samples if diagnostic else None,'tags':tags})
        validate_shared_graph(frames,[name for name,_ in active])
        edge_blocks.append({'set':source.name,'frames':frames})
        if decisions:
            decision_blocks.append({'set':source.name,'frames':decisions})
            risk_changes.extend(risk_decision_changes(decisions,{'set':source.name,'source_cluster':groups[source.name]}))
    test_platt = None
    if diagnostic:
        # Pooled test fitting is in-sample diagnostic only; no calibration JSON is published.
        scores,labels,weights = [],[],[]
        for item in case_graphs:
            samples,source = item['samples'],item['source']
            scores.extend(samples.df.score)
            labels.extend(match_labels(samples,source.groundtruth))
            counts = samples.df.groupby(['col_i','col_j']).score.transform('size')
            weights.extend(1/counts)
        test_platt = fit_platt(scores,labels,weights)
    grid = (spec.get('lambda_grid',[0.,.1,.25,.5]) if lambda_validation else
            spec.get('prior_grid',[.4,.45,.5]) if validation else spec.get('frontier_priors',[.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95]))
    if phase=='frontier' and spec.get('prior_policy')=='fit_prevalence': grid=sorted(set([*grid,spec['prior']]))
    if not grid or any(not np.isfinite(p) or (p<0 if lambda_validation else not 0<p<1) for p in grid):
        raise ValueError('Prior grid must be nonempty, finite and inside (0,1)')
    for item in case_graphs:
        source,graph,tags = item['source'],item['graph'],item['tags']
        variants = [('frozen',graph)]
        if diagnostic:
            variants.append(('test_fitted',diagnostic_graph(item['samples'],graph.columns,test_platt,graph)))
        for condition,base in variants:
            points = grid if validation or phase in ('frontier','diagnostic-calibration') else [base.meta['prior']]
            for point in dict.fromkeys(points):
                prior=base.meta['prior'] if lambda_validation else point
                evidence = risk_graph(base,point) if lambda_validation else prior_graph(base,prior)
                tagged = {**tags,'prior':float(prior),'calibration_condition':condition,
                          **({'risk_lambda':float(point)} if lambda_validation else {}),
                          'oracle':diagnostic,'test_label_fit':condition=='test_fitted','claim_eligible':not diagnostic}
                if validation:
                    metrics,record,pair_rows,_ = pairwise_decisions(evidence,source)
                    rows.append({**tagged,**({'risk_lambda':float(point)} if lambda_validation else {}),'optimizer':'pairwise',**metrics})
                    per_pair.extend({**tagged,**r} for r in pair_rows)
                    continue
                for name in ('local','global'):
                    from .threshold_selection import offset_graph
                    offset=0. if phase in ('frontier','diagnostic-calibration') else (chosen or {}).get('decision_offsets',{}).get(tags['missing'],{}).get(name,0.)
                    decision_evidence=offset_graph(evidence,offset) if offset else evidence
                    optimizer = OPTIMIZERS.build(**cfg['ablation']['optimizers'][name])
                    started = time.perf_counter()
                    result = optimizer.optimize(decision_evidence,ctx)
                    validate_matching(result,evidence,f'{phase}/{source.name}/{prior}/{name}')
                    record = matching_record(result,source.groundtruth,evidence.columns)
                    rows.append({**tagged,'optimizer':name,'optimizer_time_s':time.perf_counter()-started,'decision_offset':float(offset),
                                 **{k:record[k] for k in ('TP','FP','FN')},**evaluate(result,source.groundtruth,evidence.columns),
                                 'all_optimal':result.info.get('all_optimal'),'global_exact':result.info.get('global_exact')})
                    predictions.append({**tagged,'optimizer':name,'matching':record})
    selection = chosen
    if validation:
        axis='lambda' if lambda_validation else 'prior'
        scores = pd.DataFrame(rows).groupby('risk_lambda' if lambda_validation else 'prior').pair_f1.mean()
        best = scores.max()
        choices=[float(p) for p,s in scores.items() if s>=best-1e-12]
        value = min(choices) if lambda_validation else max(choices)
        selection = {'axis':axis,'value':value,'score':float(best),'base_strategy':'Q_risk' if lambda_validation else 'observed_only',
                     'held_fixed':spec['prior'] if lambda_validation else 0.,'decision_protocol':'pairwise',
                     'tie_rule':'smaller lambda within 1e-12' if lambda_validation else 'larger prior within 1e-12',
                     'objective':'macro-task pair_f1 on independent table-pair Hungarian decisions, validation only'}
        validation_quality = [r for r in edge_rows if r['missing']==('Q_risk' if lambda_validation else 'observed_only') and r['score_space']=='mu' and r['missing_bin']=='all'
                              and r.get('column_type_pair','all')=='all']
        selection['validation_edge_nll'] = float(np.mean([r['nll'] for r in validation_quality]))
        selection['validation_edge_brier'] = float(np.mean([r['brier'] for r in validation_quality]))
    probability_ci = paired_bootstrap(edge_blocks,bootstrap) if phase in ('edges','pairwise') else []
    decisions_ci = paired_bootstrap(decision_blocks,bootstrap,decision=True) if phase=='pairwise' else []
    if phase=='factorial' and spec.get('factorial_levels'):
        from .factorial_analysis import factorial_4x2
        interactions,interaction_ci=factorial_4x2(rows,names,bootstrap)
    else:
        interactions,interaction_ci = factorial_effects(rows,spec.get('factorial_q','Q_mean'),bootstrap) if phase=='factorial' else ([],[])
    frontier = frontier_rows(rows) if phase in ('frontier','diagnostic-calibration') else []
    actual_rows = [{**r,'optimizer':'none'} for r in edge_rows] if phase=='edges' else rows
    recovery = recovery_rows(rows) if phase=='pairwise' else []
    typed_decisions=[]
    if bootstrap['type_strata']:
        from .isolated_metrics import decision_metrics
        for block in decision_blocks:
            for method,frame in block['frames'].items():
                for kind,part in frame.groupby('column_type_pair',dropna=False):
                    typed_decisions.append({'set':block['set'],'missing':method,'column_type_pair':kind,
                                            **decision_metrics(part)})
    analysis = {'Edge probability quality (before any decisions)':edge_rows,
                'Pairwise decisions by column type':typed_decisions,
                'Pre-mask oracle recovery (pairwise, no BFS or ILP)':recovery,
                'Recovery ratio source-cluster CI':recovery_cluster_ci(recovery,bootstrap),
                'Risk effect on Hungarian accepted edges':risk_changes,
                'Paired edge cluster bootstrap':probability_ci,'Pairwise decisions by table pair':per_pair,
                'Paired pairwise-decision cluster bootstrap':decisions_ci,
                'All prior points and Pareto membership':frontier,'Factorial effects by task':interactions,
                'Recall at prespecified precision targets':recall_at_precision(rows,spec.get('precision_targets',[.7,.8,.9])) if frontier else [],
                'Factorial cluster CI and Holm tests':interaction_ci,
                'Frozen shared prior selection':[selection] if selection else [],
                'TEST-FITTED Platt — in-sample diagnostic, NOT a held-out result':[test_platt] if test_platt else []}
    if not spec.get('factorial_levels'):
        analysis['2x2 interaction by task']=interactions
        analysis['2x2 interaction task-cluster CI']=interaction_ci
    payload = {'run_id':run_id,'name':spec['name']+' '+phase+(' — TEST-FITTED DIAGNOSTIC' if diagnostic else ''),
               'kind':'research_diagnostic' if diagnostic else 'research','config':cfg,
               'protocol':{'role':'diagnostic_test_fit' if diagnostic else 'validation' if validation else 'test',
                    'phase':phase,'evaluation_layer':phase,'fingerprints':[task_signature(s) for s in sets],
                    'shared_candidate_pairs_checked':True,'diagnostic_test_fit':diagnostic,'claim_eligible':not diagnostic,
                    'bootstrap':bootstrap,
                    'note':'No test-selected operating point. Frontier is a prespecified grid, not an edge-ranking AP curve. '
                           'Column CI conditions on one task/fixed predictions; task CI resamples declared task clusters.'},
               'calibration':snapshots,'calibration_sha256':shared_platt,'selection':selection,
               'rows':actual_rows,'summary':isolation_summary(actual_rows),'predictions':predictions,
               'analysis_tables':analysis,'diagnostics':details,'isolated_figures':bool(spec.get('plots',True))}
    from .experiment import environment_info
    payload['environment']=environment_info(str(root))
    write_report(path,payload)
    print('Result:',path)
    return {'result_file':str(path),**payload}


def isolation_plots(payload):
    # 输入：结构化隔离实验结果；输出：仅由本地数值生成的内嵌 PR 前沿 SVG；继承：无（模块函数）。
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = []
    frontier = pd.DataFrame(payload.get('analysis_tables',{}).get('All prior points and Pareto membership',[]))
    if not frontier.empty:
        for (case,condition),block in frontier.groupby(['set','calibration_condition'],sort=False):
            fig,ax = plt.subplots(figsize=(5,3.5),constrained_layout=True)
            for name,points in block.groupby('optimizer'):
                points = points[points.pareto].sort_values('pair_recall')
                ax.plot(points.pair_recall,points.pair_precision,'o-',label=name)
            ax.set(xlabel='Pair recall',ylabel='Pair precision',xlim=(0,1),ylim=(0,1),title=case+' / '+condition)
            ax.legend(); ax.grid(alpha=.2)
            stream=io.StringIO(); fig.savefig(stream,format='svg'); plt.close(fig)
            raw=stream.getvalue(); result.append(raw[raw.index('<svg'):])
    return result
