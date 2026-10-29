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
                               paired_frame, pairwise_decisions, quality_rows)
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


def diagnostic_graph(samples, columns, params, original):
    # 输入：已有原始样本、列、测试内 Platt 和正式图；输出：明确带测试拟合标记的诊断图；继承：无（模块函数）。
    frame = samples.df.copy()
    frame['score'] = calibrate_scores(frame.score,params)
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
    keys = [c for c in ('missing','optimizer','score_space','missing_bin','prior','calibration_condition') if c in data]
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
    from .research import task_signature
    spec = cfg['research']
    validation = phase=='validate-prior'
    diagnostic = phase=='diagnostic-calibration'
    if validation and spec.get('prior_strategy','observed_only')!='observed_only':
        raise ValueError('Isolated shared-prior selection is fitted on observed_only only')
    names = (['observed_only'] if validation or phase in ('frontier','diagnostic-calibration') else
             ['observed_only',spec.get('factorial_q','Q_mean')] if phase=='factorial' else spec.get('missing',list(cfg['ablation']['missing'])))
    if phase=='factorial' and names[1] not in ('Q_mean','Q_risk'):
        raise ValueError('factorial_q must be Q_mean or Q_risk')
    bootstrap = spec.get('bootstrap',{})
    rows,predictions,per_pair,edge_rows,details,edge_blocks,decision_blocks,case_graphs = [],[],[],[],[],[],[],[]
    snapshots = {}
    shared_platt = None
    for source in sets:
        frames,decisions = {},{}
        for name in names:
            scfg = strategy_config(cfg,cfg['ablation']['missing'][name])
            scfg['dataset'] = copy.deepcopy(workcfg['dataset'])
            components,realization = build_components(scfg)
            snapshot = calibration_snapshot(components,ctx)
            fingerprint = snapshot['uncertainty_propagation']['sha256']
            if shared_platt is not None and shared_platt!=fingerprint:
                raise ValueError('Isolation requires one shared frozen Platt calibrator for all strategies')
            shared_platt = fingerprint
            snapshots[name] = snapshot
            started = time.perf_counter()
            data,samples,graph,hit = compute_edges(source,components,realization,scfg,ctx)
            compute_time = time.perf_counter()-started
            frame = edge_records(graph,source)
            frames[name] = frame
            tags = {'set':source.name,'variant':workcfg['dataset'].get('variant','original'),'missing':name,
                    'k':len(source.tables),'draw':0,'prior':graph.meta['prior'],'oracle':diagnostic,
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
                continue
            case_graphs.append({'source':source,'graph':graph,'samples':samples if diagnostic else None,'tags':tags})
        validate_shared_graph(frames,names)
        edge_blocks.append({'set':source.name,'frames':frames})
        if decisions:
            decision_blocks.append({'set':source.name,'frames':decisions})
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
    grid = spec.get('prior_grid',[.4,.45,.5]) if validation else spec.get('frontier_priors',[.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95])
    if not grid or any(not np.isfinite(p) or not 0<p<1 for p in grid):
        raise ValueError('Prior grid must be nonempty, finite and inside (0,1)')
    for item in case_graphs:
        source,graph,tags = item['source'],item['graph'],item['tags']
        variants = [('frozen',graph)]
        if diagnostic:
            variants.append(('test_fitted',diagnostic_graph(item['samples'],graph.columns,test_platt,graph)))
        for condition,base in variants:
            priors = grid if validation or phase in ('frontier','diagnostic-calibration') else [base.meta['prior']]
            for prior in dict.fromkeys(priors):
                evidence = prior_graph(base,prior)
                tagged = {**tags,'prior':float(prior),'calibration_condition':condition,
                          'oracle':diagnostic,'test_label_fit':condition=='test_fitted','claim_eligible':not diagnostic}
                if validation:
                    metrics,record,pair_rows,_ = pairwise_decisions(evidence,source)
                    rows.append({**tagged,'optimizer':'pairwise',**metrics})
                    per_pair.extend({**tagged,**r} for r in pair_rows)
                    continue
                for name in ('local','global'):
                    optimizer = OPTIMIZERS.build(**cfg['ablation']['optimizers'][name])
                    started = time.perf_counter()
                    result = optimizer.optimize(evidence,ctx)
                    validate_matching(result,evidence,f'{phase}/{source.name}/{prior}/{name}')
                    record = matching_record(result,source.groundtruth,evidence.columns)
                    rows.append({**tagged,'optimizer':name,'optimizer_time_s':time.perf_counter()-started,
                                 **{k:record[k] for k in ('TP','FP','FN')},**evaluate(result,source.groundtruth),
                                 'all_optimal':result.info.get('all_optimal'),'global_exact':result.info.get('global_exact')})
                    predictions.append({**tagged,'optimizer':name,'matching':record})
    selection = chosen
    if validation:
        scores = pd.DataFrame(rows).groupby('prior').pair_f1.mean()
        best = scores.max()
        value = max(float(p) for p,s in scores.items() if s>=best-1e-12)
        selection = {'axis':'prior','value':value,'score':float(best),'base_strategy':'observed_only',
                     'held_fixed':0.,'decision_protocol':'pairwise','tie_rule':'larger prior within 1e-12',
                     'objective':'macro-task pair_f1 on independent table-pair Hungarian decisions, validation only'}
    probability_ci = paired_bootstrap(edge_blocks,bootstrap) if phase in ('edges','pairwise') else []
    decisions_ci = paired_bootstrap(decision_blocks,bootstrap,decision=True) if phase=='pairwise' else []
    interactions,interaction_ci = factorial_effects(rows,spec.get('factorial_q','Q_mean'),bootstrap) if phase=='factorial' else ([],[])
    frontier = frontier_rows(rows) if phase in ('frontier','diagnostic-calibration') else []
    actual_rows = [{**r,'optimizer':'none'} for r in edge_rows] if phase=='edges' else rows
    analysis = {'Edge probability quality (before any decisions)':edge_rows,
                'Paired edge cluster bootstrap':probability_ci,'Pairwise decisions by table pair':per_pair,
                'Paired pairwise-decision cluster bootstrap':decisions_ci,
                'All prior points and Pareto membership':frontier,'2x2 interaction by task':interactions,
                '2x2 interaction task-cluster CI':interaction_ci,
                'Frozen shared prior selection':[selection] if selection else [],
                'TEST-FITTED Platt — in-sample diagnostic, NOT a held-out result':[test_platt] if test_platt else []}
    payload = {'run_id':run_id,'name':spec['name']+' '+phase+(' — TEST-FITTED DIAGNOSTIC' if diagnostic else ''),
               'kind':'research_diagnostic' if diagnostic else 'research','config':cfg,
               'protocol':{'role':'diagnostic_test_fit' if diagnostic else 'validation' if validation else 'test',
                    'phase':phase,'evaluation_layer':phase,'fingerprints':[task_signature(s) for s in sets],
                    'shared_candidate_pairs_checked':True,'diagnostic_test_fit':diagnostic,'claim_eligible':not diagnostic,
                    'note':'No test-selected operating point. Frontier is a prespecified grid, not an edge-ranking AP curve. '
                           'Column CI conditions on one task/fixed predictions; task CI resamples declared task clusters.'},
               'calibration':snapshots,'calibration_sha256':shared_platt,'selection':selection,
               'rows':actual_rows,'summary':isolation_summary(actual_rows),'predictions':predictions,
               'analysis_tables':analysis,'diagnostics':details,'isolated_figures':bool(spec.get('plots',True))}
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
