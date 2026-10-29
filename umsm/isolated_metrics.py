"""Decision-free probabilities and paired cluster uncertainty; never runs a matcher/ILP."""
from itertools import combinations

import numpy as np
import pandas as pd

from .global_optimizer.local_merge import hungarian_per_table_pair
from .research_analysis import pair_set, probability_metrics

BINS = ('all','0-20%','20-50%','50-80%','80-100%')


def decision_metrics(frame, weights=None):
    # 输入：含 label/decision 的评估边及可选簇权重；输出：逐对 TP/FP/FN/P/R/F1；继承：无（模块函数）。
    y, p = frame.label.to_numpy(bool), frame.decision.to_numpy(bool)
    w = np.ones(len(frame)) if weights is None else np.asarray(weights,float)
    tp,fp,fn = float(w[y&p].sum()),float(w[~y&p].sum()),float(w[y&~p].sum())
    precision,recall = tp/(tp+fp) if tp+fp else 0.,tp/(tp+fn) if tp+fn else 0.
    return {'TP':tp,'FP':fp,'FN':fn,'pair_precision':precision,'pair_recall':recall,
            'pair_f1':2*precision*recall/(precision+recall) if precision+recall else 0.}


def decision_frame(graph, source, accepted):
    # 输入：闭包边、真值原表、直接接受的无向对；输出：决策评估表（闭包外真边计 FN）；继承：无（模块函数）。
    from .research_analysis import edge_records
    frame = edge_records(graph,source)
    present = {frozenset((a,b)) for a,b in zip(frame.col_i,frame.col_j)}
    owner = {c.id:c.table for c in source.columns}
    # These rows are evaluation false negatives, never fabricated scoring/optimization evidence.
    absent = pair_set(source.groundtruth or [])-present
    if absent:
        extra = pd.DataFrame([{'col_i':a,'col_j':b,'table_i':owner[a],'table_j':owner[b],'label':1}
                              for a,b in (sorted(p) for p in absent)])
        rates = {c.id:float(t.df[c.name].isna().mean()) for t in source.tables for c in t.columns}
        extra['missing_bin'] = pd.cut([max(rates[a],rates[b]) for a,b in zip(extra.col_i,extra.col_j)],
                                      [-1e-12,.2,.5,.8,1.],labels=BINS[1:]).astype(str)
        frame = pd.concat([frame,extra],ignore_index=True)
    frame['decision'] = [int(frozenset((a,b)) in accepted) for a,b in zip(frame.col_i,frame.col_j)]
    return frame


def pairwise_decisions(graph, source):
    # 输入：含 signed a 的边表和真值原表；输出：无 BFS 的 Hungarian 接受对、指标、逐表对记录及评估表；继承：无（模块函数）。
    if 'a' not in graph.df or not np.isfinite(graph.df.a.to_numpy(float)).all():
        raise ValueError('Pairwise Hungarian requires finite signed a')
    accepted_df = hungarian_per_table_pair(graph.df,0.,'a')
    accepted = {frozenset((a,b)) for a,b in zip(accepted_df.col_i,accepted_df.col_j)}
    frame = decision_frame(graph,source,accepted)
    rows = []
    for a,b in combinations([t.name for t in source.tables],2):
        block = frame[((frame.table_i==a)&(frame.table_j==b))|((frame.table_i==b)&(frame.table_j==a))]
        rows.append({'table_i':a,'table_j':b,**decision_metrics(block)})
    truth = pair_set(source.groundtruth or [])
    record = {'accepted_pairs':[sorted(p) for p in sorted(accepted,key=lambda p:sorted(p))],
              'false_positives':[sorted(p) for p in accepted-truth],
              'false_negatives':[sorted(p) for p in truth-accepted],
              'semantics':'independent table-pair assignments; no transitive closure and no partition groups'}
    return decision_metrics(frame),record,rows,frame


def quality_rows(frame, tags):
    # 输入：含 mu/p_tilde 的闭包表与任务标识；输出：两种概率空间、所有缺失层的 AUC/AP/Brier/NLL；继承：无（模块函数）。
    rows = []
    for space in ('mu','p_tilde'):
        for name in BINS:
            block = frame if name=='all' else frame[frame.missing_bin==name]
            rows.append({**tags,'score_space':space,'missing_bin':name,**probability_metrics(block,space),
                         'mean_sigma':float(block.sigma.mean()) if len(block) else None,
                         'max_sigma':float(block.sigma.max()) if len(block) else None,
                         'mean_abs_risk_shift':float((block.mu-block.p_tilde).abs().mean()) if len(block) else None})
    return rows


def probability_weights(frame, column, weights=None):
    # 输入：标签、指定概率列、非负簇权重；输出：加权 AUC/AP/Brier/NLL；继承：无（模块函数）。
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    w = np.ones(len(frame)) if weights is None else np.asarray(weights,float)
    keep = w>0
    if not keep.any():
        return {k:np.nan for k in ('auc','average_precision','brier','nll')}
    y,p,w = frame.label.to_numpy(int)[keep],frame[column].to_numpy(float)[keep],w[keep]
    both = len(set(y))==2
    return {'auc':float(roc_auc_score(y,p,sample_weight=w)) if both else np.nan,
            'average_precision':float(average_precision_score(y,p,sample_weight=w)) if both else np.nan,
            'brier':float(brier_score_loss(y,p,sample_weight=w)),
            'nll':float(log_loss(y,p.clip(1e-12,1-1e-12),labels=[0,1],sample_weight=w))}


def paired_frame(method, baseline):
    # 输入：同一任务的两个方法边表；输出：按无向列对严格对齐的两表；继承：无（模块函数）。
    def keyed(frame):
        # 输入：一份边表；输出：以规范无向对为索引的副本；继承：无（内部函数）。
        result = frame.copy()
        result.index = [tuple(sorted((a,b))) for a,b in zip(frame.col_i,frame.col_j)]
        if not result.index.is_unique:
            raise ValueError('Duplicate evaluation pairs')
        return result
    a,b = keyed(method),keyed(baseline)
    if set(a.index)!=set(b.index):
        raise ValueError('Controlled contrasts require exactly the same candidate/evaluation pairs')
    b = b.reindex(a.index)
    if not np.array_equal(a.label,b.label) or not np.array_equal(a.missing_bin,b.missing_bin):
        raise ValueError('Controlled contrasts require identical labels and missingness bins')
    return a.reset_index(drop=True),b.reset_index(drop=True)


def paired_bootstrap(blocks, settings, decision=False):
    # 输入：同任务方法/基线边表、簇重采样设置；输出：配对差值百分位区间，单任务不伪造任务区间；继承：无（模块函数）。
    unit,n,seed = settings.get('unit','task'),int(settings.get('resamples',500)),int(settings.get('seed',2026))
    confidence = float(settings.get('confidence',.95))
    if unit not in ('task','column') or n<0 or not 0<confidence<1:
        raise ValueError('Bootstrap requires unit task/column, nonnegative resamples and confidence in (0,1)')
    if unit=='column' and len(blocks)!=1:
        raise ValueError('Column bootstrap is conditional on one task; use task bootstrap for multiple tasks')
    contrasts = ([('Q_mean-observed','Q_mean','decision','observed_only','decision'),
                  ('Q_mean-top1','Q_mean','decision','top1','decision'),
                  ('risk-mean','Q_risk','decision','Q_mean','decision')] if decision else
                 [('Q_mean-observed','Q_mean','mu','observed_only','mu'),
                  ('Q_mean-top1','Q_mean','mu','top1','mu'),
                  ('risk-mean','Q_risk','p_tilde','Q_risk','mu')])
    metrics = ('pair_precision','pair_recall','pair_f1') if decision else ('auc','average_precision','brier','nll')
    output = []
    for contrast,method,space,baseline,bspace in contrasts:
        eligible = [b for b in blocks if method in b['frames'] and baseline in b['frames']]
        if not eligible:
            continue
        aligned = [paired_frame(b['frames'][method],b['frames'][baseline]) for b in eligible]
        group_names = [settings.get('task_groups',{}).get(b['set'],b['set']) for b in eligible]
        groups = list(dict.fromkeys(group_names))
        for bin_name in BINS:
            local = [(a,b,np.ones(len(a),bool) if bin_name=='all' else a.missing_bin.to_numpy()==bin_name) for a,b in aligned]
            def differences(weights):
                # 输入：各任务共享的方法/基线边权重；输出：各任务的配对指标差；继承：无（内部函数）。
                values = []
                for (a,b,mask),w in zip(local,weights):
                    if not mask.any():
                        values.append({metric:np.nan for metric in metrics}); continue
                    left = decision_metrics(a.loc[mask],w[mask]) if decision else probability_weights(a.loc[mask],space,w[mask])
                    right = decision_metrics(b.loc[mask],w[mask]) if decision else probability_weights(b.loc[mask],bspace,w[mask])
                    values.append({metric:left[metric]-right[metric] for metric in metrics})
                return pd.DataFrame(values)
            point = differences([np.ones(len(a)) for a,_ in aligned])
            samples = {metric:[] for metric in metrics}
            rng = np.random.default_rng(seed)
            columns = []
            if unit=='column':
                columns = list(dict.fromkeys(c for p in zip(aligned[0][0].col_i,aligned[0][0].col_j) for c in p))
                owners = {c:t for a in (aligned[0][0],) for c,t in list(zip(a.col_i,a.table_i))+list(zip(a.col_j,a.table_j))}
                by_table = [[c for c in columns if owners[c]==table] for table in dict.fromkeys(owners.values())]
            can_sample = n>0 and (len(groups)>=2 if unit=='task' else len(columns)>=2)
            for _ in range(n if can_sample else 0):
                if unit=='task':
                    picked = rng.choice(groups,size=len(groups),replace=True)
                    counts = {g:int((picked==g).sum()) for g in groups}
                    task_weights = np.array([counts[g] for g in group_names],float)
                    values = {metric:np.average(point.loc[point[metric].notna(),metric],weights=task_weights[point[metric].notna()])
                              if task_weights[point[metric].notna()].sum()>0 else np.nan for metric in metrics}
                else:
                    counts = {}
                    for cols in by_table:
                        selected = rng.choice(cols,size=len(cols),replace=True)
                        counts.update({c:int((selected==c).sum()) for c in cols})
                    a = aligned[0][0]
                    weight = np.array([counts[x]*counts[y] for x,y in zip(a.col_i,a.col_j)],float)
                    values = differences([weight]).iloc[0].to_dict()
                for metric in metrics:
                    samples[metric].append(values[metric])
            for metric in metrics:
                valid = np.array([x for x in samples[metric] if np.isfinite(x)],float)
                valid_clusters = len({g for g,v in zip(group_names,point[metric].notna()) if v})
                enough = len(valid)>=max(20,int(.5*n)) and (unit!='task' or valid_clusters>=2)
                alpha = (1-confidence)/2
                lo,hi = np.quantile(valid,[alpha,1-alpha]) if enough else (None,None)
                output.append({'contrast':contrast,'method_space':space,'baseline_space':bspace,'missing_bin':bin_name,
                    'metric':metric,'difference':float(point[metric].mean()) if point[metric].notna().any() else None,
                    'ci_low':lo,'ci_high':hi,'confidence':confidence,'unit':unit,'n_tasks':len(eligible),
                    'n_clusters':len(groups) if unit=='task' else len(columns),'n_valid_tasks':int(point[metric].notna().sum()),
                    'n_valid_task_clusters':valid_clusters,
                    'resamples':n,'valid_resamples':len(valid),'status':'ok' if enough else 'insufficient_clusters_or_valid_resamples',
                    'better_direction':'negative' if metric in ('brier','nll') else 'positive',
                    'scope':'macro-task metric; repeated task clusters' if unit=='task' else 'conditional fixed-prediction, table-stratified two-endpoint column bootstrap'})
    return output


def frontier_rows(rows):
    # 输入：预先固定 prior 网格上的决策指标；输出：保留全部点并标记非支配 PR 点；继承：无（模块函数）。
    result = []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return result
    for _,block in frame.groupby(['set','missing','optimizer','calibration_condition'],sort=False):
        for _,row in block.iterrows():
            dominated = ((block.pair_precision>=row.pair_precision)&(block.pair_recall>=row.pair_recall)&
                         ((block.pair_precision>row.pair_precision)|(block.pair_recall>row.pair_recall))).any()
            result.append({**row.to_dict(),'pareto':not bool(dominated)})
    return result


def factorial_effects(rows, q='Q_mean', settings=None):
    # 输入：同任务 2×2 决策结果、Q 策略、任务簇设置；输出：四格 F1、主效应、交互差及任务区间；继承：无（模块函数）。
    effects = []
    data = pd.DataFrame(rows)
    for key,block in data.groupby(['set','draw','k'],sort=False):
        values = {(r.missing,r.optimizer):r.pair_f1 for r in block.itertuples()}
        needed = [('observed_only','local'),('observed_only','global'),(q,'local'),(q,'global')]
        if not all(p in values for p in needed):
            raise ValueError('Factorial interaction requires all four cells on each identical task')
        ol,og,ql,qg = [values[p] for p in needed]
        effects.append({'set':key[0],'draw':key[1],'k':key[2],'q_strategy':q,'observed_local':ol,'observed_ilp':og,
                        'q_local':ql,'q_ilp':qg,'imputation_at_local':ql-ol,'imputation_at_ilp':qg-og,
                        'ilp_without_imputation':og-ol,'ilp_with_imputation':qg-ql,'interaction':(qg-og)-(ql-ol)})
    settings = settings or {}
    clusters = list(dict.fromkeys(settings.get('task_groups',{}).get(r['set'],r['set']) for r in effects))
    samples = []
    rng = np.random.default_rng(settings.get('seed',2026))
    if len(clusters)>=2:
        for _ in range(settings.get('resamples',500)):
            chosen = rng.choice(clusters,size=len(clusters),replace=True)
            weights = [sum(g==settings.get('task_groups',{}).get(r['set'],r['set']) for g in chosen) for r in effects]
            samples.append(float(np.average([r['interaction'] for r in effects],weights=weights)))
    alpha = (1-settings.get('confidence',.95))/2
    lo,hi = np.quantile(samples,[alpha,1-alpha]) if len(samples)>=20 else (None,None)
    summary = {'n_tasks':len(effects),'n_clusters':len(clusters),'mean_interaction':float(np.mean([r['interaction'] for r in effects])),
               'ci_low':lo,'ci_high':hi,'unit':'task','status':'ok' if lo is not None else 'insufficient_clusters_or_resamples',
               'definition':'(Q_ILP - observed_ILP) - (Q_local - observed_local); F1 fractions'}
    return effects,[summary]
