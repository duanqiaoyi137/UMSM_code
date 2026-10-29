"""Descriptive edge/partition diagnostics; no parameter fitting on test labels."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .global_optimizer.local_merge import hungarian_per_table_pair


def pair_set(groups):
    # 输入：语义组列表；输出：组内全部无向列对集合；继承：无（模块函数）。
    return {frozenset(p) for group in groups for p in combinations(group, 2)}


def edge_records(edges, source, decisions=None):
    # 输入：边表、原表与可选最终分组；输出：逐边标签、自然缺失率、概率与决策表；继承：无（模块函数）。
    frame = edges.df.copy()
    truth = pair_set(source.groundtruth or [])
    rates = {c.id: float(t.df[c.name].isna().mean()) for t in source.tables for c in t.columns}
    frame['label'] = [int(frozenset((a,b)) in truth) for a,b in zip(frame.col_i, frame.col_j)]
    frame['missing_i'] = frame.col_i.map(rates)
    frame['missing_j'] = frame.col_j.map(rates)
    frame['missing_rate'] = frame[['missing_i', 'missing_j']].max(axis=1)
    frame['missing_bin'] = pd.cut(frame.missing_rate, [-1e-12, .2, .5, .8, 1.],
                                  labels=['0-20%', '20-50%', '50-80%', '80-100%']).astype(str)
    if decisions is not None:
        selected = pair_set(decisions.groups)
        frame['decision'] = [int(frozenset((a,b)) in selected) for a,b in zip(frame.col_i, frame.col_j)]
        frame['correct'] = frame.decision == frame.label
    return frame


def probability_metrics(frame, column='mu'):
    # 输入：含 label 与指定概率列的边表；输出：边数、正例数、AUC/AP/Brier/NLL；继承：无（模块函数）。
    if frame.empty:
        return {'n_edges': 0, 'n_positive': 0, 'auc': None, 'average_precision': None, 'brier': None, 'nll': None}
    y, p = frame.label.to_numpy(int), frame[column].to_numpy(float)
    both = len(set(y)) == 2
    return {'n_edges': len(y), 'n_positive': int(y.sum()),
            'auc': float(roc_auc_score(y,p)) if both else None,
            'average_precision': float(average_precision_score(y,p)) if both else None,
            'brier': float(brier_score_loss(y,p)), 'nll': float(log_loss(y,p.clip(1e-12,1-1e-12),labels=[0,1]))}


def missing_strata(frame):
    # 输入：逐边记录；输出：按列对最大缺失率分箱的概率与最终决策指标；继承：无（模块函数）。
    rows = []
    for name, block in frame.groupby('missing_bin', sort=False):
        row = {'missing_bin': name, **probability_metrics(block), 'mean_sigma': float(block.sigma.mean())}
        if 'correct' in block:
            row.update(decision_accuracy=float(block.correct.mean()),
                       TP=int(((block.label==1)&(block.decision==1)).sum()),
                       FP=int(((block.label==0)&(block.decision==1)).sum()),
                       FN=int(((block.label==1)&(block.decision==0)).sum()))
        rows.append(row)
    return rows


def uncertainty_diagnostics(frame, prior=.5):
    # 输入：含 mu/sigma/label 的边表、概率决策阈值；输出：风险覆盖曲线、μ 分层与描述性回归；继承：无（模块函数）。
    if frame.empty:
        return {'risk_coverage': [], 'conditional_errors': [], 'regression': {'status': 'empty'}}
    mu, sigma, y = frame.mu.to_numpy(float), frame.sigma.to_numpy(float), frame.label.to_numpy(int)
    errors = ((mu >= prior).astype(int) != y).astype(int)
    order = np.argsort(sigma, kind='stable')
    tied = pd.DataFrame({'sigma':sigma[order],'error':errors[order]})
    # Expected error under a random order within equal-sigma ties; no spurious ranking at sigma=0.
    expected_errors = tied.groupby('sigma',sort=False).error.transform('mean').to_numpy()
    cumulative = np.cumsum(expected_errors) / np.arange(1,len(frame)+1)
    curve = []
    for coverage in np.linspace(.1, 1., 10):
        n = max(1, int(np.ceil(coverage*len(frame))))
        curve.append({'coverage': n/len(frame), 'risk': float(cumulative[n-1]), 'n_edges': n})
    table = pd.DataFrame({'mu_bin': np.minimum((mu*10).astype(int),9), 'sigma': sigma, 'error': errors})
    conditional = []
    for key, block in table.groupby('mu_bin'):
        # Quantiles are within each mu stratum, rather than confounding certainty with mu.
        codes = pd.qcut(block.sigma, 4, duplicates='drop', labels=False)
        for quartile, group in block.assign(sigma_quartile=codes).dropna().groupby('sigma_quartile'):
            conditional.append({'mu_low': key/10, 'sigma_quartile': int(quartile), 'n_edges': len(group),
                                'mean_sigma': float(group.sigma.mean()), 'error_rate': float(group.error.mean())})
    regression = {'status': 'not_identifiable', 'n_edges': len(frame),
                  'note': 'Descriptive regularized error regression; dependent edges, no iid p-values.'}
    if len(set(errors)) == 2 and np.ptp(sigma) > 1e-12:
        confidence = logit(mu.clip(1e-6,1-1e-6))
        x = np.column_stack([confidence, np.abs(confidence), sigma])
        model = LogisticRegression(C=1., max_iter=1000).fit(StandardScaler().fit_transform(x), errors)
        regression.update(status='fitted', standardized_coefficients=dict(zip(
            ('logit_mu','abs_logit_mu','sigma'), model.coef_[0].tolist())), intercept=float(model.intercept_[0]))
    return {'risk_coverage': curve, 'conditional_errors': conditional, 'regression': regression,
            'aurc': float(cumulative.mean()), 'threshold': prior, 'baseline_error_rate': float(errors.mean())}


def local_conflicts(edges):
    # 输入：含 signed a 的完整边表；输出：BFS 前局部选边的开放楔形与同表分量冲突；继承：无（模块函数）。
    accepted = hungarian_per_table_pair(edges.df, 0., 'a')
    columns = [c.id for c in edges.columns]
    pos = {c:i for i,c in enumerate(columns)}
    ii = [pos[c] for c in accepted.col_i]; jj = [pos[c] for c in accepted.col_j]
    graph = csr_matrix((np.ones(2*len(ii),dtype=np.int64),(ii+jj,jj+ii)), shape=(len(pos),len(pos)))
    degree = np.asarray(graph.sum(axis=1)).ravel()
    wedges = int((degree*(degree-1)//2).sum())
    triangles = int(graph.multiply(graph@graph).sum()//6)
    inconsistent = wedges - 3*triangles
    _, labels = connected_components(graph, directed=False)
    counts = pd.DataFrame({'component': labels, 'table': [c.table for c in edges.columns]}).groupby(['component','table']).size()
    return {'local_accepted_edges': len(accepted), 'inconsistent_triangles_before_bfs': inconsistent,
            'wedge_count': wedges, 'conflict_rate': inconsistent/wedges if wedges else 0.,
            'same_table_pairs_in_local_components': int((counts*(counts-1)//2).sum())}


def partition_objective(matching, edges):
    # 输入：最终分组、ILP 同一套边势；输出：目标值与闭包外配对数；继承：无（模块函数）。
    lookup = {frozenset((a,b)): float(v) for a,b,v in zip(edges.df.col_i,edges.df.col_j,edges.df.a)}
    selected = pair_set(matching.groups)
    return {'partition_objective': sum(lookup[p] for p in selected if p in lookup),
            'pairs_outside_closure': len(selected-set(lookup))}


def research_summary(rows):
    # 输入：逐任务指标记录；输出：按协议条件和方法分组的均值表；继承：无（模块函数）。
    data = pd.DataFrame(rows)
    if data.empty:
        return []
    keys = [c for c in ('missing','optimizer','k','mechanism','rate','row_cap') if c in data]
    metrics = [c for c in ('pair_precision','pair_recall','pair_f1','optimizer_time_s','compute_edges_s',
                          'edge_auc','mean_sigma','objective_gap','n_variables','n_triangle_constraints') if c in data]
    data[metrics] = data[metrics].apply(pd.to_numeric,errors='coerce')
    summary = data.groupby(keys, dropna=False, sort=False)[metrics].mean().reset_index()
    summary['n_cases'] = data.groupby(keys, dropna=False, sort=False).size().to_numpy()
    return summary.to_dict('records')


def render_plots(rows, strata, risk):
    # 输入：任务、缺失分层、风险覆盖记录；输出：嵌入单 HTML 的 matplotlib SVG 图字符串；继承：无（模块函数）。
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = []
    data = pd.DataFrame(rows)
    if not data.empty and {'pair_f1','optimizer','missing','set','draw'} <= set(data):
        pivot = data.pivot_table(index=['set','draw','missing','k'],columns='optimizer',values='pair_f1')
        if {'global','local'} <= set(pivot):
            paired = pivot.dropna(subset=['global','local']).copy()
            paired['delta'] = 100*(paired['global']-paired['local'])
            covariates = data[data.optimizer=='local'].set_index(['set','draw','missing','k'])
            paired = paired.join(covariates[['edge_auc','conflict_rate']], how='left')
            for x in ('k','edge_auc','conflict_rate'):
                clean = paired.reset_index().dropna(subset=[x,'delta'])
                if len(clean):
                    panels.append((x, 'ILP - local Pair-F1 (pp)', clean[x],clean.delta,None))
    if strata:
        frame = pd.DataFrame(strata)
        for method, block in frame.groupby('missing'):
            block = block.groupby('missing_bin',sort=False).brier.mean().reindex(['0-20%','20-50%','50-80%','80-100%']).dropna()
            panels.append(('Missingness bin ('+method+')','Brier',np.arange(len(block)),block.to_numpy(),block.index.tolist()))
    if risk:
        frame = pd.DataFrame(risk)
        for method, block in frame.groupby('missing'):
            block = block.groupby('coverage').risk.mean()
            panels.append(('Coverage ('+method+')','Error risk',block.index.to_numpy(),block.to_numpy(),None))
    figures = []
    for xlabel,ylabel,x,y,ticks in panels:
        fig, ax = plt.subplots(figsize=(5,3.2), constrained_layout=True)
        ax.scatter(x,y,s=15,alpha=.65)
        ax.set(xlabel=xlabel,ylabel=ylabel)
        if ticks:
            ax.set_xticks(x,ticks)
        ax.grid(alpha=.2)
        buffer = io.StringIO()
        fig.savefig(buffer,format='svg')
        plt.close(fig)
        svg = buffer.getvalue()
        figures.append(svg[svg.index('<svg'):])
    return figures
