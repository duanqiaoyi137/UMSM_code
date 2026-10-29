"""One self-contained HTML result: readable tables plus the complete JSON payload."""
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _clean(value):
    # 输入：value；输出：可安全序列化的值；继承：无（模块函数）。
    if isinstance(value, pd.DataFrame):
        return _clean(value.to_dict('records'))
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def matching_record(matching, groundtruth=None, columns=None):
    # 输入：Matching、真值和可选全部列；输出：含单例的完整 Ω、非单例组及 TP/FP/FN；继承：无（模块函数）。
    """Keep decisions and pair errors, so a small report still supports error analysis."""
    from itertools import combinations
    record = {'groups': [sorted(g) for g in matching.groups], 'info': matching.info}
    if columns is not None:
        record['partition'] = [sorted(g) for g in matching.partition(columns)]
        record['unmatched_columns'] = [g[0] for g in record['partition'] if len(g)==1]
    if groundtruth is not None:
        pairs = lambda groups: {tuple(sorted(p)) for g in groups for p in combinations(g, 2)}
        predicted, truth = pairs(matching.groups), pairs(groundtruth)
        record.update(TP=len(predicted & truth), FP=len(predicted - truth), FN=len(truth - predicted),
                      false_positives=sorted(predicted - truth), false_negatives=sorted(truth - predicted))
    return record


def diagnostics(data, samples, edges, cfg):
    # 输入：data、samples、edges、cfg；输出：按开关嵌入 HTML 的诊断数组；继承：无（模块函数）。
    """Optional large arrays stay inside the same report; no sidecar files."""
    exp, result = cfg['experiment'], {}
    if exp.get('save_edges'):
        result['edges'] = edges.df
        result['blocking'] = samples.meta.get('candidates', {})
    if exp.get('save_score_samples'):
        result['score_samples'] = samples.df
    if exp.get('save_uncertainty'):
        result['distributions'] = [
            {'table': t.name, 'row': int(r), 'column': c, 'candidates': t.distributions.get((r, c), []),
             **t.distribution_info.get((r, c), {})}
            for t in data.tables for r, c in sorted(set(t.distributions) | set(t.distribution_info))]
        result['column_sampling'] = data.sampling_records()
    if exp.get('save_realizations'):
        result['realizations'] = {c.id: [vars(s) for s in data.states(c)] for c in data.columns}
    return result


def _table(rows, columns=None):
    # 输入：rows、columns；输出：HTML 表格字符串；继承：无（模块函数）。
    frame = pd.DataFrame(rows)
    if frame.empty:
        return '<p>No entries.</p>'
    if columns:
        frame = frame[[c for c in columns if c in frame]]
    frame = frame.fillna('—')
    return '<div class="scroll">' + frame.to_html(index=False, escape=True, border=0,
              na_rep='—', float_format=lambda x: f'{x:.3g}' if 0<abs(x)<1e-4 else f'{x:.4f}') + '</div>'


def write_report(path, payload):
    # 输入：path、payload；输出：原子写入的单个 HTML 路径；继承：无（模块函数）。
    """Atomic write; escaping prevents dataset text from becoming active HTML/script."""
    path = Path(path)
    payload = _clean({'format': 'umsm-report-v1', **payload})
    title = html.escape(payload.get('name') or payload['run_id'])
    sections = [f'<h1>{title}</h1>', '<p>UMSM · ' + html.escape(payload['run_id']) + '</p>',
                '<p>Pair precision / recall / F1 are fractions in [0,1]. Delta_pp is percentage points. '
                'Expand a case to inspect groups and individual false positives/negatives.</p>']
    if payload.get('protocol',{}).get('diagnostic_test_fit'):
        sections.append('<p style="border:2px solid #a33;padding:12px;color:#a00"><strong>TEST-FITTED CALIBRATION DIAGNOSTIC. '
                        'In-sample labels are used. This is not a held-out result or a guaranteed F1 upper bound. '
                        'Do not use this report to select formal parameters.</strong></p>')
    for heading, key in [('Frozen Q validation', 'qx_diagnostics'), ('Summary', 'summary'), ('Component effects', 'component_effects'),
                         ('Reference comparisons', 'comparisons'), ('Thresholds (validation only)', 'thresholds')]:
        if payload.get(key):
            columns = None
            if key == 'summary':
                columns = ['missing', 'optimizer', 'score_space', 'missing_bin', 'prior', 'calibration_condition',
                           'auc', 'average_precision', 'brier', 'nll', 'pair_precision_mean', 'pair_recall_mean',
                           'pair_f1_mean', 'pair_f1_std', 'n_cases', 'pair_precision', 'pair_recall', 'pair_f1', 'n_sets']
            elif key in ('component_effects', 'comparisons'):
                columns = ['component', 'claim', 'method', 'baseline', 'reference', 'other',
                           'n', 'delta_pp', 'mean_diff', 'wins', 'ties', 'losses']
            sections += [f'<h2>{heading}</h2>', _table(payload[key], columns)]
    if payload.get('rows'):
        case_heading = ('Diagnostic cases (test-fitted; not held-out)' if payload.get('protocol',{}).get('diagnostic_test_fit')
                        else 'Validation cases' if payload.get('protocol', {}).get('role') == 'validation' else 'Test cases')
        sections += [f'<h2>{case_heading}</h2>', _table(payload['rows'], ['missing', 'optimizer', 'variant', 'set',
                     'score_space', 'missing_bin', 'prior', 'calibration_condition', 'auc', 'average_precision', 'brier', 'nll',
                     'TP', 'FP', 'FN', 'pair_precision', 'pair_recall', 'pair_f1', 'bcubed_f1',
                     'calibration_brier', 'calibration_nll', 'oracle'])]
    for case in payload.get('predictions', []):
        label = ' / '.join(str(case[k]) for k in ('missing', 'optimizer', 'variant', 'set', 'prior', 'calibration_condition') if k in case)
        content = html.escape(json.dumps(case.get('matching',case.get('pairwise_decisions')), indent=2, ensure_ascii=False))
        sections.append(f'<details><summary>{html.escape(label)}</summary><pre>{content}</pre></details>')
    for heading, rows in payload.get('analysis_tables', {}).items():
        sections += [f'<details><summary>{html.escape(heading)}</summary>', _table(rows), '</details>']
    if payload.get('figures'):
        # Figures are generated by our numeric plotting code; never render raw supplied SVG.
        from .research_analysis import render_plots
        plots = render_plots(payload.get('rows', []),
                             payload.get('analysis_tables', {}).get('Missingness strata', []),
                             payload.get('analysis_tables', {}).get('Risk coverage', []))
        sections += ['<h2>Research diagnostics</h2>', *plots]
    if payload.get('isolated_figures'):
        from .isolation import isolation_plots
        plots = isolation_plots(payload)
        if plots:
            sections += ['<h2>Prespecified precision-recall frontier</h2>', *plots]
    sections += ['<h2>Configuration, calibration and diagnostics</h2>',
                 '<details><summary>Complete machine-readable result</summary><pre id="full-result"></pre></details>']
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    document = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{title}</title><style>'
                'body{font:15px/1.6 system-ui,sans-serif;max-width:1300px;margin:32px auto;padding:0 20px;color:#172434}'
                'h1,h2{line-height:1.3}.scroll{overflow:auto}table{border-collapse:collapse;font-variant-numeric:tabular-nums}'
                'td,th{padding:8px 12px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:right}'
                'th{background:#edf3f8}td:first-child,th:first-child{text-align:left}'
                'summary{cursor:pointer;padding:10px;background:#f3f6f8;margin-top:8px}'
                'pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}</style><body>'
                + ''.join(sections) + f'<script id="umsm-result" type="application/json">{encoded}</script>'
                '<script>document.getElementById("full-result").textContent=JSON.stringify('
                'JSON.parse(document.getElementById("umsm-result").textContent),null,2);</script></body></html>')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    try:
        temporary.write_text(document, encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def read_report(path):
    # 输入：path；输出：HTML 内嵌完整结果字典；继承：无（模块函数）。
    import re
    text = Path(path).read_text(encoding='utf-8')
    match = re.search(r'<script id="umsm-result" type="application/json">(.*?)</script>', text, re.S)
    if match is None:
        raise ValueError(f'{path}: not a UMSM result report')
    return json.loads(match.group(1))
