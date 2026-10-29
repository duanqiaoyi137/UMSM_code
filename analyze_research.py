"""Combine research HTMLs without rerunning matchers or APIs; one analysis HTML output."""
import argparse
from datetime import datetime
import glob
from pathlib import Path

import pandas as pd

from umsm.report import read_report, write_report
from umsm.research_analysis import research_summary
from umsm.isolation import isolation_summary


def analyze(paths, output):
    # 输入：报告路径/通配符、输出 HTML 路径；输出：跨 matcher/阶段描述统计与单个 HTML；继承：无（模块函数）。
    files = sorted({Path(p).resolve() for pattern in paths for p in glob.glob(pattern,recursive=True)})
    if not files:
        raise FileNotFoundError('No matching research HTML files')
    effects, tables, details, sources = [],{},[],[]
    has_isolated = False
    for path in files:
        report = read_report(path)
        if report.get('kind')!='research':
            raise ValueError(f'{path}: select research.py reports, not legacy/other ablations')
        cfg=report['config']['research']
        label=cfg['name']+'/'+cfg['fold']+'/'+cfg['matcher']+'/'+report['protocol']['phase']+'/'+report['run_id']
        sources.append({'report':str(path),'protocol':report['protocol'],'config':report['config'],
                        'calibration_sha256':report['calibration_sha256']})
        block=pd.DataFrame(report['rows'])
        isolated = report['protocol'].get('evaluation_layer') is not None
        has_isolated = has_isolated or isolated
        # Pair within each exact run/case, never pair unrelated source populations by position.
        keys=['set','draw','k','missing']+[c for c in ('prior','calibration_condition') if c in block]
        for case,part in block.groupby(keys,sort=False):
            local=part[part.optimizer=='local']; global_=part[part.optimizer=='global']
            if len(local)==1 and len(global_)==1:
                a,b=local.iloc[0],global_.iloc[0]
                effects.append({'report':label,**dict(zip(keys,case)), 'matcher':cfg['matcher'],
                    'delta_f1_pp':100*(b.pair_f1-a.pair_f1),'edge_auc':a.get('edge_auc'),
                    'conflict_rate':a.get('conflict_rate'),'objective_gap':a.get('objective_gap')})
        tables[label+' summary']=(isolation_summary if isolated else research_summary)(report['rows'])
        for name,values in report.get('analysis_tables',{}).items():
            if isolated or name in ('Missingness strata','Uncertainty regressions','Conditional errors','Risk coverage'):
                tables[label+' / '+name]=values
        # Prefix methods for SVG facet isolation; repeated files never merge case keys silently.
        for row in report['rows']:
            details.append({**row,'missing':label+' / '+row['missing']})
    effect_frame=pd.DataFrame(effects)
    means=[]
    if len(effect_frame):
        group_keys=['report','missing']+[c for c in ('prior','calibration_condition') if c in effect_frame]
        for group,block in effect_frame.groupby(group_keys,sort=False,dropna=False):
            means.append({**dict(zip(group_keys,group)),'n_paired_cases':len(block),
                          'n_source_tasks':block['set'].nunique(),'delta_f1_pp':block.delta_f1_pp.mean(),
                          'wins':int((block.delta_f1_pp>1e-10).sum()),'ties':int((block.delta_f1_pp.abs()<=1e-10).sum()),
                          'losses':int((block.delta_f1_pp < -1e-10).sum())})
    payload={'run_id':datetime.now().strftime('%Y%m%d-%H%M%S-%f'),'name':'Research comparison',
             'kind':'research_analysis','sources':sources,'rows':details,'figures':not has_isolated,
             'analysis_tables':{'Paired ILP-local effects':effects,'Paired descriptive means':means,**tables},
             'note':'No pooled iid significance test: source subsets and MC seeds overlap. '
                    'Rows remain separated by report/matcher/protocol/prior. No test-grid best point is selected. '
                    'Cluster CIs are retained from each input; this overview does not pool folds as independent tasks. '
                    'Raw inputs and frozen parameters are embedded.'}
    write_report(output,payload)
    print(pd.DataFrame(means).to_string(index=False))
    print('Result:',output)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',nargs='+',required=True,help='Quote glob patterns, e.g. results/research/**/*.html')
    parser.add_argument('--output',default='results/research_comparison.html')
    args=parser.parse_args()
    analyze(args.reports,args.output)
