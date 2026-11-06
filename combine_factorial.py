"""Pool explicit held-out factorial reports, regroup source dependence, and render paired results."""
import argparse
from datetime import datetime
import glob
from pathlib import Path

from umsm.factorial_analysis import factorial_4x2,comparable_protocol
from umsm.report import read_report,write_report
from umsm.research import role_config,source_clusters


def combine(patterns, output, root):
    # Input: preregistered factorial report globs; output: one report per matcher with paired effects and source clusters.
    root=Path(root); paths=sorted({Path(p).resolve() for pattern in patterns for p in glob.glob(pattern,recursive=True)})
    if not paths: raise FileNotFoundError('No factorial reports matched')
    # Each phase folder has timestamped reruns; use only its newest report.
    latest={}
    for path in paths:
        if path.parent not in latest or path.stat().st_mtime_ns>latest[path.parent].stat().st_mtime_ns: latest[path.parent]=path
    by_matcher={}
    for path in latest.values():
        report=read_report(path); cfg=report['config']; protocol=report.get('protocol',{})
        if protocol.get('role')!='test' or protocol.get('phase')!='factorial' or protocol.get('diagnostic_test_fit'):
            raise ValueError(f'Expected a held-out factorial report: {path}')
        levels=cfg['research'].get('factorial_levels')
        if not levels: raise ValueError(f'Report does not use 4x2: {path}')
        matcher=cfg['research']['matcher']; group=by_matcher.setdefault(matcher,{'levels':levels,'rows':[],'sources':{},'paths':[],
                                                                               'protocol':comparable_protocol(cfg)})
        if group['levels']!=levels: raise ValueError('Do not pool different factorial protocols')
        if group['protocol']!=comparable_protocol(cfg): raise ValueError('Do not pool different matcher, MICE, Q or validation-grid settings')
        _,sources=role_config(cfg,'test',root)
        for source in sources:
            if source.name in group['sources']: raise ValueError('Duplicate task across reports; choose one protocol per task')
            group['sources'][source.name]=source
        group['rows'].extend(report['rows']); group['paths'].append(str(path))
    all_rows=[]; all_cases=[]
    for matcher,group in sorted(by_matcher.items()):
        clusters=source_clusters(list(group['sources'].values()))
        cases,effects=factorial_4x2(group['rows'],group['levels'],{'unit':'task','resamples':2000,
            'confidence':.95,'seed':2026,'min_clusters':10,'task_groups':clusters})
        all_rows.extend({'matcher':matcher,**r} for r in effects)
        all_cases.extend({'matcher':matcher,**r} for r in cases)
    output=Path(output); output=output if output.is_absolute() else root/output
    write_report(output,{'run_id':datetime.now().strftime('%Y%m%d-%H%M%S-%f'),'name':'Pooled 4x2 factorial comparisons',
        'kind':'factorial_pooled','rows':all_rows,'factorial_figures':True,'analysis_tables':{'Task-level cells and effects':all_cases,
        'Effects with source-cluster confidence intervals and Holm correction':all_rows},
        'protocol':{'reports':[str(p) for p in latest.values()],'test':'two-sided paired source-cluster sign-flip; Holm within each matcher',
          'note':'Task CIs condition on the already fitted models. Cross-validation folds can share fitted sources; task bootstrap does not include model-training variability. 10 source clusters is a minimum, not proof of independence.'}})
    print('Pooled factorial report:',output,flush=True); return str(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--reports',nargs='+',required=True)
    p.add_argument('--output',default='output/results/full_pipeline_factorial.html'); args=p.parse_args()
    combine(args.reports,args.output,Path(__file__).resolve().parent)
