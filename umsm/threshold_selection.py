"""Validation-only, same-grid decision offsets for complete pipelines."""
import copy
from datetime import datetime
import hashlib

import numpy as np
import pandas as pd
from scipy.special import expit,logit

from .ablation import strategy_config
from .calibration import load_artifact
from .context import RunContext
from .evaluation import evaluate
from .pipeline import build_components,compute_edges
from .registry import OPTIMIZERS
from .report import write_report
from .types import EdgeTable


def offset_graph(graph,offset):
    # Input: calibrated probabilities and additive log-odds offset; output: decision graph, leaving mu/prior intact.
    if not np.isfinite(offset): raise ValueError('Decision offset must be finite')
    frame=graph.df.copy(); frame['a']=frame['a']-offset
    frame['w']=expit(logit(frame.p_tilde.to_numpy(float))-offset)
    return EdgeTable(frame,graph.columns,{**graph.meta,'decision_offset':float(offset)})


def validate_offsets(cfg,root,selection):
    # Input: frozen fit and lambda selection plus validation tasks; output: per-pipeline offsets chosen only on validation.
    from .research import (role_config,ensure_disjoint,calibration_sources,apply_fit_prior,select_parameter,
                           report_folder,source_clusters,task_signature)
    if not selection: raise ValueError('Run validate-lambda before validate-offsets')
    artifact=load_artifact(str(root/cfg['research']['calibration'])); apply_fit_prior(cfg,artifact)
    chosen=select_parameter(selection,cfg,root)
    if chosen['axis']!='lambda': raise ValueError('Expected a frozen validation lambda')
    cfg['ablation']['missing']['Q_risk']['uncertainty_propagation']['params']['risk_lambda']=chosen['value']
    work,validation=role_config(cfg,'validation',root); _,test=role_config(cfg,'test',root)
    site_check=cfg['research'].get('require_site_disjoint',False)
    ensure_disjoint(validation,test,by_site=site_check); ensure_disjoint(validation,calibration_sources(artifact,cfg,root),by_site=site_check)
    clusters=source_clusters(validation); spec=cfg['research']; rows=[]
    levels=spec.get('missing',list(cfg['ablation']['missing'])); optimizers=spec.get('optimizers',['local','global'])
    grid=spec.get('decision_offset_grid',[-2.,-1.,-.5,0.,.5,1.,2.])
    if not grid or any(not np.isfinite(x) for x in grid): raise ValueError('Invalid offset grid')
    ctx=RunContext(seed=cfg['experiment']['seed'],project_root=str(root),cache_dir=str(root/cfg['paths']['cache_dir']))
    for source in validation:
        for missing in levels:
            scfg=strategy_config(cfg,cfg['ablation']['missing'][missing]); scfg['dataset']=copy.deepcopy(work['dataset'])
            comp,real=build_components(scfg); _,_,graph,_=compute_edges(source,comp,real,scfg,ctx)
            for offset in grid:
                adjusted=offset_graph(graph,float(offset))
                for name in optimizers:
                    match=OPTIMIZERS.build(**cfg['ablation']['optimizers'][name]).optimize(adjusted,ctx)
                    rows.append({'set':source.name,'source_cluster':clusters[source.name],'missing':missing,
                                 'optimizer':name,'offset':float(offset),**evaluate(match,source.groundtruth,adjusted.columns)})
    frame=pd.DataFrame(rows)
    grouped=frame.groupby(['missing','optimizer','offset','source_cluster'],sort=True).pair_f1.mean().reset_index()
    grouped=grouped.groupby(['missing','optimizer','offset'],sort=True).pair_f1.mean().reset_index()
    offsets={}
    for (missing,optimizer),block in grouped.groupby(['missing','optimizer'],sort=True):
        best=block.pair_f1.max(); tied=block[block.pair_f1>=best-1e-12]
        point=min(tied.offset,key=lambda x:(abs(x),x))
        offsets.setdefault(missing,{})[optimizer]=float(point)
    eligible=grouped[grouped.missing.isin(['observed_only','stat_mean_mode','stat_knn','stat_mice','stat_missforest','mice_multiple'])]
    eligible=eligible[[float(r.offset)==offsets[r.missing][r.optimizer] for r in eligible.itertuples()]]
    if not len(eligible): raise ValueError('No preregistered pipeline baseline on validation')
    winner=eligible.sort_values(['pair_f1','missing','optimizer','offset'],ascending=[False,True,True,True]).iloc[0]
    frozen={**chosen,'decision_offsets':offsets,'decision_offset_grid':grid,
            'strongest_validation_baseline':{'missing':str(winner.missing),'optimizer':str(winner.optimizer),
                'validation_pair_f1':float(winner.pair_f1),'offset':float(winner.offset)},
            'offset_definition':'a = logit(p_tilde)-logit(fit_positive_prior)-offset',
            'offset_selection':'same grid; macro source-cluster pair F1 on validation; tie closest to zero'}
    run_id=datetime.now().strftime('%Y%m%d-%H%M%S-%f'); path=report_folder(cfg,root)/'validate-offsets'/(run_id+'.html')
    write_report(path,{'run_id':run_id,'name':'Frozen validation decision offsets','kind':'validation',
        'config':cfg,'selection':frozen,'calibration_sha256':hashlib.sha256((root/spec['calibration']).read_bytes()).hexdigest(),
        'protocol':{'role':'validation','phase':'validate-offsets','fingerprints':[task_signature(s) for s in validation]},
        'rows':rows,'analysis_tables':{'Validation grid':rows,'Selected offsets':[{'missing':m,'optimizer':o,'offset':v} for m,d in offsets.items() for o,v in d.items()]}})
    print('Frozen decision offsets:',path,flush=True); return {'result_file':str(path),'selection':frozen,'rows':rows}
