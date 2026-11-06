"""Frozen retrieval-aware calibration selection on independent validation data."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .calibration import calibrate_scores, load_artifact
from .context import RunContext
from .llm import llm_stats
from .report import write_report
from .training import collect_matcher_scores


def retriever_bp(graph,source):
    # Input: shared Magneto retrieval evidence; output: direct pairwise bipartite decisions, without BFS.
    from .global_optimizer.local_merge import hungarian_per_table_pair
    from .isolated_metrics import decision_frame,decision_metrics
    from .research_analysis import pair_set
    frame=graph.df.copy()
    if not {'retriever_score','retrieved'}<=set(frame): raise ValueError('Magneto retrieval metadata is missing')
    frame['retriever_weight']=np.where(frame.retrieved>0,frame.retriever_score,0.)
    selected=hungarian_per_table_pair(frame,0.,'retriever_weight')
    accepted={frozenset((a,b)) for a,b in zip(selected.col_i,selected.col_j)}; truth=pair_set(source.groundtruth or [])
    record={'accepted_pairs':[sorted(p) for p in sorted(accepted,key=lambda p:sorted(p))],
            'false_positives':[sorted(p) for p in accepted-truth],'false_negatives':[sorted(p) for p in truth-accepted],
            'semantics':'retrieval-score Hungarian independently per table pair; no BFS; symmetric retrieval union'}
    return decision_metrics(decision_frame(graph,source,accepted)),record


def select_calibrator(cfg, root):
    # Input: fitted candidates and separate validation tasks; output: frozen selected artifact and diagnostic HTML.
    from .research import role_config,ensure_disjoint,calibration_sources,report_folder,task_signature
    root=Path(root); spec=cfg['research']; fit_path=root/spec['calibration']; artifact=load_artifact(str(fit_path))
    candidates=artifact.get('calibration_candidates')
    if not candidates: raise ValueError('Fit retrieval_candidates before selecting a calibrator')
    work,validation=role_config(cfg,'validation',root); _,test=role_config(cfg,'test',root)
    site_check=spec.get('require_site_disjoint',False)
    ensure_disjoint(validation,test,by_site=site_check); ensure_disjoint(validation,calibration_sources(artifact,cfg,root),by_site=site_check)
    ctx=RunContext(seed=int(cfg['experiment']['seed']),project_root=str(root),cache_dir=str(root/cfg['paths']['cache_dir']),
                   embedding_cache=cfg['paths'].get('embedding_cache',True))
    data=collect_matcher_scores(validation,work,artifact,ctx)
    s=np.asarray(data['scores'],float); y=np.asarray(data['labels'],float); w=np.asarray(data['weights'],float)
    r=np.asarray(data['retrieved'],float)
    if not len(s) or r.shape!=s.shape or w.sum()<=0: raise ValueError('Validation retrieval scores are missing')
    w=w/w.sum(); rows=[]; bins=[]
    for name,params in candidates.items():
        p=np.clip(calibrate_scores(s,params,r),1e-12,1-1e-12)
        rows.append({'calibrator':name,'nll':float(-np.dot(w,y*np.log(p)+(1-y)*np.log1p(-p))),
                     'brier':float(np.dot(w,(p-y)**2)),'n_samples':len(s)})
        for i in range(10):
            mask=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
            if not mask.any(): continue
            local=w[mask]/w[mask].sum()
            bins.append({'calibrator':name,'bin':i,'mean_probability':float(np.dot(local,p[mask])),
                         'positive_fraction':float(np.dot(local,y[mask])),'n_samples':int(mask.sum())})
    pool=spec.get('calibrator_selection_pool',['platt_retrieved','isotonic_retrieved'])
    if not pool or any(name not in candidates for name in pool): raise ValueError('Invalid validation selection pool')
    winner=min((row for row in rows if row['calibrator'] in pool),key=lambda row:(row['nll'],pool.index(row['calibrator'])))
    selected=copy.deepcopy(artifact); selected['platt']=copy.deepcopy(candidates[winner['calibrator']])
    selected['calibrator_selection']={'chosen':winner['calibrator'],'selection_metric':'validation equal-edge NLL',
        'selection_pool':pool,'validation_fingerprints':[task_signature(s) for s in validation],
        'source_artifact_sha256':hashlib.sha256(fit_path.read_bytes()).hexdigest(),
        'selected_at_utc':datetime.now(timezone.utc).isoformat()}
    output=fit_path.with_name(fit_path.stem+'_selected.json')
    if output.exists():
        existing=load_artifact(str(output)); prior=existing.get('calibrator_selection',{})
        required=('chosen','selection_pool','validation_fingerprints','source_artifact_sha256')
        if any(prior.get(k)!=selected['calibrator_selection'].get(k) for k in required) or existing.get('platt')!=selected['platt']:
            raise FileExistsError(f'Existing frozen calibration differs: {output}')
        selected=existing  # interrupted publication: reuse only an identical, auditable selection
    output.parent.mkdir(parents=True,exist_ok=True)
    run_id=datetime.now().strftime('%Y%m%d-%H%M%S-%f'); report=report_folder(cfg,root)/'select-calibrator'/(run_id+'.html')
    write_report(report,{'run_id':run_id,'name':'Held-out calibration selection','kind':'calibrator_selection',
        'config':cfg,'rows':rows,'calibration_figures':True,'calibration_bins':bins,
        'protocol':selected['calibrator_selection'],'llm_usage':llm_stats(ctx),
        'analysis_tables':{'Validation NLL and Brier':rows,'Reliability bins':bins},
        'note':'Single-feature Platt is reported as an ablation; the prespecified selection pool determines the final model. Scores and labels are from independent validation tasks.'})
    if not output.exists():
        temp=output.with_suffix('.tmp')
        try:
            temp.write_text(json.dumps(selected,indent=2,ensure_ascii=False),encoding='utf-8'); temp.replace(output)
        finally: temp.unlink(missing_ok=True)
    print('Frozen calibrator:',output,'selected:',winner['calibrator'],flush=True)
    return {'result_file':str(report),'calibration_file':str(output),'selection':selected['calibrator_selection']}


def calibration_plots(payload):
    # Input: held-out weighted calibration bins; output: one SVG reliability plot.
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=payload.get('calibration_bins',[])
    if not rows: return []
    fig,ax=plt.subplots(figsize=(5,4),constrained_layout=True)
    ax.plot([0,1],[0,1],':',color='black',label='ideal')
    for name in dict.fromkeys(r['calibrator'] for r in rows):
        block=sorted((r for r in rows if r['calibrator']==name),key=lambda r:r['mean_probability'])
        ax.plot([r['mean_probability'] for r in block],[r['positive_fraction'] for r in block],'o-',label=name)
    ax.set(xlabel='Predicted probability',ylabel='Observed match fraction',xlim=(0,1),ylim=(0,1)); ax.legend(); ax.grid(alpha=.2)
    buf=io.StringIO(); fig.savefig(buf,format='svg'); plt.close(fig)
    content=buf.getvalue(); return [content[content.index('<svg'):]]
