"""Server study diagnostics: isolated resource workers, constraint ablations and outer union."""
import copy
from dataclasses import replace
from datetime import datetime
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

from .ablation import strategy_config
from .calibration import load_artifact
from .context import RunContext
from .evaluation import evaluate,validate_matching
from .pipeline import build_components,compute_edges,load_sets
from .profiling import Profiler
from .registry import OPTIMIZERS
from .report import write_report
from .research_analysis import pair_set,local_conflicts
from .isolated_metrics import decision_frame,decision_metrics


def triangle_counts(columns, accepted):
    # Input: column IDs and accepted undirected edges; output: open wedges/total wedges, not BFS partition artifacts; module function.
    ids=[c.id for c in columns]; positions={c:i for i,c in enumerate(ids)}; ii=[]; jj=[]
    for pair in accepted:
        a,b=tuple(pair); ii.extend((positions[a],positions[b])); jj.extend((positions[b],positions[a]))
    matrix=csr_matrix((np.ones(len(ii),np.int64),(ii,jj)),shape=(len(ids),len(ids)))
    common=(matrix@matrix).tocoo(); adjacency=set(zip(ii,jj))
    violations=sum(int(v) for i,j,v in zip(common.row,common.col,common.data) if i<j and (i,j) not in adjacency)
    degree=np.asarray(matrix.sum(axis=1)).ravel(); wedges=int(np.sum(degree*(degree-1)//2))
    return {'open_wedges':violations,'wedges':wedges,'triangle_violation_rate':violations/wedges if wedges else 0.}


def no_same_table_constraint(graph,ctx,params):
    # Input: fixed full-closure graph; output: transitive ILP partition with same-table variables free and zero potential.
    from .global_optimizer.ilp import complete_component,get_components,pair_lookup
    from .types import Matching
    ids=[c.id for c in graph.columns]; owners={c.id:c.table for c in graph.columns}
    weights=pair_lookup(graph.df,'w'); potentials=pair_lookup(graph.df,'a'); candidates=set(weights)
    components=get_components(ids,graph.df); result=[]; diagnostics=[]
    for cols in components:
        problem=complete_component(cols,owners,weights,candidates,0.,None,a_lookup=potentials)
        problem.same_table[:]=False
        solver=OPTIMIZERS.build('ilp_logit',params); y=solver._solve_component(problem)
        parent=list(range(len(cols)))
        def find(x):
            while parent[x]!=x:
                parent[x]=parent[parent[x]]; x=parent[x]
            return x
        for index in np.flatnonzero(y):
            a,b=find(int(problem.I[index])),find(int(problem.J[index])); parent[a]=b
        groups={}
        for i,col in enumerate(cols): groups.setdefault(find(i),set()).add(col)
        result.extend(frozenset(g) for g in groups.values() if len(g)>1); diagnostics.append(problem.stats)
    conflicts=sum(len(g)-len({owners[c] for c in g}) for g in result)
    return Matching(result,{'same_table_conflicts':conflicts,'problem_feasible':conflicts==0,
        'all_optimal':all(r['status']==0 for r in diagnostics),'components':diagnostics})


def outer_union_metrics(source, matching):
    # Input: original observed cells and predicted schema; output: oracle-aligned cell accuracy/count/mean query errors; module function.
    from .uncertainty_model.typed import profiles
    truth_groups=[set(g) for g in source.groundtruth or []]; covered=set().union(*truth_groups) if truth_groups else set()
    truth_groups.extend({c.id} for c in source.columns if c.id not in covered)
    truth={c:i for i,g in enumerate(truth_groups) for c in g}
    partition=[set(g) for g in matching.partition(source.columns)]; pred={c:i for i,g in enumerate(partition) for c in g}
    weights=np.zeros((len(partition),len(truth_groups))); counts=np.zeros(len(truth_groups)); values={i:[] for i in range(len(truth_groups))}
    pred_counts=np.zeros(len(partition)); pred_values={i:[] for i in range(len(partition))}
    split=source.meta.get('preparation',{}).get('split',{}).get('columns',{})
    for table in source.tables:
        prof=profiles(table.df,table.name)
        for column in table.columns:
            series=table.df[column.name].dropna(); actual=truth[column.id]; assigned=pred[column.id]
            n=len(series); weights[assigned,actual]+=n; counts[actual]+=n; pred_counts[assigned]+=n
            if prof[column.name].kind in ('numeric','ordered_numeric','date'):
                meta=split.get(table.name,{}).get(column.name,{})
                parser=copy.copy(prof[column.name])
                if parser.kind=='date' and meta.get('date_format'): parser.date_format=meta['date_format']
                numbers=[parser.number(v)/meta.get('format_scale',1.) for v in series]
                numbers=[v for v in numbers if np.isfinite(v)]
                values[actual].extend(numbers); pred_values[assigned].extend(numbers)
    rows,cols=linear_sum_assignment(-weights); alignment=dict(zip(cols,rows))
    correct=float(weights[rows,cols].sum()); total=float(counts.sum()); queries=[]
    for g in range(len(truth_groups)):
        assigned=alignment.get(g); predicted=pred_counts[assigned] if assigned is not None else 0.
        record={'semantic_columns':sorted(truth_groups[g]),'oracle_count':counts[g],'predicted_count':predicted,
            'count_relative_error':abs(predicted-counts[g])/max(counts[g],1.)}
        if values[g]:
            expected=float(np.mean(values[g])); actual_values=pred_values.get(assigned,[])
            record.update(oracle_mean=expected,predicted_mean=float(np.mean(actual_values)) if actual_values else None,
                mean_relative_error=abs(float(np.mean(actual_values))-expected)/max(abs(expected),1e-9) if actual_values else None,
                mean_query_available=bool(actual_values))
        queries.append(record)
    mean_errors=[q['mean_relative_error'] for q in queries if q.get('mean_relative_error') is not None]
    return {'attribute_cell_accuracy':correct/total if total else None,'n_observed_cells':int(total),
        'mean_count_relative_error':float(np.mean([q['count_relative_error'] for q in queries])) if queries else None,
        'mean_numeric_query_relative_error':float(np.mean(mean_errors)) if mean_errors else None,
        'n_unavailable_mean_queries':sum(q.get('mean_query_available') is False for q in queries),
        'queries':queries,'semantics':'bag outer union of observed source rows; one-to-one oracle schema alignment for evaluation only; no entity resolution/full disjunction'}


def measured_worker(job):
    # Input: serialized frozen config/task/point; output: cold-process resource/accuracy/edge moments; module function.
    from .research import attach_preparation
    root=Path(job['root']); cfg=job['config']; source=next(s for s in attach_preparation(load_sets(cfg,str(root))) if s.name==job['set'])
    if job.get('tables'):
        tables=[t for t in source.tables if t.name in job['tables']]; allowed={c.id for t in tables for c in t.columns}
        source=replace(source,tables=tables,groundtruth=[frozenset(g&allowed) for g in source.groundtruth if g&allowed])
    cfg['experiment']['online']=job.get('online',True)
    cfg['matcher']['cache_scores']=False; cfg['paths'].update(embedding_cache=False,raw_score_cache=False)
    if job.get('R'): cfg['uncertainty_model']['realization']['n_samples']=job['R']
    for key in ('k_max','rho'):
        if key in job: cfg['uncertainty_model']['params'][key]=job[key]
    if 'prior' in job: cfg['uncertainty_propagation']['params']['prior']=job['prior']
    if 'lambda' in job: cfg['uncertainty_propagation']['params']['risk_lambda']=job['lambda']
    import torch
    gpu=torch.cuda.is_available()
    if gpu: torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    profiler=Profiler().start(); ctx=RunContext(project_root=str(root),seed=cfg['experiment']['seed'],cache_dir=str(root/'output/cache'),embedding_cache=False,profiler=profiler)
    started=time.perf_counter()
    try:
        components,real=build_components(cfg)
        data,samples,graph,hit=compute_edges(source,components,real,cfg,ctx)
        if job.get('decision_offset'):
            from .threshold_selection import offset_graph
            graph=offset_graph(graph,job['decision_offset'])
        output=components['global_optimizer'].optimize(graph,ctx)
        validate_matching(output,graph,'resource-worker')
        if gpu: torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
    finally:
        resources=profiler.stop()
    resources.update(gpu_peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if gpu else 0.,
        gpu_peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20 if gpu else 0.)
    from .experiment import environment_info
    return {'set':source.name,'k':len(source.tables),'R':real.n_samples,'online':cfg['experiment']['online'],
        'prior':graph.meta['prior'],'risk_lambda':graph.meta['risk_lambda'],'k_max':cfg['uncertainty_model']['params'].get('k_max'),
        'rho':cfg['uncertainty_model']['params'].get('rho'),'time_s':elapsed,'edge_pairs_per_second':len(graph.df)/elapsed if elapsed else None,
        'score_samples_per_second':len(samples)/elapsed if elapsed else None,'n_edges':len(graph.df),
        'score_cache_hit':hit,'environment':environment_info(str(root)),**resources,**evaluate(output,source.groundtruth,graph.columns),
        'mean_sigma':float(graph.df.sigma.mean()) if len(graph.df) else 0.,
        'mean_estimated_standard_error':float(graph.df.sigma.mean()/np.sqrt(real.n_samples)) if len(graph.df) else 0.,
        'all_optimal':output.info.get('all_optimal'),'n_variables':output.info.get('n_variables'),
        'n_triangle_constraints':output.info.get('n_triangle_constraints'),'actual_mip_gap':output.info.get('actual_mip_gap'),
        'edges':[{'pair':sorted((r.col_i,r.col_j)),'mu':r.mu,'sigma':r.sigma} for r in graph.df.itertuples()]}


def run_extended(cfg, phase, root, selection=None):
    # Input: research config/phase/frozen selection; output: one diagnostics HTML, never test-selected parameters; module function.
    from .research import (role_config,ensure_disjoint,calibration_sources,apply_fit_prior,select_parameter,report_folder,source_clusters)
    root=Path(root); cfg=copy.deepcopy(cfg); artifact=load_artifact(str(root/cfg['research']['calibration']))
    apply_fit_prior(cfg,artifact)
    workcfg,sets=role_config(cfg,'test',root); _,validation=role_config(cfg,'validation',root)
    site_check=cfg['research'].get('require_site_disjoint',False)
    ensure_disjoint(sets,validation,by_site=site_check); ensure_disjoint(sets,calibration_sources(artifact,cfg,root),by_site=site_check)
    chosen=select_parameter(selection,cfg,root) if selection else None
    if chosen and chosen['axis']=='lambda':
        cfg['ablation']['missing']['Q_risk']['uncertainty_propagation']['params']['risk_lambda']=chosen['value']
    elif chosen and chosen['axis']=='prior':
        from .research import set_shared_prior
        set_shared_prior(cfg,chosen['value'])
    spec=cfg['research']; rows=[]; details=[]; groups=source_clusters(sets)
    if phase in ('benchmark','sensitivity'):
        if spec['matcher']=='magneto' and not spec.get('allow_paid_resource_runs',False):
            raise ValueError('Cold resource runs of Magneto need explicit research.allow_paid_resource_runs=true')
        settings=spec.get('sweeps',{}); mode='Q_risk'; scfg=strategy_config(cfg,cfg['ablation']['missing'][mode]); scfg['dataset']=workcfg['dataset']
        collected=[]
        for source in sets:
            ks=settings.get('tables',[len(source.tables)])
            for k in ks:
                k=len(source.tables) if k=='all' else int(k)
                if k>len(source.tables): continue
                table_names=[t.name for t in source.tables[:k]]
                if phase=='benchmark':
                    points=[{'R':r,'online':online,'axis':'resources'} for r in settings.get('R',[4,8,16,32,64]) for online in (True,False)]
                else:
                    points=[{'R':r,'axis':'R'} for r in settings.get('R',[4,8,16,32,64])]
                    points += [{key:v,'axis':key} for key in ('k_max','rho','lambda','prior') for v in settings.get(key,[])]
                for point in points:
                    jobcfg=copy.deepcopy(scfg); jobcfg['dataset']['sets']=[source.name.split('/')[-1]]
                    offset=(chosen or {}).get('decision_offsets',{}).get(mode,{}).get('global',0.)
                    job={'root':str(root),'config':jobcfg,'set':source.name,'tables':table_names,'decision_offset':offset,**point}
                    process=subprocess.run([sys.executable,'-m','umsm.study_eval','--worker'],input=json.dumps(job),text=True,
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=Path(__file__).resolve().parents[1])
                    if process.returncode: raise RuntimeError(f'Resource worker failed: {process.stderr[-6000:]}')
                    result=json.loads(process.stdout); result.update(source_cluster=groups[source.name],axis=point['axis'])
                    collected.append(result)
                    print(f'{phase}: {source.name} k={k} {point} completed',flush=True)
        references={}
        for result in collected:
            key=(result['set'],result['k'],result['online'],result['axis'])
            if key not in references or result['R']>references[key]['R']: references[key]=result
        reference_edges={key:{tuple(r['pair']):r for r in ref['edges']} for key,ref in references.items()}
        equal_run_edges={(r['set'],r['k'],r['R'],r['online']):{tuple(e['pair']):e for e in r['edges']}
                         for r in collected if r['axis']=='resources'}
        for result in collected:
            refs=[r for r in collected if r['set']==result['set'] and r['k']==result['k'] and r['online']==result['online'] and r['axis']==result['axis']]
            reference=max(refs,key=lambda r:r['R']) if result['axis'] in ('R','resources') else None
            if reference:
                lookup=reference_edges[(result['set'],result['k'],result['online'],result['axis'])]
                result['mu_error_vs_largest_R']=float(np.mean([abs(r['mu']-lookup[tuple(r['pair'])]['mu']) for r in result['edges']])) if result['edges'] else 0.
                result['sigma_error_vs_largest_R']=float(np.mean([abs(r['sigma']-lookup[tuple(r['pair'])]['sigma']) for r in result['edges']])) if result['edges'] else 0.
            if result['axis']=='resources':
                # Compare equal-R runs, not the largest-R convergence reference.
                other=next((r for r in collected if r['set']==result['set'] and r['k']==result['k'] and r['R']==result['R']
                    and r['online']!=result['online'] and r['axis']=='resources'),None)
                if other:
                    # Store a snapshot before edge arrays are discarded below.
                    other_edges=equal_run_edges[(other['set'],other['k'],other['R'],other['online'])]
                    result['online_batch_max_mu_difference']=max((abs(r['mu']-other_edges[tuple(r['pair'])]['mu']) for r in result['edges']),default=0.)
                    result['online_batch_max_sigma_difference']=max((abs(r['sigma']-other_edges[tuple(r['pair'])]['sigma']) for r in result['edges']),default=0.)
            result.pop('edges'); rows.append(result)
    else:
        ctx=RunContext(project_root=str(root),seed=cfg['experiment']['seed'],cache_dir=str(root/'output/cache'))
        for source in sets:
            strategies=spec.get('constraints_missing',['observed_only']) if phase=='constraints' else spec.get('missing',['observed_only','Q_mean'])
            for missing in strategies:
                scfg=strategy_config(cfg,cfg['ablation']['missing'][missing]); components,real=build_components(scfg)
                data,samples,graph,hit=compute_edges(source,components,real,scfg,ctx)
                if phase=='constraints':
                    from .threshold_selection import offset_graph
                    offset=(chosen or {}).get('decision_offsets',{}).get(missing,{}).get('global',0.)
                    graph=offset_graph(graph,offset) if offset else graph
                    accepted={frozenset((a,b)) for a,b,w in zip(graph.df.col_i,graph.df.col_j,graph.df.a) if w>0}
                    frame=decision_frame(graph,source,accepted)
                    rows.append({'set':source.name,'missing':missing,'optimizer':'no_triangle','evaluation_layer':'edge_only_not_partition',
                        **decision_metrics(frame),**triangle_counts(graph.columns,accepted),'source_cluster':groups[source.name]})
                    for triangles in ('full','lazy'):
                        params={**cfg['global_optimizer']['params'],'triangles':triangles}
                        started=time.perf_counter()
                        try: matching=OPTIMIZERS.build('ilp_logit',params).optimize(graph,ctx)
                        except ValueError as exc:
                            if triangles!='full' or 'max_full_triples' not in str(exc): raise
                            rows.append({'set':source.name,'missing':missing,'optimizer':triangles,
                                         'status':'full_expansion_limit','reason':str(exc),'source_cluster':groups[source.name]})
                            continue
                        rows.append({'set':source.name,'missing':missing,'optimizer':triangles,**evaluate(matching,source.groundtruth,graph.columns),
                            **triangle_counts(graph.columns,pair_set(matching.groups)),'time_s':time.perf_counter()-started,
                            'objective':matching.info['objective'],'n_triangle_constraints':matching.info['n_triangle_constraints'],'actual_mip_gap':matching.info.get('actual_mip_gap'),
                            'source_cluster':groups[source.name],**local_conflicts(graph)})
                    params={**cfg['global_optimizer']['params'],'triangles':'lazy'}
                    started=time.perf_counter(); unconstrained=no_same_table_constraint(graph,ctx,params)
                    rows.append({'set':source.name,'missing':missing,'optimizer':'no_same_table',
                        **evaluate(unconstrained,source.groundtruth,graph.columns),
                        **triangle_counts(graph.columns,pair_set(unconstrained.groups)),
                        'time_s':time.perf_counter()-started,'same_table_conflicts':unconstrained.info['same_table_conflicts'],
                        'problem_feasible':unconstrained.info['problem_feasible'],'all_optimal':unconstrained.info['all_optimal'],
                        'source_cluster':groups[source.name]})
                elif phase=='downstream':
                    from .types import Matching
                    matches={'oracle_schema':Matching(source.groundtruth,{})}
                    for label in spec.get('downstream_optimizers',['local','global']):
                        from .threshold_selection import offset_graph
                        offset=(chosen or {}).get('decision_offsets',{}).get(missing,{}).get(label,0.)
                        decision_graph=offset_graph(graph,offset) if offset else graph
                        matches[label]=OPTIMIZERS.build(**cfg['ablation']['optimizers'][label]).optimize(decision_graph,ctx)
                    for label,matching in matches.items():
                        result=outer_union_metrics(source,matching); queries=result.pop('queries')
                        rows.append({'set':source.name,'missing':missing,'optimizer':label,'source_cluster':groups[source.name],**result})
                        details.extend({'set':source.name,'missing':missing,'optimizer':label,**q} for q in queries)
    run_id=datetime.now().strftime('%Y%m%d-%H%M%S-%f'); path=report_folder(cfg,root)/phase/(run_id+'.html')
    write_report(path,{'run_id':run_id,'name':spec['name']+' '+phase,'kind':'extended_research','config':cfg,'rows':rows,
        'calibration_sha256':hashlib.sha256((root/spec['calibration']).read_bytes()).hexdigest(),
        'study_figures':bool(spec.get('plots',True)),
        'protocol':{'phase':phase,'selection':chosen,'prior_policy':spec.get('prior_policy'),'source_clusters':groups,
            'note':'No held-out operating point selection. Sigma is world variability; sigma/sqrt(R) is estimated MC mean standard error. Resource workers are fresh processes with score/embedding disk caches disabled.'},
        'analysis_tables':{'Results':rows,'Outer-union queries':details}})
    print('Result:',path,flush=True)
    return {'result_file':str(path),'rows':rows}


def render_study_plots(payload):
    # Input: trusted numeric study rows; output: embedded SVG resource/convergence/sensitivity figures; module function.
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=pd.DataFrame(payload.get('rows',[])); images=[]
    if data.empty or 'axis' not in data: return images
    for (task,k,axis),block in data.groupby(['set','k','axis'],sort=False):
        if axis=='resources':
            metrics=[('time_s','Time (s)'),('peak_rss_minus_baseline_mb','Peak RSS above baseline (MiB)'),('edge_pairs_per_second','Pairs / second')]
            x='R'; groups=list(block.groupby('online',sort=False))
        elif axis=='R':
            metrics=[('mu_error_vs_largest_R','Mean error vs largest R'),('mean_sigma','World variability sigma'),('mean_estimated_standard_error','Estimated mean SE')]
            x='R'; groups=[('frozen',block)]
        else:
            x={'lambda':'risk_lambda'}.get(axis,axis)
            metrics=[('pair_f1','Pair F1'),('time_s','Time (s)')]; groups=[('frozen',block)]
        fig,axes=plt.subplots(1,len(metrics),figsize=(4*len(metrics),3.4),constrained_layout=True)
        for ax,(metric,label) in zip(np.atleast_1d(axes),metrics):
            for mode,points in groups:
                label_mode=('online' if bool(mode) else 'batch') if isinstance(mode,(bool,np.bool_)) else str(mode)
                points=points.sort_values(x); ax.plot(points[x],points[metric],'o-',label=label_mode)
            ax.set(xlabel=x,ylabel=label); ax.grid(alpha=.2); ax.legend()
        fig.suptitle(f'{task} / k={k} / {axis}')
        buffer=io.StringIO(); fig.savefig(buffer,format='svg'); plt.close(fig)
        raw=buffer.getvalue(); images.append(raw[raw.index('<svg'):])
    return images


if __name__=='__main__':
    if sys.argv[1:]!=['--worker']: raise SystemExit('Use run.py --phase benchmark/sensitivity')
    result=measured_worker(json.loads(sys.stdin.read()))
    print(json.dumps(result,allow_nan=False,default=str))
