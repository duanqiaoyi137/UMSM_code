"""Prespecified 4-by-2 paired effects, source-cluster bootstrap and Holm adjustment."""
from itertools import product
import copy
import numpy as np
import pandas as pd


def comparable_protocol(cfg):
    # Input: frozen report configuration; output: protocol fields that must agree when pooling folds.
    model=copy.deepcopy(cfg['uncertainty_model']); model.get('params',{}).pop('calibration',None)
    r=cfg['research']
    return {'matcher':cfg['matcher'],'model':model,'factorial_levels':r.get('factorial_levels'),
        'multiple_mice_m':r.get('multiple_mice_m'),'decision_offset_grid':r.get('decision_offset_grid'),
        'prior_policy':r.get('prior_policy'),'lambda_grid':r.get('lambda_grid'),'calibration_method':cfg.get('calibration',{}).get('method')}


def factorial_4x2(rows, levels, settings=None):
    # Input: matched task/method rows and cluster settings; output: per-task 4x2 contrasts and family-wise results.
    settings=settings or {}; levels=list(levels)
    if len(levels)!=4 or len(set(levels))!=4 or levels[0]!='observed_only':
        raise ValueError('4x2 requires four distinct levels with observed_only first')
    data=pd.DataFrame(rows); cases=[]
    for key,block in data.groupby(['set','draw','k'],sort=False):
        if block.duplicated(['missing','optimizer']).any(): raise ValueError('Repeated factorial cell')
        cells={(r.missing,r.optimizer):float(r.pair_f1) for r in block.itertuples()}
        if any((level,opt) not in cells for level in levels for opt in ('local','global')):
            raise ValueError('A 4x2 case requires every level and both optimizers')
        ol,og=cells[(levels[0],'local')],cells[(levels[0],'global')]
        effects={}
        for level in levels[1:]:
            a,b=cells[(level,'local')],cells[(level,'global')]
            effects['imputation_main:'+level]=((a-ol)+(b-og))/2
            effects['interaction:'+level]=(b-a)-(og-ol)
        effects['optimizer_main']=np.mean([cells[(level,'global')]-cells[(level,'local')] for level in levels])
        q=levels[-1]
        effects['waterfall:add_Q']=cells[(q,'local')]-ol
        effects['waterfall:add_ILP']=cells[(q,'global')]-cells[(q,'local')]
        effects['pipeline:Q_ILP_vs_observed_local']=cells[(q,'global')]-ol
        for reference in levels[1:-1]:
            effects['Q_vs:'+reference]=np.mean([cells[(q,o)]-cells[(reference,o)] for o in ('local','global')])
        cases.append({'set':key[0],'draw':int(key[1]),'k':int(key[2]),
                      'source_cluster':settings.get('task_groups',{}).get(key[0],block.iloc[0].get('source_cluster',key[0])),
                      'cells':{level:{o:cells[(level,o)] for o in ('local','global')} for level in levels},'effects':effects})
    if not cases: return [],[]
    confidence=float(settings.get('confidence',.95)); n=int(settings.get('resamples',1000)); seed=int(settings.get('seed',2026))
    minimum=int(settings.get('min_clusters',10)); rng=np.random.default_rng(seed); summary=[]
    for effect in cases[0]['effects']:
        d=pd.DataFrame({'cluster':[c['source_cluster'] for c in cases],'value':[c['effects'][effect] for c in cases]})
        values=d.groupby('cluster',sort=True).value.mean().to_numpy(float); size=len(values)
        boot=rng.choice(values,size=(n,size),replace=True).mean(axis=1) if size>=2 and n>=20 else np.array([])
        low,high=np.quantile(boot,[(1-confidence)/2,(1+confidence)/2]) if len(boot) else (None,None)
        observed=abs(float(values.mean()))
        if size<=16:
            signs=np.asarray(list(product((-1.,1.),repeat=size)))
            p=float(np.mean(np.abs((signs*values).mean(axis=1))>=observed-1e-12))
        else:
            signs=rng.choice((-1.,1.),size=(10000,size))
            p=float((1+np.sum(np.abs((signs*values).mean(axis=1))>=observed-1e-12))/(len(signs)+1))
        summary.append({'effect':effect,'difference':float(values.mean()),'ci_low':low,'ci_high':high,
            'confidence':confidence,'n_cases':len(cases),'n_source_clusters':size,'minimum_clusters':minimum,
            'p_unadjusted':p,'test':'two-sided paired sign-flip on source-cluster means',
            'ci_scope':'unadjusted percentile bootstrap, source clusters','eligible_n':size>=minimum})
    order=sorted(range(len(summary)),key=lambda i:summary[i]['p_unadjusted']); adjusted=0.
    for rank,index in enumerate(order):
        adjusted=max(adjusted,min(1.,(len(order)-rank)*summary[index]['p_unadjusted']))
        summary[index]['p_holm']=adjusted
    for item in summary:
        item['supported']=bool(item['eligible_n'] and item['ci_low'] is not None and item['ci_low']>0 and item['p_holm']<1-confidence)
    return cases,summary


def factorial_plots(payload):
    # Input: pooled source-cluster effects; output: waterfall increments with 95% task-bootstrap intervals.
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=payload.get('rows',[]); images=[]
    for matcher in dict.fromkeys(r.get('matcher') for r in rows):
        block=[r for r in rows if r.get('matcher')==matcher and r['effect'] in ('waterfall:add_Q','waterfall:add_ILP')]
        if len(block)!=2: continue
        block.sort(key=lambda r:0 if r['effect']=='waterfall:add_Q' else 1)
        fig,ax=plt.subplots(figsize=(6,3.5),constrained_layout=True)
        for i,row in enumerate(block):
            value=row['difference']; ax.bar(i,value,color=['#3c78a8','#68a56a'][i])
            if row['ci_low'] is not None:
                # Percentile intervals may not enclose the point; draw their endpoints explicitly.
                ax.vlines(i,row['ci_low'],row['ci_high'],color='black'); ax.hlines([row['ci_low'],row['ci_high']],i-.08,i+.08,color='black')
        ax.axhline(0,color='black',linewidth=.7); ax.set_xticks([0,1],['Add Q','Add ILP']); ax.set_ylabel('Pair F1 increment')
        ax.set_title(f'{matcher}: paired source-cluster effects (unadjusted 95% CI)')
        buf=io.StringIO(); fig.savefig(buf,format='svg'); plt.close(fig)
        svg=buf.getvalue(); images.append(svg[svg.index('<svg'):])
    return images
