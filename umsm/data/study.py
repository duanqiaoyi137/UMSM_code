"""C2-C4 reproducible derived tasks; immutable natural inputs and complete pre-mask oracle."""
import fnmatch
import itertools
import json
import math
import hashlib
from dataclasses import replace,asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import load_dataset
from .masking import stable_seed
from .preparation import publish_task
from .transplant import transplant_mask,column_tokens
from ..calibration import set_fingerprint
from ..types import IntegrationSet,Table
from ..uncertainty_model.typed import profiles


def source_pool(specifications, root):
    # Input: dataset/glob specifications and root; output: provenance-checked original tasks; module function.
    from ..research import attach_preparation
    result=[]
    for spec in specifications:
        folder=root/'data'/spec['dataset']
        if not folder.exists(): raise FileNotFoundError(folder)
        patterns=spec.get('sets',['*'])
        selected=[p.name for p in folder.iterdir() if p.is_dir() and any(fnmatch.fnmatchcase(p.name,q) for q in patterns)
                  and not any(fnmatch.fnmatchcase(p.name,q) for q in spec.get('exclude',[]))]
        result.extend(attach_preparation(load_dataset(spec['dataset'],str(root/'data'),sets=sorted(selected),variant=spec.get('variant'))))
    return result


def dense_frames(source, max_missing=.2, min_rows=10):
    # Input: original task and predeclared density rules; output: complete eligible frames and original row positions; module function.
    frames={}; positions={}
    for t in source.tables:
        selected=t.df.loc[:,t.df.isna().mean()<=max_missing]
        keep=np.flatnonzero(selected.notna().all(axis=1).to_numpy())
        if len(selected.columns)>=2 and len(keep)>=min_rows:
            frames[t.name]=selected.iloc[keep].reset_index(drop=True)
            positions[t.name]=keep.tolist()
    return frames,positions


def donor_pool(sources):
    # Input: natural tasks; output: donor tables with measured bands and content identity; module function.
    result=[]
    for source in sources:
        for table in source.tables:
            rate=float(table.df.isna().to_numpy().mean())
            if rate<=0 or table.df.empty: continue
            result.append({'table':table,'task':source.name,'dataset':source.dataset,'rate':rate,
                'band':'low' if rate<.3 else 'medium' if rate<=.6 else 'high',
                'fingerprint':set_fingerprint(replace(source,tables=[table])),
                'sites':source.meta.get('source_sites',[])})
    return sorted(result,key=lambda d:(d['dataset'],d['task'],d['table'].name))


def choose_donor(frame, donors, band, index, explicit=None):
    # Input: complete base, natural donors/band and stable index; output: deterministic donor, without F1 selection; module function.
    available=[d for d in donors if d['band']==band]
    if not available: raise ValueError(f'No natural donor in band {band}')
    # Rotate priority to avoid always reusing the first source. Maps can pin a particular donor.
    if explicit and explicit.get('donor'):
        available=[d for d in available if f"{d['task']}/{d['table'].name}"==explicit['donor']]
        if not available: raise ValueError('Declared donor unavailable in this band')
    return available[index%len(available)]


def choose_compatible_donor(frame, donors, band, index, explicit, seed):
    # Input: observed data/metadata only; output: first donor satisfying preregistered alignment, without using matching labels.
    available=[d for d in donors if d['band']==band]
    if explicit.get('donor'):
        available=[d for d in available if f"{d['task']}/{d['table'].name}"==explicit['donor']]
    if not available: raise ValueError(f'No declared natural donor in band {band}')
    start=index%len(available); reasons=[]
    for donor in available[start:]+available[:start]:
        try:
            masked,mask,audit=transplant_mask(frame,donor['table'].df,seed,'aligned',
                column_map=explicit.get('columns'),anchors=explicit.get('anchors'),require_anchors=True)
            return donor,masked,mask,audit
        except ValueError as exc: reasons.append(str(exc))
    raise ValueError(f'No compatible aligned donor in {band}; checked {len(available)}; last reason: {reasons[-1]}')


def split_siblings(frame, table_name, k, rows, overlap, coverage, sibling_ratio, seed,
                   mixed_format=False, aliases=None):
    # Input: dense real table and preregistered split/derivation settings; output: oracle frames/GT/exact construction audit; module function.
    if k<2 or rows<1 or not 0<overlap<1 or not 0<coverage[0]<=coverage[1]<1 or not 0<=sibling_ratio<1:
        raise ValueError('Invalid split parameters')
    rng=np.random.default_rng(seed); prof=profiles(frame,table_name)
    semantics={str(c):{'base_column':c,'profile':prof[c],'formula':'identity','scale':1.,'offset_days':0}
               for c in frame}
    eligible=[c for c in frame if prof[c].kind in ('numeric','ordered_numeric','date')]
    n_siblings=min(2*len(eligible),max(1,round(len(frame.columns)*sibling_ratio/max(1-sibling_ratio,1e-12)))) if sibling_ratio else 0
    for index in range(n_siblings):
        c=eligible[index%len(eligible)]; p=prof[c]
        suffix=('update' if index<len(eligible) else 'update_late') if p.kind=='date' else 'sale' if index<len(eligible) else 'max'
        key=f'{c}__{suffix}'
        scale=float(rng.uniform(.7,.95) if suffix=='sale' else rng.uniform(1.05,1.3)) if p.kind!='date' else 1.
        days=int(rng.integers(30,401)) if p.kind=='date' else 0
        semantics[key]={'base_column':c,'profile':p,'formula':f'x*{scale:.17g}' if not days else f'date+{days} days',
                        'scale':scale,'offset_days':days}
    lo=max(1,math.ceil(k*coverage[0])); hi=min(k-1,max(lo,math.floor(k*coverage[1])))
    memberships={s:set(map(int,rng.choice(k,int(rng.integers(lo,hi+1)),replace=False))) for s in semantics}
    # Guarantee one both/original-only/sibling-only source when possible, without changing semantics.
    if k>=3:
        sibling_anchors={}
        for s,spec in semantics.items():
            base=str(spec['base_column'])
            if s==base: continue
            proposed=tuple(map(int,rng.permutation(k)[:3]))
            a,b,c=sibling_anchors.setdefault(base,proposed); count=max(2,lo)
            if count>hi: continue
            memberships[base]={a,b}|set(map(int,rng.choice([i for i in range(k) if i not in (a,b,c)],count-2,replace=False)))
            memberships[s]={a,c}|set(map(int,rng.choice([i for i in range(k) if i not in (a,b,c)],count-2,replace=False)))
    for i in range(k):
        if not any(i in v for v in memberships.values()):
            possible=[s for s,v in memberships.items() if len(v)<hi]
            if not possible: raise ValueError('Column coverage leaves an empty source')
            memberships[min(possible,key=lambda s:len(memberships[s]))].add(i)
    n=min(rows,int(len(frame)/(2-overlap)))
    if n<2: raise ValueError('Too few complete entities for the requested overlap')
    previous=None; output={}; groups={s:set() for s in semantics}; audit={'source_rows':{},'columns':{},'derivations':{},'adjacent_entity_overlap':[]}
    for i in range(k):
        if previous is None: indices=rng.choice(len(frame),n,replace=False)
        else:
            retained=rng.choice(previous,min(n,round(overlap*n)),replace=False)
            new=rng.choice(np.setdiff1d(np.arange(len(frame)),previous),n-len(retained),replace=False)
            indices=np.r_[retained,new]
            audit['adjacent_entity_overlap'].append(len(set(indices)&set(previous))/n)
        previous=indices; name=f'source_{i:02d}.csv'; columns={}; audit['source_rows'][name]=indices.tolist(); audit['columns'][name]={}
        for s,spec in semantics.items():
            if i not in memberships[s]: continue
            c=spec['base_column']; p=spec['profile']
            choices=(aliases or {}).get(s) or [str(s),''.join(part[:2] for part in str(s).split('_')),f'field_{list(semantics).index(s)}']
            label=f's{i}_'+choices[i%len(choices)]
            if label in columns: label+=f'_{len(columns)}'
            values=[]; format_scale=1.
            for v in frame[c].iloc[indices]:
                if p.kind in ('numeric','ordered_numeric','date'):
                    number=p.number(v)
                    if not np.isfinite(number): raise ValueError('Dense numerical base has incompatible units/format')
                    number=number*spec['scale']+spec['offset_days']
                    # Freeze semantic rounding before changing only the presentation format.
                    number=p.number(p.render(number))
                    if mixed_format and i%2 and p.kind=='date': value=pd.Timestamp(number,unit='D').strftime('%d/%m/%Y')
                    elif mixed_format and i%2 and p.kind!='date':
                        # Unit changes only where the observed unit is explicitly inches.
                        if p.suffix.strip().lower() in ('in','inch','inches'):
                            format_scale=2.54; value=f'{number*format_scale:.4f} cm'
                        else: value=f'{number:.4e}'
                    else: value=p.render(number)
                else: value=v
                values.append(value)
            columns[label]=values; groups[s].add(f'{name}.{label}')
            audit['columns'][name][label]={'semantic':s,'base_column':str(c),'format_scale':format_scale,
                                           'column_type':p.kind,'date_format':'%d/%m/%Y' if mixed_format and i%2 and p.kind=='date' else p.date_format}
        output[name]=pd.DataFrame(columns)
    audit['derivations']={s:{**{key:value for key,value in spec.items() if key!='profile'},
                            'rendering':asdict(spec['profile']),
                            'rounding_rule':'parse(render(original column template, derived value)) before format variants'}
                          for s,spec in semantics.items()}
    audit['semantic_source_coverage']={s:len(v)/k for s,v in memberships.items()}
    audit['actual_sibling_ratio']=(len(semantics)-len(frame.columns))/len(semantics)
    return output,[frozenset(g) for g in groups.values() if g],audit


def prepare_study(spec, root):
    # Input: preregistered construction configuration and project root; output: immutable C2/C3/C4 tasks plus one audit report; module function.
    from ..report import write_report
    root=Path(root); bases=source_pool(spec['bases'],root); donors=donor_pool(source_pool(spec['donors'],root))
    if spec.get('max_base_tasks'): bases=bases[:int(spec['max_base_tasks'])]
    spec_hash=hashlib.sha256(json.dumps(spec,sort_keys=True,default=str).encode()).hexdigest()
    made=[]; rejected=[]; row_caps=spec.get('rows',[10,30,100,300]); seeds=spec.get('seeds',[42,43,44])
    maps=spec.get('alignment_maps',{})
    for source_index,source in enumerate(bases):
        dense,positions=dense_frames(source,spec.get('dense_max_missing',.2),spec.get('min_rows',10))
        if source.name in spec.get('base_tables',{}):
            selected_names=set(spec['base_tables'][source.name])
            dense={name:f for name,f in dense.items() if name in selected_names}
        origin=set_fingerprint(source)
        if len(dense)<2:
            rejected.append({'source':source.name,'construction':'C2','reason':'fewer than two dense complete-row sources'})
        for seed,band,alignment in itertools.product(seeds,spec.get('donor_bands',['low','medium','high']),spec.get('alignments',['random','aligned'])):
            if len(dense)>=2 and spec.get('c2',True) and alignment!='none':
                frames={}; audits={}; dependencies=[origin]; sites=set(source.meta.get('source_sites',[]))
                try:
                    for i,(name,frame) in enumerate(dense.items()):
                        selected=frame.sample(min(max(row_caps),len(frame)),random_state=stable_seed(seed,name)).sort_index()
                        map_spec=maps.get(f'{source.name}/{name}',{})
                        if alignment=='aligned' and spec.get('donor_selection')=='compatible':
                            donor,result,mask,audit=choose_compatible_donor(selected.reset_index(drop=True),donors,band,source_index,map_spec,stable_seed(seed,name))
                        else:
                            donor=choose_donor(selected,donors,band,source_index,map_spec)
                            result,mask,audit=transplant_mask(selected.reset_index(drop=True),donor['table'].df,stable_seed(seed,name),
                                alignment,column_map=map_spec.get('columns'),anchors=map_spec.get('anchors'),require_anchors=True)
                        audit.update(donor=f"{donor['task']}/{donor['table'].name}",donor_fingerprint=donor['fingerprint'],
                                     original_source_rows=[positions[name][int(r)] for r in selected.index])
                        frames[name]=(result,selected.reset_index(drop=True)); audits[name]=audit
                        dependencies.append(donor['fingerprint']); sites.update(donor['sites'])
                    for cap in row_caps:
                        masked={}; oracle={}; subset_audit={}
                        for name,(observed,complete) in frames.items():
                            chosen=np.sort(np.random.default_rng(stable_seed(seed,name)).choice(len(complete),min(cap,len(complete)),replace=False))
                            masked[name]=observed.iloc[chosen].reset_index(drop=True); oracle[name]=complete.iloc[chosen].reset_index(drop=True)
                            subset_audit[name]={**audits[name],'retained_rows_in_transplanted_table':chosen.tolist(),
                                'cells':np.argwhere(masked[name].isna().to_numpy()).tolist()}
                        allowed={f'{t}.{c}' for t,f in masked.items() for c in f}
                        groups=[frozenset(g&allowed) for g in source.groundtruth or [] if len(g&allowed)>0]
                        if not any(len(g)>1 for g in groups): raise ValueError('No positive schema group after density filtering')
                        name=f'{source.slug[:45]}_{band}_{alignment}_n{cap}_s{seed}'
                        protocol={'adapter_version':1,'mechanism':'transplant','construction':'C2','donor_band':band,'alignment':alignment,
                            'row_cap':cap,'mask_seed':seed,'source_dataset':source.dataset,'source_set':source.name,
                            'source_fingerprint':origin,'dependency_fingerprints':sorted(set(dependencies)),'source_sites':sorted(sites),
                            'dense_max_missing':spec.get('dense_max_missing',.2),'complete_rows':True,'rate':None,
                            'construction_spec_sha256':spec_hash}
                        manifest=publish_task(root/'data'/spec.get('c2_name','Transplant_Real')/name,masked,groups,protocol,
                            variants={'oracle':oracle},cell_masks=subset_audit)
                        made.append({'construction':'C2','case':name,'tables':len(masked),'GT_pairs':manifest['gt_pairs'],'source':source.name})
                except ValueError as exc:
                    rejected.append({'source':source.name,'construction':'C2','band':band,'alignment':alignment,'seed':seed,'reason':str(exc)})
            if not spec.get('c3',True): continue
            c3_sources=spec.get('c3_sources',['*'])
            if not any(fnmatch.fnmatchcase(source.name,pattern) for pattern in c3_sources): continue
            widest=spec.get('c3_selection','rows')=='widest'
            eligible_tables=sorted(dense.items(),key=lambda item:(-len(item[1].columns),-len(item[1]),item[0]) if widest
                                   else (-len(item[1]),item[0]))[:spec.get('c3_tables_per_source',1)]
            for table_index,(table_name,frame) in enumerate(eligible_tables):
                for k,ratio,mixed,overlap in itertools.product(spec.get('tables',[3,4,6,8,12,16]),spec.get('sibling_ratios',[0,.1,.3,.5]),
                        spec.get('mixed_formats',[False,True]),spec.get('entity_overlaps',[.5])):
                    point={'tables':k,'sibling_ratios':ratio,'mixed_formats':mixed,'entity_overlaps':overlap,
                           'donor_bands':band,'alignments':alignment}
                    conditions=[c for c in spec.get('c3_conditions',[{}]) if all(value in c.get(key,[value]) for key,value in point.items())]
                    if not conditions: continue
                    caps=sorted({cap for c in conditions for cap in c.get('rows',row_caps)})
                    coverages={tuple(c.get('column_coverage',spec.get('column_coverage',[.4,.9]))) for c in conditions}
                    if len(coverages)!=1: raise ValueError('Overlapping C3 conditions must agree on column coverage')
                    coverage=list(next(iter(coverages)))
                    try:
                        oracle,groups,construction=split_siblings(frame,table_name,k,max(caps),overlap,coverage,
                            ratio,stable_seed(seed,table_name),mixed,spec.get('header_aliases'))
                        if min(map(len,oracle.values()))<spec.get('min_rows',10): raise ValueError('Too few split rows after enforcing entity overlap')
                        base_map=maps.get(f'{source.name}/{table_name}',{})
                        if alignment=='none': donor=None
                        elif alignment=='aligned' and spec.get('donor_selection')=='compatible':
                            donor,_,_,_=choose_compatible_donor(frame,donors,band,source_index+table_index,base_map,stable_seed(seed,table_name))
                        else: donor=choose_donor(frame,donors,band,source_index+table_index,base_map)
                        masked={}; audits={}; base_map=maps.get(f'{source.name}/{table_name}',{})
                        # Map renamed/sibling columns through their declared base-column semantics.
                        for name,f in oracle.items():
                            semantic_columns=construction['columns'][name]
                            original_names={c:semantic_columns[c]['base_column'] for c in f}
                            # Construction can use the base-row context before schema projection; inference never sees it.
                            context=frame.iloc[construction['source_rows'][name]].reset_index(drop=True)
                            if donor is None:
                                mask=np.zeros(context.shape,bool); audit={'mechanism':'none','alignment':'none','cells':[]}
                            else:
                                _,mask,audit=transplant_mask(context,donor['table'].df,stable_seed(seed,name),alignment,
                                    column_map=base_map.get('columns'),anchors=base_map.get('anchors'),require_anchors=True)
                            expanded=np.column_stack([mask[:,context.columns.get_loc(original_names[c])] for c in f])
                            masked[name]=f.astype(object).mask(expanded)
                            audits[name]={**audit,'cells':np.argwhere(expanded).tolist(),'base_column_names':original_names}
                        for cap in caps:
                            capped={}; complete={}; audit_cap={}; construction_cap=json.loads(json.dumps(construction,default=str))
                            for name,f in masked.items():
                                take=np.sort(np.random.default_rng(stable_seed(seed,name)).choice(len(f),min(cap,len(f)),replace=False))
                                capped[name]=f.iloc[take].reset_index(drop=True); complete[name]=oracle[name].iloc[take].reset_index(drop=True)
                                audit_cap[name]={**audits[name],'retained_split_rows':take.tolist(),'cells':np.argwhere(capped[name].isna().to_numpy()).tolist()}
                                construction_cap['source_rows'][name]=[positions[table_name][construction['source_rows'][name][int(r)]] for r in take]
                            row_sets=[set(construction_cap['source_rows'][name]) for name in sorted(capped)]
                            construction_cap['adjacent_overlap_before_row_cap']=construction_cap['adjacent_entity_overlap']
                            construction_cap['adjacent_entity_overlap']=[len(a&b)/max(1,min(len(a),len(b))) for a,b in zip(row_sets,row_sets[1:])]
                            band_label=band if donor else 'no_donor'
                            name=f'{source.slug[:40]}_t{table_index}_k{k}_b{ratio:g}_fmt{int(mixed)}_o{overlap:g}_{band_label}_{alignment}_n{cap}_s{seed}'
                            protocol={'adapter_version':1,'construction':'C3','mechanism':'transplant' if donor else None,
                                'donor_band':band if donor else None,'alignment':alignment,
                                'row_cap':cap,'mask_seed':seed,'source_dataset':source.dataset,'source_set':source.name,'source_table':table_name,
                                'source_fingerprint':origin,'dependency_fingerprints':sorted({origin,donor['fingerprint']} if donor else {origin}),
                                'source_sites':sorted(set(source.meta.get('source_sites',[]))|(set(donor['sites']) if donor else set())),
                                'tables':k,'requested_sibling_ratio':ratio,'mixed_format':mixed,'entity_overlap':overlap,
                                'donor':f"{donor['task']}/{donor['table'].name}" if donor else None,'rate':None,'split':construction_cap,
                                'construction_spec_sha256':spec_hash}
                            manifest=publish_task(root/'data'/spec.get('c3_name','Split_Sibling')/name,capped,groups,protocol,
                                variants={'oracle':complete},cell_masks=audit_cap)
                            made.append({'construction':'C3','case':name,'tables':k,'GT_pairs':manifest['gt_pairs'],'source':source.name})
                    except ValueError as exc:
                        rejected.append({'source':source.name,'table':table_name,'construction':'C3','k':k,'ratio':ratio,
                            'band':band,'alignment':alignment,'seed':seed,'reason':str(exc)})
    path=root/spec.get('audit','output/results/data_preparation_study.html')
    write_report(path,{'run_id':'study-preparation','name':'Derived data preparation audit','kind':'preparation_audit','rows':[],
        'config':spec,'analysis_tables':{'Prepared cases':made,'Rejected cases (never silently replaced)':rejected,
        'Donor pool':[{'task':d['task'],'table':d['table'].name,'band':d['band'],'missing_rate':d['rate'],'fingerprint':d['fingerprint']} for d in donors]}})
    print(f'Prepared {len(made)} cases; rejected {len(rejected)} constructions. Audit: {path}',flush=True)
    return {'prepared':made,'rejected':rejected,'result_file':str(path)}
