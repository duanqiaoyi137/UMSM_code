"""Auditable natural masks: explicit/lexical column correspondence and optional row alignment."""
import re
import numpy as np
import pandas as pd
from ..uncertainty_model.typed import profiles

ALIASES={'cost':'price','amount':'price','brand_name':'brand','manufacturer':'brand','colour':'color',
         'company_name':'company','country_name':'country','year_published':'year','launch_yr':'year'}


def column_tokens(name):
    # Input: raw column label; output: deterministic lexical tokens, never matching labels; module function.
    label=re.sub(r'[^a-z0-9]+','_',str(name).lower()).strip('_')
    return set(ALIASES.get(label,label).split('_'))-{'id','name','value','column','col'}


def correspond_columns(base, donor, mapping=None):
    # Input: two observed frames and optional explicit base->donor map; output: column map with reasons; module function.
    bp,dp=profiles(base),profiles(donor); mapping=mapping or {}; chosen={}; reasons={}; used=set()
    for c in base:
        if c in mapping:
            if mapping[c] not in donor: raise ValueError(f'Unknown explicit donor column: {mapping[c]}')
            chosen[c]=mapping[c]; reasons[c]='explicit'; used.add(mapping[c]); continue
        compatible=[d for d in donor if dp[d].kind==bp[c].kind or {dp[d].kind,bp[c].kind}<={'numeric','ordered_numeric'}]
        if not compatible: reasons[c]='no_compatible_type'; continue
        def score(d):
            a,b=column_tokens(c),column_tokens(d)
            return len(a&b)/max(1,len(a|b))
        best=min(compatible,key=lambda d:(-score(d),d in used,str(d)))
        chosen[c]=best; used.add(best); reasons[c]='lexical' if score(best)>0 else 'type_fallback'
    return chosen,reasons


def transplant_mask(base, donor, seed, alignment='random', column_map=None, anchors=None,
                    require_anchors=True):
    # Input: real complete/incomplete base and natural donor, fixed alignment; output: masked copy/mask/exact audit; module function.
    if donor.empty or alignment not in ('random','aligned'): raise ValueError('Invalid donor/alignment')
    rng=np.random.default_rng(seed); mapping,reasons=correspond_columns(base,donor,column_map)
    bp,dp=profiles(base),profiles(donor)
    anchor_pairs=list((anchors or {}).items())
    if not anchors:
        anchor_pairs=[(c,d) for c,d in mapping.items() if reasons[c]=='lexical'
            and bp[c].kind in ('categorical','ordered_numeric') and dp[d].kind in ('categorical','ordered_numeric')]
    if any(c not in base or d not in donor for c,d in anchor_pairs): raise ValueError('Unknown alignment anchor')
    if alignment=='aligned' and not anchor_pairs and require_anchors:
        raise ValueError('No shared discrete row anchors; supply an explicit mapping or report random transplant')
    rows=[]; overlaps=[]; fallback=[]
    for r in range(len(base)):
        agree=np.zeros(len(donor)); comparable=np.zeros(len(donor))
        if alignment=='aligned':
            for c,d in anchor_pairs:
                value=base.iloc[r][c]
                if pd.isna(value): continue
                observed=donor[d].notna().to_numpy()
                same=donor[d].astype(str).str.strip().str.casefold().to_numpy()==str(value).strip().casefold()
                comparable+=observed; agree+=observed&same
        similarity=np.divide(agree,comparable,out=np.full(len(donor),-1.),where=comparable>0)
        eligible=np.flatnonzero(similarity==similarity.max()) if alignment=='aligned' and similarity.max()>0 else np.arange(len(donor))
        no_match=alignment=='aligned' and similarity.max()<=0
        if no_match and require_anchors:
            raise ValueError(f'No compatible donor row for base row {r}; do not claim conditional alignment')
        chosen=int(rng.choice(eligible)); rows.append(chosen); overlaps.append(float(max(similarity[chosen],0.))); fallback.append(no_match)
    mask=np.zeros(base.shape,bool)
    for j,c in enumerate(base):
        if c in mapping: mask[:,j]=donor[mapping[c]].isna().to_numpy()[rows]&base[c].notna().to_numpy()
    result=base.astype(object).mask(mask)
    audit={'mechanism':'transplant','alignment':alignment,'seed':seed,'donor_rows':rows,
        'donor_columns':mapping,'column_mapping_reasons':reasons,'anchors':dict(anchor_pairs),
        'row_similarity':overlaps,'unaligned_rows':int(sum(fallback)),
        'donor_natural_missing_rate':float(donor.isna().to_numpy().mean()),
        'n_observed_before':int(base.notna().to_numpy().sum()),'n_added_missing':int(mask.sum()),
        'actual_total_missing_rate':float(result.isna().to_numpy().mean()),'cells':np.argwhere(mask).tolist(),
        'note':'No mask rescaling. Alignment approximates conditional structure; it does not prove mechanism preservation.'}
    return result,mask,audit
