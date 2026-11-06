"""Table-local typed candidate generators; finite Q remains the module boundary.

All fitting below sees observed cells only. Never pass a complete oracle frame or
matching labels here. Continuous candidates are quantile representatives, not
claims of exact missing-value recovery or calibrated posterior probabilities.
"""
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import re
import zlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from ..registry import UNCERTAINTY_MODELS
from .base import CellDistribution, DeterministicModel, normalise

KINDS = ('categorical', 'ordered_numeric', 'numeric', 'date', 'text', 'identifier')
CONTINUOUS = ('numeric', 'date')
DEFAULT_ROUTES = {
    'categorical': ['column_marginal', 'gower_knn', 'cluster_marginal', 'nb_classifier'],
    'ordered_numeric': ['column_marginal', 'gower_knn', 'cluster_marginal', 'numeric_kernel'],
    'numeric': ['quantile_marginal', 'gower_knn', 'mice_posterior'],
    'date': ['quantile_marginal', 'gower_knn', 'mice_posterior'],
    'text': ['row_embedding_knn', 'column_marginal'], 'identifier': []}
NUMBER = re.compile(r'^\s*([^\d+\-.,]*)([+\-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+\-]?\d+)?)([^\d]*)\s*$')
DATE_FORMATS = ('%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S%z', '%d-%b-%Y')


def parse_number(value):
    # Input: observed value; output: finite numeric value or NaN; module function.
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        return float(value) if np.isfinite(value) else np.nan
    match = NUMBER.fullmatch(str(value))
    if not match or match[1].strip().upper() not in ('','$','€','£','¥','￥','USD','EUR','GBP','CNY','RMB'):
        return np.nan
    try:
        number = float(match[2].replace(',', ''))
        return number if np.isfinite(number) else np.nan
    except ValueError:
        return np.nan


@dataclass
class ColumnProfile:
    """Observed-only type and a dominant formatting template; no hidden truth."""
    kind: str
    observed: int
    unique_ratio: float
    scale: float = 1.
    prefix: str = ''
    suffix: str = ''
    decimals: int = 0
    thousands: bool = False
    as_string: bool = True
    integer: bool = False
    date_format: str = '%Y-%m-%d'

    def number(self, value):
        # Input: candidate; output: numeric value/days since epoch or NaN; ColumnProfile.
        if self.kind != 'date':
            if self.as_string and isinstance(value,str):
                match=NUMBER.fullmatch(value)
                if not match or match[1]!=self.prefix or match[3]!=self.suffix:
                    return np.nan  # Never silently mix incompatible units/currencies.
            return parse_number(value)
        try:
            stamp = pd.Timestamp(datetime.strptime(str(value), self.date_format)) if isinstance(value, str) else pd.Timestamp(value)
            return stamp.timestamp()/86400.
        except (ValueError, TypeError, OverflowError):
            return np.nan

    def render(self, value):
        # Input: finite numeric value/days; output: original-style scalar/string; ColumnProfile.
        if self.kind == 'date':
            # Observed calendars can predate pandas' nanosecond range (1677-2262).
            # Format through the stdlib datetime in that case instead of dropping a
            # legitimate candidate or changing its day offset.
            try:
                stamp = pd.Timestamp(value, unit='D', tz='UTC')
                formatted, plain = stamp.strftime(self.date_format), stamp.tz_localize(None)
            except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
                epoch = datetime(1970, 1, 1)
                lowest = (datetime.min - epoch).days
                highest = (datetime.max - epoch).days
                plain = epoch + timedelta(days=min(max(float(value), lowest), highest))
                formatted = plain.strftime(self.date_format)
            return formatted if self.as_string else plain
        value = round(float(value), self.decimals) if self.as_string or self.integer else float(value)
        if not self.as_string:
            return int(round(value)) if self.integer else value
        return self.prefix+format(value, f'{"," if self.thousands else ""}.{self.decimals}f')+self.suffix


def profile_column(series, table='', options=None):
    # Input: observed series, table name, thresholds/overrides; output: ColumnProfile; module function.
    options = options or {}
    values = series.dropna().tolist()
    ratio = len(set(values))/len(values) if values else 0.
    p = ColumnProfile('identifier', len(values), ratio)
    override = options.get('overrides', {}).get(f'{table}.{series.name}', options.get('overrides', {}).get(str(series.name)))
    if override is not None and override not in KINDS:
        raise ValueError(f'Unknown column type override: {override}')
    if not values:
        if override: p.kind=override
        return p
    p.as_string = isinstance(values[0], str)
    numeric = np.array([parse_number(v) for v in values])
    number_fraction = np.isfinite(numeric).mean()
    low, high = options.get('categorical_ratio', .3), options.get('identifier_ratio', .8)
    threshold = options.get('parse_fraction', .8)
    if not 0 < low < high <= 1 or not .5 < threshold <= 1:
        raise ValueError('Require 0 < categorical_ratio < identifier_ratio <= 1 and .5 < parse_fraction <= 1')
    date_counts = {}
    if number_fraction < threshold or override == 'date':
        for fmt in DATE_FORMATS:
            count = 0
            for v in values:
                try:
                    datetime.strptime(str(v), fmt)
                    count += 1
                except ValueError:
                    pass
            date_counts[fmt] = count
    date_fmt = max(date_counts, key=date_counts.get) if date_counts else None
    if override:
        p.kind = override
    elif pd.api.types.is_datetime64_any_dtype(series.dtype) or (date_fmt and date_counts[date_fmt]/len(values) >= threshold):
        p.kind = 'date'
    elif number_fraction >= threshold:
        p.kind = 'ordered_numeric' if ratio <= low else 'numeric'
    else:
        p.kind = 'categorical' if ratio <= low else 'text' if ratio <= high else 'identifier'
    if p.kind == 'date':
        if not date_fmt or date_counts[date_fmt] == 0:
            if p.as_string:
                raise ValueError(f'No supported date format for {table}.{series.name}; normalize dates explicitly')
        else:
            p.date_format = date_fmt
    if p.kind in ('numeric', 'ordered_numeric'):
        observed = numeric[np.isfinite(numeric)]
        if len(observed):
            p.integer = bool(np.all(observed == np.round(observed)))
            if p.as_string:
                templates = []
                for v in values:
                    m = NUMBER.fullmatch(str(v))
                    if m:
                        mantissa = re.split('[eE]', m[2])[0]
                        templates.append((m[1], m[3], len(mantissa.split('.')[1]) if '.' in mantissa else 0, ',' in mantissa))
                if templates:
                    p.prefix, p.suffix, p.decimals, p.thousands = Counter(templates).most_common(1)[0][0]
    if p.kind in (*CONTINUOUS, 'ordered_numeric'):
        numbers = np.array([p.number(v) for v in values]); numbers = numbers[np.isfinite(numbers)]
        if len(numbers):
            iqr = float(np.diff(np.quantile(numbers, [.25, .75]))[0])
            p.scale = iqr if iqr > 1e-12 else float(np.ptp(numbers)) if np.ptp(numbers) > 1e-12 else 1.
    return p


def profiles(frame, table='', options=None):
    # Input: incomplete frame and fixed typing options; output: column->profile; module function.
    return {c: profile_column(frame[c], table, options) for c in frame}


def route_specs(specs, kind):
    # Input: route dictionary and inferred type; output: normalized generator specs; module function.
    key = kind if kind in specs else 'categorical' if kind=='ordered_numeric' else 'numeric' if kind=='date' else kind
    result = specs.get(key, [])
    return [{'name':s} if isinstance(s, str) else dict(s) for s in result]


def empirical(values):
    # Input: donor values; output: unique candidates with log-frequency scores; module function.
    counts = Counter(values)
    return [(v, float(np.log(n))) for v, n in sorted(counts.items(), key=lambda x:(type(x[0]).__name__,str(x[0])))]


def quantile_q(values, probabilities, k):
    # Input: finite numeric support/probabilities and K; output: K equal-mass midpoint representatives; module function.
    values, probabilities = np.asarray(values, float), np.asarray(probabilities, float)
    keep = np.isfinite(values) & (probabilities > 0)
    values, probabilities = values[keep], probabilities[keep]
    if not len(values):
        return []
    order = np.argsort(values, kind='stable'); values, probabilities = values[order], probabilities[order]
    probabilities /= probabilities.sum()
    # Inverse empirical CDF: each representative receives its bin mass 1/K.
    indices = np.searchsorted(np.cumsum(probabilities), (np.arange(k)+.5)/k, side='left')
    counts = Counter(values[np.minimum(indices,len(values)-1)])
    return [(float(v), n/k) for v,n in sorted(counts.items())]


def gower_distances(frame, prof, target, device='cpu', rows=None, block_size=256):
    # Input: table/profile/excluded target/device/query rows; output: query-to-all distances, Inf without overlap.
    n=len(frame); queries=np.arange(n) if rows is None else np.asarray(rows,dtype=int)
    if device!='cpu':
        import torch
        from ..numerics import torch_device
        columns=[]; dev=torch_device(device)
        for c,p in prof.items():
            if c==target or p.kind=='identifier': continue
            if p.kind in (*CONTINUOUS,'ordered_numeric'):
                values=np.array([p.number(v) if pd.notna(v) else np.nan for v in frame[c]])
                observed=np.isfinite(values); width=np.ptp(values[observed]) if observed.any() else 0.
            else:
                observed=frame[c].notna().to_numpy()
                values=pd.factorize(frame[c].fillna('').astype(str),sort=False)[0]; width=None
            columns.append((torch.as_tensor(np.nan_to_num(values),dtype=torch.float64,device=dev),
                            torch.tensor(observed,device=dev),width))
        result=np.empty((len(queries),n))
        with torch.inference_mode():
            for start in range(0,len(queries),block_size):
                idx=torch.as_tensor(queries[start:start+block_size],device=dev)
                total=torch.zeros((len(idx),n),dtype=torch.float64,device=dev); count=torch.zeros_like(total)
                for values,observed,width in columns:
                    shared=observed[idx,None]&observed[None,:]
                    delta=(values[idx,None]!=values[None,:]).to(torch.float64) if width is None else (
                        torch.abs(values[idx,None]-values[None,:])/width if width>0 else torch.zeros_like(total))
                    total+=torch.where(shared,delta,0.); count+=shared
                result[start:start+len(idx)]=torch.where(count>0,total/count.clamp_min(1.),torch.inf).cpu().numpy()
        return result
    total=np.zeros((len(queries),n)); count=np.zeros_like(total)
    for c,p in prof.items():
        if c == target or p.kind == 'identifier':
            continue
        if p.kind in (*CONTINUOUS, 'ordered_numeric'):
            values = np.array([p.number(v) if pd.notna(v) else np.nan for v in frame[c]])
            observed = np.isfinite(values); width = np.ptp(values[observed]) if observed.any() else 0.
            delta = np.abs(values[queries,None]-values[None,:])/width if width>0 else np.zeros_like(total)
        else:
            observed = frame[c].notna().to_numpy(); values = frame[c].fillna('').astype(str).to_numpy()
            delta = (values[queries,None]!=values[None,:]).astype(float)
        shared = observed[queries,None]&observed[None,:]
        total += np.where(shared, delta, 0.); count += shared
    return np.divide(total,count,out=np.full_like(total,np.inf),where=count>0)


def predictor_matrix(frame, prof, target):
    # Input: incomplete table excluding target and identifiers; output: numeric features retaining numeric NaNs; module function.
    blocks = []
    for c,p in prof.items():
        if c == target or p.kind == 'identifier':
            continue
        if p.kind in (*CONTINUOUS, 'ordered_numeric'):
            values = np.array([p.number(v) if pd.notna(v) else np.nan for v in frame[c]])
            good = np.isfinite(values)
            if good.any():
                blocks.append(((values-np.nanmedian(values))/p.scale)[:,None])
        else:
            blocks.append(pd.get_dummies(frame[c].fillna('__MISSING__').astype(str), dtype=float).to_numpy())
    return np.concatenate(blocks,axis=1) if blocks else np.empty((len(frame),0))


class TypedGenerator:
    """One configured local proposal; shared calibration is handled by mixture.py."""
    def __init__(self, name, k=10, quantiles=32, smoothing=1., cluster_threshold=.35,
                 n_draws=32, max_iter=10, model=None, device='cpu', predictor_missing='iterative',
                 numerical_backend='sklearn', **options):
        # Input: generator settings; output: configured generator; no inheritance.
        if k<1 or quantiles<1 or n_draws<2 or smoothing<=0 or not 0<cluster_threshold<=1 or max_iter<1:
            raise ValueError('Invalid typed generator settings')
        if options:
            raise TypeError(f'Unknown typed generator options: {sorted(options)}')
        if predictor_missing not in ('iterative','median'):
            raise ValueError('predictor_missing must be iterative or median')
        if numerical_backend not in ('sklearn','torch'): raise ValueError('Unknown numerical_backend')
        self.name,self.k,self.quantiles,self.smoothing = name,k,quantiles,smoothing
        self.cluster_threshold,self.n_draws,self.max_iter = cluster_threshold,n_draws,max_iter
        self.model,self.device = model,device
        self.predictor_missing=predictor_missing
        self.numerical_backend=numerical_backend

    def generate(self, frame, table, iset, ctx, prof=None, keys=None, distance_cache=None):
        # Input: masked table, context, optional profiles/selected cells; output: candidate/raw-score dictionary; TypedGenerator.
        prof = prof or profiles(frame,table.name)
        out={}
        for c,p in prof.items():
            missing=np.flatnonzero(frame[c].isna().to_numpy())
            if keys is not None:
                missing=np.array([r for r in missing if (int(r),c) in keys],dtype=int)
            if not len(missing) or p.kind=='identifier':
                continue
            donors=np.flatnonzero(frame[c].notna().to_numpy()); vals=frame[c].to_numpy(object)
            if not len(donors):
                continue
            fallback=empirical(vals[donors])
            if self.name=='quantile_marginal':
                if p.kind not in CONTINUOUS: continue
                points=quantile_q([p.number(v) for v in vals[donors]],np.ones(len(donors)),self.quantiles)
                q=empirical([p.render(v) for v,w in points for _ in range(max(1,round(w*self.quantiles)))])
                out.update({(int(r),c):q for r in missing}); continue
            if self.name=='mice_posterior':
                if p.kind not in CONTINUOUS: continue
                out.update(self._mice(frame,prof,c,p,missing,ctx)); continue
            if self.name=='nb_classifier':
                if p.kind not in ('categorical','ordered_numeric'): continue
                from sklearn.naive_bayes import CategoricalNB
                x=[]
                for other,op in prof.items():
                    if other==c or op.kind=='identifier': continue
                    if op.kind in (*CONTINUOUS,'ordered_numeric'):
                        z=np.array([op.number(v) if pd.notna(v) else np.nan for v in frame[other]])
                        finite=z[np.isfinite(z)]
                        cuts=np.unique(np.quantile(finite,[.25,.5,.75])) if len(finite) else []
                        x.append(np.where(np.isfinite(z),np.searchsorted(cuts,np.nan_to_num(z),side='right')+1,0))
                    else:
                        codes,_=pd.factorize(frame[other].fillna('__MISSING__').astype(str),sort=True)
                        x.append(codes)
                if not x:
                    out.update({(int(r),c):fallback for r in missing}); continue
                x=np.stack(x,axis=1); labels,classes=pd.factorize(frame[c].iloc[donors],sort=True)
                model=CategoricalNB(alpha=self.smoothing,min_categories=x.max(axis=0)+1).fit(x[donors],labels)
                for r,probs in zip(missing,model.predict_proba(x[missing])):
                    out[(int(r),c)]=[(classes[int(label)],float(np.log(max(prob,1e-300)))) for label,prob in zip(model.classes_,probs)]
                continue
            if self.name!='row_embedding_knn':
                distance_cache={} if distance_cache is None else distance_cache
                full=self.name=='cluster_marginal' or c in distance_cache.get('full_targets',set())
                query=None if full else missing
                signature=(c,self.device,None if full else tuple(missing))
                if signature not in distance_cache:
                    distance_cache[signature]=gower_distances(frame,prof,c,self.device,query)
                distances=distance_cache[signature]
            if self.name=='cluster_marginal':
                from scipy.cluster.hierarchy import linkage,fcluster
                from scipy.spatial.distance import squareform
                d=np.where(np.isfinite(distances),distances,1.); np.fill_diagonal(d,0.)
                groups=fcluster(linkage(squareform(d,checks=False),method='average'),self.cluster_threshold,criterion='distance') if len(frame)>1 else np.ones(len(frame))
                prior=Counter(vals[donors]); total=sum(prior.values())
                for r in missing:
                    local=Counter(vals[donors[groups[donors]==groups[r]]])
                    out[(int(r),c)]=[(v,float(np.log(local[v]+self.smoothing*n/total))) for v,n in sorted(prior.items(),key=lambda x:str(x[0]))]
                continue
            if self.name=='row_embedding_knn':
                if p.kind!='text': continue
                from ..embedding import DEFAULT_MODEL,get_encoder
                # Observed unique titles may help retrieval even though they are never themselves imputed.
                columns=[other for other in frame if other!=c]
                texts=['; '.join(f'{other}: {frame.iloc[r][other]}' for other in columns if pd.notna(frame.iloc[r][other])) for r in range(len(frame))]
                if not columns or not any(texts):
                    out.update({(int(r),c):fallback for r in missing}); continue
                encoder=get_encoder(ctx,model=self.model or DEFAULT_MODEL,device=self.device)
                vectors=encoder.encode_map(texts,namespace='typed-row-excluding-target')
                embeddings=np.stack([vectors[t] for t in texts]).astype(float)
                embeddings/=np.maximum(np.linalg.norm(embeddings,axis=1,keepdims=True),1e-12)
                from ..numerics import matrix_product
                distances=np.clip(1-matrix_product(embeddings[missing],embeddings,self.device),0,2)
                empty=np.array([not t for t in texts]); distances[empty[missing],:]=np.inf; distances[:,empty]=np.inf
            for query_index,r in enumerate(missing):
                row=distances[query_index] if self.name=='row_embedding_knn' or not full else distances[r]
                ordered=sorted((int(d) for d in donors if np.isfinite(row[d])),key=lambda d:(row[d],str(vals[d])))[:self.k]
                if not ordered:
                    out[(int(r),c)]=fallback; continue
                weights={}
                for d in ordered:
                    w=float(np.exp(-row[d]))
                    weights[vals[d]]=weights.get(vals[d],0.)+w
                if self.name=='numeric_kernel':
                    if p.kind!='ordered_numeric': continue
                    observed=list(dict.fromkeys(vals[donors])); donor_x=np.array([p.number(v) for v in weights])
                    donor_w=np.array(list(weights.values())); bandwidth=max(p.scale*.2,1e-9)
                    weights={v:float(np.exp(-.5*((p.number(v)-donor_x)/bandwidth)**2)@donor_w) for v in observed if np.isfinite(p.number(v))}
                out[(int(r),c)]=[(v,float(np.log(w))) for v,w in weights.items() if w>0]
        return out

    def _mice(self, frame, prof, target, p, missing, ctx):
        # Input: masked table/target; output: empirical posterior candidate scores; TypedGenerator.
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        x=predictor_matrix(frame,prof,target)
        y=np.array([p.number(v) if pd.notna(v) else np.nan for v in frame[target]])
        valid=np.isfinite(y)
        if valid.sum()<3 or not x.shape[1]:
            q=empirical(frame[target].dropna().tolist())
            return {(int(r),target):q for r in missing}
        if self.predictor_missing=='median':
            # Complete only predictors before fitting; the hidden target is never used to fill them.
            from sklearn.impute import SimpleImputer
            x=SimpleImputer(strategy='median').fit_transform(x)
            if not x.shape[1]:
                q=empirical(frame[target].dropna().tolist())
                return {(int(r),target):q for r in missing}
        center=np.nanmedian(y); matrix=np.column_stack([(y-center)/p.scale,x])
        seed=(ctx.seed+zlib.crc32(str(target).encode()))%2**32
        from .torch_bayes import mice_estimator
        model=IterativeImputer(estimator=mice_estimator(self.numerical_backend,self.device),
            sample_posterior=True,max_iter=self.max_iter,random_state=seed,skip_complete=True)
        model.fit(matrix)
        draws=[model.transform(matrix)[missing,0]*p.scale+center for _ in range(self.n_draws)]
        out={}
        for r,values in zip(missing,np.asarray(draws).T):
            points=quantile_q(values,np.ones(len(values)),self.quantiles)
            weights={}
            for v,w in points:
                value=p.render(v); weights[value]=weights.get(value,0.)+w
            out[(int(r),target)]=[(v,float(np.log(w))) for v,w in weights.items()]
        return out


def register_generators(registry):
    # Input: existing generator registry; output: in-place registration of typed proposal factories; module function.
    for name in ('gower_knn','cluster_marginal','quantile_marginal','mice_posterior','row_embedding_knn','nb_classifier','numeric_kernel'):
        registry[name]=lambda params,name=name:TypedGenerator(name,**params)


def candidate_examples(specs, frame, table, iset, ctx, options=None, keys=None):
    # Input: routes and masked table; output: missing cell -> proposals/profile/route; module function.
    from .mixture import GENERATORS
    prof=profiles(frame,table.name,options)
    result={}; cache={}
    distance_cache={'full_targets':{c for c,p in prof.items()
        if any(s['name']=='cluster_marginal' for s in route_specs(specs,p.kind))}}
    for c,p in prof.items():
        specs_for_type=route_specs(specs,p.kind)
        for r in np.flatnonzero(frame[c].isna().to_numpy()):
            if keys is not None and (int(r),c) not in keys: continue
            result[(int(r),c)]={'generators':[],'profile':asdict(p),'kind':p.kind,'route':p.kind}
        relevant={key for key in result if key[1]==c}
        if not relevant: continue
        for spec in specs_for_type:
            # Each target is computed in a frame where every pseudo-mask is already hidden.
            signature=repr(spec)
            if signature not in cache:
                generator=GENERATORS[spec['name']](spec.get('params',{}))
                if isinstance(generator,TypedGenerator):
                    cache[signature]=generator.generate(frame,table,iset,ctx,prof,keys,distance_cache)
                else:
                    cache[signature]=generator.generate(frame,table,iset,ctx)
            for key in relevant:
                result[key]['generators'].append(cache[signature].get(key,[]))
    return result


def numeric_support(q, profile, dispersion=1.):
    # Input: finite candidate Q, formatting profile, spread factor; output: numeric support/probabilities; module function.
    p=ColumnProfile(**profile); pairs=[(p.number(v),w) for v,w in q if np.isfinite(p.number(v))]
    if not pairs: return np.array([]),np.array([])
    values,weights=np.array(pairs,float).T; weights/=weights.sum()
    center=float(values@weights)
    return center+float(dispersion)*(values-center),weights


def crps(values, probabilities, truth):
    # Input: finite atomic distribution and observed numeric truth; output: CRPS in target units; module function.
    if not len(values): return np.nan
    values,probabilities=np.asarray(values,float),np.asarray(probabilities,float)
    if (values.shape!=probabilities.shape or not np.isfinite(values).all()
            or not np.isfinite(probabilities).all() or (probabilities<0).any()
            or probabilities.sum()<=0 or not np.isfinite(truth)):
        raise ValueError('CRPS requires finite values/truth and nonnegative probability mass')
    order=np.argsort(values); x=values[order]; p=probabilities[order]/probabilities.sum()
    cdf=np.cumsum(p)[:-1]
    # Half E|X-X'| through adjacent gaps avoids cancellation of large epoch dates.
    return float(p@np.abs(x-truth)-np.sum(np.diff(x)*cdf*(1-cdf)))


def fit_continuous(examples, n_generators):
    # Input: fit-only numeric/date masks; output: bounded mixture/spread fit and convergence diagnostics; module function.
    prepared=[]
    for ex in examples:
        profile=ColumnProfile(**ex['profile']); values,probabilities,owners=[],[],[]
        for index,proposal in enumerate(ex['generators']):
            if not proposal: continue
            mass=softmax(np.array([score for value,score in proposal],float))
            for (value,score),weight in zip(proposal,mass):
                number=profile.number(value)
                if np.isfinite(number):
                    # Center before mixing/scaling: date origin must not affect calibration.
                    values.append((number-ex['numeric_truth'])/profile.scale)
                    probabilities.append(weight); owners.append(index)
        if values:
            prepared.append((np.array(values),np.array(probabilities),np.array(owners,int)))
    if not prepared:
        return {**default_parameters(n_generators,'no_valid_numeric_candidates'),'n_usable_masks':0}

    def unpack(z):
        # Input: log spread and bounded mixture logits; output: positive spread and simplex weights; nested function.
        return np.exp(z[0]),softmax(z[1:])

    def loss(z):
        # Input: bounded parameters; output: mean IQR-normalized CRPS; nested function.
        spread,alpha=unpack(z); scores=[]
        for x,base,owner in prepared:
            p=base*alpha[owner]; p/=p.sum()
            center=float(x@p)
            scores.append(crps(center+spread*(x-center),p,0.))
        return float(np.mean(scores))

    # Atomic CRPS contains absolute-value kinks. Powell does not require a smooth gradient.
    initial=np.zeros(n_generators+1); initial_loss=loss(initial)
    fit=minimize(loss,initial,method='Powell',
        bounds=[(np.log(.25),np.log(4.))]+[(-12.,12.)]*n_generators,
        options={'xtol':1e-6,'ftol':1e-8,'maxiter':1000,'maxfev':20000})
    if not fit.success or not np.isfinite(fit.x).all() or not np.isfinite(fit.fun):
        raise RuntimeError(f'CRPS calibration failed with Powell: {fit.message}')
    # A converged search can offer no improvement; retain the documented initial point then.
    selected=fit.x if loss(fit.x)<=initial_loss else initial
    spread,weights=unpack(selected)
    return {'temperatures':[1.]*n_generators,'weights':weights.tolist(),'dispersion':float(spread),
        'fit_status':'fitted_crps','fit_crps_initial':initial_loss,'fit_crps':loss(selected),
        'optimizer':'Powell','optimizer_success':bool(fit.success),'optimizer_message':str(fit.message),
        'optimizer_nfev':int(fit.nfev),'retained_initial':bool(selected is initial),'n_usable_masks':len(prepared)}


def default_parameters(n, status='unseen_type'):
    # Input: generator count and status; output: explicit unfitted neutral fallback; module function.
    return {'temperatures':[1.]*n,'weights':[1/n]*n if n else [],'dispersion':1.,'fit_status':status}


def fit_typed(examples, specs, type_options=None):
    # Input: fit-only pseudo-masks and fixed routes; output: per-type frozen NLL/CRPS parameters; module function.
    from .mixture import fit_mixture
    fitted={}
    for kind in KINDS:
        n=len(route_specs(specs,kind)); group=[ex for ex in examples if ex['kind']==kind and any(ex['generators'])]
        params=default_parameters(n,'no_usable_fit_masks')
        if group and n:
            if kind not in CONTINUOUS:
                covered=[ex for ex in group if any(ex['truth'] in dict(q) for q in ex['generators'])]
                if covered:
                    params.update(fit_mixture(group,n),fit_status='fitted_nll')
                else: params['fit_status']='no_exact_truth_coverage'
            else:
                group=[ex for ex in group if np.isfinite(ex.get('numeric_truth',np.nan))]
                if group:
                    params.update(fit_continuous(group,n))
        params['n_fit_masks']=len(group); fitted[kind]=params
    return {'routing':'by_type','typed_version':1,'generators':specs,'type_options':type_options or {},'by_type':fitted}


def calibrated_q(example, parameters, k_max=10):
    # Input: one typed proposal bundle and frozen parameters; output: finite Q plus approximation metadata; module function.
    from .mixture import build_qx
    kind=example['kind']; params=parameters['by_type'][kind]
    q,alpha=build_qx(example['generators'],params['temperatures'],params['weights']) if example['generators'] else ([],[])
    info={'column_type':kind,'active_weights':alpha,'type_fit_status':params['fit_status']}
    if kind in CONTINUOUS and q:
        from scipy.stats import wasserstein_distance
        p=ColumnProfile(**example['profile']); valid_mass=sum(w for v,w in q if np.isfinite(p.number(v)))
        values,weights=numeric_support(q,example['profile'],params['dispersion'])
        points=quantile_q(values,weights,k_max); rendered={}
        for v,w in points:
            value=p.render(v); rendered[value]=rendered.get(value,0.)+w
        q=list(rendered.items())
        represented=np.array([p.number(v) for v,w in q]); represented_weights=np.array([w for v,w in q])
        info.update(quantile_compressed=True,dispersion=params['dispersion'],representation='equal-mass midpoint inverse CDF',
                    quantization_w1_normalized=float(wasserstein_distance(values,represented,weights,represented_weights)/p.scale) if len(values) and len(q) else None,
                    invalid_numeric_mass=float(max(0.,1-valid_mass)))
    return CellDistribution(q,info)


def typed_diagnostics(examples, parameters, rho=.95, k_max=10):
    # Input: held-out masked truths and frozen Q; output: per-type categorical scores or numeric CRPS/coverage/PIT; module function.
    from .base import truncate_and_normalize
    rows=[]
    for kind in KINDS:
        group=[ex for ex in examples if ex['kind']==kind]; outcomes=[]; pits=[]
        for ex in group:
            proposal=calibrated_q(ex,parameters,k_max)
            kept,stats=truncate_and_normalize(proposal.candidates,1. if kind in CONTINUOUS else rho,k_max)
            if kind in CONTINUOUS:
                p=ColumnProfile(**ex['profile']); y=p.number(ex['truth'])
                values,weights=numeric_support(kept,ex['profile']); values0,w0=numeric_support(
                    normalise([(v,np.exp(s)) for v,s in ex.get('marginal_baseline',[])]),ex['profile'])
                baseline_points=quantile_q(values0,w0,k_max)
                values0=np.array([p.number(p.render(v)) for v,w in baseline_points])
                w0=np.array([w for v,w in baseline_points])
                if len(values) and np.isfinite(y):
                    order=np.argsort(values); x=values[order]; w=weights[order]; cdf=np.cumsum(w)
                    low,high=x[np.minimum(np.searchsorted(cdf,[.1,.9]),len(x)-1)]
                    score=crps(x,w,y)/p.scale; baseline=crps(values0,w0,y)/p.scale
                    outcomes.append({'crps_normalized':score,'baseline_crps_normalized':baseline,
                        'crps_improvement':baseline-score,'coverage_80':float(low<=y<=high),'interval_width_normalized':float((high-low)/p.scale)})
                    outcomes[-1]['quantization_w1_normalized']=proposal.info.get('quantization_w1_normalized',0.)
                    # Mid-PIT is deterministic for atoms; it need not be uniform for coarse finite Q.
                    pits.append(float(w[x<y].sum()+.5*w[x==y].sum()))
            elif kind!='identifier':
                q,_=build_raw_q(ex,parameters)
                before,after=dict(q).get(ex['truth'],0.),dict(kept).get(ex['truth'],0.)
                outcomes.append({'union_covered':float(before>0),'recall_at_k':float(after>0),
                    'top1_accuracy':float(bool(kept and kept[0][0]==ex['truth'])),
                    'nll_union':float(-np.log(max(before,1e-12))),'nll_retained':float(-np.log(max(after,1e-12))),**stats})
        row={'column_type':kind,'n_masks':len(group),'n_evaluated':len(outcomes),
             'n_unavailable':len(group)-len(outcomes),'status':'observed_only' if kind=='identifier' else 'ok' if outcomes else 'no_usable_masks'}
        if outcomes:
            row.update({k:float(np.mean([o[k] for o in outcomes])) for k in outcomes[0]})
        if kind in CONTINUOUS:
            row['pit_histogram']=np.histogram(pits,bins=np.linspace(0,1,11))[0].tolist()
            row['pit_definition']='mid-PIT for finite atomic Q; coarse discretization can prevent uniformity'
        rows.append(row)
    return {'n_masks':len(examples),'routing':'by_type','by_type':rows}


def build_raw_q(example,parameters):
    # Input: typed example and frozen artifact; output: untruncated mixture/active weights; module function.
    from .mixture import build_qx
    p=parameters['by_type'][example['kind']]
    return build_qx(example['generators'],p['temperatures'],p['weights']) if example['generators'] else ([],[])


@UNCERTAINTY_MODELS.register('statistical')
class StatisticalImputer(DeterministicModel):
    """Typed mean/mode, numeric sklearn KNN/MICE, or mixed-type iterative random forests."""
    def __init__(self, method='mean_mode', k=5, max_iter=5, n_estimators=50, type_options=None,
                 numerical_backend='sklearn',device='auto'):
        # Input: baseline method/settings; output: configured deterministic baseline; inherits DeterministicModel.
        if method not in ('mean_mode','median_mode','knn','mice','missforest'):
            raise ValueError('Unknown statistical imputer')
        if k<1 or max_iter<1 or n_estimators<1: raise ValueError('Baseline settings must be positive')
        self.method,self.k,self.max_iter,self.n_estimators,self.type_options=method,k,max_iter,n_estimators,type_options or {}
        if numerical_backend not in ('sklearn','torch'): raise ValueError('Unknown numerical_backend')
        self.numerical_backend,self.device=numerical_backend,device

    def fill_table(self, frame, table, iset, ctx):
        # Input: incomplete local table; output: filled copy preserving observed cells/identifier missingness; StatisticalImputer override.
        prof=profiles(frame,table.name,self.type_options); out=frame.copy()
        numeric=[c for c,p in prof.items() if p.kind in (*CONTINUOUS,'ordered_numeric') and p.observed]
        if self.method in ('knn','mice') and numeric:
            # Numeric-only sklearn baselines: categorical cells use the same explicit mode fallback.
            matrix=np.array([[prof[c].number(v) if pd.notna(v) else np.nan for v in frame[c]] for c in numeric]).T
            centers=np.nanmedian(matrix,axis=0); scales=np.array([prof[c].scale for c in numeric]); scaled=(matrix-centers)/scales
            if self.method=='knn':
                if self.numerical_backend=='torch':
                    from ..numerics import knn_impute
                    values=knn_impute(scaled,self.k,self.device)
                else:
                    from sklearn.impute import KNNImputer
                    values=KNNImputer(n_neighbors=self.k,weights='distance').fit_transform(scaled)
            else:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer
                from .torch_bayes import mice_estimator
                values=IterativeImputer(estimator=mice_estimator(self.numerical_backend,self.device),
                    max_iter=self.max_iter,sample_posterior=True,random_state=ctx.seed).fit_transform(scaled)
            values=values*scales+centers
            for j,c in enumerate(numeric):
                observed=[prof[c].number(v) for v in frame[c].dropna()]
                observed=[v for v in observed if np.isfinite(v)]
                for r in np.flatnonzero(frame[c].isna().to_numpy()):
                    value=min(observed,key=lambda v:abs(v-values[r,j])) if prof[c].kind=='ordered_numeric' else values[r,j]
                    out.iat[r,out.columns.get_loc(c)]=prof[c].render(value)
        for c,p in prof.items():
            if p.kind=='identifier' or not p.observed: continue
            missing=frame[c].isna()
            if p.kind in (*CONTINUOUS,'ordered_numeric'):
                if self.method in ('knn','mice'): continue
                numbers=[p.number(v) for v in frame[c].dropna()]; numbers=np.array(numbers); numbers=numbers[np.isfinite(numbers)]
                if not len(numbers): continue
                fill=np.median(numbers) if self.method=='median_mode' else np.mean(numbers)
                if p.kind=='ordered_numeric': fill=min(numbers,key=lambda x:abs(x-fill))
                out.loc[missing,c]=p.render(fill)
            else:
                counts=Counter(frame[c].dropna()); fill=sorted(counts,key=lambda v:(-counts[v],str(v)))[0]
                out.loc[missing,c]=fill
        if self.method=='missforest':
            # MissForest-style fixed-iteration implementation; deliberately not labelled original package reproduction.
            from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
            targets=sorted((c for c in frame if prof[c].kind!='identifier' and frame[c].isna().any() and prof[c].observed),key=lambda c:frame[c].isna().sum())
            for iteration in range(self.max_iter):
                for c in targets:
                    p=prof[c]; observed=frame[c].notna().to_numpy(); missing=~observed
                    x=predictor_matrix(out,prof,c)
                    if not x.shape[1]: continue
                    x=np.nan_to_num(x)
                    # Parallel tree construction is numerically identical for a fixed
                    # random_state; it only removes the single-core bottleneck of the
                    # MissForest-style baseline.
                    kwargs=dict(n_estimators=self.n_estimators,random_state=ctx.seed+iteration,
                                n_jobs=getattr(self, 'n_jobs', -1))
                    if p.kind in (*CONTINUOUS,'ordered_numeric'):
                        y=np.array([p.number(v) for v in frame.loc[observed,c]])
                        good=np.isfinite(y)
                        if not good.any(): continue
                        model=RandomForestRegressor(**kwargs).fit(x[observed][good],y[good]); predictions=model.predict(x[missing])
                        if p.kind=='ordered_numeric': predictions=[min(y[good],key=lambda v:abs(v-z)) for z in predictions]
                        predictions=[p.render(z) for z in predictions]
                    else:
                        labels,classes=pd.factorize(frame.loc[observed,c],sort=True)
                        model=RandomForestClassifier(**kwargs).fit(x[observed],labels)
                        predictions=[classes[int(z)] for z in model.predict(x[missing])]
                    out.loc[missing,c]=predictions
        return out
