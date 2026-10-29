"""Reproducible additional masking on real cells; never overwrite natural-missing inputs."""
from dataclasses import replace
import hashlib

import numpy as np
import pandas as pd
from scipy.special import expit


def stable_seed(seed, name):
    # 输入：运行种子、字符串标识；输出：稳定派生整数种子；继承：无（模块函数）。
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:4], 'little')


def value_feature(series):
    # 输入：一列取值；输出：数值秩或稳定字符串哈希形成的标准化特征；继承：无（模块函数）。
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().sum() >= max(2, int(.8*series.notna().sum())):
        values = numeric.rank(pct=True).fillna(.5).to_numpy(float)
    else:
        values = np.array([stable_seed(0,str(v))/2**32 if pd.notna(v) else .5 for v in series])
    std = values.std()
    return (values-values.mean())/std if std > 0 else values*0


def mask_frame(frame, mechanism, rate, seed, donor=None):
    # 输入：真实表、机制、期望额外遮蔽率、种子与可选自然掩码来源；输出：遮蔽表、布尔掩码、机制审计；继承：无（模块函数）。
    if mechanism not in ('mcar','mar','mnar','transplant') or not 0 < rate < 1:
        raise ValueError('Unknown mechanism or rate outside (0,1)')
    rng = np.random.default_rng(seed)
    observed = frame.notna().to_numpy()
    mask = np.zeros(frame.shape, dtype=bool)
    audit = {'mechanism': mechanism, 'requested_rate_on_observed': rate, 'seed': seed}
    if mechanism == 'transplant':
        if donor is None or donor.empty:
            raise ValueError('Transplant requires a real donor table')
        row_index = rng.choice(len(donor), size=len(frame), replace=len(donor)<len(frame))
        col_order = rng.permutation(len(donor.columns))
        col_index = np.resize(col_order, len(frame.columns))
        mask = donor.isna().to_numpy()[np.ix_(row_index,col_index)] & observed
        audit.update(donor_rows=row_index.tolist(), donor_columns=col_index.tolist(),
                     note='Preserve sampled donor row/column pattern; rate argument does not rescale this mask.')
    else:
        # MAR drivers stay observed, so the mechanism really depends on observed context.
        driver = max(range(len(frame.columns)), key=lambda i: int(observed[:,i].sum())) if mechanism=='mar' else None
        if mechanism=='mar' and len(frame.columns) < 2:
            raise ValueError('MAR requires at least two columns')
        audit['mar_driver'] = str(frame.columns[driver]) if driver is not None else None
        for c in range(len(frame.columns)):
            if c == driver:
                continue
            eligible = observed[:,c].copy()
            if driver is not None:
                eligible &= observed[:,driver]
            if not eligible.any():
                continue
            feature = np.zeros(len(frame)) if mechanism=='mcar' else value_feature(frame.iloc[:,driver if driver is not None else c])
            low, high = -40.,40.
            for _ in range(80):
                middle = (low+high)/2
                if expit(middle+feature[eligible]).mean() < rate:
                    low = middle
                else:
                    high = middle
            probability = expit((low+high)/2+feature)
            mask[:,c] = (rng.random(len(frame)) < probability) & eligible
    result = frame.astype(object).mask(mask)
    audit.update(n_observed_before=int(observed.sum()), n_added_missing=int(mask.sum()),
                 achieved_additional_rate=float(mask.sum()/observed.sum()) if observed.sum() else 0.,
                 actual_total_missing_rate=float(result.isna().to_numpy().mean()),
                 cells=np.argwhere(mask).tolist())
    return result, mask, audit


def base_sample(source, rows, seed, dense_max_missing=1., complete_rows=False, min_rows=2, min_columns=1):
    # 输入：真实多表、行数、种子、密度筛选和完整行选项；输出：带同列真值的 oracle 基准输入；继承：无（模块函数）。
    tables = []
    for table in source.tables:
        keep = table.df.isna().mean() <= dense_max_missing
        frame = table.df.loc[:,keep].copy()
        if complete_rows:
            frame = frame.dropna()
        if len(frame.columns)<min_columns or len(frame)<min_rows:
            continue
        if len(frame)>rows:
            frame = frame.sample(rows,random_state=stable_seed(seed,table.name)).sort_index()
        tables.append(replace(table,df=frame.reset_index(drop=True),path=None))
    if len(tables)<2:
        raise ValueError(f'{source.name}: fewer than two eligible tables after density/complete-row selection')
    allowed = {c.id for t in tables for c in t.columns}
    gold = [frozenset(g&allowed) for g in (source.groundtruth or []) if g&allowed]
    if not any(len(g)>1 for g in gold):
        raise ValueError('No positive correspondence left after predeclared density selection')
    return replace(source,tables=tables,groundtruth=gold)
