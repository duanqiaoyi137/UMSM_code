"""Disjoint natural Alaska source tasks; no masking or synthetic values."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from umsm.calibration import set_fingerprint
from umsm.data.loader import load_dataset
from umsm.data.preparation import publish_task
from umsm.research import attach_preparation
from umsm.report import write_report


def site(table):
    # Input: Alaska source filename; output: normalized observed vendor hostname.
    return table.name.removesuffix('.csv').lower().removeprefix('www.')


def prepare(root, dataset='Alaska_Partition', min_tables=6, seed=2026):
    # Input: existing full natural domains; output: immutable disjoint pool/test tasks and preparation audit.
    root=Path(root); domains=('camera','monitor','notebook'); used=set(); audit=[]; exclusions=[]
    for domain in domains:
        source=attach_preparation(load_dataset('Alaska_full',str(root/'data'),sets=[domain]))[0]
        selected=[]
        for table in sorted(source.tables,key=lambda t:t.name):
            host=site(table)
            if host in used: exclusions.append({'domain':domain,'table':table.name,'reason':'website assigned to earlier domain'}); continue
            used.add(host); selected.append(table)
        if len(selected)<min_tables: raise ValueError(f'{domain}: too few unique sites for one task')
        count=len(selected)//min_tables
        rng=np.random.default_rng(seed); shuffled=[selected[i] for i in rng.permutation(len(selected))]
        groups=[list(g) for g in np.array_split(np.asarray(shuffled,dtype=object),count)]
        for name,tables in [(domain+'_pool',selected)]+[(f'{domain}_test_{i:02d}',g) for i,g in enumerate(groups,1)]:
            subset=replace(source,tables=tables); allowed={c.id for c in subset.columns}
            truth=[frozenset(g&allowed) for g in source.groundtruth or [] if g&allowed]
            if not any(len(g)>1 for g in truth): raise ValueError(f'{name}: no positive match groups')
            protocol={'adapter_version':1,'construction':'natural_disjoint_source_partition','source_dataset':'Alaska_full',
                'source_set':source.name,'source_task_fingerprint':set_fingerprint(source),
                'source_fingerprint':set_fingerprint(subset),'source_sites':sorted(site(t) for t in tables),
                'seed':seed,'min_tables':min_tables,'domain':domain,'missingness':'unchanged natural missingness',
                'site_policy':'normalized exact hostname, first domain owns duplicated sites'}
            manifest=publish_task(root/'data'/dataset/name,{t.name:t.df for t in tables},truth,protocol)
            audit.append({'set':name,'domain':domain,'tables':len(tables),'gt_pairs':manifest['gt_pairs'],
                          'sites':protocol['source_sites']})
    path=root/'output/results/alaska_partition_preparation.html'
    write_report(path,{'run_id':'alaska-partitions','name':'Alaska natural disjoint source tasks','kind':'preparation_audit',
        'rows':[],'analysis_tables':{'Prepared tasks':audit,'Excluded repeated websites':exclusions},
        'note':'Different normalized hostnames are a practical source boundary, not proof of statistical independence; inspect linked companies and cross-fold shared fitted models.'})
    print('Prepared Alaska tasks:',len(audit),'test tasks:',sum('_test_' in a['set'] for a in audit),'audit:',path,flush=True)
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='Alaska_Partition'); parser.add_argument('--min-tables',type=int,default=6)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args(); prepare(Path(__file__).resolve().parent,args.name,args.min_tables,args.seed)
