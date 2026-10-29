"""Convert pinned Alaska JSON specifications to annotation-covered multi-table tasks."""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
import pandas as pd

from umsm.data.preparation import clean_cell, fetch_file, file_sha256, partition_annotations, publish_task

ROOT = Path(__file__).resolve().parent
REVISION = 'fb452aef32cd648eae4bcbdc984789b7f6f07960'
SOURCE = 'https://github.com/merialdo/research.alaska'


def read_archive(path, domain):
    # 输入：官方 tar.gz、领域名；输出：源表字典、官方属性映射；继承：无（模块函数）。
    rows, gold = defaultdict(list), None
    with tarfile.open(path, mode='r|gz') as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile():
                continue
            if member.name.endswith('_schema_matching_gt.csv'):
                gold = pd.read_csv(io.BytesIO(archive.extractfile(member).read()))
            elif len(parts) == 4 and parts[:2] == (domain, domain+'_specs') and parts[-1].endswith('.json'):
                record = json.load(archive.extractfile(member))
                rows[parts[2]+'.csv'].append({str(k): clean_cell(v) for k, v in record.items()})
    if gold is None:
        raise ValueError('Schema-matching ground truth missing from archive')
    mappings = []
    for row in gold.itertuples(index=False):
        source, column = row.source_attribute_id.split('//', 1)
        mappings.append((source+'.csv', column, row.target_attribute_name))
    return {name: pd.DataFrame(records) for name, records in sorted(rows.items())}, mappings


def prepare(domain, name='Alaska', max_rows=300, seed=42, alias_policy='most_observed', root=ROOT):
    # 输入：领域、输出数据集名、行上限、种子、别名规则；输出：全源及不交叠源划分的来源清单；继承：无（模块函数）。
    if not name.replace('_', '').replace('-', '').isalnum() or max_rows < 1:
        raise ValueError('Use a safe dataset name and positive max_rows')
    archive = fetch_file(f'https://raw.githubusercontent.com/merialdo/research.alaska/{REVISION}/{domain}.tar.gz',
                         root / 'cache/datasets/new_benchmarks' / (domain+'.tar.gz'))
    frames, mappings = read_archive(archive, domain)
    frames, groups, audit = partition_annotations(frames, mappings, alias_policy, drop_absent=True)
    original_rows = {n: len(f) for n, f in frames.items()}
    frames = {n: f.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)
              if len(f) > max_rows else f.reset_index(drop=True) for n, f in frames.items()}
    names = np.random.default_rng(seed).permutation(sorted(frames)).tolist()
    parts = np.array_split(np.array(names, dtype=object), 3)
    cases = [(domain, names, 'all')] + [(domain+'_'+role, list(selected), role)
             for role, selected in zip(('fit', 'validation', 'test'), parts) if len(selected) >= 2]
    manifests = []
    for case, selected, role in cases:
        allowed = {f'{n}.{c}' for n in selected for c in frames[n]}
        gold = [frozenset(g & allowed) for g in groups if g & allowed]
        protocol = {'adapter_version': 1, 'source': SOURCE, 'revision': REVISION, 'domain': domain,
                    'role': role, 'max_rows': max_rows, 'seed': seed, 'alias_policy': alias_policy,
                    'ambiguous_source_policy': 'drop source attributes mapping to multiple targets',
                    'features': 'annotated source names/values only; target names are labels only',
                    'missingness': 'natural JSON attribute absence and original nulls'}
        manifest = publish_task(root / 'data' / name / case, {n: frames[n] for n in selected}, gold, protocol,
                                archive_sha256=file_sha256(archive), annotation_audit=audit,
                                source_tables=selected, original_rows=original_rows)
        manifests.append(manifest)
        missing = np.mean([t['missing_rate'] for t in manifest['tables']])
        print(f'{name}/{case}: {len(selected)} tables; {manifest["gt_pairs"]} GT pairs; mean table nulls={missing:.1%}')
    return manifests


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=('camera', 'monitor', 'notebook', 'all'), default='all')
    parser.add_argument('--name', default='Alaska')
    parser.add_argument('--max-rows', type=int, default=300, help='Use a large value for complete rows')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alias-policy', choices=('most_observed', 'drop_target'), default='most_observed')
    args = parser.parse_args()
    for domain in ('camera', 'monitor', 'notebook') if args.domain == 'all' else (args.domain,):
        prepare(domain, args.name, args.max_rows, args.seed, args.alias_policy)
