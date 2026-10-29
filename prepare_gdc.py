"""Derive source-to-source GDC schema partitions from the official GDC-SM release."""
import argparse
import io
import json
from pathlib import Path
import zipfile

import pandas as pd

from umsm.data.preparation import clean_cell, fetch_file, file_sha256, partition_annotations, publish_task

ROOT = Path(__file__).resolve().parent
RECORD = 'https://zenodo.org/api/records/14963588'


def read_source(name, info, cache, source_dir=None):
    # 输入：论文表名、官方下载元数据、缓存、可选已下载 CSV 目录；输出：与官方转换一致的临床 DataFrame；继承：无（模块函数）。
    if source_dir:
        path = Path(source_dir) / name
        if not path.exists():
            raise FileNotFoundError(f'Provide all official input CSV tables; missing {path}')
        return pd.read_csv(path, skip_blank_lines=False), path
    url = info['Dataset URL']
    path = fetch_file(url, cache / Path(url).name)
    try:
        frame = pd.read_excel(path, sheet_name=info['Sheet Name'])
    except ImportError as exc:
        raise ImportError('GDC Excel conversion requires: python -m pip install openpyxl') from exc
    if name == 'Vasaikar.csv':
        frame = frame.iloc[1:-1]
    if name == 'McDermott.csv':
        frame = frame.rename(columns={'Age in Months at Time of Tissue Procurement \t': 'Age in Months at Time of Tissue Procurement'})
    if name == 'Clark.csv':
        frame['Ethnicity_Self_Identify'] = frame['Ethnicity_Self_Identify'].replace('Whte', 'White')
    if name == 'Wang.csv':
        frame['secondhand_smoke_exposure'] = frame['secondhand_smoke_exposure'].replace({
            'No or minimal exposure to secondhand smoke': 'No', 'Exposed in childhood houshold': 'Yes',
            'Exposed in current household': 'Yes'})
    return frame, path


def prepare(name='GDC', max_rows=300, seed=42, alias_policy='most_observed', source_dir=None, root=ROOT):
    # 输入：数据集名、行上限、种子、别名规则、本地 CSV 来源；输出：GDC 多源任务清单；继承：无（模块函数）。
    if not name.replace('_', '').replace('-', '').isalnum() or max_rows < 1:
        raise ValueError('Use a safe dataset name and positive max_rows')
    cache = root / 'cache/datasets/new_benchmarks'
    meta = fetch_file(RECORD+'/files/papers_info.json/content', cache / 'gdc_papers_info.json')
    archive = fetch_file(RECORD+'/files/gdc-sm-data.zip/content', cache / 'gdc-sm-data.zip')
    papers, frames, mappings, source_files = json.loads(meta.read_bytes()), {}, [], {}
    with zipfile.ZipFile(archive) as zipped:
        for table, info in sorted(papers.items()):
            frame, path = read_source(table, info, cache / 'gdc_sources', source_dir)
            if not frame.columns.is_unique:
                raise ValueError(f'Duplicate source headers: {table}')
            frames[table] = frame.map(clean_cell)
            source_files[table] = {'url': info['Dataset URL'], 'sha256': file_sha256(path), 'sheet': info['Sheet Name']}
            gold = pd.read_csv(io.BytesIO(zipped.read('data/ground-truth/'+table)))
            mappings.extend((table, str(row.iloc[0]), str(row.iloc[1])) for _, row in gold.iterrows())
    frames, groups, audit = partition_annotations(frames, mappings, alias_policy)
    frames = {n: f.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)
              if len(f) > max_rows else f.reset_index(drop=True) for n, f in frames.items()}
    protocol = {'adapter_version': 1, 'source': 'https://doi.org/10.5281/zenodo.14963588',
                'max_rows': max_rows, 'seed': seed, 'alias_policy': alias_policy,
                'task': 'derived source-source partition; not official source-to-736-column benchmark',
                'ambiguous_source_policy': 'drop multi-target source columns; never join target aliases transitively',
                'source_conversion': 'official sheet selection and four documented cleanups; preserve other values'}
    manifest = publish_task(root / 'data' / name / 'clinical', frames, groups, protocol,
                            archive_sha256=file_sha256(archive), metadata_sha256=file_sha256(meta),
                            source_files=source_files, annotation_audit=audit)
    print(f'{name}/clinical: {len(frames)} tables; {manifest["gt_pairs"]} GT pairs')
    for profile in manifest['tables']:
        print(profile['table'], f'nulls={profile["missing_rate"]:.1%}', 'columns=', profile['columns'])
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='GDC')
    parser.add_argument('--max-rows', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alias-policy', choices=('most_observed', 'drop_target'), default='most_observed')
    parser.add_argument('--source-dir', help='Fallback: official data/input-tables CSV folder, never the target table')
    args = parser.parse_args()
    prepare(args.name, args.max_rows, args.seed, args.alias_policy, args.source_dir)
