"""Shared cell parsing and content hashes for public-data preparation."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
from urllib.request import Request, urlopen

import pandas as pd

from .groundtruth import write_groundtruth


def clean_cell(value):
    # 输入：value；输出：统一空值后的原值、JSON 文本或 None；继承：无（模块函数）。
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False) if value else None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True) if value else None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null", "nan", "[]"):
        return None
    return value


def sha256(raw):
    # 输入：raw；输出：内容 SHA256 十六进制字符串；继承：无（模块函数）。
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path):
    # 输入：path；输出：文件内容 SHA256 字符串；继承：无（模块函数）。
    return sha256(Path(path).read_bytes())


def fetch_file(url, path, expected=None):
    # 输入：公开 URL、缓存路径、可选 SHA256；输出：已验证的本地 Path；继承：无（模块函数）。
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={'User-Agent': 'UMSM-data-preparation'}), timeout=60) as response:
                    with temporary.open('wb') as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                temporary.replace(path)
                break
            except (OSError, EOFError):
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    if expected and file_sha256(path) != expected:
        raise ValueError(f'Content hash mismatch: {path}')
    return path


def partition_annotations(frames, mappings, alias_policy='most_observed', drop_absent=False):
    # 输入：原始表、(表名,源列,目标属性) 映射、同表别名策略；输出：保留表、语义组、排除审计；继承：无（模块函数）。
    from collections import defaultdict
    if alias_policy not in ('most_observed', 'drop_target'):
        raise ValueError('alias_policy must be most_observed or drop_target')
    targets, audit = defaultdict(set), []
    for table, column, target in mappings:
        if table not in frames or column not in frames[table]:
            if drop_absent:
                audit.append({'table':table,'column':column,'target':target,'reason':'annotation_absent_from_source'})
                continue
            raise ValueError(f'Annotated column absent: {table}.{column}')
        targets[table, column].add(target)
    unique = defaultdict(list)
    for (table, column), values in sorted(targets.items()):
        if len(values) != 1:
            audit.append({'table': table, 'column': column, 'reason': 'source_column_has_multiple_targets',
                          'targets': sorted(values)})
        else:
            unique[table, next(iter(values))].append(column)
    dropped_targets = {target for (_, target), columns in unique.items() if len(columns) > 1}
    selected, groups = defaultdict(list), defaultdict(set)
    for (table, target), columns in sorted(unique.items()):
        if alias_policy == 'drop_target' and target in dropped_targets:
            audit.append({'table': table, 'target': target, 'columns': columns, 'reason': 'ambiguous_target_excluded_globally'})
            continue
        # Decide using all source observations, before pilot row sampling; ties use source name.
        keep = min(columns, key=lambda c: (-int(frames[table][c].notna().sum()), c))
        selected[table].append(keep)
        groups[target].add(f'{table}.{keep}')
        if len(columns) > 1:
            audit.append({'table': table, 'target': target, 'kept': keep,
                          'excluded': sorted(set(columns)-{keep}), 'reason': 'same_table_alias'})
    output = {name: frame[[c for c in frame if c in selected[name]]].copy()
              for name, frame in frames.items() if selected[name]}
    if len(output) < 2:
        raise ValueError('Fewer than two annotated sources after ambiguity filtering')
    return output, [frozenset(g) for _, g in sorted(groups.items())], audit


def publish_task(output, frames, groups, protocol, variants=None, **metadata):
    # 输入：任务路径、CSV 表、完整语义组、协议与来源；输出：不可静默覆盖的 provenance 字典；继承：无（模块函数）。
    output = Path(output)
    if output.exists():
        manifest = json.loads((output / 'provenance.json').read_text(encoding='utf-8'))
        if manifest['protocol'] != protocol or any(file_sha256(output / f) != h for f, h in manifest['files'].items()):
            raise ValueError(f'Existing task differs; choose a new output name: {output}')
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare_', dir=output.parent) as temporary:
        stage = Path(temporary) / 'task'
        (stage / 'original').mkdir(parents=True)
        profiles = []
        for variant, tables in (variants or {}).items():
            if not variant.replace('_', '').isalnum() or variant == 'original':
                raise ValueError('Unsafe/additionally repeated variant name')
            (stage / variant).mkdir()
            for table, frame in tables.items():
                if Path(table).name != table:
                    raise ValueError('Unsafe variant table name')
                frame.to_csv(stage / variant / table, index=False)
        for name, frame in sorted(frames.items()):
            if Path(name).name != name or not name.endswith('.csv') or not frame.columns.is_unique:
                raise ValueError(f'Unsafe table name or repeated header: {name}')
            frame.to_csv(stage / 'original' / name, index=False)
            loaded = pd.read_csv(stage / 'original' / name, skip_blank_lines=False)
            profiles.append({'table': name, 'rows': len(loaded), 'columns': len(loaded.columns),
                             'missing_rate': float(loaded.isna().to_numpy().mean()),
                             'column_missing_rates': loaded.isna().mean().to_dict()})
        write_groundtruth(groups, str(stage / 'groundtruth.txt'))
        manifest = {'protocol': protocol, **metadata, 'tables': profiles,
                    'gt_pairs': sum(len(g)*(len(g)-1)//2 for g in groups),
                    'files': {p.relative_to(stage).as_posix(): file_sha256(p) for p in stage.rglob('*') if p.is_file()}}
        (stage / 'provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        stage.rename(output)
    return manifest
