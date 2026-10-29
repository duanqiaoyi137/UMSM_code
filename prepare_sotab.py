"""Derive small, auditable multi-table tasks from official SOTAB V2 CPA tables.

Official validation websites are split into calibration/validation; official test
websites remain test-only. This is a derived benchmark, not the official CPA task.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

import pandas as pd

from umsm.data.groundtruth import write_groundtruth
from umsm.data.preparation import clean_cell, file_sha256 as digest

ROOT = Path(__file__).resolve().parent
SOURCE = "https://data.dws.informatik.uni-mannheim.de/structureddata/sotab/v2/"


def identity(name):
    # 输入：name；输出：(实体类别, 网站标识)；继承：无（模块函数）。
    match = re.fullmatch(r"([^_]+)_(.+)_September2020_CPA\.json\.gz", name)
    if not match:
        raise ValueError(f"Unknown SOTAB table identifier: {name}")
    topic, host = match.groups()
    return topic, host.lower().removeprefix("www.")


def rank(text, seed):
    # 输入：text、seed；输出：固定种子的排序哈希；继承：无（模块函数）。
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def download(split, folder):
    # 输入：split、folder；输出：缓存的官方 ZIP 文件 Path；继承：无（模块函数）。
    path = folder / f"CPA_{split}_schemaorg.zip"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        print(f"Downloading {SOURCE + path.name}", flush=True)
        with urlopen(SOURCE + path.name, timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Corrupt download: {temporary}")
        temporary.replace(path)
    return path


def annotations(archive, split):
    # 输入：archive、split；输出：去重的官方属性标注 DataFrame；继承：无（模块函数）。
    path = f"sotab_v2_cpa_{split}_set.csv"
    frame = pd.read_csv(io.BytesIO(archive.read(path)), dtype={"table_name": str, "label": str})
    required = {"table_name", "main_column_index", "column_index", "label"}
    if not required.issubset(frame):
        raise ValueError(f"Missing annotation fields in {path}")
    return frame.drop_duplicates()


def table_record(archive, members, name, labels, max_rows, seed):
    # One subject context and one unambiguous property per evaluated column/table.
    # 输入：archive、members、name、labels、max_rows、seed；输出：符合条件的原表记录或 None；继承：无（模块函数）。
    if labels.main_column_index.nunique() != 1 or labels.column_index.duplicated().any() or labels.label.duplicated().any():
        return None
    if len(labels) < 3 or len(labels) > 16:
        return None
    records = [json.loads(line) for line in gzip.decompress(archive.read(members[name])).splitlines() if line.strip()]
    frame = pd.DataFrame(records)
    frame.columns = frame.columns.map(str)
    indices = [str(int(i)) for i in labels.column_index]
    if not set(indices).issubset(frame):
        raise ValueError(f"Column index missing in {name}")
    frame = frame[indices].map(clean_cell)
    if len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed).sort_index()
    frame = frame.reset_index(drop=True)
    # Use the same CSV parsing rules that the experiment will use.
    frame.columns = [f"c{i}" for i in indices]
    frame = pd.read_csv(io.StringIO(frame.to_csv(index=False)), engine="python", skip_blank_lines=False)
    rates, counts, unique = frame.isna().mean(), frame.notna().sum(), frame.nunique()
    informative = (rates >= .3) & (rates <= .9) & (counts >= 10) & (unique >= 2)
    if len(frame) < 20 or not informative.any():
        return None
    topic, host = identity(name)
    return {"source": name, "topic": topic, "host": host, "frame": frame,
            "labels": dict(zip(frame.columns, labels.label)),
            "filename": "t_" + hashlib.sha256(name.encode()).hexdigest()[:12] + ".csv",
            "profile": {"rows": len(frame), "columns": len(frame.columns),
                        "missing_rate": float(frame.isna().to_numpy().mean()),
                        "column_missing_rates": rates.to_dict(), "observed_unique": unique.to_dict(),
                        "informative_missing_columns": list(frame.columns[informative])}}


def group_records(records, tables_per_set, max_sets, seed, grouping='chunks'):
    # 输入：records、tables_per_set、max_sets、seed；输出：按类别与网站组成的多表任务列表；继承：无（模块函数）。
    by_topic = defaultdict(list)
    for record in records:
        by_topic[record["topic"]].append(record)
    batches = defaultdict(list)
    for topic, members in sorted(by_topic.items()):
        members.sort(key=lambda r: rank(r["source"], seed))
        if grouping == 'distinct_sites':
            remaining, groups = list(members), []
            while len({r['host'] for r in remaining}) >= tables_per_set:
                group, hosts = [], set()
                for record in remaining:
                    if record['host'] not in hosts:
                        group.append(record)
                        hosts.add(record['host'])
                    if len(group) == tables_per_set:
                        break
                chosen = {r['source'] for r in group}
                remaining = [r for r in remaining if r['source'] not in chosen]
                groups.append(group)
        else:
            groups = [members[offset:offset+tables_per_set] for offset in range(0,len(members),tables_per_set)]
        for group in groups:
            if len(group) != tables_per_set or len({r["host"] for r in group}) != tables_per_set:
                continue
            counts = Counter(label for r in group for label in r["labels"].values())
            if sum(n >= 2 for n in counts.values()) >= 2:
                batches[topic].append(group)
    # Round robin over classes prevents the first alphabetical class dominating.
    selected = []
    while len(selected) < max_sets and any(batches.values()):
        for topic in sorted(batches):
            if batches[topic] and len(selected) < max_sets:
                selected.append(batches[topic].pop(0))
    return selected


def prepare(name="SOTAB_CPA", max_rows=300, max_sets=6, tables_per_set=3, seed=42, root=ROOT, grouping='chunks', roles=('fit','validation','test')):
    # 输入：name、max_rows、max_sets、tables_per_set、seed、root；输出：数据来源清单；生成并校验数据目录；继承：无（模块函数）。
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or min(max_rows, max_sets) < 1 or tables_per_set < 3 or grouping not in ('chunks','distinct_sites'):
        raise ValueError("Use a safe dataset name, positive sizes, and at least 3 tables per task")
    output = root / "data" / name
    protocol = {"adapter_version": 1, "max_rows": max_rows, "max_sets_per_role": max_sets,
                "tables_per_set": tables_per_set, "seed": seed,
                "selection": "3-16 unique CPA properties; >=20 rows; >=1 column with 30-90% nulls, >=10 observed cells and >=2 values",
                "split": "official validation -> fit/validation by website hash; official test, excluding validation websites",
                "groundtruth": "same CPA property within same table entity class and unique subject context; derived equivalence, review required"}
    if grouping != 'chunks':
        protocol.update(adapter_version=2, grouping=grouping)
    if tuple(roles) != ('fit','validation','test'):
        if not roles or set(roles)-{'fit','validation','test'}:
            raise ValueError('Unknown/empty role selection')
        protocol.update(adapter_version=2, roles=list(roles))
    if output.exists():
        manifest = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        if manifest["protocol"] != protocol or any(digest(output / f) != sha for f, sha in manifest["files"].items()):
            raise ValueError("Existing prepared dataset differs; choose a new --name rather than overwrite")
        print_summary(manifest)
        return manifest
    folder = root / "cache/datasets/sotab_v2"
    paths = {split: download(split, folder) for split in ("validation", "test")}
    pools, excluded = defaultdict(list), Counter()
    with zipfile.ZipFile(paths["validation"]) as archive:
        val_hosts = {identity(n)[1] for n in annotations(archive, "validation").table_name.unique()}
    for split, path in paths.items():
        with zipfile.ZipFile(path) as archive:
            members = {Path(n).name: n for n in archive.namelist() if n.endswith(".json.gz")}
            for source, labels in annotations(archive, split).groupby("table_name", sort=True):
                _, host = identity(source)
                if split == "test" and host in val_hosts:
                    excluded["test_website_seen_in_validation"] += 1
                    continue
                role = "test" if split == "test" else ("fit" if int(rank(host, seed), 16) % 2 == 0 else "validation")
                record = table_record(archive, members, source, labels, max_rows, seed)
                if record is not None:
                    pools[role].append(record)
                else:
                    excluded[f"{role}_ineligible"] += 1
    selected = {role: group_records(pools[role], tables_per_set, max_sets, seed, grouping)
                for role in roles}
    if any(not groups for groups in selected.values()):
        raise ValueError(f"Insufficient eligible groups: { {k: len(v) for k, v in selected.items()} }")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sotab_", dir=output.parent) as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        cases = []
        for role, groups in selected.items():
            for index, records in enumerate(groups, 1):
                case = f"{role}_{index:02d}_{records[0]['topic']}"
                directory = stage / case
                (directory / "original").mkdir(parents=True)
                gold, tables = defaultdict(set), []
                for record in records:
                    filename = record["filename"]
                    record["frame"].to_csv(directory / "original" / filename, index=False)
                    for column, label in record["labels"].items():
                        gold[label].add(f"{filename}.{column}")
                    tables.append({k: v for k, v in record.items() if k != "frame"})
                write_groundtruth([frozenset(gold[k]) for k in sorted(gold)], str(directory / "groundtruth.txt"))
                cases.append({"name": case, "role": role, "topic": records[0]["topic"], "tables": tables,
                              "gt_pairs": sum(len(g) * (len(g)-1)//2 for g in gold.values())})
        manifest = {"source": SOURCE, "protocol": protocol,
                    "archives": {p.name: digest(p) for p in paths.values()}, "excluded": dict(excluded),
                    "cases": cases, "files": {p.relative_to(stage).as_posix(): digest(p) for p in stage.rglob("*") if p.is_file()}}
        (stage / "provenance.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        stage.rename(output)
    print_summary(manifest)
    return manifest


def print_summary(manifest):
    # 输入：manifest；输出：None；打印准备后任务规模；继承：无（模块函数）。
    print("Derived SOTAB CPA tasks; inspect provenance.json labels, sites and missingness before running.")
    for case in manifest["cases"]:
        profiles = [t["profile"] for t in case["tables"]]
        print(case["name"], "tables=", len(profiles), "columns=", sum(p["columns"] for p in profiles),
              "GT pairs=", case["gt_pairs"], "missing=", [round(p["missing_rate"], 3) for p in profiles])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="SOTAB_CPA")
    parser.add_argument("--max-rows", type=int, default=300)
    parser.add_argument("--max-sets", type=int, default=6)
    parser.add_argument("--tables-per-set", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grouping", choices=('chunks','distinct_sites'), default='chunks')
    parser.add_argument('--roles',nargs='+',choices=('fit','validation','test'),default=['fit','validation','test'])
    args = parser.parse_args()
    prepare(args.name, args.max_rows, args.max_sets, args.tables_per_set, args.seed, grouping=args.grouping, roles=args.roles)
