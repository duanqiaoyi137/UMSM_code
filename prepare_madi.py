"""Download pinned MaDI base sources and derive an annotation-covered multi-table task.

No artificial masking, imputation or canonical renaming. Gold is used only to define
the evaluated columns/groups; IDs and unannotated metadata never become features.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from umsm.data.loader import _read_table
from umsm.data.groundtruth import write_groundtruth
from umsm.data.preparation import clean_cell, sha256

ROOT = Path(__file__).resolve().parent
REPOSITORY = "https://github.com/wbsg-uni-mannheim/MaDI-Bench"
REVISION = "062058d09c8abd97026e6d4e9908b433753f58d3"
ADAPTER_VERSION = 1


def download(repo_path, revision, cache):
    # 输入：repo_path、revision、cache；输出：下载的原始 bytes；继承：无（模块函数）。
    path = cache / repo_path
    if path.is_file():
        return path.read_bytes()
    url = f"https://raw.githubusercontent.com/wbsg-uni-mannheim/MaDI-Bench/{revision}/{quote(repo_path)}"
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "UMSM-data-preparation"}), timeout=60) as response:
                raw = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(raw)
            temporary.replace(path)
            return raw
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def convert_sources(gold, sources, max_rows=None, seed=42):
    # 输入：gold、sources、max_rows、seed；输出：(评估列 DataFrame 字典, 语义组, 源表概况)；继承：无（模块函数）。
    """Validate full annotation coverage of the retained evaluation universe."""
    columns = defaultdict(dict)
    identifiers = defaultdict(set)
    for entry in gold["mappings"]:
        if entry.get("label") is not True:
            continue
        source, col, target = entry["source_dataset"], entry["source_column"], entry["target_column"]
        if source not in sources or col not in sources[source].columns:
            raise ValueError(f"Gold column absent from source: {source}.{col}")
        if target == "id":
            identifiers[source].add(col)
            continue
        if col in columns[source] and columns[source][col] != target:
            raise ValueError(f"One source column has multiple targets: {source}.{col}")
        columns[source][col] = target
    frames, groups, profiles = {}, defaultdict(set), []
    for source, original in sources.items():
        if not original.columns.is_unique or not columns[source]:
            raise ValueError(f"Duplicate columns or no annotated attributes: {source}")
        mapping = columns[source]
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(f"Gold violates at-most-one-column-per-table: {source}")
        # Retain all annotated attributes in source order, not a subset chosen by results.
        keep = [col for col in original if col in mapping]
        if set(keep) & identifiers[source] or "cluster_id" in keep:
            raise ValueError("Identifiers/record-cluster labels must not become features")
        frame = original.loc[:, keep].map(clean_cell)
        if max_rows is not None and len(frame) > max_rows:
            frame = frame.sample(n=max_rows, random_state=seed).sort_index()
        frame = frame.reset_index(drop=True)
        name = source + ".csv"
        # CSV serialization is followed by the standard UMSM reader; profile that representation.
        frames[name] = frame
        for col, target in mapping.items():
            groups[target].add(f"{name}.{col}")
        profiles.append({"table": name, "original_rows": len(original), "rows": len(frame),
                         "columns": keep, "excluded_columns": [c for c in original if c not in keep]})
    if len(frames) < 3:
        raise ValueError("The multi-table experiment needs at least three sources")
    return frames, [frozenset(g) for _, g in sorted(groups.items())], profiles


def prepare(domain, revision=REVISION, max_rows=None, seed=42):
    # 输入：domain、revision、max_rows、seed；输出：数据来源清单；生成并校验数据目录；继承：无（模块函数）。
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a fixed 40-character Git commit, not a moving branch")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    case = f"{domain}_base" if max_rows is None else f"{domain}_pilot{max_rows}"
    parent = ROOT / "data" / "MaDI"
    output = parent / case
    protocol = {"adapter_version": ADAPTER_VERSION, "repository": REPOSITORY, "revision": revision,
                "domain": domain, "max_rows": max_rows, "seed": seed,
                "feature_policy": "all gold-annotated non-ID source columns; original column names",
                "missing_policy": "preserve nulls; empty lists and null placeholders become empty cells",
                "task": "derived multi-table schema matching; canonical targets used only for ground truth"}
    manifest_path = output / "provenance.json"
    if output.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol"] != protocol:
            raise ValueError(f"Existing data uses a different preparation protocol: {output}")
        expected = {str(Path(p).name) for p in manifest["files"] if p.startswith("original/")}
        actual = {p.name for p in (output / "original").glob("*.csv")}
        if actual != expected or any(sha256((output / p).read_bytes()) != h for p, h in manifest["files"].items()):
            raise ValueError(f"Prepared data has changed; inspect before reusing: {output}")
        print_profile(case, manifest)
        return manifest
    cache = ROOT / "cache" / "datasets" / "madi" / revision
    folder = PurePosixPath("use cases") / domain / "base" / "input"
    gold_path = str(folder / "schemamatching" / "sm_mapping_gold.json")
    gold_raw = download(gold_path, revision, cache)
    gold = json.loads(gold_raw)
    hashes, sources = {gold_path: sha256(gold_raw)}, {}
    for source, relative in gold["source_files"].items():
        path = (folder / "schemamatching" / relative)
        # Gold paths are resolved under this one case; reject external/traversing paths.
        resolved = (cache / str(path)).resolve()
        repo_path = resolved.relative_to(cache.resolve()).as_posix()
        if not repo_path.startswith(str(folder / "data") + "/") or not re.fullmatch(r"[\w-]+", source):
            raise ValueError(f"Unsupported source path/name: {relative}, {source}")
        raw = download(repo_path, revision, cache)
        hashes[repo_path] = sha256(raw)
        sources[source] = (pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", keep_default_na=False,
                                       skip_blank_lines=False) if repo_path.endswith(".csv")
                           else pd.read_json(io.BytesIO(raw), lines=True) if repo_path.endswith('.jsonl')
                           else pd.DataFrame(json.loads(raw)))
    frames, groups, profiles = convert_sources(gold, sources, max_rows, seed)
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare_madi_", dir=parent) as temporary:
        stage = Path(temporary) / "prepared"
        (stage / "original").mkdir(parents=True)
        for profile in profiles:
            path = stage / "original" / profile["table"]
            frames[profile["table"]].to_csv(path, index=False, encoding="utf-8")
            loaded = _read_table(str(path))
            profile["missing_cells"] = int(loaded.isna().to_numpy().sum())
            profile["missing_rate"] = profile["missing_cells"] / loaded.size
            profile["column_missing_rates"] = loaded.isna().mean().to_dict()
        write_groundtruth(groups, str(stage / "groundtruth.txt"))
        files = {p.relative_to(stage).as_posix(): sha256(p.read_bytes()) for p in stage.rglob("*") if p.is_file()}
        manifest = {"protocol": protocol, "upstream_sha256": hashes, "files": files,
                    "tables": profiles, "gt_groups": len(groups),
                    "gt_pairs": sum(len(g) * (len(g) - 1) // 2 for g in groups)}
        (stage / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.rename(output)
    print_profile(case, manifest)
    return manifest


def print_profile(case, manifest):
    # 输入：case、manifest；输出：None；打印 MaDI 数据概况；继承：无（模块函数）。
    print(f"MaDI/{case}: {len(manifest['tables'])} tables, {manifest['gt_pairs']} GT pairs")
    for table in manifest["tables"]:
        print(f"  {table['table']}: {table['rows']} rows, {len(table['columns'])} columns, "
              f"missing={table['missing_rate']:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=["companies", "products", "games", "music", "papers"])
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max-rows", type=int, help="fixed-seed row sample for a pilot; no artificial missingness")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.domain, args.revision, args.max_rows, args.seed)
