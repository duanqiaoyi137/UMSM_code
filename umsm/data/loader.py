"""Unified loader for the local benchmarks under data/.

    single set   data/<Dataset>/<variant>/*.csv          + data/<Dataset>/groundtruth.txt
    multi set    data/<Dataset>/<set>/<variant>/*.csv    + data/<Dataset>/<set>/groundtruth.txt

Restaurant, Shopping, Laptop and Synthetic are single-set datasets, Real and Align multi-set.
Synthetic variants: "<MCAR|MAR|MNAR> missing original/<5_|10_|40_>/seed<1-5>".
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import pandas as pd

from ..types import IntegrationSet, Table
from .groundtruth import check_groundtruth, read_groundtruth


@dataclass
class DatasetSpec:
    variant: str = "original"
    tables: list[str] | None = None  # whitelist of table files (None = all csv files)


DATASETS: dict[str, DatasetSpec] = {
    "Restaurant": DatasetSpec(),
    # The folder contains 6 shopping tables; the paper (Table 2: 4 tables, 16 columns,
    # 1.2k tuples) and the ground truth use these four.
    "Shopping": DatasetSpec(tables=["aliexpress.csv", "amazon.csv", "ebay.csv", "walmart.csv"]),
    "Laptop": DatasetSpec(),
    "Real": DatasetSpec(),
    "Align": DatasetSpec(),
    "Synthetic": DatasetSpec(variant="MCAR missing original/5_/seed1"),
}


def _read_table(path: str) -> pd.DataFrame:
    # Same reader as the original code; dtype inference is kept because cell texts such as
    # "10.0" vs "10" change the embeddings.
    # A missing row in a one-column CSV is an empty line, not an absent tuple.
    # 输入：path；输出：保留空行的 DataFrame；继承：无（模块函数）。
    return pd.read_csv(path, encoding="utf-8-sig", engine="python", skip_blank_lines=False)


def _load_set(dataset: str, set_name: str, table_dir: str, gt_dir: str, tables: list[str] | None) -> IntegrationSet:
    # 输入：dataset、set_name、table_dir、gt_dir、tables；输出：含标注和元信息的 IntegrationSet；继承：无（模块函数）。
    files = sorted(glob.glob(os.path.join(table_dir, "*.csv")))
    if tables is not None:
        by_name = {os.path.basename(f): f for f in files}
        absent = [t for t in tables if t not in by_name]
        if absent:
            raise FileNotFoundError(f"{dataset}: tables {absent} not found in {table_dir}")
        files = [by_name[t] for t in tables]
    if len(files) < 2:
        raise FileNotFoundError(f"{dataset}/{set_name}: need >= 2 csv files in {table_dir}")
    tabs = [Table(os.path.basename(f), _read_table(f), f) for f in files]
    iset = IntegrationSet(dataset, set_name, tabs, meta={"table_dir": table_dir})
    gt_path = os.path.join(gt_dir, "groundtruth.txt")
    if os.path.exists(gt_path):
        iset.groundtruth = check_groundtruth(read_groundtruth(gt_path), iset.columns, f"{dataset}/{set_name}")
        iset.meta["groundtruth_path"] = gt_path
    return iset


def load_dataset(name: str, root: str, variant: str | None = None, sets: list[str] | None = None,
                 tables: list[str] | None = None, max_sets: int | None = None) -> list[IntegrationSet]:
    # 输入：name、root、variant、sets、tables、max_sets；输出：list[IntegrationSet]；继承：无（模块函数）。
    """Return the integration sets of data/<name>.

    variant:   sub-folder holding the csv files (default from DATASETS, usually "original").
    sets:      for multi-set datasets (Real, Align) restrict to these set names.
    tables:    whitelist of table files (overrides the dataset default).
    """
    spec = DATASETS.get(name, DatasetSpec())
    base = os.path.join(root, name)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Dataset folder {base} not found (available: {sorted(DATASETS)})")
    variant = variant or spec.variant
    tables = tables if tables is not None else spec.tables

    direct = os.path.join(base, variant)
    if glob.glob(os.path.join(direct, "*.csv")):
        set_name = name if variant == "original" else f"{name}/{variant}"
        return [_load_set(name, set_name, direct, base, tables)]

    result = []
    for set_dir in sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)):
        set_name = os.path.basename(set_dir)
        if sets is not None and set_name not in sets:
            continue
        table_dir = os.path.join(set_dir, variant)
        if not glob.glob(os.path.join(table_dir, "*.csv")):
            continue
        result.append(_load_set(name, f"{name}/{set_name}", table_dir, set_dir, tables))
        if max_sets is not None and len(result) >= max_sets:
            break
    if not result:
        raise FileNotFoundError(f"No integration set with csv files found under {base} (variant={variant!r})")
    return result
