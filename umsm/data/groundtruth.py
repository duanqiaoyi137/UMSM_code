"""Ground truth in the benchmark format:

    Cluster 1:
      openrice.csv.Taste
      tripadvisor.csv.food

Internally a ground truth is a list of frozensets of column ids.
"""
from __future__ import annotations

import re
import warnings

from ..types import ColumnRef

_HEADER = re.compile(r"^Cluster\s+(\d+)\s*:\s*$")


def read_groundtruth(path: str) -> list[frozenset[str]]:
    # 输入：path；输出：list[frozenset[str]]；继承：无（模块函数）。
    clusters: dict[int, set[str]] = {}
    current = None
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = _HEADER.match(line)
            if m:
                current = int(m.group(1))
                clusters.setdefault(current, set())
            elif current is not None:
                clusters[current].add(line)
    return [frozenset(clusters[k]) for k in sorted(clusters) if clusters[k]]


def write_groundtruth(groups: list[frozenset[str]], path: str) -> None:
    # 输入：groups、path；输出：None；写入 Cluster 格式标注；继承：无（模块函数）。
    with open(path, "w", encoding="utf-8") as f:
        for i, g in enumerate(groups, 1):
            f.write(f"Cluster {i}:\n")
            for c in sorted(g):
                f.write(f"  {c}\n")
            f.write("\n")


def check_groundtruth(groups: list[frozenset[str]], columns: list[ColumnRef], where: str) -> list[frozenset[str]]:
    # 输入：groups、columns、where；输出：与已加载列标识对齐的标注组；继承：无（模块函数）。
    """Map ground-truth ids to the loaded columns and warn about the ones that do not exist.

    Lines of groundtruth.txt are stripped, so a column name with trailing blanks (e.g.
    ' Base Sales including tax ' in Align) is recovered by comparing stripped ids.
    """
    known = {c.id for c in columns}
    by_stripped: dict[str, list[str]] = {}
    for cid in known:
        by_stripped.setdefault(cid.strip(), []).append(cid)

    def resolve(cid: str) -> str:
        # 输入：cid；输出：绝对路径字符串；继承：无（check_groundtruth 内部函数）。
        if cid in known:
            return cid
        cands = by_stripped.get(cid.strip(), [])
        return cands[0] if len(cands) == 1 else cid

    groups = [frozenset(resolve(c) for c in g) for g in groups]
    unknown = sorted({c for g in groups for c in g} - known)
    if unknown:
        warnings.warn(f"{where}: {len(unknown)} ground-truth columns not found in tables, e.g. {unknown[:3]}")
    return groups
