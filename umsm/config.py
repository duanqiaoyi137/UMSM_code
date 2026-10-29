"""Experiment configuration: YAML file + defaults + command-line overrides (a.b.c=value)."""
from __future__ import annotations

import copy
import os

import yaml

MODULES = ("uncertainty_model", "matcher", "uncertainty_propagation", "global_optimizer")

DEFAULTS = {
    "experiment": {
        "name": None,                # run name; default "<dataset>__<model>+<matcher>+<propagation>+<optimizer>"
        "seed": 42,
        "output_dir": "results",
        "verbose": False,
        "online": True,            # Algorithm 1; false is an explicit batch reference only
        "save_edges": False,        # optional diagnostics embedded in the same HTML result
        "save_score_samples": False,
        "save_uncertainty": False,
        "save_realizations": False,
        "notes": "",
    },
    "dataset": {
        "name": "Restaurant",
        "root": "data",
        "variant": None,        # sub-folder holding the csv files, e.g. "original"
        "sets": None,           # Real / Align: restrict to these integration sets
        "max_sets": None,
        "tables": None,         # whitelist of table files
    },
    # Module 1. realization: how Q is turned into data states for the matcher (see RealizationSpec)
    "uncertainty_model": {"name": "column_marginal", "params": {},
                          "realization": {"n_samples": 32, "mode": "mc", "max_exact_states": 8}},
    "candidates": {"mode": "name_top_l", "top_l": 10, "min_score": 0.0},
    # Module 2: sbert_quantile (S-BERT) | alite | magneto | llm | valentine | precomputed
    "matcher": {"name": "sbert_quantile", "params": {}, "cache_scores": True},
    # Module 3
    "uncertainty_propagation": {"name": "mean_std", "params": {}},
    # Raw-score baseline defaults; current-method YAML selects calibrated_logit + ilp_logit.
    "global_optimizer": {"name": "ilp", "params": {}},
    "profiling": {"interval_s": 0.02, "tracemalloc": False},
    # For runtime/scalability experiments set embedding_cache: false and matcher.cache_scores: false,
    # otherwise the matcher time excludes encoding / scoring.
    "paths": {"cache_dir": "cache", "embedding_cache": True, "raw_score_cache": False},
}


def deep_merge(base: dict, override: dict) -> dict:
    # 输入：base、override；输出：独立的合并配置字典；继承：无（模块函数）。
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "params":
            out[k] = deep_merge(out[k], v)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_override(cfg: dict, expr: str) -> None:
    # 输入：cfg、expr；输出：None；应用命令行键值覆盖；继承：无（模块函数）。
    if "=" not in expr:
        raise ValueError(f"override must look like a.b.c=value, got {expr!r}")
    key, raw = expr.split("=", 1)
    value = yaml.safe_load(raw) if raw != "" else None
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    module_name = (len(parts)==2 and parts[0] in MODULES) or (len(parts)==3 and parts[0]=='modules' and parts[1] in MODULES)
    if parts[-1]=='name' and module_name and node.get('name')!=value:
        node['params'] = {}
    node[parts[-1]] = value


def read_yaml(path: str, _chain: tuple = ()) -> dict:
    # 输入：path、_chain；输出：解析继承并校正组件参数的配置字典；继承：无（模块函数）。
    """A YAML config; `base: other.yaml` (relative to this file) is loaded first and overridden.

    A module that names another component than its base starts from that component's defaults
    (the params of the base component are not inherited).
    """
    path = os.path.abspath(path)
    if path in _chain:
        raise ValueError(f"config base cycle: {' -> '.join(_chain + (path,))}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("base", None)
    if not base:
        return data
    parent = read_yaml(os.path.join(os.path.dirname(path), base), _chain + (path,))
    merged = deep_merge(parent, data)
    for m in MODULES:
        child = data.get(m) or {}
        if "name" in child and child["name"] != (parent.get(m) or {}).get("name"):
            merged[m]["params"] = copy.deepcopy(child.get("params") or {})
    for m,child in data.get('modules',{}).items():
        old = parent.get('modules',{}).get(m,{})
        if 'name' in child and child['name']!=old.get('name'):
            merged['modules'][m]['params'] = copy.deepcopy(child.get('params') or {})
    # The same component replacement rule applies inside ablation strategies.
    # Otherwise e.g. replacing column_marginal by knn_tuple silently keeps the
    # former model's params, making future grid configurations fail or misbehave.
    for name, strategy in (data.get("ablation", {}).get("missing", {}) or {}).items():
        for m in ("uncertainty_model", "uncertainty_propagation"):
            child = strategy.get(m) or {}
            old = parent.get("ablation", {}).get("missing", {}).get(name, {}).get(m) or {}
            if "name" in child and child["name"] != old.get("name"):
                merged["ablation"]["missing"][name][m]["params"] = copy.deepcopy(child.get("params") or {})
    return merged


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    # 输入：path、overrides；输出：合并默认值与覆盖项的完整配置；继承：无（模块函数）。
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        cfg = deep_merge(cfg, read_yaml(path))
    for o in overrides or []:
        apply_override(cfg, o)
    for section in MODULES:
        cfg[section]["params"] = cfg[section].get("params") or {}
    return cfg
