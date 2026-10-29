"""Fixed-matcher ablations, held-out threshold selection and one result report.

Modules 1-3 run once per strategy/case; optimizers reuse identical edge evidence.
"""
from __future__ import annotations

import copy
import itertools
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .context import RunContext, set_global_seed
from .evaluation import METRIC_KEYS, evaluate, validate_matching
from .experiment import _edge_stats, _setup_logger, _slug, environment_info
from .calibration import calibration_snapshot, probability_diagnostics
from .report import diagnostics, matching_record, write_report
from .pipeline import build_components, compute_edges, load_sets
from .registry import OPTIMIZERS
from .tuning import best_theta, default_grid, search_threshold

MAIN_METRICS = tuple(METRIC_KEYS)


# ---- variants and tuning groups --------------------------------------------------------------
def expand_variants(spec: dict | None, default_variant: str | None) -> list[dict]:
    # 输入：spec、default_variant；输出：实验变体字典列表；继承：无（模块函数）。
    """[{"variant": str, <grid keys>}] from {"template": "...{key}...", "grid": {key: [values]}}."""
    if not spec:
        return [{"variant": default_variant}]
    if "list" in spec:
        return [{"variant": v} for v in spec["list"]]
    keys = list(spec["grid"])
    out = []
    for values in itertools.product(*(spec["grid"][k] for k in keys)):
        keyed = dict(zip(keys, values))
        out.append({"variant": spec["template"].format(**keyed), **keyed})
    return out


def tuning_groups(variants: list[dict], group_by: list[str] | None, tune_on: dict | None) -> list[dict]:
    # 输入：variants、group_by、tune_on；输出：分组后的调参与测试变体列表；继承：无（模块函数）。
    """[{"group": label, "tune": [variant], "test": [variant], "oracle": bool}].

    Variants with equal values of group_by form a group; within a group the variants matching
    tune_on ({key: [values]}) select the thresholds and the others are evaluated. Without
    tune_on every variant is used for both (oracle).
    """
    group_by = list(group_by or [])
    groups: dict[tuple, list[dict]] = {}
    for v in variants:
        groups.setdefault(tuple(v.get(k) for k in group_by), []).append(v)
    out = []
    for key, members in groups.items():
        label = "/".join(f"{k}={x}" for k, x in zip(group_by, key)) or "all"
        if tune_on:
            tune = [v for v in members if all(v.get(k) in vals for k, vals in tune_on.items())]
            test = [v for v in members if v not in tune]
            if not tune or not test:
                raise ValueError(f"group {label}: tune_on {tune_on} must select some but not all variants")
        else:
            tune = test = members
        out.append({"group": label, "keys": dict(zip(group_by, key)), "tune": [v["variant"] for v in tune],
                    "test": [v["variant"] for v in test], "oracle": not tune_on})
    return out


def strategy_config(cfg: dict, spec: dict) -> dict:
    # 输入：cfg、spec；输出：替换模块1/3后的独立配置字典；继承：无（模块函数）。
    """cfg with module 1 / module 3 replaced by a missing-value strategy."""
    out = copy.deepcopy(cfg)
    for m in ("uncertainty_model", "uncertainty_propagation"):
        if m in spec:
            section = {"name": spec[m]["name"], "params": copy.deepcopy(spec[m].get("params") or {})}
            if m == "uncertainty_model":
                section["realization"] = copy.deepcopy(spec[m].get("realization")
                                                       or cfg["uncertainty_model"].get("realization") or {})
            out[m] = section
    return out


# ---- statistics ------------------------------------------------------------------------------
def paired_comparison(ref: pd.Series, other: pd.Series) -> dict:
    # 输入：ref、other；输出：配对差值、胜负数和检验结果字典；继承：无（模块函数）。
    """Paired comparison on the common cases (index): ref - other."""
    common = ref.index.intersection(other.index)
    d = (ref.loc[common] - other.loc[common]).to_numpy(float)
    if len(d) == 0:
        return {"n": 0}
    nz = d[np.abs(d) > 1e-12]
    p = float(wilcoxon(nz).pvalue) if len(nz) > 1 else 1.0
    return {"n": len(d), "ref_mean": float(ref.loc[common].mean()), "other_mean": float(other.loc[common].mean()),
            "mean_diff": float(d.mean()), "wins": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()),
            "losses": int((d < -1e-12).sum()), "wilcoxon_p": p}


# ---- runner ----------------------------------------------------------------------------------
def run_ablation(cfg: dict, project_root: str, only_missing: list[str] | None = None,
                 only_optimizers: list[str] | None = None, max_variants: int | None = None,
                 oracle: bool = False) -> dict:
    # 输入：cfg、project_root、only_missing、only_optimizers、max_variants、oracle；输出：结果表、比较项和单个 HTML 路径；继承：无（模块函数）。
    abl = cfg.get("ablation") or {}
    for selected, key in ((only_missing, "missing"), (only_optimizers, "optimizers")):
        unknown = set(selected or []) - set(abl[key])
        if unknown:
            raise ValueError(f"unknown ablation.{key}: {sorted(unknown)}")
    if max_variants is not None and max_variants < 1:
        raise ValueError("max_variants must be positive")
    resolve = lambda p: p if os.path.isabs(p) else os.path.join(project_root, p)  # noqa: E731
    missing = {k: v for k, v in abl["missing"].items() if not only_missing or k in only_missing}
    optimizers = {k: v for k, v in abl["optimizers"].items() if not only_optimizers or k in only_optimizers}
    if not missing or not optimizers:
        raise ValueError("no missing-value strategy / optimizer selected")
    objective = abl.get("objective", "pair_f1")
    if objective not in METRIC_KEYS:
        raise ValueError(f"objective must be one of {METRIC_KEYS}")
    ref = abl.get("reference") or {}
    ref_missing = ref.get("missing", list(missing)[-1])
    ref_optimizer = ref.get("optimizer", list(optimizers)[0])

    variants = expand_variants(abl.get("variants"), cfg["dataset"].get("variant"))
    groups = tuning_groups(variants, abl.get("group_by"), abl.get("tune_on"))
    tune_sets = set(abl.get("tune_sets") or [])
    has_tunable = any(spec.get("tune") for spec in optimizers.values())
    if not has_tunable and not abl.get("tune_on") and not tune_sets:
        # Fixed-parameter pilots need only held-out tests: no threshold is fitted.
        groups = [{**g, "tune": [], "oracle": False} for g in groups]
    if tune_sets:
        groups = [{**g, "oracle": False} for g in groups]
    if oracle:
        # Keep the held-out evaluation population unchanged. Oracle only changes
        # where thresholds are selected; it must not add seed1 to the test cases.
        groups = [{**g, "tune": list(g["test"]), "oracle": True} for g in groups]
        tune_sets = set()
    if max_variants:  # quick check: first groups, one tuning and one test variant each
        groups = [{**g, "tune": g["tune"][:1], "test": g["test"][:1]} for g in groups[:max_variants]]
    needed = list(dict.fromkeys(v for g in groups for v in g["tune"] + g["test"]))

    name = cfg["experiment"].get("name") or f"ablation_{cfg['dataset']['name']}"
    run_id = f"{datetime.now():%Y%m%d-%H%M%S-%f}_{_slug(name + '__' + cfg['matcher']['name'])}"
    result_file = os.path.join(resolve(cfg["experiment"]["output_dir"]), "ablation", run_id + ".html")
    logger = _setup_logger(cfg["experiment"].get("verbose", False))
    protocol = {"objective": objective, "missing": list(missing), "optimizers": list(optimizers),
                "role": abl.get("role", "test"),
                "groups": groups, "tune_sets": sorted(tune_sets), "quick_check": max_variants,
                "oracle_requested": oracle}
    if abl.get("selection"):
        protocol["selection"] = copy.deepcopy(abl["selection"])
    calibration, details, predictions = {}, [], []
    seed = int(cfg["experiment"]["seed"])
    set_global_seed(seed)
    ctx = RunContext(seed=seed, project_root=project_root, cache_dir=resolve(cfg["paths"]["cache_dir"]),
                     embedding_cache=bool(cfg["paths"].get("embedding_cache", True)), logger=logger)
    logger.debug(f"ablation {run_id}: matcher {cfg['matcher']['name']} {cfg['matcher']['params']}, "
                f"{len(missing)} missing-value strategies x {len(optimizers)} optimizers, "
                f"{len(needed)} variant(s) in {len(groups)} group(s)"
                + (" [ORACLE thresholds: tuned on the test variants]" if groups[0]["oracle"] else ""))

    # 1. edge weights of every (strategy, variant): modules 1-3
    cases: dict[tuple[str, str], list[dict]] = {}
    for s_name, s_spec in missing.items():
        scfg = strategy_config(cfg, s_spec)
        comp, realization = build_components(scfg)
        calibration[s_name] = calibration_snapshot(comp, ctx)
        logger.debug(f"[{s_name}] {scfg['uncertainty_model']['name']} {comp['uncertainty_model'].params()} + "
                    f"{scfg['uncertainty_propagation']['name']} {comp['uncertainty_propagation'].params()}")
        for variant in needed:
            vcfg = copy.deepcopy(scfg)
            vcfg["dataset"]["variant"] = variant
            t0 = time.perf_counter()
            cases[(s_name, variant)] = []
            for iset in load_sets(vcfg, project_root):
                if iset.groundtruth is None:
                    raise ValueError(f"{iset.name} has no ground truth")
                data, samples, edges, cache_hit = compute_edges(iset, comp, realization, vcfg, ctx)
                detail = diagnostics(data, samples, edges, cfg)
                if detail:
                    details.append({"missing": s_name, "variant": variant, "set": iset.name, **detail})
                ds, es = data.stats(), _edge_stats(edges)
                cases[(s_name, variant)].append({"set": iset.name, "edges": edges, "gt": iset.groundtruth,
                                                 "n_uncertain_cells": ds.get("n_uncertain_cells", 0),
                                                 "n_filled_cells": ds.get("n_filled_cells", 0),
                                                 "n_uncertain_edges": es.get("n_uncertain_edges", 0),
                                                 "mean_sigma": es.get("mean_sigma", 0.0),
                                                 "n_score_samples": len(samples), "R": data.realization.n_samples,
                                                 "matcher_cache_hit": cache_hit})
            logger.debug(f"  [{s_name}] {variant}: {len(cases[(s_name, variant)])} set(s) "
                        f"({time.perf_counter() - t0:.1f}s)")

    # 2. thresholds on the tuning variants, 3. evaluation on the test variants
    quiet = RunContext(seed=seed, project_root=ctx.project_root, cache_dir=ctx.cache_dir,
                       embedding_cache=ctx.embedding_cache, resources=ctx.resources,
                       logger=logging.getLogger("umsm.ablation.quiet"))
    quiet.logger.setLevel(logging.WARNING)
    def partition(g, role, strategy):
        # 输入：g、role、strategy；输出：指定策略的校准/测试任务记录列表；继承：无（run_ablation 内部函数）。
        selected = [c for v in g[role] for c in cases[(strategy, v)]]
        if tune_sets:
            def is_tuning(c):
                # 输入：c；输出：当前任务是否用于调参的 bool；继承：无（partition 内部函数）。
                return c["set"] in tune_sets or c["set"].split("/")[-1] in tune_sets
            selected = [c for c in selected if is_tuning(c) == (role == "tune")]
        if not selected:
            raise ValueError(f"group {g['group']}: empty {role} partition; check tune_sets/tune_on")
        return selected

    if tune_sets:
        available = {c["set"] for cc in cases.values() for c in cc}
        unknown = tune_sets - available - {s.split("/")[-1] for s in available}
        if unknown:
            raise ValueError(f"tune_sets not loaded: {sorted(unknown)}")
    for s_name in missing:
        for g in groups:
            if g["tune"]:
                partition(g, "tune", s_name)
            partition(g, "test", s_name)

    results, thresholds, grid_rows = [], [], []
    for s_name in missing:
        for o_name, o_spec in optimizers.items():
            params = copy.deepcopy(o_spec.get("params") or {})
            param = o_spec.get("tune")
            if param:
                params.setdefault(param, 0.5)
            optimizer = OPTIMIZERS.build(o_spec["name"], params)
            for g in groups:
                value = None
                if param:
                    selected_tune = partition(g, "tune", s_name)
                    tune_cases = [(c["edges"], c["gt"]) for c in selected_tune]
                    w = np.concatenate([e.df["w"].to_numpy(float) for e, _ in tune_cases])
                    explicit_grid = abl.get("threshold_grid")
                    grid = np.array(sorted(explicit_grid), dtype=float) if explicit_grid is not None else default_grid(
                        w, abl.get("min_quantile", 0.5))
                    if len(grid) == 0 or not np.isfinite(grid).all():
                        raise ValueError("threshold_grid must be a nonempty list of finite values")
                    save_solution = getattr(optimizer, "save_solution", None)
                    if save_solution is not None:
                        optimizer.save_solution = False  # no need to build every y_ij CSV at each theta
                    res = search_threshold(optimizer, param, tune_cases, grid, objective,
                                           "full" if explicit_grid is not None else abl.get("search", "coarse_fine"))
                    if save_solution is not None:
                        optimizer.save_solution = save_solution
                    value = best_theta(res[objective])
                    setattr(optimizer, param, value)
                    grid_rows.extend({"missing": s_name, "optimizer": o_name, "group": g["group"], **r}
                                     for r in res.reset_index().to_dict("records"))
                    thresholds.append({"missing": s_name, "optimizer": o_name, "group": g["group"], "param": param,
                                       "value": value, "tune_score": float(res.loc[value, objective]),
                                       "n_evaluated": len(res), "tune_variants": json.dumps(g["tune"]),
                                       "tune_sets": json.dumps(sorted({c["set"] for c in selected_tune})),
                                       "oracle": g["oracle"]})
                for v in g["test"]:
                    for c in cases[(s_name, v)]:
                        if tune_sets and (c["set"] in tune_sets or c["set"].split("/")[-1] in tune_sets):
                            continue
                        t0 = time.perf_counter()
                        m = optimizer.optimize(c["edges"], quiet)
                        validate_matching(m, c["edges"], f"{s_name}/{o_name}/{c['set']}")
                        optimizer_time = time.perf_counter() - t0
                        prediction = matching_record(m, c["gt"], c['edges'].columns)
                        predictions.append({"missing": s_name, "optimizer": o_name, "variant": v,
                                            "set": c["set"], "matching": prediction})
                        if cfg["experiment"].get("save_edges"):
                            predictions[-1]["optimizer_artifacts"] = m.artifacts
                        results.append({"missing": s_name, "optimizer": o_name, "group": g["group"], **g["keys"],
                                        "variant": v, "set": c["set"], "threshold": value,
                                        "oracle": g["oracle"],
                                        "matcher_cache_hit": c["matcher_cache_hit"],
                                        "n_score_samples": c["n_score_samples"], "R": c["R"],
                                        "solver_all_optimal": m.info.get("all_optimal"),
                                        "global_exact": m.info.get("global_exact"),
                                        **{k: prediction[k] for k in ("TP", "FP", "FN")},
                                        "optimizer_time_s": optimizer_time, "n_groups": len(m.groups),
                                        **evaluate(m, c["gt"]),
                                        **probability_diagnostics(c["edges"], c["gt"]),
                                        **{k: c[k] for k in ("n_uncertain_cells", "n_filled_cells",
                                                             "n_uncertain_edges", "mean_sigma")}})
            logger.debug(f"  {s_name:>16s} + {o_name:<14s} {objective} = "
                        f"{np.mean([r[objective] for r in results if r['missing'] == s_name and r['optimizer'] == o_name]):.4f}")

    from .embedding import encoder_stats
    from .llm import llm_stats
    res, thr = pd.DataFrame(results), pd.DataFrame(thresholds)
    summary, by_group, comparisons = summarize(res, objective, ref_missing, ref_optimizer, missing, optimizers)
    payload = {"run_id": run_id, "name": name, "kind": "ablation", "config": cfg,
               "environment": environment_info(project_root), "protocol": protocol, "calibration": calibration,
               "rows": res, "summary": summary.reset_index(), "by_group": by_group.reset_index(),
               "comparisons": comparisons, "component_effects": component_effects(res, objective),
               "thresholds": thr, "threshold_grid": grid_rows, "predictions": predictions,
               "diagnostics": details, "resources": {"embedding": encoder_stats(ctx), "llm": llm_stats(ctx)}}
    write_report(result_file, payload)
    print(res.groupby(["missing", "optimizer"], sort=False)[objective].mean().unstack().round(4).to_string())
    print("Result:", result_file)
    return {"result_file": result_file, "results": res, "summary": summary,
            "comparisons": comparisons, "thresholds": thr}


def summarize(res, objective, ref_missing, ref_optimizer, missing, optimizers):
    # 输入：res、objective、ref_missing、ref_optimizer、missing、optimizers；输出：(按方法汇总表, 按组结果, 配对比较)；继承：无（模块函数）。
    metrics = [m for m in MAIN_METRICS if m in res]
    summary = res.groupby(["missing", "optimizer"], sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary["n_cases"] = res.groupby(["missing", "optimizer"], sort=False).size()
    by_group = res.pivot_table(index=["missing", "optimizer"], columns="group", values=objective, sort=False)

    unit = ["variant", "set"]
    score = {k: g.set_index(unit)[objective] for k, g in res.groupby(["missing", "optimizer"], sort=False)}
    comps = []
    for s in missing:
        if s != ref_missing and (s, ref_optimizer) in score and (ref_missing, ref_optimizer) in score:
            comps.append({"claim": "missing values", "reference": f"{ref_missing} + {ref_optimizer}",
                          "other": f"{s} + {ref_optimizer}",
                          **paired_comparison(score[(ref_missing, ref_optimizer)], score[(s, ref_optimizer)])})
    for o in optimizers:
        if o != ref_optimizer and (ref_missing, o) in score and (ref_missing, ref_optimizer) in score:
            comps.append({"claim": "global optimization", "reference": f"{ref_missing} + {ref_optimizer}",
                          "other": f"{ref_missing} + {o}",
                          **paired_comparison(score[(ref_missing, ref_optimizer)], score[(ref_missing, o)])})
    comps = pd.DataFrame(comps)
    return summary, by_group, comps


def component_effects(data, metric="pair_f1"):
    # 输入：data、metric；输出：独立模块贡献的配对比较列表；继承：无（模块函数）。
    contrasts = [("single_imputation", "top1", "global", "observed_only", "global"),
                 ("candidate_distribution", "Q_mean", "global", "top1", "global"),
                 ("risk_penalty", "Q_risk", "global", "Q_mean", "global"),
                 ("global_optimizer", "Q_risk", "global", "Q_risk", "greedy"),
                 ("full_pipeline", "Q_risk", "global", "observed_only", "greedy"),
                 ("global_vs_local_no_imputation", "observed_only", "global", "observed_only", "local"),
                 ("global_vs_greedy_no_imputation", "observed_only", "global", "observed_only", "greedy"),
                 ("probabilistic_imputation_local", "Q_mean", "local", "observed_only", "local"),
                 ("probabilistic_imputation_global", "Q_mean", "global", "observed_only", "global"),
                 ("candidate_distribution_local", "Q_mean", "local", "top1", "local"),
                 ("global_vs_local_qmean", "Q_mean", "global", "Q_mean", "local")]
    scores = {key: block.set_index(["variant", "set"])[metric]
              for key, block in data.groupby(["missing", "optimizer"], sort=False)}
    rows = []
    for name, s, o, bs, bo in contrasts:
        if (s, o) in scores and (bs, bo) in scores:
            effect = paired_comparison(scores[(s, o)], scores[(bs, bo)])
            rows.append({"component": name, "method": f"{s} + {o}", "baseline": f"{bs} + {bo}",
                         "metric": metric, **effect, "delta_pp": 100 * effect["mean_diff"]})
    return rows
