"""Build paper-ready CSV/LaTeX tables and figures from frozen UMSM research reports.

Reads only completed HTML reports (never re-runs matchers or changes parameters).
Outputs go to output/paper by default.  Every table records the exact report paths
used, so results can be traced back to a run.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from umsm.factorial_analysis import factorial_4x2
from umsm.report import read_report

ROOT = Path(__file__).resolve().parent

METHODS = [
    ("Q_risk", "global", "Q-risk + ILP (ours)"),
    ("Q_risk", "local", "Q-risk + local"),
    ("Q_mean", "global", "Q-mean + ILP"),
    ("top1", "global", "Top-1 imputation + ILP"),
    ("mice_multiple", "global", "Multiple MICE + ILP"),
    ("stat_mice", "global", "Single MICE + ILP"),
    ("stat_missforest", "global", "MissForest-style + ILP"),
    ("stat_knn", "global", "KNN + ILP"),
    ("stat_mean_mode", "global", "Mean/mode + ILP"),
    ("observed_only", "global", "Observed-only + ILP"),
    ("observed_only", "local", "Observed-only + local"),
    ("Q_risk", "greedy", "Q-risk + Greedy"),
    ("Q_risk", "alite", "Q-risk + ALITE cluster"),
    ("Q_risk", "kwik", "Q-risk + Kwik"),
    ("Q_risk", "threshold", "Q-risk + threshold"),
]
METHOD_LOOKUP = {(m, o): label for m, o, label in METHODS}


def latest_by_folder(paths):
    latest = {}
    for path in sorted({Path(p).resolve() for p in paths}):
        parent = path.parent
        if parent not in latest or path.stat().st_mtime_ns > latest[parent].stat().st_mtime_ns:
            latest[parent] = path
    return [latest[p] for p in sorted(latest)]


def collect(pattern, kind_filter=None):
    reports = []
    for path in latest_by_folder(ROOT.glob(pattern)):
        try:
            report = read_report(path)
        except Exception:
            continue
        if kind_filter and report.get("kind") not in kind_filter:
            continue
        report["_path"] = str(path)
        reports.append(report)
    return reports


def family_of(report):
    cfg = report.get("config", {})
    # The embedded config keeps the base dataset; the explicit test role is authoritative.
    role = (cfg.get("research", {}).get("roles", {}) or {}).get("test", {})
    name = str(role.get("name") or cfg.get("dataset", {}).get("name", ""))
    rows = report.get("rows", [])
    if name in ("", "Real") and rows:
        name = str(rows[0].get("set", "")).split("/")[0]
    if name.startswith("Alaska"):
        return "Alaska"
    if name.startswith("SOTAB"):
        return "SOTAB"
    if name == "GDC":
        return "GDC"
    if name == "MaDI":
        return "MaDI"
    return name or "other"


def matcher_of(report):
    return report.get("config", {}).get("research", {}).get("matcher") or report.get("config", {}).get("matcher", {}).get("name", "?")


def task_clusters(report):
    protocol = report.get("protocol", {})
    groups = protocol.get("bootstrap", {}).get("task_groups")
    if not groups:
        groups = protocol.get("source_clusters", {})
    return groups or {}


def cluster_mean(frame, value, cluster_column="cluster"):
    if frame.empty:
        return float("nan")
    means = frame.groupby(cluster_column, sort=True)[value].mean(numeric_only=True)
    return float(means.mean()) if len(means) else float("nan")


def collect_missing_rows():
    """One row per (family, matcher, task, cluster, missing, optimizer)."""
    records, sources = [], []
    for report in collect("output/results/research/full_*_gpu_v1/**/missing/*.html", kind_filter={"research"}):
        cfg = report.get("config", {})
        protocol = report.get("protocol", {})
        if protocol.get("role") != "test" or protocol.get("phase") != "missing":
            continue
        rows = pd.DataFrame(report.get("rows", []))
        if rows.empty:
            continue
        groups = task_clusters(report)
        rows = rows[rows.get("oracle", False) != True].copy()
        rows["family"] = family_of(report)
        rows["matcher"] = matcher_of(report)
        rows["cluster"] = rows["set"].map(lambda s: groups.get(s, s))
        q_risk = cfg.get("ablation", {}).get("missing", {}).get("Q_risk", {})
        rows["risk_lambda"] = np.where(
            rows["missing"] == "Q_risk",
            q_risk.get("uncertainty_propagation", {}).get("params", {}).get("risk_lambda", np.nan),
            np.nan,
        )
        rows["report"] = report["_path"]
        records.append(rows)
        sources.append({"path": report["_path"], "family": family_of(report), "matcher": matcher_of(report),
                        "kind": "main_matcher"})
    # Independent ALITE-style baseline (observed evidence + ALITE clustering).
    for report in collect("output/results/research/full_alite_*/**/missing/*.html", kind_filter={"research"}):
        cfg = report.get("config", {})
        protocol = report.get("protocol", {})
        if protocol.get("role") != "test" or protocol.get("phase") != "missing":
            continue
        rows = pd.DataFrame(report.get("rows", []))
        if rows.empty:
            continue
        alite = rows[(rows["missing"] == "observed_only") & (rows["optimizer"] == "alite")].copy()
        if alite.empty:
            continue
        groups = task_clusters(report)
        alite["cluster"] = alite["set"].map(lambda s: groups.get(s, s))
        alite["family"] = family_of(report)
        alite["matcher"] = "alite_native"
        alite["report"] = report["_path"]
        records.append(alite)
        sources.append({"path": report["_path"], "family": family_of(report), "matcher": "alite_native",
                        "kind": "alite_baseline"})
    frame = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    return frame, sources


def method_summary(frame):
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for (family, matcher), block in frame.groupby(["family", "matcher"], sort=True):
        for missing, optimizer, label in METHODS:
            part = block[(block["missing"] == missing) & (block["optimizer"] == optimizer)]
            if part.empty:
                continue
            rows.append({
                "family": family, "matcher": matcher, "method": label,
                "missing": missing, "optimizer": optimizer,
                "pair_f1": cluster_mean(part, "pair_f1"),
                "pair_precision": cluster_mean(part, "pair_precision"),
                "pair_recall": cluster_mean(part, "pair_recall"),
                "bcubed_f1": cluster_mean(part, "bcubed_f1"),
                "ari": cluster_mean(part, "ari"),
                "n_tasks": part["set"].nunique(), "n_clusters": part["cluster"].nunique(),
                "n_edges": part["n_edges"].mean() if "n_edges" in part else np.nan,
                "compute_edges_s": part["compute_edges_s"].mean() if "compute_edges_s" in part else np.nan,
                "optimizer_s": part["optimizer_time_s"].mean() if "optimizer_time_s" in part else np.nan,
                "risk_lambda": part["risk_lambda"].dropna().iloc[0] if "risk_lambda" in part and part["risk_lambda"].notna().any() else np.nan,
            })
        alite = block[block["matcher"] == "alite_native"]
        if not alite.empty:
            rows.append({
                "family": family, "matcher": "alite_native", "method": "ALITE-style (independent)",
                "missing": "observed_only", "optimizer": "alite",
                "pair_f1": cluster_mean(alite, "pair_f1"),
                "pair_precision": cluster_mean(alite, "pair_precision"),
                "pair_recall": cluster_mean(alite, "pair_recall"),
                "bcubed_f1": cluster_mean(alite, "bcubed_f1"),
                "ari": cluster_mean(alite, "ari"),
                "n_tasks": alite["set"].nunique(), "n_clusters": alite["cluster"].nunique(),
                "n_edges": alite["n_edges"].mean() if "n_edges" in alite else np.nan,
                "compute_edges_s": alite["compute_edges_s"].mean() if "compute_edges_s" in alite else np.nan,
                "optimizer_s": alite["optimizer_time_s"].mean() if "optimizer_time_s" in alite else np.nan,
            })
    return pd.DataFrame(rows)


def paired_contrast(frame, family, matcher, method, baseline, seed=2026, resamples=2000,
                    baseline_matcher=None):
    baseline_matcher = baseline_matcher or matcher
    subset = frame[(frame["family"] == family) & (frame["matcher"] == matcher)]
    ours = subset[(subset["missing"] == method[0]) & (subset["optimizer"] == method[1])]
    base_block = frame[(frame["family"] == family) & (frame["matcher"] == baseline_matcher)]
    base = base_block[(base_block["missing"] == baseline[0]) & (base_block["optimizer"] == baseline[1])]
    if ours.empty or base.empty:
        return None
    left = ours.groupby("cluster", sort=True)["pair_f1"].mean()
    right = base.groupby("cluster", sort=True)["pair_f1"].mean()
    common = left.index.intersection(right.index)
    d = (left.loc[common] - right.loc[common]).to_numpy(float)
    if not len(d):
        return None
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(resamples, len(d)), replace=True).mean(axis=1) if len(d) >= 2 else np.array([])
    lo, hi = np.quantile(boot, [0.025, 0.975]) if len(boot) else (None, None)
    if len(d) <= 16:
        signs = np.asarray(list(product((-1.0, 1.0), repeat=len(d))))
        p = float(np.mean(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12))
    else:
        signs = rng.choice((-1.0, 1.0), size=(10000, len(d)))
        p = float((1 + np.sum(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12)) / (len(signs) + 1))
    return {"family": family, "matcher": matcher, "ours": "Q-risk + ILP (ours)",
            "baseline": METHOD_LOOKUP.get(
                baseline, "ALITE-style (independent)" if baseline == ("observed_only", "alite") else str(baseline)),
            "difference_f1": float(d.mean()), "ci_low": lo, "ci_high": hi,
            "n_clusters": len(d), "p_unadjusted": p}


def contrast_table(frame, method=("Q_risk", "global"), baselines=None):
    if frame.empty:
        return pd.DataFrame()
    baselines = baselines or [(m, o) for m, o, _ in METHODS if (m, o) != method]
    rows = []
    for family in sorted(frame["family"].unique()):
        for matcher in sorted(frame[frame["family"] == family]["matcher"].unique()):
            if matcher == "alite_native":
                continue
            for baseline in baselines:
                item = paired_contrast(frame, family, matcher, method, baseline)
                if item:
                    rows.append(item)
            alite = paired_contrast(frame, family, matcher, method,
                                    ("observed_only", "alite"), baseline_matcher="alite_native")
            if alite:
                rows.append(alite)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Holm correction across all non-primary baseline contrasts per family/matcher.
    result["p_holm"] = np.nan
    for (family, matcher), block in result.groupby(["family", "matcher"], sort=True):
        order = block.sort_values("p_unadjusted").index
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, min(1.0, (len(order) - rank) * result.loc[index, "p_unadjusted"]))
            result.loc[index, "p_holm"] = running
    result["supported"] = (result["ci_low"].notna() & (result["ci_low"] > 0) &
                           (result["p_holm"] < 0.05) & (result["n_clusters"] >= 10))
    return result


def factorial_table():
    # Pool all folds of a dataset before testing, so three Alaska folds produce
    # one effect per family rather than three independent-looking rows.
    groups = {}
    for report in collect("output/results/research/full_*_gpu_v1/**/factorial/*.html", kind_filter={"research"}):
        protocol = report.get("protocol", {})
        if protocol.get("role") != "test" or protocol.get("phase") != "factorial":
            continue
        levels = report.get("config", {}).get("research", {}).get("factorial_levels")
        if not levels:
            continue
        family, matcher = family_of(report), matcher_of(report)
        key = (family, matcher)
        group = groups.setdefault(key, {"levels": levels, "rows": [], "task_groups": {}, "reports": []})
        if group["levels"] != levels:
            raise ValueError(f"Inconsistent factorial levels within {family}/{matcher}")
        group["rows"].extend(report.get("rows", []))
        group["task_groups"].update(task_clusters(report))
        group["reports"].append(report["_path"])

    rows = []
    for (family, matcher), group in sorted(groups.items()):
        try:
            _, effects = factorial_4x2(
                group["rows"], group["levels"],
                {"unit": "task", "resamples": 2000, "confidence": 0.95,
                 "seed": 2026, "min_clusters": 10, "task_groups": group["task_groups"]},
            )
        except ValueError:
            # A family without complete 4x2 cells should not fabricate a pooled row.
            continue
        for row in effects:
            effect = str(row.get("effect", ""))
            if effect.startswith("imputation_main:"):
                label = "Q imputation main effect (" + effect.split(":", 1)[1] + ")"
            elif effect == "optimizer_main":
                label = "ILP vs local main effect"
            elif effect.startswith("interaction:"):
                label = "Imputation x ILP interaction (" + effect.split(":", 1)[1] + ")"
            elif effect == "waterfall:add_Q":
                label = "Waterfall: add Q"
            elif effect == "waterfall:add_ILP":
                label = "Waterfall: add ILP"
            elif effect == "pipeline:Q_ILP_vs_observed_local":
                label = "Full pipeline vs observed-only local"
            elif effect.startswith("Q_vs:"):
                label = "Q-risk vs " + effect.split(":", 1)[1]
            else:
                label = effect
            rows.append({"family": family, "matcher": matcher, "effect": effect, "label": label,
                         "difference": row.get("difference"), "ci_low": row.get("ci_low"),
                         "ci_high": row.get("ci_high"), "p_unadjusted": row.get("p_unadjusted"),
                         "p_holm": row.get("p_holm"), "supported": row.get("supported"),
                         "n_cases": row.get("n_cases"), "n_source_clusters": row.get("n_source_clusters"),
                         "report": ";".join(group["reports"])})
    return pd.DataFrame(rows)


def constraint_table():
    records = []
    for report in collect("output/results/research/full_ablation_*_gpu_v1/**/constraints/*.html"):
        block = pd.DataFrame(report.get("rows", []))
        if block.empty:
            continue
        block["family"] = family_of(report)
        block["matcher"] = matcher_of(report)
        block["report"] = report["_path"]
        records.append(block)
    if not records:
        return pd.DataFrame()
    frame = pd.concat(records, ignore_index=True)
    keep = [c for c in ("family", "matcher", "set", "missing", "optimizer", "pair_precision", "pair_recall",
                        "pair_f1", "bcubed_f1", "time_s", "objective", "n_triangle_constraints",
                        "same_table_conflicts", "problem_feasible", "all_optimal", "source_cluster", "report")
            if c in frame]
    return frame[keep].copy()


def runtime_table(missing_frame):
    if missing_frame.empty:
        return pd.DataFrame()
    rows = []
    for (family, matcher), block in missing_frame.groupby(["family", "matcher"], sort=True):
        if matcher == "alite_native":
            continue
        for missing, part in block.groupby("missing", sort=True):
            # compute_edges_s is repeated for every optimizer that reuses the same evidence.
            unique = part.drop_duplicates(subset=["set", "missing"])
            rows.append({"family": family, "matcher": matcher, "missing": missing,
                         "tables": unique["k"].mean() if "k" in unique else np.nan,
                         "n_edges": unique["n_edges"].mean() if "n_edges" in unique else np.nan,
                         "R": unique["R"].mean() if "R" in unique else np.nan,
                         "mean_compute_edges_s": unique["compute_edges_s"].mean(),
                         "max_compute_edges_s": unique["compute_edges_s"].max(),
                         "mean_matcher_s": unique["matcher_s"].mean() if "matcher_s" in unique else np.nan,
                         "mean_uncertainty_model_s": unique["uncertainty_model_s"].mean() if "uncertainty_model_s" in unique else np.nan,
                         "mean_optimizer_s": part["optimizer_time_s"].mean()})
    return pd.DataFrame(rows)


def make_figures(method_frame, factorial, runtime, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # Figure 1: main F1 by dataset family and matcher.
    selected = ["Q-risk + ILP (ours)", "Q-risk + local", "Observed-only + ILP", "Top-1 imputation + ILP",
                "Single MICE + ILP", "Multiple MICE + ILP", "ALITE-style (independent)"]
    colors = dict(zip(selected, ["#1f4e79", "#76a5d3", "#999999", "#d08b3e", "#7fa65a", "#a64d79", "#444444"]))
    if not method_frame.empty:
        matchers = [m for m in method_frame["matcher"].unique() if m != "alite_native"]
        families = ["Alaska", "SOTAB", "GDC", "MaDI"]
        families = [f for f in families if f in set(method_frame["family"])]
        fig, axes = plt.subplots(len(matchers), 1, figsize=(9, 2.6 * len(matchers)), squeeze=False)
        for ax, matcher in zip(axes[:, 0], matchers):
            # ALITE is an independent matcher; include it as a reference in every panel.
            block = method_frame[method_frame["matcher"].isin([matcher, "alite_native"])]
            x = np.arange(len(families))
            available = [m for m in selected if m in set(block["method"])]
            width = 0.8 / max(1, len(available))
            for j, method in enumerate(available):
                values = [block[(block["family"] == f) & (block["method"] == method)]["pair_f1"].mean() for f in families]
                ax.bar(x + (j - (len(available) - 1) / 2) * width, values, width, label=method,
                       color=colors.get(method, None))
            ax.set_xticks(x, families)
            ax.set_ylabel("Pair F1")
            ax.set_title(f"Matcher: {matcher}", loc="left")
            ax.grid(axis="y", alpha=0.25)
        axes[0, 0].legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.35))
        fig.tight_layout()
        fig.savefig(out_dir / "fig_main_f1.png", dpi=220)
        fig.savefig(out_dir / "fig_main_f1.pdf")
        plt.close(fig)

    # Figure 2: factorial forest (Q effect, ILP effect).  Waterfall is easier to read.
    if not factorial.empty:
        chosen = factorial[factorial["effect"].isin(["waterfall:add_Q", "waterfall:add_ILP"])].copy()
        chosen = chosen.dropna(subset=["difference"])
        if not chosen.empty:
            labels = [f"{r.family} | {r.label.replace('Waterfall: ', '')}" for r in chosen.itertuples()]
            fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.28 * len(chosen))))
            y = np.arange(len(chosen))
            colors = ["#3c78a8" if "Q" in str(r.effect) else "#68a56a" for r in chosen.itertuples()]
            ax.barh(y, chosen["difference"], color=colors)
            for yi, row in zip(y, chosen.itertuples()):
                if pd.notna(row.ci_low) and pd.notna(row.ci_high):
                    ax.plot([row.ci_low, row.ci_high], [yi, yi], color="black", linewidth=1)
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color="#3c78a8", label="Add Q (vs observed-only)"),
                               Patch(color="#68a56a", label="Add ILP (vs Q + local)")],
                      frameon=False, fontsize=7, loc="lower right")
            ax.set_yticks(y, labels, fontsize=7)
            ax.axvline(0, color="black", linewidth=0.7)
            ax.set_xlabel("Paired pair-F1 increment (source-cluster mean)")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "fig_waterfall_f1.png", dpi=220)
            fig.savefig(out_dir / "fig_waterfall_f1.pdf")
            plt.close(fig)

    # Figure 3: runtime by family/matcher (cold-ish mean per task).
    if not runtime.empty and not runtime["mean_compute_edges_s"].isna().all():
        block = runtime[runtime["missing"].isin(["observed_only", "Q_risk", "stat_mice", "mice_multiple"])].copy()
        if not block.empty:
            fig, ax = plt.subplots(figsize=(8, 3.2))
            labels = sorted(set(block["family"] + " | " + block["matcher"]))
            methods = ["observed_only", "stat_mice", "mice_multiple", "Q_risk"]
            method_labels = {"observed_only": "Observed-only", "stat_mice": "Single MICE",
                             "mice_multiple": "Multiple MICE", "Q_risk": "Q-risk"}
            width = 0.8 / len(methods)
            x = np.arange(len(labels))
            for j, method in enumerate(methods):
                vals = []
                for label in labels:
                    fam, mat = label.split(" | ")
                    part = block[(block["family"] == fam) & (block["matcher"] == mat) & (block["missing"] == method)]
                    vals.append(part["mean_compute_edges_s"].mean() if not part.empty else np.nan)
                ax.bar(x + (j - (len(methods) - 1) / 2) * width, vals, width,
                       label=method_labels.get(method, method))
            ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Mean edges+Q time (s)")
            ax.set_yscale("log")
            ax.legend(frameon=False, fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / "fig_runtime.png", dpi=220)
            fig.savefig(out_dir / "fig_runtime.pdf")
            plt.close(fig)


def write_latex_tables(method_frame, factorial, constraints, runtime, out_dir):
    # Compact main table: ours vs strongest observed/MICE baseline per family/matcher.
    if not method_frame.empty:
        rows = []
        for (family, matcher), block in method_frame.groupby(["family", "matcher"], sort=True):
            ours = block[block["method"] == "Q-risk + ILP (ours)"]
            if ours.empty:
                continue
            # The independent ALITE baseline lives under matcher=alite_native;
            # include it when choosing the strongest baseline for this family.
            candidates = method_frame[(method_frame["family"] == family) &
                                      (method_frame["matcher"].isin([matcher, "alite_native"]))]
            baseline_methods = [m for m in candidates["method"].unique() if m != "Q-risk + ILP (ours)"]
            best_method, best_f1 = None, -np.inf
            for method in baseline_methods:
                value = candidates[candidates["method"] == method]["pair_f1"].mean()
                if pd.notna(value) and value > best_f1:
                    best_method, best_f1 = method, value
            rows.append({"Dataset": family, "Matcher": matcher,
                         "Ours lambda": ours["risk_lambda"].dropna().iloc[0] if ours["risk_lambda"].notna().any() else np.nan,
                         "Q-risk+ILP F1": ours["pair_f1"].mean(),
                         "Best baseline": best_method, "Best baseline F1": best_f1,
                         "Delta": ours["pair_f1"].mean() - best_f1})
        table = pd.DataFrame(rows)
        table.to_csv(out_dir / "table_main.csv", index=False)
        (out_dir / "table_main.tex").write_text(
            table.to_latex(index=False, float_format="%.4f", escape=True,
                           caption="Held-out pair F1 (equal source-cluster mean) of the full pipeline against the strongest available baseline, including the independent ALITE baseline.",
                           label="tab:main"), encoding="utf-8")

    if not factorial.empty:
        chosen = factorial[factorial["effect"].isin(["waterfall:add_Q", "waterfall:add_ILP", "optimizer_main",
                                                     "pipeline:Q_ILP_vs_observed_local"])].copy()
        chosen = chosen[["family", "matcher", "label", "difference", "ci_low", "ci_high", "p_holm", "supported"]]
        chosen.to_csv(out_dir / "table_factorial.csv", index=False)
        (out_dir / "table_factorial.tex").write_text(
            chosen.to_latex(index=False, float_format="%.4f", escape=True,
                            caption="Prespecified 4x2 factorial effects (pair-F1 increments) with source-cluster 95\\% intervals and Holm-adjusted sign-flip p-values.",
                            label="tab:factorial"), encoding="utf-8")

    if not constraints.empty:
        constraints.to_csv(out_dir / "table_constraints.csv", index=False)
    if not runtime.empty:
        runtime.to_csv(out_dir / "table_runtime.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default="output/paper")
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing_frame, sources = collect_missing_rows()
    methods = method_summary(missing_frame)
    contrasts = contrast_table(missing_frame)
    factorial = factorial_table()
    constraints = constraint_table()
    runtime = runtime_table(missing_frame)

    methods.to_csv(out_dir / "main_methods.csv", index=False)
    contrasts.to_csv(out_dir / "main_contrasts.csv", index=False)
    pd.DataFrame(sources).to_csv(out_dir / "sources.csv", index=False)
    make_figures(methods, factorial, runtime, out_dir)
    write_latex_tables(methods, factorial, constraints, runtime, out_dir)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "missing_reports": len(sources),
        "main_rows": len(methods),
        "factorial_rows": len(factorial),
        "constraint_reports": len(list(ROOT.glob("output/results/research/full_ablation_*_gpu_v1/**/constraints/*.html"))),
        "note": "All values are read from frozen HTML reports; no test-set parameter selection was performed here. "
                "Equality/confidence intervals follow the source reports' source-cluster bootstrap protocol.",
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Paper assets:", out_dir)
    if not methods.empty:
        print(methods.sort_values(["family", "matcher", "pair_f1"], ascending=[True, True, False]).to_string(index=False))
    if not contrasts.empty:
        print(contrasts.sort_values(["matcher", "p_unadjusted"]).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
