"""Compare the frozen distribution main run with the S-BERT/Magneto optimization pilots.

Reads only completed HTML reports.  No matcher is re-run and no test label is used
for parameter selection; lambda comes from each report's validation selection.
"""
from __future__ import annotations

from pathlib import Path
from itertools import product
import glob
import json

import numpy as np
import pandas as pd

from umsm.report import read_report

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/paper_optimized"


def latest(paths):
    by_parent = {}
    for path in sorted({Path(p).resolve() for p in paths}):
        parent = path.parent
        if parent not in by_parent or path.stat().st_mtime_ns > by_parent[parent].stat().st_mtime_ns:
            by_parent[parent] = path
    return [by_parent[p] for p in sorted(by_parent)]


def family(report):
    cfg = report.get("config", {})
    name = str((cfg.get("research", {}).get("roles", {}).get("test", {}) or {}).get("name", ""))
    if name.startswith("Alaska"):
        return "Alaska"
    if name.startswith("SOTAB"):
        return "SOTAB"
    if name == "GDC":
        return "GDC"
    if name == "MaDI":
        return "MaDI"
    return name or "other"


def matcher(report):
    return report.get("config", {}).get("research", {}).get("matcher", "?")


def lambda_of(report):
    q = report.get("config", {}).get("ablation", {}).get("missing", {}).get("Q_risk", {})
    return q.get("uncertainty_propagation", {}).get("params", {}).get("risk_lambda")


def load_reports(pattern, exclude=()):
    out = []
    for p in latest(glob.glob(str(ROOT / pattern), recursive=True)):
        if any(token in str(p) for token in exclude):
            continue
        try:
            r = read_report(p)
        except Exception:
            continue
        if r.get("protocol", {}).get("role") != "test" or r.get("protocol", {}).get("phase") != "missing":
            continue
        r["_path"] = str(p)
        out.append(r)
    return out


def metric(df, missing, optimizer, column="pair_f1"):
    x = df[(df["missing"] == missing) & (df["optimizer"] == optimizer)]
    return float(x[column].mean()) if len(x) else np.nan


def summarize(reports):
    rows = []
    per_task = []
    for report in reports:
        df = pd.DataFrame(report.get("rows", []))
        if df.empty:
            continue
        fam, mat = family(report), matcher(report)
        lam = lambda_of(report)
        ours = df[(df.missing == "Q_risk") & (df.optimizer == "global")]
        rows.append({
            "family": fam, "matcher": mat,
            "tasks": int(ours.set.nunique()),
            "lambda": lam,
            "ours_f1": metric(df, "Q_risk", "global"),
            "ours_precision": metric(df, "Q_risk", "global", "pair_precision"),
            "ours_recall": metric(df, "Q_risk", "global", "pair_recall"),
            "ours_cluster_f1": metric(df, "Q_risk", "global", "cluster_f1"),
            "ours_cluster_precision": metric(df, "Q_risk", "global", "cluster_precision"),
            "ours_cluster_recall": metric(df, "Q_risk", "global", "cluster_recall"),
            "qmean_f1": metric(df, "Q_mean", "global"),
            "observed_ilp_f1": metric(df, "observed_only", "global"),
            "observed_local_f1": metric(df, "observed_only", "local"),
            "qrisk_local_f1": metric(df, "Q_risk", "local"),
            "alite_optimizer_f1": metric(df, "observed_only", "alite"),
            "mean_sigma": float(ours["mean_sigma"].mean()) if "mean_sigma" in ours else np.nan,
            "n_uncertain_edges": float(ours["n_uncertain_edges"].mean()) if "n_uncertain_edges" in ours else np.nan,
            "report": report["_path"],
        })
        for task, block in df.groupby("set", sort=True):
            per_task.append({
                "family": fam, "matcher": mat, "set": task, "lambda": lam,
                "ours_f1": metric(block, "Q_risk", "global"),
                "qmean_f1": metric(block, "Q_mean", "global"),
                "observed_ilp_f1": metric(block, "observed_only", "global"),
                "observed_local_f1": metric(block, "observed_only", "local"),
                "alite_optimizer_f1": metric(block, "observed_only", "alite"),
            })
    return pd.DataFrame(rows), pd.DataFrame(per_task)


def native_alite():
    reports = load_reports("output/results/research/full_alite_*/**/missing/*.html")
    rows = []
    for report in reports:
        df = pd.DataFrame(report.get("rows", []))
        if df.empty:
            continue
        rows.append({"family": family(report),
                     "tasks": int(df["set"].nunique()),
                     "alite_native_f1": metric(df, "observed_only", "alite"),
                     "alite_native_precision": metric(df, "observed_only", "alite", "pair_precision"),
                     "alite_native_recall": metric(df, "observed_only", "alite", "pair_recall")})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["family", "alite_native_f1", "alite_native_precision", "alite_native_recall"])
    for column in ("alite_native_f1", "alite_native_precision", "alite_native_recall"):
        frame[column] = frame[column] * frame["tasks"]
    pooled = frame.groupby("family", as_index=False)[
        ["tasks", "alite_native_f1", "alite_native_precision", "alite_native_recall"]].sum()
    for column in ("alite_native_f1", "alite_native_precision", "alite_native_recall"):
        pooled[column] /= pooled["tasks"]
    return pooled.drop(columns="tasks")


def paired_vs_distribution(distribution_tasks, sbert_tasks, seed=2026, resamples=2000):
    rows = []
    for fam in sorted(set(sbert_tasks["family"]) & set(distribution_tasks["family"])):
        left = sbert_tasks[sbert_tasks.family == fam].set_index("set")["ours_f1"]
        right = distribution_tasks[distribution_tasks.family == fam].set_index("set")["ours_f1"]
        common = left.index.intersection(right.index)
        d = (left.loc[common] - right.loc[common]).to_numpy(float)
        if not len(d):
            continue
        rng = np.random.default_rng(seed)
        boot = rng.choice(d, size=(resamples, len(d)), replace=True).mean(axis=1) if len(d) >= 2 else np.array([])
        lo, hi = np.quantile(boot, [0.025, 0.975]) if len(boot) else (None, None)
        if len(d) <= 16:
            signs = np.asarray(list(product((-1.0, 1.0), repeat=len(d))))
            p = float(np.mean(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12))
        else:
            signs = rng.choice((-1.0, 1.0), size=(10000, len(d)))
            p = float((1 + np.sum(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12)) / (len(signs) + 1))
        rows.append({"family": fam, "n_tasks": len(d), "delta_sbert_minus_distribution": float(d.mean()),
                     "ci_low": lo, "ci_high": hi, "p_unadjusted": p})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    distribution, distribution_tasks = summarize(load_reports("output/results/research/full_*_gpu_v1/*/distribution/missing/*.html"))
    sbert, sbert_tasks = summarize(load_reports("output/results/research/sbert_*_sbert_mean_v1/*/mean/missing/*.html"))
    sbert_wide, sbert_wide_tasks = summarize(load_reports("output/results/research/sbert_*_lambda_wide_v1/*/mean/missing/*.html"))
    sbert_wide["matcher"] = "mean_wide"
    sbert_wide_tasks["matcher"] = "mean_wide"
    sbert_fixed, sbert_fixed_tasks = summarize(load_reports(
        "output/results/research/sbert_*_sbert_fix_v1/*/mean/missing/*.html",
        exclude=("_prior_", "_cold_")))
    sbert_fixed["matcher"] = "mean_fixed"
    sbert_fixed_tasks["matcher"] = "mean_fixed"
    magneto, magneto_tasks = summarize(load_reports("output/results/research/magneto_*_magneto_v1/*/magneto/missing/*.html"))

    alite = native_alite()
    all_summary = pd.concat([distribution, sbert, sbert_wide, sbert_fixed, magneto], ignore_index=True)
    all_summary = all_summary.merge(alite, on="family", how="left")
    ref = distribution.groupby("family")["ours_f1"].mean()
    all_summary["delta_vs_distribution"] = all_summary["ours_f1"] - all_summary["family"].map(ref)

    all_summary.to_csv(OUT / "dataset_matcher_summary.csv", index=False)
    sbert_tasks.to_csv(OUT / "sbert_mean_per_task.csv", index=False)
    magneto_tasks.to_csv(OUT / "magneto_per_task.csv", index=False)
    paired_vs_distribution(distribution_tasks, sbert_tasks).to_csv(
        OUT / "paired_sbert_vs_distribution.csv", index=False)
    paired_vs_distribution(distribution_tasks, sbert_wide_tasks).to_csv(
        OUT / "paired_sbert_wide_vs_distribution.csv", index=False)
    paired_vs_distribution(distribution_tasks, sbert_fixed_tasks).to_csv(
        OUT / "paired_sbert_fixed_vs_distribution.csv", index=False)
    effective = all_summary[["family", "matcher", "tasks", "lambda", "mean_sigma",
                             "n_uncertain_edges"]].copy()
    effective["risk_adjustment"] = effective["lambda"] * effective["mean_sigma"]
    effective.to_csv(OUT / "lambda_effect.csv", index=False)

    # Pool folds with task weights for the paper-facing comparison.
    metrics = ["ours_f1", "ours_precision", "ours_recall", "ours_cluster_f1",
               "ours_cluster_precision", "ours_cluster_recall", "qmean_f1", "observed_ilp_f1",
               "observed_local_f1", "qrisk_local_f1", "alite_optimizer_f1",
               "alite_native_f1", "alite_native_precision", "alite_native_recall"]
    weighted = all_summary.copy()
    for column in metrics:
        weighted[column] = weighted[column] * weighted["tasks"]
    family_summary = weighted.groupby(["family", "matcher"], as_index=False)[
        metrics + ["tasks"]].sum()
    for column in metrics:
        family_summary[column] /= family_summary["tasks"]
    lam = all_summary.groupby(["family", "matcher"], as_index=False)["lambda"].first()
    family_summary = family_summary.drop(columns=["lambda"], errors="ignore").merge(
        lam, on=["family", "matcher"], how="left")
    dist_pooled = family_summary[family_summary["matcher"] == "distribution"].set_index("family")["ours_f1"]
    family_summary["delta_vs_distribution"] = family_summary["ours_f1"] - family_summary["family"].map(dist_pooled)
    family_summary.to_csv(OUT / "family_matcher_summary.csv", index=False)
    print(family_summary.sort_values(["family", "matcher"]).to_string(index=False))

    # Validation lambda + effective uncertainty scale.
    lam_rows = []
    for pattern, label in [("output/results/research/sbert_*_sbert_mean_v1/*/mean/validate-lambda/*.html", "sbert_mean"),
                           ("output/results/research/magneto_*_magneto_v1/*/magneto/validate-lambda/*.html", "magneto"),
                           ("output/results/research/sweep_sotab_k5_matcher_sweep_v1/external/*/validate-lambda/*.html", "sweep")]:
        for p in latest(glob.glob(str(ROOT / pattern), recursive=True)):
            r = read_report(p)
            sel = r.get("selection", {})
            cfg = r.get("config", {})
            lam_rows.append({
                "matcher": label,
                "name": cfg.get("research", {}).get("name", Path(p).parts[-4]),
                "selected_lambda": sel.get("value"),
                "validation_pair_f1": sel.get("score"),
                "validation_nll": sel.get("validation_edge_nll"),
                "report": str(p),
            })
    pd.DataFrame(lam_rows).to_csv(OUT / "lambda_validation.csv", index=False)

    # Compact figure: task-level F1 by family and matcher.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    families = [f for f in ["Alaska", "SOTAB", "GDC", "MaDI"]
                if f in set(all_summary["family"])]
    matchers = [m for m in ["distribution", "mean", "mean_wide", "mean_fixed", "magneto"]
                if m in set(all_summary["matcher"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(families))
    width = 0.8 / max(1, len(matchers) + 1)
    for j, matcher_name in enumerate(matchers):
        vals = [all_summary[(all_summary.family == f) & (all_summary.matcher == matcher_name)]["ours_f1"].mean()
                for f in families]
        ax.bar(x + (j - len(matchers) / 2) * width, vals, width, label=matcher_name)
    alite_vals = [alite.loc[alite.family == f, "alite_native_f1"].mean() for f in families]
    ax.bar(x + (len(matchers) / 2) * width, alite_vals, width, label="ALITE native")
    ax.set_xticks(x, families)
    ax.set_ylabel("Held-out pair F1")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_f1_by_matcher.png", dpi=220)
    fig.savefig(OUT / "fig_f1_by_matcher.pdf")
    plt.close(fig)

    manifest = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "distribution_reports": len(distribution),
        "sbert_reports": len(sbert),
        "sbert_wide_reports": len(sbert_wide),
        "sbert_fixed_reports": len(sbert_fixed),
        "magneto_reports": len(magneto),
        "note": "Lambda is the validation-selected value recorded in each frozen report; test labels were not used for selection. "
                "SOTAB column names are anonymized (c4, c5, ...), so Magneto's equal-name prior is not informative there.",
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(all_summary.sort_values(["family", "matcher"]).to_string(index=False))
    print("Assets:", OUT)


if __name__ == "__main__":
    main()
