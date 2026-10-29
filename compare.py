"""Read-only comparison of single HTML reports or legacy experimental results.

Examples
    python compare.py --dataset Real
    python compare.py --dataset Real --all            # every run, not only the latest per name
    python compare.py --dataset Real --sort pair_f1
    python compare.py --ablation results/mcar40/ablation/<run_id>.html
"""
import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import yaml
from umsm.report import read_report
from umsm.ablation import component_effects

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ["name", "theta", "cluster_precision", "cluster_recall", "cluster_f1", "pair_f1", "bcubed_f1",
           "time_total_s", "time_matcher_s", "llm_calls", "llm_cost_usd", "matcher_cache_hit", "run_dir"]


def load_result(path):
    # 输入：path；输出：(结果 DataFrame, 配置字典)；继承：无（模块函数）。
    """Read a new single report or an existing, untouched legacy result directory."""
    source = Path(path)
    if source.is_file():
        payload = read_report(source)
        return pd.DataFrame(payload["rows"]), payload.get("config", {})
    config = source / "config.yaml"
    cfg = yaml.safe_load(config.read_text(encoding="utf-8")) if config.exists() else {}
    return pd.read_csv(source / "results.csv"), cfg


def compare_ablation(directory, metric="pair_f1"):
    # 输入：directory、metric；输出：(方法均值矩阵, 模块贡献表)；继承：无（模块函数）。
    """Display the method matrix and paired component ablations for V2 strategy names."""
    data, _ = load_result(directory)
    mean = data.pivot_table(index="missing", columns="optimizer", values=metric, aggfunc="mean", sort=False)
    print(f"\nMean {metric} on test cases (higher is better):\n{mean.round(4).to_string()}")
    effects = pd.DataFrame(component_effects(data, metric))
    if not effects.empty:
        print("\nPaired effects: delta_pp > 0 means improvement in percentage points.")
        print(effects[["component", "n", "delta_pp", "wins", "ties", "losses"]].round(4).to_string(index=False))
    return mean, effects


def compare_against(directory, baseline_directory, metric="pair_f1"):
    # 输入：directory、baseline_directory、metric；输出：完整方法与各基线的配对比较 DataFrame；继承：无（模块函数）。
    """Pair the full UMSM method against each baseline on exactly the same test cases."""
    from umsm.ablation import paired_comparison
    labels = []
    for folder in (directory, baseline_directory):
        _, cfg = load_result(folder)
        matcher = cfg.get("matcher", {})
        params = matcher.get("params", {})
        if params.get("llm_backend") == "mock":
            raise ValueError("mock LLM results are only for checking execution; use real API runs for accuracy comparisons")
        label = matcher.get("name", "unknown")
        if params.get("llm_model"):
            label += "/" + params["llm_model"]
        labels.append(label)
    data, _ = load_result(directory)
    baseline, _ = load_result(baseline_directory)
    target = data[(data.missing == "Q_risk") & (data.optimizer == "global")]
    if target.empty:
        raise ValueError("target experiment needs Q_risk + global to compare against Magneto")
    keys = ["variant", "set"]
    score = target.set_index(keys)[metric]
    if not score.index.is_unique:
        raise ValueError("target has duplicate test cases")
    rows = []
    for (strategy, optimizer), block in baseline.groupby(["missing", "optimizer"], sort=False):
        other = block.set_index(keys)[metric]
        if not other.index.is_unique or set(other.index) != set(score.index):
            raise ValueError("comparison requires identical held-out variants/sets; do not mix quick and full runs")
        effect = paired_comparison(score, other)
        rows.append({"method": "Q_risk + global", "baseline": f"{strategy} + {optimizer}",
                     "method_matcher": labels[0], "baseline_matcher": labels[1],
                     "metric": metric, "method_run": os.path.abspath(directory),
                     "baseline_run": os.path.abspath(baseline_directory), **effect,
                     "delta_pp": 100 * effect["mean_diff"]})
    result = pd.DataFrame(rows)
    print(f"\n{labels[0]} full method vs {labels[1]} baseline ({metric}); positive delta_pp means improvement:")
    print(result[["baseline", "n", "ref_mean", "other_mean", "delta_pp", "wins", "ties", "losses"]]
          .round(4).to_string(index=False))
    return result


def compare_same_methods(directory, baseline_directory, metric="pair_f1"):
    # 输入：directory、baseline_directory、metric；输出：相同方法和任务的配对比较 DataFrame；继承：无（模块函数）。
    """Align identical strategy/optimizer/case rows to isolate a changed experimental factor."""
    data, _ = load_result(directory)
    baseline, _ = load_result(baseline_directory)
    keys = ["missing", "optimizer", "variant", "set"]
    if data.duplicated(keys).any() or baseline.duplicated(keys).any():
        raise ValueError("paired reports must have unique strategy/optimizer/test cases")
    if set(map(tuple, data[keys].to_numpy())) != set(map(tuple, baseline[keys].to_numpy())):
        raise ValueError("paired reports require identical strategies, optimizers and test cases")
    paired = data.merge(baseline, on=keys, suffixes=("_method", "_baseline"), validate="one_to_one")
    rows = []
    for (strategy, optimizer), block in paired.groupby(["missing", "optimizer"], sort=False):
        row = {"missing": strategy, "optimizer": optimizer, "n": len(block),
               "baseline": block[metric + "_baseline"].mean(), "method": block[metric + "_method"].mean()}
        row["delta_pp" if metric in ("pair_f1", "pair_precision", "pair_recall", "cluster_f1", "bcubed_f1")
            else "delta"] = (row["method"] - row["baseline"]) * (100 if metric in
                ("pair_f1", "pair_precision", "pair_recall", "cluster_f1", "bcubed_f1") else 1)
        for name in ("calibration_brier", "calibration_nll"):
            if name + "_method" in block and name + "_baseline" in block:
                row[name + "_delta"] = (block[name + "_method"] - block[name + "_baseline"]).mean()
        rows.append(row)
    result = pd.DataFrame(rows)
    print("\nPaired identical methods/cases: method - baseline; positive F1 delta_pp is better; negative Brier/NLL delta is better.")
    print(result.round(6).to_string(index=False))
    return result


def main():
    # 输入：命令行参数；输出：None；解析命令并执行对应工作流；继承：无（模块函数）。
    p = argparse.ArgumentParser(description="compare HTML reports or legacy result directories")
    p.add_argument("--dataset", default="Real")
    p.add_argument("--variant", default=None)
    p.add_argument("--summary", help="legacy summary.csv (optional)")
    p.add_argument("--root", default=os.path.join(PROJECT_ROOT, "results"), help="search single-run HTML reports here")
    p.add_argument("--sort", default=None, help="metric; defaults to pair_f1 for ablations, cluster_f1 for runs")
    p.add_argument("--ablation", help="HTML result file or legacy ablation directory")
    p.add_argument("--against", help="another ablation directory, e.g. the native Magneto baseline")
    p.add_argument("--paired", action="store_true", help="compare the same strategy/optimizer rows in both reports")
    p.add_argument("--all", action="store_true", help="show every run instead of the latest one per name")
    args = p.parse_args()
    if args.paired and not (args.ablation and args.against):
        p.error("--paired requires --ablation and --against")
    if args.ablation:
        compare_ablation(args.ablation, args.sort or "pair_f1")
        if args.against:
            (compare_same_methods if args.paired else compare_against)(args.ablation, args.against, args.sort or "pair_f1")
        return
    if args.against:
        p.error("--against requires --ablation")
    records = []
    if args.summary:
        records = pd.read_csv(args.summary).to_dict("records")
    else:
        for path in Path(args.root).rglob("*.html"):
            result = read_report(path)
            if result.get("kind") == "run":
                config = result["config"]
                records.append({"name": result["name"], "run_id": result["run_id"],
                                "timestamp": result["run_id"], "dataset": config["dataset"]["name"],
                                "variant": config["dataset"].get("variant"),
                                **result["metrics_mean_over_sets"], "result_file": str(path)})
        legacy = Path(args.root) / "summary.csv"
        if legacy.exists():
            records.extend(pd.read_csv(legacy).to_dict("records"))
    if not records:
        sys.exit("No results found")
    df = pd.DataFrame(records)
    df = df[df["dataset"] == args.dataset]
    if args.variant is not None:
        df = df[df["variant"].fillna("") == args.variant]
    if df.empty:
        sys.exit(f"no runs of {args.dataset} in {args.summary}")
    derived = df["run_id"].map(lambda r: re.sub(r"^\d{8}-\d{6}_", "", str(r)))
    df["name"] = df["name"].fillna(derived) if "name" in df else derived
    df = df.sort_values("timestamp")
    if not args.all:
        df = df.groupby("name", as_index=False).tail(1)
    df = df.sort_values(args.sort or "cluster_f1", ascending=False)
    cols = [c for c in COLUMNS + ["result_file"] if c in df.columns]
    out = df[cols].reset_index(drop=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 60)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
