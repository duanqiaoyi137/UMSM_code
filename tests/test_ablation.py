"""Checks of the ablation runner (umsm/ablation.py); no language model needed.

    python tests/test_ablation.py        or        python -m pytest tests
"""
import os
import json
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from umsm import MATCHERS, UNCERTAINTY_MODELS
from umsm.ablation import expand_variants, paired_comparison, run_ablation, tuning_groups
from umsm.config import load_config  # noqa: E402
from umsm.context import RunContext  # noqa: E402
from umsm.data import write_groundtruth  # noqa: E402
from umsm.matcher.base import PairwiseMatcher  # noqa: E402
from umsm.types import IntegrationSet, Table  # noqa: E402
from umsm.report import read_report


@MATCHERS.register("value_overlap_test")
class _ValueOverlap(PairwiseMatcher):
    registry_name = "value_overlap_test"

    def similarity(self, ci, si, cj, sj, ctx):
        a, b = {str(v) for v in si.values}, {str(v) for v in sj.values}
        return len(a & b) / len(a | b) if a | b else 0.0


def _toy_tables(drop: int):
    t1 = pd.DataFrame({"City": ["paris", "rome", "oslo", "rome"], "Kind": ["a", "b", "a", "b"]})
    t2 = pd.DataFrame({"Town": ["rome", "paris", "rome", "lima"], "Type": ["b", "a", "b", "a"]})
    t3 = pd.DataFrame({"Place": ["oslo", "rome", "paris", "rome"]})
    t1.loc[drop % 4, "City"] = None
    t3.loc[(drop + 1) % 4, "Place"] = None
    return {"t1.csv": t1, "t2.csv": t2, "t3.csv": t3}


GT = [frozenset({"t1.csv.City", "t2.csv.Town", "t3.csv.Place"}), frozenset({"t1.csv.Kind", "t2.csv.Type"})]


def test_argmax_imputes_most_probable_candidate():
    t = Table("t.csv", pd.DataFrame({"x": ["a", "a", "b", None], "y": [1, 2, 3, 4]}))
    iset = IntegrationSet("S", "S", [t], None)
    d = UNCERTAINTY_MODELS.build("argmax", {"model": "column_marginal"}).estimate(iset, RunContext())
    assert d.is_deterministic and d.table("t.csv").frame.loc[3, "x"] == "a"
    assert iset.tables[0].df["x"].isna().sum() == 1


def test_variants_groups_and_comparisons():
    v = expand_variants({"template": "{m}/{r}_/seed{seed}", "grid": {"m": ["MCAR", "MAR"], "r": [5], "seed": [1, 2, 3]}},
                        None)
    assert len(v) == 6 and v[0] == {"variant": "MCAR/5_/seed1", "m": "MCAR", "r": 5, "seed": 1}
    assert expand_variants(None, "original") == [{"variant": "original"}]
    g = tuning_groups(v, ["m", "r"], {"seed": [1]})
    assert [x["group"] for x in g] == ["m=MCAR/r=5", "m=MAR/r=5"]
    assert g[0]["tune"] == ["MCAR/5_/seed1"] and g[0]["test"] == ["MCAR/5_/seed2", "MCAR/5_/seed3"]
    oracle = tuning_groups(v, ["m"], None)
    assert oracle[0]["tune"] == oracle[0]["test"] and oracle[0]["oracle"]
    c = paired_comparison(pd.Series([0.9, 0.8, 0.7, 0.5]), pd.Series([0.8, 0.8, 0.6, 0.6]))
    assert (c["wins"], c["ties"], c["losses"]) == (2, 1, 1) and np.isclose(c["mean_diff"], 0.025)



def raw_ablation_config():
    cfg = load_config(None)
    cfg['ablation'] = {
        'objective':'pair_f1', 'search':'full', 'min_quantile':0.,
        'reference':{'missing':'Q_mu_sigma','optimizer':'ilp'},
        'missing':{
            'observed_only':{'uncertainty_model':{'name':'observed_only'},'uncertainty_propagation':{'name':'expected'}},
            'top1':{'uncertainty_model':{'name':'argmax','params':{'model':'column_marginal'}},'uncertainty_propagation':{'name':'expected'}},
            'Q_mu_sigma':{'uncertainty_model':{'name':'column_marginal'},'uncertainty_propagation':{'name':'mean_std','params':{'alpha':.5}}}},
        'optimizers':{
            'ilp':{'name':'ilp','params':{'top_l':None},'tune':'theta'},
            'greedy_threshold':{'name':'greedy_merge','params':{'selection':'threshold'},'tune':'threshold'},
            'pairwise_hungarian':{'name':'local_merge','params':{'local_decision':'hungarian'},'tune':'min_weight'},
            'greedy_topk_gap':{'name':'greedy_merge','params':{'selection':'topk_gap'}},
            'alite_cluster':{'name':'alite_cluster','params':{'linkage':'complete'}}}}
    return cfg


def test_ablation_end_to_end():
    with tempfile.TemporaryDirectory() as root:
        for seed in (1, 2, 3):
            d = os.path.join(root, "data", "Toy", "MCAR", f"seed{seed}")
            os.makedirs(d)
            for name, df in _toy_tables(seed).items():
                df.to_csv(os.path.join(d, name), index=False)
        write_groundtruth(GT, os.path.join(root, "data", "Toy", "groundtruth.txt"))
        cfg = raw_ablation_config()
        cfg["dataset"] = {**cfg["dataset"], "name": "Toy", "root": os.path.join(root, "data")}
        cfg["matcher"] = {"name": "value_overlap_test", "params": {}, "cache_scores": False}
        cfg["candidates"] = {"mode": "all"}  # this test isolates ablation; blocking is tested separately
        cfg["experiment"]["output_dir"] = os.path.join(root, "results")
        cfg["paths"]["cache_dir"] = os.path.join(root, "cache")
        cfg["ablation"]["variants"] = {"template": "{m}/seed{seed}", "grid": {"m": ["MCAR"], "seed": [1, 2, 3]}}
        cfg["ablation"]["group_by"] = ["m"]
        cfg["ablation"]["tune_on"] = {"seed":[1]}
        out = run_ablation(cfg, root, only_missing=["observed_only", "top1", "Q_mu_sigma"])
        res = out["results"]
        assert set(res["missing"]) == {"observed_only", "top1", "Q_mu_sigma"} and set(res["optimizer"]) == set(
            cfg["ablation"]["optimizers"])
        assert set(res["variant"]) == {"MCAR/seed2", "MCAR/seed3"}  # seed1 only selects the thresholds
        assert (res.loc[res["missing"] == "Q_mu_sigma", "n_uncertain_cells"] > 0).all()
        assert (res.loc[res["missing"] == "top1", "mean_sigma"] == 0).all()
        assert (res.loc[res["missing"] == "top1", "n_filled_cells"] > 0).all()
        report = read_report(out["result_file"])
        thr = pd.DataFrame(report["thresholds"])
        assert set(thr["optimizer"]) == {"ilp", "greedy_threshold", "pairwise_hungarian"}
        comps = out["comparisons"]
        assert len(comps) == 2 + 4 and (comps["n"] == 2).all()
        assert [f for _, _, files in os.walk(cfg["experiment"]["output_dir"]) for f in files] == [os.path.basename(out["result_file"])]
        assert not res["oracle"].any()
        assert res.loc[res.optimizer == "ilp", "global_exact"].all()
        assert report["threshold_grid"]
        assert len(report["predictions"]) == len(res)
        assert all("groups" in p["matching"] for p in report["predictions"])
        # Oracle selection changes the tuning population, never the test population.
        cfg["ablation"]["threshold_grid"] = [.4, .6, .8]
        oracle = run_ablation(cfg, root, only_missing=["observed_only"],
                              only_optimizers=["greedy_threshold"], oracle=True)
        assert set(oracle["results"].variant) == {"MCAR/seed2", "MCAR/seed3"}
        assert oracle["results"].oracle.all()
        threshold = pd.DataFrame(read_report(oracle["result_file"])["thresholds"])
        assert "seed1" not in threshold.iloc[0].tune_variants


def test_ablation_holds_out_integration_sets():
    with tempfile.TemporaryDirectory() as root:
        for name in ("train", "test1", "test2"):
            d = os.path.join(root, "data", "Multi", name, "original")
            os.makedirs(d)
            for table, frame in _toy_tables(1).items():
                frame.to_csv(os.path.join(d, table), index=False)
            write_groundtruth(GT, os.path.join(root, "data", "Multi", name, "groundtruth.txt"))
        cfg = raw_ablation_config()
        cfg["dataset"] = {"name": "Multi", "root": os.path.join(root, "data")}
        cfg["matcher"] = {"name": "value_overlap_test", "params": {}, "cache_scores": False}
        cfg["candidates"] = {"mode": "all"}
        cfg["experiment"]["output_dir"] = os.path.join(root, "results")
        cfg["ablation"]["tune_sets"] = ["train"]
        cfg["ablation"]["threshold_grid"] = [.4, .6, .8]
        out = run_ablation(cfg, root, only_missing=["observed_only"], only_optimizers=["greedy_threshold"])
        assert set(out["results"]["set"]) == {"Multi/test1", "Multi/test2"}
        assert not out["results"].oracle.any()
        thr = pd.DataFrame(read_report(out["result_file"])["thresholds"])
        assert json.loads(thr.iloc[0].tune_sets) == ["Multi/train"]


def test_component_effect_report_uses_paired_differences_and_percentage_points():
    from compare import compare_ablation
    with tempfile.TemporaryDirectory() as root:
        rows = []
        methods = [("observed_only", "global", .4), ("top1", "global", .5),
                   ("Q_mean", "global", .6), ("Q_risk", "global", .7),
                   ("Q_risk", "greedy", .55), ("observed_only", "greedy", .3)]
        for seed in (3, 4):
            for strategy, optimizer, f1 in methods:
                rows.append({"missing": strategy, "optimizer": optimizer,
                             "variant": f"seed{seed}", "set": f"Toy/seed{seed}", "pair_f1": f1 - .02 * seed})
        pd.DataFrame(rows).to_csv(os.path.join(root, "results.csv"), index=False)
        _, effects = compare_ablation(root)
        effect = effects.set_index("component").loc["global_optimizer"]
        assert effect["n"] == 2 and effect["wins"] == 2
        assert np.isclose(effect["mean_diff"], .15) and np.isclose(effect["delta_pp"], 15.)
        assert not os.path.exists(os.path.join(root, "component_effects.csv"))


def test_cross_matcher_comparison_pairs_identical_heldout_cases():
    from compare import compare_against
    with tempfile.TemporaryDirectory() as root:
        ours, baseline = os.path.join(root, "ours"), os.path.join(root, "magneto")
        os.makedirs(ours)
        os.makedirs(baseline)
        a, b = [], []
        for seed in (3, 4, 5):
            common = {"variant": f"seed{seed}", "set": f"Toy/seed{seed}"}
            a.append({**common, "missing": "Q_risk", "optimizer": "global", "pair_f1": .8})
            b.append({**common, "missing": "observed_only", "optimizer": "magneto_local", "pair_f1": .7})
        pd.DataFrame(a).to_csv(os.path.join(ours, "results.csv"), index=False)
        pd.DataFrame(b).to_csv(os.path.join(baseline, "results.csv"), index=False)
        effects = compare_against(ours, baseline)
        assert effects.iloc[0]["n"] == 3 and np.isclose(effects.iloc[0].delta_pp, 10.)
        assert effects.iloc[0].wins == 3 and not os.path.exists(os.path.join(ours, "cross_method_comparisons.csv"))
        pd.DataFrame(b[:1]).to_csv(os.path.join(baseline, "results.csv"), index=False)
        try:
            compare_against(ours, baseline)
        except ValueError as exc:
            assert "identical" in str(exc)
        else:
            raise AssertionError("a quick single-seed run must not be compared against three test seeds")
        pd.DataFrame(b).to_csv(os.path.join(baseline, "results.csv"), index=False)
        with open(os.path.join(baseline, "config.yaml"), "w", encoding="utf-8") as f:
            f.write("matcher: {name: magneto, params: {llm_backend: mock}}\n")
        try:
            compare_against(ours, baseline)
        except ValueError as exc:
            assert "mock" in str(exc)
        else:
            raise AssertionError("mock LLM smoke scores cannot serve as real Magneto baselines")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
