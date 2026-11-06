"""Four-module algorithm: build components, load tables, then Q -> sampled scores -> edges.

Module 4 is optimizer.optimize(edges, ctx). Experiment runners own output files and metrics.
"""
import hashlib
import json
import os
from pathlib import Path

from .calibration import check_inference_split, check_matcher, set_fingerprint
from .candidates import compact_info
from .config import MODULES
from .data import load_dataset
from .registry import MATCHERS, OPTIMIZERS, PROPAGATIONS, UNCERTAINTY_MODELS
from .score_cache import score_with_cache
from .types import RealizationSpec

REGISTRIES = {"uncertainty_model": UNCERTAINTY_MODELS, "matcher": MATCHERS,
              "uncertainty_propagation": PROPAGATIONS, "global_optimizer": OPTIMIZERS}


def build_components(cfg: dict) -> tuple[dict, RealizationSpec]:
    # 输入：cfg；输出：组件字典与 RealizationSpec；继承：无（模块函数）。
    comp = {m: REGISTRIES[m].build(cfg[m]["name"], cfg[m]["params"]) for m in MODULES}
    return comp, RealizationSpec(**(cfg["uncertainty_model"].get("realization") or {}))


def load_sets(cfg: dict, project_root: str):
    # 输入：cfg、project_root；输出：list[IntegrationSet]；继承：无（模块函数）。
    dcfg = cfg["dataset"]
    root = dcfg["root"] if os.path.isabs(dcfg["root"]) else os.path.join(project_root, dcfg["root"])
    return load_dataset(dcfg["name"], root, variant=dcfg.get("variant"), sets=dcfg.get("sets"),
                        tables=dcfg.get("tables"), max_sets=dcfg.get("max_sets"))


def uncertainty_cache_key(iset, model, ctx, cfg):
    # Input: task/model/config/seed; output: content-addressed Q-state cache key.
    params = model.params()
    clean = {k: v for k, v in params.items() if k != "calibration"}
    calibration = params.get("calibration")
    calibration_identity = None
    if isinstance(calibration, str):
        path = Path(ctx.resolve(calibration))
        if path.is_file():
            calibration_identity = {"path": str(path.resolve()),
                                    "size": path.stat().st_size,
                                    "mtime_ns": path.stat().st_mtime_ns}
        else:
            calibration_identity = {"path": calibration}
    payload = {"version": "qstate_v1", "model": model.registry_name, "params": clean,
               "calibration": calibration_identity, "seed": int(ctx.seed),
               "dataset": cfg.get("dataset", {}).get("name"),
               "task": set_fingerprint(iset)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def estimate_uncertainty(iset, model, ctx, cfg):
    # Input: tables/model/config; output: Q state (cached before any MC configure); module function.
    if not cfg.get("paths", {}).get("uncertainty_cache", True):
        return model.estimate(iset, ctx)
    key = uncertainty_cache_key(iset, model, ctx, cfg)
    path = Path(ctx.cache_dir) / "uncertainty" / (key + ".pkl")
    if path.is_file():
        cached = __import__("pandas").read_pickle(path)
        # The cached object predates configure(); realization and RNG state are local to callers.
        return cached
    # Statistical imputations are matcher-independent, so concurrent study workers
    # would otherwise repeat the same expensive MICE/MissForest fit.  The lock is
    # separate from atomic_pickle's own publish lock.
    from filelock import FileLock
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".compute.lock"):
        if path.is_file():
            return __import__("pandas").read_pickle(path)
        data = model.estimate(iset, ctx)
        from .online import atomic_pickle
        atomic_pickle(data, path)
        return data


def compute_edges(iset, comp: dict, realization: RealizationSpec, cfg: dict, ctx):
    # 输入：iset、comp、realization、cfg、ctx；输出：(UncertainData, ScoreSamples, EdgeTable, 缓存命中标志)；继承：无（模块函数）。
    """Modules 1-3 for one integration set: (UncertainData, ScoreSamples, EdgeTable, matcher cache hit)."""
    from .execution import configure_context
    configure_context(ctx,cfg)
    profiler, logger = ctx.profiler, ctx.logger
    model, base_matcher = comp["uncertainty_model"], comp["matcher"]
    propagation = comp["uncertainty_propagation"]
    if propagation.registry_name == "calibrated_logit":
        artifact = check_inference_split(propagation.calibration, iset, ctx)
        check_matcher(artifact, base_matcher)
    if realization.mode != "mc":
        raise ValueError("Real experiments require realization.mode=mc; exact is a direct unit-test reference only")

    # Module 1: Missing-Value Uncertainty Model
    with profiler.stage("uncertainty_model", set=iset.name):
        # Independent of strategy/variant iteration order. In particular Q_mean and
        # Q_risk, and argmax's base model, must see the same stochastic Q.
        ctx.reset_rng()
        data = estimate_uncertainty(iset, model, ctx, cfg).configure(realization, ctx.seed)
    logger.debug(f"[{iset.name}] uncertainty: {data.stats()}")
    with profiler.stage("sampling", set=iset.name):
        if propagation.registry_name != 'calibrated_logit' or not cfg['experiment'].get('online',True):
            for col in data.columns:
                data.states(col)  # explicit batch references / raw baseline compatibility
    if propagation.registry_name == "calibrated_logit":
        context_version = getattr(base_matcher, "context_sampling_version", 1)
        if (artifact.get("context_sampling_version", 1) != context_version
                and any(data.state_count(c) > 1 for c in data.columns)):
            raise ValueError("table-context MC scoring changed; refit this matcher's calibration before Q experiments; "
                             "observed-only scores and LLM response caches remain reusable")

    if propagation.registry_name=='calibrated_logit' and cfg['experiment'].get('online',True):
        from .online import estimate_online
        with profiler.stage('online_evidence',set=iset.name):
            with profiler.compact():
                samples,edges,hit = estimate_online(data,model,base_matcher,propagation,artifact,cfg,ctx)
        return data,samples,edges,hit

    # Module 2: Base Matcher (score samples are cached)
    with profiler.stage("matcher", set=iset.name):
        samples, cache_hit = score_with_cache(data, model, base_matcher, cfg, ctx)
    logger.debug(f"[{iset.name}] candidate/closure: {compact_info(samples.meta.get('candidates', {}))}; "
                f"{samples.n_pairs} pairs, {len(samples)} score samples")

    # Module 3: Uncertainty Propagation
    if propagation.registry_name == "calibrated_logit":
        expected = base_matcher.expected_sample_counts(data)
        actual = {frozenset((a, b)): len(group) for (a, b), group in samples.df.groupby(["col_i", "col_j"])}
        if expected != actual:
            raise ValueError("V2 requires every closure pair and every aligned MC draw; matcher output is incomplete")
    with profiler.stage("uncertainty_propagation", set=iset.name):
        edges = comp["uncertainty_propagation"].propagate(samples, ctx)
    return data, samples, edges, cache_hit
