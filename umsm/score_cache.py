"""Scoring and blocking caches; no calibration fitting or uncertainty propagation here."""
import hashlib
import json
import os
import copy

import pandas as pd

from .candidates import plan_candidates
from .embedding import DEFAULT_MODEL
from .types import ScoreSamples

SCORE_CACHE_VERSION = 5


def _frame_digest(frame):
    # 输入：frame；输出：DataFrame 内容摘要；继承：无（模块函数）。
    digest = hashlib.sha256()
    digest.update(repr([(str(c), str(d)) for c, d in zip(frame.columns, frame.dtypes)]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _encoder_revision(path):
    # 输入：path；输出：模型文件修订信息或 None；继承：无（模块函数）。
    if os.path.isdir(path):
        return [(f, os.stat(os.path.join(path, f)).st_size, os.stat(os.path.join(path, f)).st_mtime_ns)
                for f in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, f))]
    return None


def _key(payload):
    # 输入：payload；输出：缓存标识字符串；继承：无（模块函数）。
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]


def candidate_cache_key(cfg, iset, ctx):
    # 输入：cfg、iset、ctx；输出：不依赖 Q 和昂贵 matcher 的 blocking 缓存键；继承：无（模块函数）。
    """Blocking depends on original inputs, never Q, R, the matcher or the optimizer."""
    spec = cfg.get("candidates", {})
    return _key({"tables": [(t.name, _frame_digest(t.df)) for t in iset.tables],
                 "spec": spec, "seed": ctx.seed,
                 "encoder": _encoder_revision(ctx.resolve(spec.get("model", DEFAULT_MODEL))),
                 "groundtruth": [sorted(group) for group in iset.groundtruth or []]})


def score_cache_key(cfg, iset, model, matcher, realization, data=None, canonical=False):
    # 输入：cfg、iset、model、matcher、realization、data；输出：原始分数缓存键；继承：无（模块函数）。
    """Hash actual table/Q contents, external scores and encoder revisions, not just sizes."""
    root = cfg.get("_project_root", ".")
    payload = {"v": SCORE_CACHE_VERSION,
               "dataset": {k: cfg["dataset"].get(k) for k in ("name", "variant", "tables")},
               "set": iset.name, "files": [(t.name, _frame_digest(t.df)) for t in iset.tables],
               "uncertainty_model": {"name": model.registry_name, "params": model.params()},
               "realization": vars(realization), "candidates": cfg.get("candidates", {}),
               "matcher": {"name": matcher.registry_name, "params": matcher.params()},
               "seed": data.seed if data is not None else cfg["experiment"]["seed"]}
    if matcher.scoring_version != 1:
        payload["matcher_scoring_version"] = matcher.scoring_version
    context_version = getattr(matcher, "context_sampling_version", None)
    if context_version is not None and data is not None and any(data.state_count(c) > 1 for c in data.columns):
        payload["context_sampling_version"] = context_version
    revision = _encoder_revision(os.path.join(root, cfg.get("candidates", {}).get("model", DEFAULT_MODEL)))
    if revision is not None:
        payload["blocking_encoder_files"] = revision
    if data is not None:
        payload["estimated_data"] = [
            (t.name, _frame_digest(t.frame.map(repr)),
             [(int(r), str(c), [(repr(v), float(p)) for v, p in q])
              for (r, c), q in sorted(t.distributions.items())]) for t in data.tables]
    if canonical:
        if data is None:
            raise ValueError('Canonical score caching requires the actual estimated Q')
        params=copy.deepcopy(payload['uncertainty_model']['params'])
        params.pop('calibration',None)
        if isinstance(params.get('model_params'),dict):
            params['model_params'].pop('calibration',None)
        payload['uncertainty_model']['params']=params
        payload['canonical_q']=1  # The actual Q is already hashed, not its calibration filename.
    if matcher.registry_name == "precomputed":
        path = os.path.join(root, matcher.template.format(set=iset.name, slug=iset.slug))
        with open(path, "rb") as f:
            payload["external_scores"] = hashlib.sha256(f.read()).hexdigest()
    model_path = getattr(matcher, "model", None)
    if model_path:
        revision = _encoder_revision(os.path.join(root, model_path))
        if revision is not None:
            payload["encoder_files"] = revision
    return _key(payload)


def score_with_cache(data, model, matcher, cfg, ctx):
    # 输入：data、model、matcher、cfg、ctx；输出：(ScoreSamples, 是否命中缓存)；继承：无（模块函数）。
    """Return raw samples and cache-hit flag. Blocking is shared across crossed methods."""
    enabled = cfg["matcher"].get("cache_scores", True)
    cache_path = None
    if enabled:
        key = score_cache_key({**cfg, "_project_root": ctx.project_root}, data.source, model, matcher, data.realization, data)
        cache_path = os.path.join(ctx.cache_dir, "scores", f"{key}.pkl")
        if os.path.exists(cache_path):
            cached = pd.read_pickle(cache_path)
            data.candidate_pairs = cached["candidate_pairs"]
            ctx.logger.debug(f"[{data.source.name}] matcher scores loaded from cache {cache_path} "
                            "(runtime of this run is NOT the scoring time)")
            return ScoreSamples(cached["df"], data.columns, cached["meta"]), True

    ctx.reset_rng()  # Q estimation must not advance the matcher's random stream.
    spec = cfg.get("candidates", {})
    make_plan = lambda: plan_candidates(data.source, ctx, spec)
    plan = ctx.shared(("candidates", candidate_cache_key(cfg, data.source, ctx)), make_plan) if enabled else make_plan()
    data.candidate_pairs, candidate_info = plan
    ctx.reset_rng()
    samples = matcher.score(data, ctx)
    samples.meta.update(data.score_meta())
    samples.meta["candidates"] = candidate_info
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        pd.to_pickle({"df": samples.df, "meta": samples.meta, "candidate_pairs": data.candidate_pairs}, cache_path)
    return samples, False
