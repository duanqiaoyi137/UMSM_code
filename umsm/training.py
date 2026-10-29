"""Explicit train/validation flow: fit qx, fit matcher calibration, then save once.

Inference imports calibration.py, never this module. Only explicit fit commands
(fit_calibration.py or the study scripts' fit phases) fit parameters.
"""
import copy
import json
import os

from .calibration import fit_platt, load_artifact, match_labels, set_fingerprint
from .candidates import compact_info
from .context import RunContext
from .pipeline import build_components, compute_edges, load_sets
from .uncertainty_model.mixture import DEFAULT_GENERATORS, fit_mixture, pseudo_mask_examples, qx_diagnostics


def _fit_qx(sets, params, ctx, max_cells, mask_seed):
    # 输入：sets、params、ctx、max_cells、mask_seed；输出：Q 校准参数与独立隐藏值诊断；继承：无（模块函数）。
    generators = params.get("generators", DEFAULT_GENERATORS)
    examples = pseudo_mask_examples(sets, generators, ctx, max_cells, mask_seed)
    if len(examples) < 2:
        raise ValueError("at least two pseudo-masks needed for fit/diagnostic split")
    cut = min(len(examples) - 1, max(1, int(.8 * len(examples))))
    qx = {"generators": generators, **fit_mixture(examples[:cut], len(generators))}
    qx["diagnostics"] = {name: qx_diagnostics(ex, qx["temperatures"], qx["weights"],
                                            params.get("rho", .95), params.get("k_max", 10))
                         for name, ex in (("fit", examples[:cut]), ("heldout_masks", examples[cut:]))}
    return qx


def _fit_matcher(sets, cfg, artifact, ctx):
    # 输入：sets、cfg、artifact、ctx；输出：(Platt 参数, blocking 诊断列表)；继承：无（模块函数）。
    params = cfg["uncertainty_model"].get("params", {})
    if cfg['uncertainty_model']['name']=='calibrated_mixture':
        cfg["uncertainty_model"] = {**cfg["uncertainty_model"],
            "params": {**params,"calibration":artifact,"generators":artifact['qx']['generators']}}
    elif cfg['uncertainty_model']['name']=='argmax' and params.get('model')=='calibrated_mixture':
        cfg['uncertainty_model']['params']['model_params']['calibration']=artifact
    cfg["uncertainty_propagation"] = {"name": "expected", "params": {}}
    cfg["matcher"]["cache_scores"] = False
    comp, realization = build_components(cfg)
    scores, labels, weights, diagnostics = [], [], [], []
    for iset in sets:
        _, samples, _, _ = compute_edges(iset, comp, realization, cfg, ctx)
        diagnostics.append({"set": iset.name, **compact_info(samples.meta.get("candidates", {}))})
        scores.extend(samples.df.score)
        labels.extend(match_labels(samples, iset.groundtruth))
        # Each edge contributes the same total mass, whether scored once or R times.
        counts = samples.df.groupby(["col_i", "col_j"])["score"].transform("size")
        weights.extend(1. / counts)
    return fit_platt(scores, labels, weights), diagnostics


def fit_pipeline(cfg, project_root, output, split_role="validation", max_cells=300, mask_seed=42,
                 qx_from=None):
    # 输入：cfg、project_root、output、split_role、max_cells、mask_seed、qx_from；输出：冻结校准字典；原子写入校准文件；继承：无（模块函数）。
    """Fit only the explicitly declared split; publish the artifact after both fits succeed."""
    cfg = copy.deepcopy(cfg)
    if split_role not in ("train", "validation"):
        raise ValueError("calibration split role must be train or validation")
    sets = load_sets(cfg, project_root)
    if not sets:
        raise ValueError("empty calibration split")
    ctx = RunContext(seed=int(cfg["experiment"]["seed"]), project_root=project_root,
                     cache_dir=os.path.join(project_root, cfg["paths"]["cache_dir"]))
    ctx.resources["calibration_fitting"] = True
    fit_sets = [{"dataset": s.dataset, "name": s.name, "variant": cfg["dataset"].get("variant"),
                 "fingerprint": set_fingerprint(s)} for s in sets]
    matcher = build_components(cfg)[0]["matcher"]
    source = load_artifact(qx_from, ctx) if qx_from else None
    if source is not None:
        # Q is matcher-independent. Only Q parameters are reused; Platt is fitted below.
        qx = copy.deepcopy(source["qx"])
        if cfg["uncertainty_model"].get("params", {}).get("generators", DEFAULT_GENERATORS) != qx["generators"]:
            raise ValueError("qx_from generators differ from the configured generators")
        qx_sets = copy.deepcopy(source.get("qx_fit_sets", source["fit_sets"]))
    else:
        qx = _fit_qx(sets, cfg["uncertainty_model"].get("params", {}), ctx, max_cells, mask_seed)
        qx_sets = copy.deepcopy(fit_sets)
    artifact = {"version": "qx-global-v2.1", "split_role": split_role, "mask_seed": mask_seed,
                "mc_seed": ctx.seed, "realization": cfg["uncertainty_model"].get("realization", {}),
                "fit_sets": list({s["fingerprint"]: s for s in qx_sets + fit_sets}.values()),
                "qx_fit_sets": qx_sets, "platt_fit_sets": fit_sets,
                "matcher": copy.deepcopy(cfg["matcher"]),
                "matcher_scoring_version": matcher.scoring_version, "qx": qx}
    artifact['uncertainty_model'] = copy.deepcopy(cfg['uncertainty_model'])
    if hasattr(matcher, "context_sampling_version"):
        artifact["context_sampling_version"] = matcher.context_sampling_version
    if qx_from:
        artifact["qx_from"] = str(qx_from)
    method = cfg.get('calibration',{}).get('method','platt')
    if method=='identity':
        artifact['platt'] = {'method':'identity','source':'explicitly declared calibrated base matcher'}
        artifact['blocking_diagnostics'] = []
    elif method=='platt':
        artifact["platt"], artifact["blocking_diagnostics"] = _fit_matcher(sets, cfg, artifact, ctx)
    else:
        raise ValueError('calibration.method must be platt or identity')
    output = ctx.resolve(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False, default=str)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return artifact
