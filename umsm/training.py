"""Explicit train/validation flow: fit qx, fit matcher calibration, then save once.

Inference imports calibration.py, never this module. Only explicit fit commands
(fit_calibration.py or the study scripts' fit phases) fit parameters.
"""
import copy
import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .calibration import fit_platt, fit_retrieval_calibrators, load_artifact, match_labels, set_fingerprint
from .candidates import compact_info
from .context import RunContext
from .pipeline import build_components, compute_edges, load_sets
from .uncertainty_model.mixture import DEFAULT_GENERATORS, fit_mixture, pseudo_mask_examples, qx_diagnostics


def _fit_qx(sets, params, ctx, max_cells, mask_seed):
    # Input: fit-only tables/Q settings/seeds; output: verified matcher-independent Q fit, cached by inputs/code.
    if not ctx.resources.get('calibration_fitting'):
        return _compute_qx(sets,params,ctx,max_cells,mask_seed)
    parameters=copy.deepcopy(params); parameters.pop('calibration',None)
    directory=Path(__file__).parent
    code={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        directory/'training.py',directory/'numerics.py',directory/'uncertainty_model/mixture.py',
        directory/'uncertainty_model/typed.py',directory/'uncertainty_model/torch_bayes.py')}
    inputs={'v':1,'parameters':parameters,'sets':[set_fingerprint(s) for s in sets],
        'max_cells':max_cells,'seed':ctx.seed,'mask_seed':mask_seed,'code':code}
    key=hashlib.sha256(json.dumps(inputs,sort_keys=True,default=str).encode()).hexdigest()
    path=Path(ctx.resolve(ctx.cache_dir))/'qx_fit'/(key+'.json')
    if path.exists():
        cached=json.loads(path.read_text(encoding='utf-8'))
        digest=hashlib.sha256(json.dumps(cached['qx'],sort_keys=True,default=str).encode()).hexdigest()
        if cached['key']!=key or digest!=cached['sha256']: raise ValueError('Q fit cache checksum mismatch')
        ctx.resources['qx_fit_cache']={'hit':True,'key':key}
        print('Q fit: reuse verified fit-only cache '+key[:12],flush=True)
        return cached['qx']
    print(f'Q fit: constructing up to {max_cells} pseudo-masks',flush=True)
    qx=_compute_qx(sets,params,ctx,max_cells,mask_seed)
    payload={'key':key,'qx':qx,'sha256':hashlib.sha256(json.dumps(qx,sort_keys=True,default=str).encode()).hexdigest()}
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload,ensure_ascii=False,default=str),encoding='utf-8'); os.replace(temporary,path)
    ctx.resources['qx_fit_cache']={'hit':False,'key':key}
    print('Q fit: cached; starting independent matcher calibration',flush=True)
    return qx


def _compute_qx(sets, params, ctx, max_cells, mask_seed):
    # 输入：sets、params、ctx、max_cells、mask_seed；输出：Q 校准参数与独立隐藏值诊断；继承：无（模块函数）。
    generators = params.get("generators", DEFAULT_GENERATORS)
    typed=params.get('routing','shared')=='by_type'
    if typed:
        from .uncertainty_model.typed import DEFAULT_ROUTES, fit_typed, typed_diagnostics, KINDS
        generators=params.get('generators',DEFAULT_ROUTES)
    examples = pseudo_mask_examples(sets, generators, ctx, max_cells, mask_seed, params.get('type_options'))
    if len(examples) < 2:
        raise ValueError("at least two pseudo-masks needed for fit/diagnostic split")
    cut = min(len(examples) - 1, max(1, int(.8 * len(examples))))
    if typed:
        # Split within inferred type; no held-out truth contributes to fitted weights/spread.
        train,heldout=[],[]
        for kind in KINDS:
            group=[ex for ex in examples if ex['kind']==kind]
            split=min(len(group),max(1,int(.8*len(group))))
            train.extend(group[:split]); heldout.extend(group[split:])
        qx=fit_typed(train,generators,params.get('type_options'))
        qx['diagnostics']={name:typed_diagnostics(ex,qx,params.get('rho',.95),params.get('k_max',10))
                           for name,ex in (('fit',train),('heldout_masks',heldout))}
        return qx
    qx = {"generators": generators, **fit_mixture(examples[:cut], len(generators))}
    qx["diagnostics"] = {name: qx_diagnostics(ex, qx["temperatures"], qx["weights"],
                                            params.get("rho", .95), params.get("k_max", 10))
                         for name, ex in (("fit", examples[:cut]), ("heldout_masks", examples[cut:]))}
    return qx


def collect_matcher_scores(sets, cfg, artifact, ctx):
    # Input: fitting or held-out validation sets and frozen Q; output: raw per-draw scores, indicators, labels and edge weights.
    cfg=copy.deepcopy(cfg)
    params = cfg["uncertainty_model"].get("params", {})
    if cfg['uncertainty_model']['name']=='calibrated_mixture':
        cfg["uncertainty_model"] = {**cfg["uncertainty_model"],
            "params": {**params,"calibration":artifact,"generators":artifact['qx']['generators']}}
    elif cfg['uncertainty_model']['name']=='argmax' and params.get('model')=='calibrated_mixture':
        cfg['uncertainty_model']['params']['model_params']['calibration']=artifact
    cfg["uncertainty_propagation"] = {"name": "expected", "params": {}}
    cfg["matcher"]["cache_scores"] = bool(cfg.get('paths',{}).get('raw_score_cache',False))
    comp, realization = build_components(cfg)
    scores, labels, weights, indicators, diagnostics = [], [], [], [], []
    for iset in sets:
        _, samples, _, _ = compute_edges(iset, comp, realization, cfg, ctx)
        diagnostics.append({"set": iset.name, **compact_info(samples.meta.get("candidates", {}))})
        scores.extend(samples.df.score)
        if 'retrieved' in samples.df: indicators.extend(samples.df.retrieved)
        labels.extend(match_labels(samples, iset.groundtruth))
        # Each edge contributes the same total mass, whether scored once or R times.
        counts = samples.df.groupby(["col_i", "col_j"])["score"].transform("size")
        weights.extend(1. / counts)
    return {'scores':scores,'labels':labels,'weights':weights,'retrieved':indicators,'diagnostics':diagnostics}


def _fit_matcher(sets, cfg, artifact, ctx):
    # Input: fit sets and config; output: Platt parameters and blocking diagnostics.
    data=collect_matcher_scores(sets,cfg,artifact,ctx)
    return fit_platt(data['scores'],data['labels'],data['weights']),data['diagnostics']


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
        model_params=cfg['uncertainty_model'].get('params',{})
        from .uncertainty_model.typed import DEFAULT_ROUTES
        defaults=DEFAULT_ROUTES if model_params.get('routing')=='by_type' else DEFAULT_GENERATORS
        if (model_params.get('generators',defaults)!=qx['generators']
                or model_params.get('routing','shared')!=qx.get('routing','shared')
                or model_params.get('type_options',{})!=qx.get('type_options',{})):
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
    elif method=='retrieval_candidates':
        data=collect_matcher_scores(sets,cfg,artifact,ctx)
        if len(data['retrieved'])!=len(data['scores']):
            raise ValueError('retrieval_candidates requires a matcher emitting retrieved for every edge')
        artifact['calibration_candidates']=fit_retrieval_calibrators(data['scores'],data['labels'],data['retrieved'],data['weights'])
        artifact['platt']=copy.deepcopy(artifact['calibration_candidates']['platt_retrieved'])
        artifact['blocking_diagnostics']=data['diagnostics']
    else:
        raise ValueError('calibration.method must be platt, identity or retrieval_candidates')
    from .llm import llm_stats,SYSTEM_PROMPT,PROMPT_TEMPLATE,COMPACT_TEMPLATE
    artifact['fit_execution']={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'llm_usage':llm_stats(ctx),
        'qx_fit_cache':ctx.resources.get('qx_fit_cache'),
        'prompt_template_sha256':hashlib.sha256((SYSTEM_PROMPT+PROMPT_TEMPLATE+COMPACT_TEMPLATE).encode()).hexdigest()}
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
