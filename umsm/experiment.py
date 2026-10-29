"""Single experiment: four modules, evaluation, one HTML result."""
import logging
import os
import platform
import re
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

import pandas as pd

from .calibration import calibration_snapshot
from .candidates import compact_info
from .context import RunContext, set_global_seed
from .embedding import encoder_stats
from .evaluation import METRIC_KEYS, evaluate, validate_matching
from .llm import llm_stats
from .pipeline import build_components, compute_edges, load_sets
from .profiling import Profiler
from .report import diagnostics, matching_record, write_report


def _slug(s):
    # 输入：s；输出：路径安全的实验名称；继承：无（模块函数）。
    return re.sub(r'[^A-Za-z0-9_.+-]+', '_', s).strip('_')


def _setup_logger(verbose=False):
    # 输入：verbose；输出：运行 logger；继承：无（模块函数）。
    logger = logging.getLogger('umsm')
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def environment_info(project_root=None):
    # 输入：project_root；输出：复现实验环境信息；继承：无（模块函数）。
    import psutil

    packages = {}
    for package in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'transformers', 'valentine'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    info = {'python': sys.version, 'platform': platform.platform(), 'packages': packages,
            'processor': platform.processor(), 'cpu_count': os.cpu_count(),
            'total_ram_gb': psutil.virtual_memory().total / 1024 ** 3}
    torch = sys.modules.get('torch')
    if torch is not None:
        info['cuda_available'] = torch.cuda.is_available()
        if info['cuda_available']:
            info['gpu'] = torch.cuda.get_device_name(0)
    if project_root:
        head = os.path.join(project_root, '.git', 'HEAD')
        if os.path.isfile(head):
            with open(head, encoding='utf-8') as source:
                info['git_head'] = source.read().strip()
    return info


def _edge_stats(edges):
    # 输入：edges；输出：边数量与 sigma 等统计；继承：无（模块函数）。
    df = edges.df
    return {'n_edges': len(df), 'n_uncertain_edges': int((df.sigma > 0).sum()),
            'mean_sigma': float(df.sigma.mean()) if len(df) else 0.,
            'max_sigma': float(df.sigma.max()) if len(df) else 0.,
            'candidates': compact_info(edges.meta.get('candidates', {}))}


def run_experiment(cfg, project_root):
    # 输入：cfg、project_root；输出：单次实验指标、资源统计及报告路径；继承：无（模块函数）。
    exp = cfg['experiment']
    name = exp.get('name') or cfg['dataset']['name']
    run_id = f'{datetime.now():%Y%m%d-%H%M%S-%f}_{_slug(name)}'
    logger = _setup_logger(exp.get('verbose', False))
    set_global_seed(exp['seed'])
    profiler = Profiler(cfg['profiling']['interval_s'], cfg['profiling']['tracemalloc']).start()
    ctx = RunContext(seed=exp['seed'], project_root=project_root,
                     cache_dir=os.path.join(project_root, cfg['paths']['cache_dir']),
                     embedding_cache=cfg['paths']['embedding_cache'], profiler=profiler, logger=logger)
    rows, predictions, details = [], [], []
    try:
        comp, realization = build_components(cfg)
        calibration = calibration_snapshot(comp, ctx)
        with profiler.stage('load_data'):
            sets = load_sets(cfg, project_root)
        for iset in sets:
            data, samples, edges, hit = compute_edges(iset, comp, realization, cfg, ctx)
            with profiler.stage('global_optimizer', set=iset.name):
                matching = comp['global_optimizer'].optimize(edges, ctx)
            validate_matching(matching, edges, iset.name)
            prediction = matching_record(matching, iset.groundtruth, edges.columns)
            metrics = evaluate(matching, iset.groundtruth) if iset.groundtruth is not None else {}
            rows.append({'set': iset.name, **iset.stats(), **data.stats(), **_edge_stats(edges), **metrics,
                         **{k: prediction[k] for k in ('TP', 'FP', 'FN') if k in prediction},
                         'matcher_cache_hit': hit, 'n_score_samples': len(samples),
                         **{k:samples.meta.get(k,False) for k in ('online_statistics','legacy_cache_import','raw_samples_retained')}})
            predictions.append({'set': iset.name, 'matching': prediction})
            detail = diagnostics(data, samples, edges, cfg)
            if detail:
                details.append({'set': iset.name, **detail, 'optimizer_artifacts': matching.artifacts})
    finally:
        resources = profiler.stop()
    frame = pd.DataFrame(rows)
    overall = {k: float(frame[k].mean()) for k in METRIC_KEYS if k in frame}
    result = {'run_id': run_id, 'name': name, 'kind': 'run', 'config': cfg,
              'environment': environment_info(project_root), 'calibration': calibration,
              'rows': rows, 'summary': [{'n_sets': len(rows), **overall}], 'predictions': predictions,
              'diagnostics': details, 'metrics_mean_over_sets': overall,
              'resources': {**resources, 'stages': profiler.stage_totals(), 'stage_records': profiler.records,
                            'embedding': encoder_stats(ctx), 'llm': llm_stats(ctx)}}
    path = ctx.resolve(os.path.join(exp['output_dir'], 'runs', run_id + '.html'))
    write_report(path, result)
    print('Pair F1:', f"{overall['pair_f1']:.4f}" if overall else 'no ground truth', '|', path)
    return {**result, 'result_file': path}
