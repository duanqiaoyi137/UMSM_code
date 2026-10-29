"""Command-line options for run.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import apply_override, load_config


def add_config_args(p: argparse.ArgumentParser) -> None:
    # 输入：p；输出：None；添加通用命令行参数；继承：无（模块函数）。
    p.add_argument("--config", "-c", help="YAML experiment config")
    p.add_argument("--dataset", help="dataset name for a direct four-module configuration")
    p.add_argument("--variant", help="dataset sub-folder, e.g. 'MCAR missing original/5_/seed1'")
    p.add_argument("--uncertainty-model", "-u", help="module 1")
    p.add_argument("--matcher", "-m", help="module 2: sbert_quantile | alite | magneto | llm | valentine | ...")
    p.add_argument("--propagation", "-p", help="module 3")
    p.add_argument("--optimizer", "-g", help="module 4")
    p.add_argument("--seed", type=int)
    p.add_argument("--no-cache", action="store_true",
                   help="disable embedding and score caches (use for runtime / memory measurements)")
    p.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config entry, e.g. uncertainty_propagation.params.risk_lambda=0.25")


def config_from_args(args) -> dict:
    # 输入：args；输出：完整配置字典；继承：无（模块函数）。
    cfg = load_config(args.config)
    for flag, section in ((args.uncertainty_model, "uncertainty_model"), (args.matcher, "matcher"),
                          (args.propagation, "uncertainty_propagation"), (args.optimizer, "global_optimizer")):
        if flag and 'research' in cfg:
            modules = cfg.setdefault('modules',{})
            if section=='matcher' and flag in ('mean','name','value','coma','jaccard','distribution','magneto'):
                cfg['research']['matcher'] = flag
                modules.pop('matcher',None)
            else:
                modules[section] = {'name':flag,'params':{}}
        elif flag and flag != cfg[section]["name"]:
            cfg[section] = {**cfg[section], "name": flag, "params": {}}
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    if args.variant:
        cfg["dataset"]["variant"] = args.variant
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.no_cache:
        cfg["paths"]["embedding_cache"] = False
        cfg["matcher"]["cache_scores"] = False
        if 'research' in cfg:
            cfg['research']['cache_scores'] = False
    for o in args.set:
        apply_override(cfg, o)
    return cfg


def main(argv=None, default_phase='run'):
    # 输入：统一命令行；输出：显式阶段的单 HTML/冻结校准，或无执行配置预览；继承：无（模块函数）。
    from .pipeline import build_components
    from .registry import MATCHERS,OPTIMIZERS,PROPAGATIONS,UNCERTAINTY_MODELS
    from .research import PHASES,research_config,role_config,run_research
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='UMSM paper experiments: one interface, four configurable modules')
    add_config_args(parser)
    parser.set_defaults(config=str(root/'configs/paper.yaml'))
    parser.add_argument('--phase',choices=('run','ablation','summarize','package',*PHASES),default=default_phase)
    parser.add_argument('--selection',help='frozen validation HTML; never a test-fitted diagnostic')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--check-config',action='store_true',help='validate components without loading data/models or calling APIs')
    parser.add_argument('--output','-o',help='fit artifact or summary HTML path')
    parser.add_argument('--max-cells',type=int)
    parser.add_argument('--mask-seed',type=int,default=42)
    parser.add_argument('--split-role',choices=('train','validation'),default='validation')
    parser.add_argument('--qx-from',help='reuse Q; fit only a new score calibrator')
    parser.add_argument('--reports',nargs='+',help='HTML paths/globs for phase summarize')
    parser.add_argument('--missing',help='comma-separated strategy subset')
    parser.add_argument('--optimizers',help='comma-separated optimizer subset')
    parser.add_argument('--max-variants',type=int)
    parser.add_argument('--oracle',action='store_true',help='legacy diagnostic only; never formal parameter selection')
    args = parser.parse_args(argv)
    if args.list:
        for title,registry in [('1. uncertainty_model',UNCERTAINTY_MODELS),('2. matcher',MATCHERS),
                               ('3. uncertainty_propagation',PROPAGATIONS),('4. global_optimizer',OPTIMIZERS)]:
            print(title)
            for name,doc in registry.describe().items():
                print(f'  {name:22s} {doc}')
        return
    if args.phase=='package':
        from .release import package_source
        return package_source(root,args.output or root/'dist/umsm_source.zip')
    if args.phase=='summarize':
        if not args.reports:
            parser.error('summarize requires --reports')
        from analyze_research import analyze
        return analyze(args.reports,args.output or str(root/'results/research_comparison.html'))
    if args.selection and (args.phase.startswith('validate-') or args.phase in ('fit','qcheck','profile')):
        parser.error('Frozen selection is used only for evaluation phases')
    cfg = config_from_args(args)
    research = 'research' in cfg
    if research:
        if args.dataset:
            parser.error('Research datasets have explicit roles; edit research.roles in YAML instead of --dataset')
        if args.output and args.phase=='fit':
            cfg['research']['calibration'] = args.output
        if args.phase=='fit':
            cfg['research'].update(mask_seed=args.mask_seed,calibration_split_role=args.split_role)
        if args.max_variants is not None:
            parser.error('Research protocols use explicit roles/test sets and subsets.repeats, not legacy --max-variants')
        if args.max_cells is not None:
            cfg['research']['max_cells'] = args.max_cells
        if args.qx_from:
            cfg['research']['qx_from'] = args.qx_from
        if args.missing:
            cfg['research']['missing'] = args.missing.split(',')
        if args.optimizers:
            cfg['research']['optimizers'] = args.optimizers.split(',')
        cfg = research_config(cfg,root)
    if args.check_config:
        build_components(cfg)
        print(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
        return cfg
    if research and args.phase!='run':
        if args.oracle:
            parser.error('Research phases never select parameters on test labels')
        phase = 'missing' if args.phase=='ablation' else args.phase
        return run_research(cfg,phase,root,args.selection)
    if args.phase=='fit':
        if not args.output:
            parser.error('Direct fit requires --output')
        if Path(args.output).exists():
            raise FileExistsError('Keep frozen artifacts; choose a new --output')
        from .training import fit_pipeline
        return fit_pipeline(cfg,str(root),args.output,args.split_role,args.max_cells or 300,args.mask_seed,args.qx_from)
    if args.phase=='ablation':
        from .ablation import run_ablation
        return run_ablation(cfg,str(root),only_missing=args.missing.split(',') if args.missing else None,
            only_optimizers=args.optimizers.split(',') if args.optimizers else None,
            max_variants=args.max_variants,oracle=args.oracle)
    if args.phase!='run':
        parser.error('Research phases need a configuration containing research.roles')
    if args.selection:
        parser.error('Use research phases missing/factorial for a frozen selection')
    if research:
        cfg,_ = role_config(cfg,'test',root)
    from .experiment import run_experiment
    return run_experiment(cfg,str(root))
