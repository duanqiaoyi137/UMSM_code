"""One default four-module configuration and its controlled ablation specifications."""
import copy

from .config import load_config


def paper_config(calibration, risk_lambda=.5, prior=.5):
    # 输入：冻结校准路径、λ、prior；输出：论文四模块默认配置；继承：无（模块函数）。
    cfg = load_config(None)
    cfg['candidates'] = {'mode':'all'}
    cfg['matcher'] = {'name':'sbert_mean','params':{},'cache_scores':True}
    cfg['uncertainty_model'] = {'name':'calibrated_mixture','params':{'calibration':calibration,'rho':.95,'k_max':10},
                                'realization':{'mode':'mc','n_samples':32}}
    cfg['uncertainty_propagation'] = {'name':'calibrated_logit','params':{'calibration':calibration,'risk_lambda':risk_lambda,'prior':prior}}
    cfg['global_optimizer'] = {'name':'ilp_logit','params':{'save_solution':False}}
    cfg['ablation'] = ablation_specs(cfg)
    return cfg


def ablation_specs(cfg):
    # 输入：四模块配置；输出：共享 matcher/校准的五种缺失策略及四种优化器配置；继承：无（模块函数）。
    model,prop = cfg['uncertainty_model'],cfg['uncertainty_propagation']
    strategies = {}
    for name in ('observed_only','mode','top1','Q_mean','Q_risk'):
        selected = copy.deepcopy(model)
        if name in ('observed_only','mode'):
            selected = {'name':name,'params':{}}
        elif name=='top1':
            selected = {'name':'argmax','params':{'model':model['name'],'model_params':copy.deepcopy(model['params'])}}
        evidence = copy.deepcopy(prop)
        if evidence['name']=='calibrated_logit' and name!='Q_risk':
            evidence['params']['risk_lambda'] = 0.
        strategies[name] = {'uncertainty_model':selected,'uncertainty_propagation':evidence}
    prior = prop.get('params',{}).get('prior',.5)
    return {'objective':'pair_f1','reference':{'missing':'Q_risk','optimizer':'global'},
            'variants':{'list':['original']},'missing':strategies,'optimizers':{
        'local':{'name':'local_merge','params':{'local_decision':'hungarian','weight':'a','min_weight':0.}},
        'greedy':{'name':'greedy_merge','params':{'selection':'threshold','threshold':prior}},
        'alite':{'name':'alite_cluster','params':{}},
        'global':copy.deepcopy(cfg['global_optimizer'])}}
