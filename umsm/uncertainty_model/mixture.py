"""Candidate distributions: generator-local temperatures, learned mixture, then truncation.

Generators return compatible raw scores, never probabilities. Fitting is separate from
inference; pseudo-masks are created only from the explicitly supplied calibration split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from ..registry import UNCERTAINTY_MODELS
from ..calibration import check_inference_split, load_artifact
from .base import CellDistribution, DistributionModel, object_frame
from .distributional import ColumnMarginal, KnnTuple


def compatible_scores(values, kind):
    # 输入：values、kind；输出：方向统一后的 NumPy 兼容分数数组；继承：无（模块函数）。
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("generator scores must be finite")
    if kind in ("similarity", "confidence"):
        return values
    if kind in ("distance", "rank"):
        return -values
    if kind in ("frequency", "density"):
        if (values <= 0).any():
            raise ValueError("frequency/density scores must be positive")
        return np.log(values)
    raise ValueError(f"unknown score kind: {kind}")


# Extension point: factory(params) -> generator implementing generate(frame, table, iset, ctx).
# Output: {(row_position, column): [(candidate, direction-normalized raw score), ...]}.
GENERATORS = {}


class VoteGenerator:
    def __init__(self, model):
        # 输入：model；输出：None；初始化实例及校验参数；继承：VoteGenerator(object)。
        self.model = model

    def generate(self, frame, table, iset, ctx):
        # 输入：frame、table、iset、ctx；输出：缺失位置到候选原始分数的字典；继承：VoteGenerator(object)。
        raw = self.model.cell_distributions(frame, table, iset, ctx)
        # Frequency and neighbour similarity votes are positive evidence counts.
        return {key: [(v, float(np.log(w))) for v, w in q if w > 0] for key, q in raw.items()}


GENERATORS.update({"column_marginal": lambda p: VoteGenerator(ColumnMarginal(**p)),
                   "knn_tuple": lambda p: VoteGenerator(KnnTuple(**p))})
DEFAULT_GENERATORS = [{"name": "knn_tuple", "params": {"k": 10}}, {"name": "column_marginal"}]
from .typed import register_generators
register_generators(GENERATORS)


def generate_candidates(specs, frame, table, iset, ctx):
    # 输入：specs、frame、table、iset、ctx；输出：按生成器排列的候选分数字典列表；继承：无（模块函数）。
    return [GENERATORS[s["name"]](s.get("params", {})).generate(frame, table, iset, ctx) for s in specs]


def build_qx(generators, temperatures, weights):
    # 输入：generators、temperatures、weights；输出：(混合分布, 可用生成器权重)；继承：无（模块函数）。
    """Union supports; unavailable generators receive zero weight, others renormalize."""
    if len(generators) != len(temperatures) or len(generators) != len(weights):
        raise ValueError("generator/temperature/weight lengths differ")
    t, alpha = np.asarray(temperatures, float), np.asarray(weights, float)
    if (t <= 0).any() or not np.isfinite(t).all() or (alpha < 0).any() or not np.isfinite(alpha).all():
        raise ValueError("temperatures must be positive; mixture weights finite and nonnegative")
    active = np.array([bool(q) for q in generators])
    alpha = alpha * active
    if not active.any():
        return [], []
    if alpha.sum() <= 0:
        raise ValueError("active generators must have positive total mixture weight")
    alpha /= alpha.sum()
    merged = {}
    for q, temperature, weight in zip(generators, t, alpha):
        if not q:
            continue
        if len({v for v, _ in q}) != len(q):
            raise ValueError("each generator must aggregate duplicate candidates")
        scores = np.array([s for _, s in q], float)
        if not np.isfinite(scores).all():
            raise ValueError("generator scores must be finite")
        for (v, _), p in zip(q, softmax(scores / temperature)):
            merged[v] = merged.get(v, 0.) + float(weight * p)
    return list(merged.items()), alpha.tolist()


@UNCERTAINTY_MODELS.register("calibrated_mixture")
class CalibratedMixture(DistributionModel):
    """Frozen temperature-calibrated generator mixture; fitted only by explicit training phases."""

    def __init__(self, calibration: str | dict, generators=None, rho=.95, k_max=10,
                 routing='shared', type_options=None):
        # 输入：calibration、generators、rho、k_max、routing、type_options；输出：配置好的 Q 模型；继承：CalibratedMixture(DistributionModel)。
        super().__init__(rho, k_max)
        if routing not in ('shared','by_type'):
            raise ValueError('routing must be shared or by_type')
        self.calibration = calibration
        self.routing, self.type_options = routing, dict(type_options or {})
        if routing=='by_type':
            from .typed import DEFAULT_ROUTES, KINDS, route_specs
            self.generators = DEFAULT_ROUTES if generators is None else generators
            if not isinstance(self.generators,dict) or set(self.generators)-set(KINDS):
                raise ValueError('by_type requires a type -> generator list dictionary')
            for kind in KINDS:
                for spec in route_specs(self.generators,kind):
                    if spec['name'] not in GENERATORS:
                        raise ValueError(f"Unknown typed generator: {spec['name']}")
            missing = [kind for kind in KINDS if kind != 'identifier' and kind not in self.generators]
            if missing:
                raise ValueError(
                    "by_type routing must declare explicit generators for every data type; "
                    f"missing routes: {missing}. `identifier` may remain empty, but categorical, "
                    "ordered_numeric, numeric, date and text must not be omitted."
                )
            if self.generators.get('identifier'):
                raise ValueError('identifier routing must remain empty')
        else:
            self.generators = DEFAULT_GENERATORS if generators is None else generators
            if not isinstance(self.generators,list):
                raise ValueError('shared routing requires a generator list')
        if not self.generators:
            raise ValueError("at least one candidate generator is required")

    def parameters(self, ctx, iset=None):
        # 输入：ctx、iset；输出：冻结的 Q 温度、权重和生成器配置；继承：CalibratedMixture(DistributionModel)。
        artifact = (load_artifact(self.calibration, ctx) if iset is None
                    else check_inference_split(self.calibration, iset, ctx))
        q = artifact["qx"]
        if q["generators"] != self.generators:
            raise ValueError("qx generator specification differs from fitted calibration")
        if q.get('routing','shared')!=self.routing or q.get('type_options',{})!=self.type_options:
            raise ValueError('qx type routing/options differ from fitted calibration')
        if self.routing=='by_type' and q.get('typed_version')!=1:
            raise ValueError('Unsupported typed Q calibration version')
        return q

    def params(self):
        # Input: instance; output: stable constructor/cache parameters; DistributionModel override.
        params=super().params()
        if self.routing=='shared' and not self.type_options:
            params.pop('routing'); params.pop('type_options')
        return params

    def cell_distributions(self, frame, table, iset, ctx):
        # 输入：frame、table、iset、ctx；输出：缺失位置到候选分布的字典；继承：CalibratedMixture(DistributionModel)，重写父类方法。
        p = self.parameters(ctx, iset)
        if self.routing=='by_type':
            from .typed import candidate_examples, calibrated_q
            examples=candidate_examples(self.generators,frame,table,iset,ctx,self.type_options)
            return {key:calibrated_q(ex,p,self.k_max) for key,ex in examples.items()}
        proposals = generate_candidates(self.generators, frame, table, iset, ctx)
        out = {}
        for key in set().union(*(g.keys() for g in proposals)):
            local = [g.get(key, []) for g in proposals]
            q, alpha = build_qx(local, p["temperatures"], p["weights"])
            out[key] = CellDistribution(q, {"active_weights": alpha, "temperatures": p["temperatures"],
                                            "generator_candidate_counts": [len(g) for g in local]})
        return out

    def prepare_distribution(self, q):
        # Input: typed/legacy Q; output: retained finite probabilities and diagnostics; DistributionModel override.
        if isinstance(q,CellDistribution) and q.info.get('quantile_compressed'):
            from .base import truncate_and_normalize
            kept,stats=truncate_and_normalize(q.candidates,1.,self.k_max)
            return kept,{**stats,**q.info,'requested_rho':self.rho,'numeric_quantile_mass_kept':1.}
        return super().prepare_distribution(q)

    def estimate(self, iset, ctx):
        # 输入：iset、ctx；输出：UncertainData（不修改原表）；继承：CalibratedMixture(DistributionModel)，重写父类方法。
        result = super().estimate(iset, ctx)
        result.info["fitted_qx"] = self.parameters(ctx)
        return result


def pseudo_mask_examples(sets, specs, ctx, max_cells=500, seed=42, type_options=None):
    # 输入：sets、specs、ctx、max_cells、seed、type_options；输出：隐藏值及无泄漏候选/类型标签；继承：无（模块函数）。
    """One hidden cell per example. Neither frame, Table nor IntegrationSet exposes its y.

    Max cells applies over the whole split; existing natural missing cells are preserved.
    """
    from dataclasses import replace
    rng = np.random.default_rng(seed)
    if max_cells < 1:
        raise ValueError("max_cells must be positive")
    blocks = [(si, ti, int(t.df.notna().to_numpy().sum())) for si, s in enumerate(sets) for ti, t in enumerate(s.tables)]
    ends = np.cumsum([n for _, _, n in blocks])
    total = int(ends[-1]) if len(ends) else 0
    chosen = rng.choice(total, min(max_cells, total), replace=False)
    if isinstance(specs,dict):
        from .typed import profiles
        # Balanced type strata prevent long categorical columns monopolizing the fit budget.
        strata={}; start=0
        for si,ti,n in blocks:
            source,table=sets[si],sets[si].tables[ti]
            prof=profiles(table.df,table.name,type_options)
            flat=np.flatnonzero(table.df.notna().to_numpy().ravel())
            for offset,index in enumerate(flat):
                ci=int(index%table.df.shape[1]); kind=prof[table.df.columns[ci]].kind
                strata.setdefault(kind,[]).append(start+offset)
            start+=n
        pools={kind:list(rng.permutation(indices)) for kind,indices in sorted(strata.items())}
        balanced=[]
        while len(balanced)<min(max_cells,total):
            for pool in pools.values():
                if pool and len(balanced)<min(max_cells,total): balanced.append(pool.pop())
        chosen=np.asarray(balanced,dtype=int)
    # Keep only selected positions, rather than a Python tuple per observed dataset cell.
    positions = {}
    for bi in set(np.searchsorted(ends, chosen, side="right")):
        si, ti, _ = blocks[bi]
        start = int(ends[bi - 1]) if bi else 0
        observed = np.flatnonzero(sets[si].tables[ti].df.notna().to_numpy().ravel())
        for ix in chosen[(chosen >= start) & (chosen < ends[bi])]:
            r, c = np.unravel_index(observed[ix - start], sets[si].tables[ti].df.shape)
            positions[int(ix)] = si, ti, int(r), int(c)
    examples = []
    for example_number,ix in enumerate(chosen,1):
        si, ti, r, ci = positions[int(ix)]
        source, table = sets[si], sets[si].tables[ti]
        frame = object_frame(table.df)
        truth, col = frame.iat[r, ci], frame.columns[ci]
        frame.iat[r, ci] = np.nan
        hidden_table = replace(table, df=frame, path=None)
        hidden_tables = [replace(t, path=None) for t in source.tables]
        hidden_tables[ti] = hidden_table
        hidden_set = replace(source, tables=hidden_tables, groundtruth=None)
        if isinstance(specs,dict):
            from .typed import candidate_examples, ColumnProfile, TypedGenerator
            proposal=candidate_examples(specs,frame,hidden_table,hidden_set,ctx,type_options,{(r,col)})[(r,col)]
            profile=ColumnProfile(**proposal['profile'])
            baseline=TypedGenerator('quantile_marginal').generate(frame,hidden_table,hidden_set,ctx,
                {col:profile},{(r,col)}).get((r,col),[]) if profile.kind in ('numeric','date') else []
            examples.append({**proposal,'truth':truth,'numeric_truth':profile.number(truth) if profile.kind in ('numeric','date') else None,
                'marginal_baseline':baseline,'set':source.name,'table':table.name,'row':r,'column':str(col)})
        else:
            proposals = generate_candidates(specs, frame, hidden_table, hidden_set, ctx)
            examples.append({"generators": [p.get((r, col), []) for p in proposals], "truth": truth,
                             "set": source.name, "table": table.name, "row": r, "column": str(col)})
        if ctx.resources.get('calibration_fitting') and (example_number==1 or example_number%10==0 or example_number==len(chosen)):
            print(f'Q fit pseudo-masks: {example_number}/{len(chosen)}',flush=True)
    return examples


def qx_diagnostics(examples, temperatures, weights, rho=.95, k_max=10):
    # 输入：examples、temperatures、weights、rho、k_max；输出：覆盖率、NLL、top1、熵等指标字典；继承：无（模块函数）。
    from .base import truncate_and_normalize
    rows = []
    for ex in examples:
        q, _ = build_qx(ex["generators"], temperatures, weights)
        kept, stats = truncate_and_normalize(q, rho, k_max)
        before, after = dict(q).get(ex["truth"], 0.), dict(kept).get(ex["truth"], 0.)
        rows.append({"union_covered": before > 0, "recall_at_k": after > 0,
                     "nll_union": -np.log(max(before, 1e-12)), "nll_retained": -np.log(max(after, 1e-12)),
                     "top1_accuracy": bool(kept and kept[0][0] == ex["truth"]), **stats})
    df = pd.DataFrame(rows)
    return {"n_masks": len(rows), "nll_floor": 1e-12, "rho": rho, "k_max": k_max,
            **({c: float(df[c].mean()) for c in df} if rows else {})}


def fit_mixture(examples, n_generators):
    # 输入：examples、n_generators；输出：学得的温度、混合权重及拟合诊断；继承：无（模块函数）。
    """Joint masked-value NLL. Uncovered examples contribute constant floored loss.

    Coverage is reported separately, so this floor cannot conceal generation failures.
    """
    if not examples or not any(any(ex["truth"] in dict(q) for q in ex["generators"]) for ex in examples):
        raise ValueError("no pseudo-masked truth covered; cannot fit qx calibration")
    def unpack(x):
        # 输入：x；输出：(正温度数组, softmax 混合权重)；继承：无（fit_mixture 内部函数）。
        return np.exp(x[:n_generators]), softmax(x[n_generators:])
    def loss(x):
        # 输入：x；输出：当前参数的负对数似然目标 float；继承：无（fit_mixture 内部函数）。
        temperatures, weights = unpack(x)
        return np.mean([-np.log(max(dict(build_qx(ex["generators"], temperatures, weights)[0])
                                           .get(ex["truth"], 0.), 1e-12)) for ex in examples])
    initial = np.zeros(2 * n_generators)
    result = minimize(loss, initial, method="L-BFGS-B", bounds=[(-4., 4.)] * n_generators + [(-12., 12.)] * n_generators)
    if not result.success:
        raise RuntimeError(f"qx calibration failed: {result.message}")
    temperatures, weights = unpack(result.x)
    return {"temperatures": temperatures.tolist(), "weights": weights.tolist(),
            "fit_nll_initial": float(loss(initial)), "fit_nll": float(result.fun), "fit_success": True,
            "log_temperature_bounds": [-4., 4.], "mixture_logit_bounds": [-12., 12.]}
