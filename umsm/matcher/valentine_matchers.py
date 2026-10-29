"""Pair-table baselines of the original paper through the Valentine library (pip install valentine):
COMA-S / COMA-I, Cupid, Similarity Flooding, Jaccard-Levenshtein (JLM), Distribution-based (DB).

As in the paper, every pair of tables is matched independently. Valentine works on whole
tables, so the data states are realizations of the tables (UncertainData.table_realizations).
Column draws are reused to construct aligned Monte Carlo tables with probability 1/R.
Only explicit exact unit-test references enumerate joint states. Pairs that Valentine does not
return get score 0. `is_match` marks Valentine's local (pair-table) decision, selected with
`local_selection` (default: one-to-one Hungarian selection).
"""
from __future__ import annotations

from itertools import combinations

from ..registry import MATCHERS
from ..types import ScoreSamples, UncertainData
from .base import BaseMatcher, sample_record

ALGORITHMS = {
    "coma": ("Coma", {}),
    "coma_instance": ("Coma", {"use_instances": True}),
    "cupid": ("Cupid", {}),
    "similarity_flooding": ("SimilarityFlooding", {}),
    "jaccard_levenshtein": ("JaccardDistanceMatcher", {}),
    "distribution_based": ("DistributionBased", {}),
}


@MATCHERS.register("valentine")
class ValentineMatcher(BaseMatcher):
    """Valentine pair-table matchers (coma, coma_instance, cupid, similarity_flooding, jaccard_levenshtein, distribution_based)."""

    scoring_version = 2  # All edges are evaluated under each aligned table context.
    sampling_scope = "table_pair"

    def __init__(self, algorithm: str = "coma", algorithm_params: dict | None = None,
                 instance_sample_size: int | None = 1000, local_selection: str = "one_to_one",
                 local_threshold: float | None = None):
        # 输入：algorithm、algorithm_params、instance_sample_size、local_selection、local_threshold；输出：None；初始化实例及校验参数；继承：ValentineMatcher(BaseMatcher)。
        if algorithm not in ALGORITHMS:
            raise KeyError(f"Unknown valentine algorithm {algorithm}; choose from {sorted(ALGORITHMS)}")
        if local_selection not in ("one_to_one", "all"):
            raise ValueError("local_selection must be 'one_to_one' or 'all'")
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params or {}
        self.instance_sample_size = instance_sample_size
        self.local_selection = local_selection
        self.local_threshold = local_threshold


    def _build(self):
        # 输入：实例状态；输出：Valentine 算法实例；继承：ValentineMatcher(BaseMatcher)。
        import valentine.algorithms as algs

        cls_name, defaults = ALGORITHMS[self.algorithm]
        return getattr(algs, cls_name)(**{**defaults, **self.algorithm_params})

    def score(self, data: UncertainData, ctx) -> ScoreSamples:
        # 输入：data、ctx；输出：ScoreSamples（对齐采样的原始分数）；继承：ValentineMatcher(BaseMatcher)，重写父类方法。
        from valentine import valentine_match

        reals = {t.name: data.table_realizations(t.name) for t in data.tables}
        records = []
        for ta, tb in combinations(data.tables, 2):
            active = [(ci, cj) for ci in ta.columns for cj in tb.columns if data.allows(ci, cj)]
            if not active:
                continue
            ra, rb = reals[ta.name], reals[tb.name]

            if data.realization.mode == "exact":
                runs = [(a, b, pa * pb) for a, (_, pa) in enumerate(ra) for b, (_, pb) in enumerate(rb)]
            else:
                n = max(len(ra), len(rb))
                runs = [(r % len(ra), r % len(rb), 1.0 / n) for r in range(n)]
            for sample_id, (a, b, prob) in enumerate(runs):
                fa, fb = ra[a][0], rb[b][0]
                res = valentine_match([fa, fb], self._build(), df_names=[ta.name, tb.name],
                                      instance_sample_size=self.instance_sample_size)
                selected = res.one_to_one_hungarian(self.local_threshold) if self.local_selection == "one_to_one" else res
                scores = {}
                for pair, sim in res.items():
                    key = _key(pair)
                    scores[key] = max(scores.get(key, 0.0), float(sim))
                local = {_key(p) for p in selected}
                for ci, cj in active:
                    na, nb = len(data.states(ci)), len(data.states(cj))
                    key = tuple(sorted((ci.id, cj.id)))
                    records.append(sample_record(ci, cj, a % na, b % nb, prob,
                                                 scores.get(key, 0.0), is_match=key in local,
                                                 sample_id=sample_id))
        return ScoreSamples.from_records(records, data.columns,
                                        meta={"matcher": f"valentine.{self.algorithm}", **data.score_meta()})


def _key(pair) -> tuple[str, str]:
    # 输入：pair；输出：缓存标识字符串；继承：无（模块函数）。
    a = f"{pair.source_table}.{pair.source_column}"
    b = f"{pair.target_table}.{pair.target_column}"
    return tuple(sorted((a, b)))
