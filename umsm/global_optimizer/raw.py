"""Raw-score baseline a=w-theta. The current method is ilp_logit, in logit.py."""
import numpy as np
import pandas as pd

from ..registry import OPTIMIZERS
from .ilp import PartitionILP, build_candidate_graph, get_components, pair_lookup


@OPTIMIZERS.register("ilp")
class ILPOptimizer(PartitionILP):
    """Raw-score ablation baseline: w-theta potential, with optional Top-L positive-evidence blocking."""

    def __init__(self, theta: float = 0.5, top_l: int | None = 5, candidate_positive_only: bool = True,
                 triangles: str = "auto", full_max_columns: int = 40, missing_w: float | None = None,
                 time_limit: float | None = None, mip_rel_gap: float = 0.0, max_lazy_rounds: int = 200,
                 save_solution: bool = True, require_optimal: bool = True):
        # 输入：theta、top_l、candidate_positive_only 等配置参数（详见签名）；输出：None；初始化实例及校验参数；继承：ILPOptimizer(PartitionILP)，重写父类方法。
        super().__init__(triangles, full_max_columns, time_limit, mip_rel_gap, max_lazy_rounds, save_solution, require_optimal)
        if top_l is not None and (not isinstance(top_l, int) or top_l < 1):
            raise ValueError("top_l must be a positive integer or null")
        if isinstance(theta, str) or not np.isfinite(theta):
            raise ValueError("raw ILP needs a finite numeric theta; tune it on validation via ablation.py")
        self.theta, self.top_l = float(theta), top_l
        self.candidate_positive_only, self.missing_w = candidate_positive_only, missing_w

    def optimize(self, graph, ctx):
        # 输入：graph、ctx；输出：Matching（互斥且跨表的语义组）；继承：ILPOptimizer(PartitionILP)。
        weights = pair_lookup(graph.df, "w")
        frame = pd.DataFrame([{"col_i": a, "col_j": b, "w": w}
                              for pair, w in weights.items() for a, b in [sorted(pair)]],
                             columns=["col_i", "col_j", "w"])
        candidates = build_candidate_graph(frame, self.theta, self.top_l, self.candidate_positive_only)
        pairs = {frozenset((a, b)) for a, b in zip(candidates.col_i, candidates.col_j)}
        components = get_components([c.id for c in graph.columns], candidates)
        return self._optimize_components(graph, ctx, components, pairs, weights,
                                         theta=self.theta, missing_w=self.missing_w,
                                         info={"theta": self.theta, "top_l": self.top_l,
                                               "blocking_exact": self.top_l is None})
