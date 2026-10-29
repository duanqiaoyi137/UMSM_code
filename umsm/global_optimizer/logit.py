"""V2.1 uses supplied calibrated log-odds, without applying a second threshold."""
import numpy as np
from itertools import combinations
from ..registry import OPTIMIZERS
from .ilp import PartitionILP, get_components, pair_lookup


@OPTIMIZERS.register("ilp_logit")
class LogitILP(PartitionILP):
    """Full closure ILP: maximize sum(a*y); same-table exclusion and triangles."""

    def optimize(self, graph, ctx):
        # 输入：graph、ctx；输出：Matching（互斥且跨表的语义组）；继承：LogitILP(PartitionILP)。
        if "a" not in graph.df or graph.meta.get("potential") != "calibrated_logit":
            raise ValueError("ilp_logit requires calibrated_logit propagation, including finite a")
        if not np.isfinite(graph.df.a.to_numpy(float)).all():
            raise ValueError("nonfinite logit potential")
        info = graph.meta.get("candidates", {})
        weights, potentials = pair_lookup(graph.df, "w"), pair_lookup(graph.df, "a")
        if "components" in info:
            owner = {c.id: c.table for c in graph.columns}
            expected = {frozenset((a, b)) for comp in info["components"] for a, b in combinations(comp, 2)
                        if owner[a] != owner[b]}
            present = {frozenset((a, b)) for a, b in zip(graph.df.col_i, graph.df.col_j)}
            if expected != present:
                raise ValueError("V2 optimizer needs exactly the full blocking component closure")
            components = [comp for comp in info["components"] if len(comp) >= 2]
            candidates = {frozenset((r["col_i"], r["col_j"])) for r in info["blocking_edges"] if r["candidate"]}
        else:
            # Direct API calls may supply already-complete components without blocking metadata.
            components = get_components([c.id for c in graph.columns], graph.df)
            candidates = set(weights)
        return self._optimize_components(graph, ctx, components, candidates, weights, a_lookup=potentials,
                                         info={"potential": "calibrated_logit", "theta": None, "top_l": None,
                                               "blocking_exact": info.get("mode", "all") == "all",
                                               "blocking": {k: v for k, v in info.items()
                                                            if k not in ("blocking_edges", "components")}})
