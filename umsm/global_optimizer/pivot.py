"""Explicit pivot and unconstrained threshold-component comparison methods."""
import numpy as np
from ..registry import OPTIMIZERS
from ..types import Matching
from .base import GlobalOptimizer
from .ilp import get_components


@OPTIMIZERS.register('kwik_cluster')
class KwikCluster(GlobalOptimizer):
    """Table-constrained KwikCluster-style random pivots; not an original-package reproduction."""
    def optimize(self,graph,ctx):
        # Input: fixed signed edges; output: feasible pivot partition with at most one column per table; GlobalOptimizer override.
        rng=np.random.default_rng(ctx.seed); available=set(c.id for c in graph.columns)
        owner={c.id:c.table for c in graph.columns}; weights={}
        for a,b,w in zip(graph.df.col_i,graph.df.col_j,graph.df.a): weights[frozenset((a,b))]=float(w)
        groups=[]
        while available:
            pivot=sorted(available)[int(rng.integers(len(available)))]; selected={pivot}; used={owner[pivot]}
            neighbors=sorted((c for c in available if c!=pivot),key=lambda c:(-weights.get(frozenset((pivot,c)),0.),c))
            for c in neighbors:
                if owner[c] not in used and weights.get(frozenset((pivot,c)),0.)>0:
                    selected.add(c); used.add(owner[c])
            available-=selected
            if len(selected)>1: groups.append(frozenset(selected))
        return Matching(groups,{'algorithm':'table-constrained KwikCluster-style pivot','seed':ctx.seed})


@OPTIMIZERS.register('threshold_components')
class ThresholdComponents(GlobalOptimizer):
    """Unconstrained positive-edge transitive closure; same-table conflicts are reported, not hidden."""
    def optimize(self,graph,ctx):
        # Input: signed edges; output: raw component partition, explicitly allowed to violate table exclusion; GlobalOptimizer override.
        components=get_components([c.id for c in graph.columns],graph.df[graph.df.a>0])
        owner={c.id:c.table for c in graph.columns}
        conflicts=sum(sum(owner[a]==owner[b] for i,a in enumerate(c) for b in c[i+1:]) for c in components)
        return Matching([frozenset(c) for c in components],{'diagnostic_unconstrained':True,
            'same_table_conflicts':conflicts,'problem_feasible':conflicts==0})
