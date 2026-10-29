"""ALITE holistic schema matching (integration ids), on the edge weights of any matcher.

As codes/align_integration_ids.py of https://github.com/northeastern-datalab/alite: agglomerative
clustering of all columns of the integration set where only columns of different tables are
connected (sklearn connectivity), for every number of clusters k in [largest table width,
#columns), keeping the k with the best silhouette score. ALITE clusters L2 distances of column
embeddings; here the distance is 1 - w (same-table pairs and pairs without an edge: 1).
Clusters that still contain two columns of one table keep, per table, the column with the
largest mean weight to the rest of the cluster; the others become singletons.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from ..registry import OPTIMIZERS
from ..types import EdgeTable, Matching
from .base import GlobalOptimizer


@OPTIMIZERS.register("alite_cluster")
class ALITECluster(GlobalOptimizer):
    """ALITE integration-id clustering: agglomerative clustering with cross-table connectivity, k by silhouette."""

    def __init__(self, linkage: str = "complete"):
        # 输入：linkage；输出：None；初始化实例及校验参数；继承：ALITECluster(GlobalOptimizer)。
        if linkage not in ("complete", "average", "single"):
            raise ValueError("linkage must be 'complete', 'average' or 'single'")
        self.linkage = linkage

    def optimize(self, graph: EdgeTable, ctx) -> Matching:
        # 输入：graph、ctx；输出：Matching（互斥且跨表的语义组）；继承：ALITECluster(GlobalOptimizer)，重写父类方法。
        cols = [c.id for c in graph.columns]
        tables = [c.table for c in graph.columns]
        n = len(cols)
        pos = {c: k for k, c in enumerate(cols)}
        w = np.zeros((n, n))
        for i, j, v in zip(graph.df["col_i"], graph.df["col_j"], graph.df["w"]):
            w[pos[i], pos[j]] = w[pos[j], pos[i]] = max(w[pos[i], pos[j]], float(v))
        dist = 1.0 - np.clip(w, 0.0, 1.0)
        np.fill_diagonal(dist, 0.0)
        tab = np.array(tables)
        conn = csr_matrix((tab[:, None] != tab[None, :]).astype(float))
        k_min = max(np.unique(tab, return_counts=True)[1])
        scores = {}
        best_labels, best_k = np.arange(n), n
        for k in range(max(k_min, 2), n):
            labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage=self.linkage,
                                             connectivity=conn).fit_predict(dist)
            if len(set(labels)) < 2:
                continue
            scores[k] = float(silhouette_score(dist, labels, metric="precomputed"))
            if scores[k] > scores.get(best_k, -np.inf):
                best_k, best_labels = k, labels
        groups, n_split = [], 0
        for lab in np.unique(best_labels):
            members = np.flatnonzero(best_labels == lab)
            keep = []
            for t in np.unique(tab[members]):
                same = members[tab[members] == t]
                if len(same) > 1:
                    n_split += len(same) - 1
                    others = members[tab[members] != t]
                    same = [max(same, key=lambda m: w[m, others].mean() if len(others) else 0.0)]
                keep.extend(same)
            if len(keep) >= 2:
                groups.append(frozenset(cols[m] for m in keep))
        info = {"k": int(best_k), "silhouette": scores.get(best_k), "k_range": [int(k_min), n - 1],
                "n_same_table_removed": n_split}
        return Matching(groups, info)
