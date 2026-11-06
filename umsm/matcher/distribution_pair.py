"""Exact one-dimensional EMD kernel for the strict two-column Distribution adapter."""
import math
import numpy as np


def quantile_emd_cdf(column1,column2,quantiles=256):
    # Input: Valentine's original ranked columns; output: 1-D uniform-bin transport distance or Inf.
    from valentine.algorithms.distribution_based.quantile_histogram import QuantileHistogram
    if column1.size==0 or column2.size==0: return math.inf
    first=column1.quantile_histogram
    second=QuantileHistogram(column2.long_name,column2.ranks,column2.size,quantiles,reference_hist=first)
    a=first.get_values; b=second.get_values
    if not a.sum() or not b.sum(): return math.inf
    a=a/a.sum(); b=b/b.sum()
    # Only n-1 intervals transport mass; the last cumulative difference is theoretically zero.
    return float(np.abs(np.cumsum(a-b)[:-1]).sum()/len(a))


def intersection_emd_cdf(column1,column2,namespace,quantiles=256,use_bloom_filters=False):
    # Input: two ranked columns; output: the original mean of the two intersection distances.
    from valentine.algorithms.distribution_based.column_model import CorrelationClusteringColumn
    if use_bloom_filters:
        from valentine.algorithms.distribution_based.emd_utils import intersection_emd
        return intersection_emd(column1,column2,namespace,quantiles,True)
    common=set(column1.data)&set(column2.data)
    if not common: return math.inf
    values1=[v for v in column1.data if v in common]; values2=[v for v in column2.data if v in common]
    one=CorrelationClusteringColumn('','intersection1',values1,'','',namespace)
    two=CorrelationClusteringColumn('','intersection2',values2,'','',namespace)
    return (quantile_emd_cdf(column1,one,quantiles)+quantile_emd_cdf(column2,two,quantiles))/2.


def distribution_pair_score(values_i,values_j,quantiles=256,t1=.15,t2=.15):
    # Input: exactly two observed/realized value arrays; output: the strict two-column raw Distribution score.
    from .distribution_fast import PreparedColumn,TwoColumnDistribution
    helper=TwoColumnDistribution({'quantiles':quantiles,'threshold1':t1,'threshold2':t2})
    columns=[]
    for table,column,values in [('A','x',values_i),('B','y',values_j)]:
        import pandas as pd
        values=tuple(pd.Series(values,dtype=object).dropna().tolist())
        columns.append(PreparedColumn((table,table,column,table+':'+column),values,frozenset(values),''))
    return helper.score(*columns)[0]
