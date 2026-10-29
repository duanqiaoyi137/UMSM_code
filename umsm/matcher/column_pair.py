"""A strict f(c_i,c_j) adapter for matchers that normally inspect entire tables.

Only the two scored columns are visible. This is deliberately a separate method
from native full-schema COMA/Magneto, and requires its own frozen calibration.
"""
import copy
from dataclasses import replace

from ..registry import MATCHERS
from ..types import ScoreSamples
from .base import BaseMatcher,cross_table_pairs,sample_record


@MATCHERS.register('column_pair')
class ColumnPairMatcher(BaseMatcher):
    """Column-local adapter: uncertainty depends only on the two scored columns."""

    def __init__(self, matcher, matcher_params=None):
        # 输入：基础 matcher 名和参数；输出：严格二列 f 的适配器；继承：ColumnPairMatcher(BaseMatcher)。
        if matcher=='column_pair':
            raise ValueError('Do not nest column_pair adapters')
        self.matcher=matcher
        self._base=MATCHERS.build(matcher,matcher_params or {})

    @property
    def scoring_version(self):
        # 输入：实例；输出：底层 matcher 的评分版本，用于冻结校准校验；继承：ColumnPairMatcher(BaseMatcher)。
        return self._base.scoring_version

    @property
    def model(self):
        # 输入：实例；输出：底层编码模型路径或 None，用于缓存修订检查；继承：ColumnPairMatcher(BaseMatcher)。
        return getattr(self._base,'model',None)

    def params(self):
        # 输入：实例；输出：可重建并覆盖底层默认值的参数；继承：ColumnPairMatcher(BaseMatcher)，重写父类方法。
        return {'matcher':self.matcher,'matcher_params':self._base.params()}

    def score(self,data,ctx):
        # 输入：列实现及固定候选对；输出：仅依赖每对两列的原始分数；继承：ColumnPairMatcher(BaseMatcher)，重写父类方法。
        records=[]
        for ci,cj in cross_table_pairs(data):
            states_i,states_j=data.states(ci),data.states(cj)
            ai,bj,prob=data.pair_indices(ci,cj)
            for r,(a,b,p) in enumerate(zip(ai,bj,prob)):
                snapshot=copy.copy(data)
                selected={ci.table:ci.name,cj.table:cj.name}
                snapshot.tables=[replace(t,frame=t.frame[[selected[t.name]]],
                    distributions={k:q for k,q in t.distributions.items() if k[1]==selected[t.name]})
                    for t in data.tables if t.name in selected]
                snapshot.source=replace(data.source,tables=[replace(t,df=t.df[[selected[t.name]]])
                    for t in data.source.tables if t.name in selected],groundtruth=None)
                snapshot._states={ci:[states_i[int(a)]],cj:[states_j[int(b)]]}
                snapshot.candidate_pairs={frozenset((ci.id,cj.id))}
                frame=self._base.score(snapshot,ctx).df
                if len(frame)!=1:
                    raise ValueError('The column_pair adapter requires one score for its two visible columns')
                row=frame.iloc[0]
                extra={k:row[k] for k in frame.columns if k not in ScoreSamples.REQUIRED and k!='sample_id'}
                records.append(sample_record(ci,cj,int(a),int(b),p,row.score,sample_id=r,**extra))
        return ScoreSamples.from_records(records,data.columns,{'matcher':self.registry_name,'evidence_scope':'pair',**data.score_meta()})
