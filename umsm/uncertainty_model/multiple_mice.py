"""Multiple MICE baseline: aligned complete-table draws, never independent cell resampling."""
import copy
import zlib

import numpy as np

from ..registry import UNCERTAINTY_MODELS
from ..types import ColumnState, UncertainData, UncertainTable, is_missing
from .base import UncertaintyModel,object_frame
from .typed import StatisticalImputer


class MultipleMICEData(UncertainData):
    # Input: original tables and MICE table draws; output: aligned column/table realizations.
    def __init__(self, source, tables, model, worlds, info):
        super().__init__(source,tables,model,info=info)
        self.worlds=worlds

    def configure(self, realization, seed):
        if realization.mode!='mc' or realization.n_samples!=self.info['m']:
            raise ValueError('Multiple MICE requires its configured m aligned realizations')
        return super().configure(realization,seed)

    def state_count(self, col):
        return self.info['m'] if self.table(col.table).uncertain_cells(col.name) else 1

    def states(self, col):
        if col not in self._states:
            frames=self.worlds[col.table] if self.state_count(col)>1 else self.worlds[col.table][:1]
            self._states[col]=[ColumnState([v for v in f[col.name] if not is_missing(v)],1/len(frames)) for f in frames]
        return self._states[col]

    def iter_realizations(self):
        rounds=max((self.state_count(c) for c in self.columns),default=1)
        for r in range(rounds):
            yield r,{c:self.states(c)[r%self.state_count(c)] for c in self.columns}

    def score_meta(self):
        return {**super().score_meta(),'aggregation':'raw_mean_then_calibrate','multiple_imputation_m':self.info['m']}


@UNCERTAINTY_MODELS.register('multiple_mice')
class MultipleMICE(UncertaintyModel):
    """Numeric posterior MICE; categorical mode fallback; preserve full draw alignment."""
    stochastic = True
    def __init__(self,m=5,max_iter=5,type_options=None,numerical_backend='sklearn',device='auto'):
        if not isinstance(m,int) or m<2 or max_iter<1: raise ValueError('Multiple MICE requires m>=2 and positive max_iter')
        self.m,self.max_iter,self.type_options=m,max_iter,type_options or {}
        self.numerical_backend,self.device=numerical_backend,device

    def estimate(self,iset,ctx):
        # Input: observed source tables; output: m independent posterior completed tables per source.
        tables=[]; worlds={}; baseline=StatisticalImputer(method='mice',max_iter=self.max_iter,type_options=self.type_options,
            numerical_backend=self.numerical_backend,device=self.device)
        for table in iset.tables:
            original=object_frame(table.df); completed=[]
            for draw in range(self.m):
                local=copy.copy(ctx)
                local.seed=int(np.random.SeedSequence([ctx.seed,zlib.crc32(table.name.encode()),draw]).generate_state(1)[0])
                completed.append(baseline.fill_table(original.copy(),table,iset,local))
            missing=original.isna(); distributions={}
            for r,c in zip(*np.where(missing.to_numpy())):
                name=original.columns[c]; values=[f.iat[r,c] for f in completed]
                if any(not is_missing(v) for v in values):
                    if any(is_missing(v) for v in values): raise ValueError('MICE draw missingness must be consistent')
                    distributions[(int(r),name)]=[(values[0],1.)]
            tables.append(UncertainTable(table.name,original,distributions=distributions)); worlds[table.name]=completed
        return MultipleMICEData(iset,tables,self.registry_name,worlds,self.params())
