"""Deterministic baselines: every missing cell gets no evidence or exactly one value (p = 1)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..registry import UNCERTAINTY_MODELS
from ..types import ZERO_EMBEDDING
from .base import DeterministicModel


@UNCERTAINTY_MODELS.register("observed_only")
class ObservedOnly(DeterministicModel):
    """No handling: missing cells provide no evidence; the matcher sees observed values only."""

    def fill_table(self, frame, table, iset, ctx):
        # 输入：frame、table、iset、ctx；输出：确定性补全后的 DataFrame；继承：ObservedOnly(DeterministicModel)，重写父类方法。
        return frame


@UNCERTAINTY_MODELS.register("zero_embedding")
class ZeroEmbedding(DeterministicModel):
    """Legacy MTSM: a missing cell is represented by the all-zero embedding."""

    def fill_table(self, frame, table, iset, ctx):
        # 输入：frame、table、iset、ctx；输出：确定性补全后的 DataFrame；继承：ZeroEmbedding(DeterministicModel)，重写父类方法。
        return frame.where(frame.notna(), ZERO_EMBEDDING)


@UNCERTAINTY_MODELS.register("mode")
class ModeImputation(DeterministicModel):
    """Deterministic imputation with the most frequent observed value of the column."""

    def fill_table(self, frame, table, iset, ctx):
        # 输入：frame、table、iset、ctx；输出：确定性补全后的 DataFrame；继承：ModeImputation(DeterministicModel)，重写父类方法。
        out = frame.copy()
        for col in out.columns:
            observed = out[col].dropna()
            if len(observed):
                out[col] = out[col].where(out[col].notna(), observed.value_counts(sort=True).index[0])
        return out


@UNCERTAINTY_MODELS.register("argmax")
class ArgmaxImputation(DeterministicModel):
    """Single imputation with the most probable candidate of a distributional model (same Q, p = 1).

    Ablation of "keep several candidates": the distribution Q_x of `model` is computed as usual
    and only its top candidate is written into the cell.
    """

    def __init__(self, model: str = "column_marginal", model_params: dict | None = None):
        # 输入：model、model_params；输出：None；初始化实例及校验参数；继承：ArgmaxImputation(DeterministicModel)。
        self.model = model
        self.model_params = dict(model_params or {})
        self._base = UNCERTAINTY_MODELS.build(model, self.model_params)
        if not hasattr(self._base, "cell_distributions"):
            raise ValueError(f"argmax: {model} is not a distributional uncertainty model")

    def fill_table(self, frame, table, iset, ctx):
        # 输入：frame、table、iset、ctx；输出：确定性补全后的 DataFrame；继承：ArgmaxImputation(DeterministicModel)，重写父类方法。
        out = frame.copy()
        for (r, c), q in self._base.cell_distributions(frame, table, iset, ctx).items():
            q, _ = self._base.prepare_distribution(q)
            if q:
                out.iat[r, out.columns.get_loc(c)] = max(q, key=lambda vp: vp[1])[0]
        return out


@UNCERTAINTY_MODELS.register("pre_imputed")
class PreImputed(DeterministicModel):
    """Deterministic imputation computed by an external tool (HyperImpute, DLM, HoloClean, ...).

    Imputed tables are read from `template`, formatted with
        root    the `root` parameter
        method  the `method` parameter
        set     the integration set name, e.g. "Restaurant" or "Real/25ksep11"
        table   the table file name, e.g. "openrice.csv"
    Imputed files must keep the column names and row order of the original tables.
    """

    def __init__(self, method: str, root: str = "imputed", template: str = "{root}/{method}/{set}/{table}",
                 only_missing: bool = True):
        # 输入：method、root、template、only_missing；输出：None；初始化实例及校验参数；继承：PreImputed(DeterministicModel)。
        self.method = method
        self.root = root
        self.template = template
        self.only_missing = only_missing

    def fill_table(self, frame, table, iset, ctx):
        # 输入：frame、table、iset、ctx；输出：确定性补全后的 DataFrame；继承：PreImputed(DeterministicModel)，重写父类方法。
        path = ctx.resolve(self.template.format(root=self.root, method=self.method, set=iset.name, table=table.name))
        if not os.path.exists(path):
            raise FileNotFoundError(f"pre_imputed: {path} not found")
        imputed = pd.read_csv(path, encoding="utf-8-sig", engine="python", skip_blank_lines=False)
        if list(imputed.columns) != list(table.df.columns) or len(imputed) != len(table.df):
            raise ValueError(f"pre_imputed: {path} does not have the shape/columns of {table.name}")
        imputed = imputed.astype(object).where(imputed.notna(), np.nan)
        if not self.only_missing:
            return imputed
        return frame.where(frame.notna(), imputed)
