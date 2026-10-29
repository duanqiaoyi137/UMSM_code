"""Global Optimizer V2.1: calibrate each draw, unbiased moments, risk and logit potential."""
import numpy as np
from scipy.special import logit

from ..calibration import calibrate_scores, load_artifact
from ..registry import PROPAGATIONS
from ..types import EdgeTable, ScoreSamples
from .base import UncertaintyPropagation, moments


@PROPAGATIONS.register("calibrated_logit")
class CalibratedLogit(UncertaintyPropagation):
    """mu/std of calibrated probabilities; a=logit(clip(mu-lambda*sigma))-logit(prior)."""

    def __init__(self, calibration: str, risk_lambda=.5, epsilon=1e-6, prior=.5):
        # 输入：calibration、risk_lambda、epsilon、prior；输出：None；初始化实例及校验参数；继承：CalibratedLogit(UncertaintyPropagation)。
        if not np.isfinite(risk_lambda) or risk_lambda < 0:
            raise ValueError("risk_lambda must be finite and nonnegative")
        if not 0 < epsilon < .5 or not 0 < prior < 1:
            raise ValueError("require 0<epsilon<.5 and 0<prior<1")
        self.calibration, self.risk_lambda = calibration, risk_lambda
        self.epsilon, self.prior = epsilon, prior

    def propagate(self, samples, ctx):
        # 输入：samples、ctx；输出：EdgeTable（统计量与最终边证据）；继承：CalibratedLogit(UncertaintyPropagation)，重写父类方法。
        artifact = load_artifact(self.calibration, ctx)
        df = samples.df.copy()
        df["score"] = calibrate_scores(df.score, artifact["platt"])
        samples.df["calibrated_probability"] = df["score"].to_numpy()
        calibrated = ScoreSamples(df, samples.columns, samples.meta)
        out = moments(calibrated)
        return self.from_moments(out,samples.columns,samples.meta,artifact['platt'])

    def from_moments(self, out, columns, meta, calibration):
        # 输入：在线/数学参照统计、列、采样元信息与固定 g；输出：风险调整概率及 logit 边；继承：CalibratedLogit(UncertaintyPropagation)。
        out = out.copy()
        out["p_tilde"] = (out.mu - self.risk_lambda * out.sigma).clip(self.epsilon, 1 - self.epsilon)
        out["a"] = logit(out.p_tilde.to_numpy(float)) - logit(self.prior)
        out["w"] = out.p_tilde
        return EdgeTable(out, columns, {**meta, "propagation": self.registry_name,
                         "potential": "calibrated_logit", "calibration": calibration,
                         "risk_lambda": self.risk_lambda, "prior": self.prior,"epsilon":self.epsilon})
