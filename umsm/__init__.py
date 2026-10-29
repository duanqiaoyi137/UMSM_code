"""UMSM: Uncertainty-Aware Multi-Table Schema Matching experiment framework.

    Omega = GlobalOptimizer( UncertaintyPropagation( BaseMatcher( UncertaintyModel(T) ) ) )
"""
from . import global_optimizer, matcher, uncertainty_model, uncertainty_propagation  # noqa: F401  (registries)
from .config import load_config
from .experiment import run_experiment
from .registry import MATCHERS, OPTIMIZERS, PROPAGATIONS, UNCERTAINTY_MODELS

__all__ = ["load_config", "run_experiment", "UNCERTAINTY_MODELS", "MATCHERS", "PROPAGATIONS", "OPTIMIZERS"]
