"""LLM-based matcher: an LLM scores every source column against all columns of the target table.

This is the LLM-only setting of the Magneto paper ("All Schema", gpt_only=True in the library):
one prompt per source column (header + priority sample of its values) listing every target
column, answered with a score in [0, 1] per target column (scoring-oriented one-shot prompt of
umsm/llm.py). Columns the LLM does not score get 0. No embedding model is loaded.

Configure the provider through llm_model / llm_base_url / llm_api_key_env (any OpenAI-compatible
endpoint); llm_backend=mock runs offline. Prompts and answers are cached under cache/llm.
llm_prompt=compact (with llm_top_n, llm_value_chars) is the low-cost layout of the same prompt;
configs/magneto_deepseek.yaml shows the provider settings.
"""
from __future__ import annotations

from ..registry import MATCHERS
from .magneto import LLM_RUNTIME_PARAMS, MagnetoMatcher


@MATCHERS.register("llm")
class LLMMatcher(MagnetoMatcher):
    """LLM-only matcher: one scoring prompt per source column against all target columns."""

    _retrieval = False

    def __init__(self, llm_model: str = "gpt-4o-mini", llm_base_url: str = "https://api.openai.com/v1",
                 llm_api_key_env: str = "OPENAI_API_KEY", llm_backend: str = "openai",
                 llm_temperature: float = 0.0, llm_max_tokens: int = 1024, llm_max_attempts: int = 3,
                 llm_sample_size: int = 10, llm_max_variants: int | None = None, llm_prompt: str = "magneto",
                 llm_top_n: int | None = None, llm_value_chars: int | None = None,
                 llm_extra_body: dict | None = None, llm_workers: int = 8, llm_price: list[float] | None = None,
                 bidirectional: bool = False):
        # 输入：llm_model、llm_base_url、llm_api_key_env 等配置参数（详见签名）；输出：None；初始化实例及校验参数；继承：LLMMatcher(MagnetoMatcher)，重写父类方法。
        super().__init__(reranker="llm", rerank_k=None, bidirectional=bidirectional, llm_model=llm_model,
                         llm_base_url=llm_base_url, llm_api_key_env=llm_api_key_env, llm_backend=llm_backend,
                         llm_temperature=llm_temperature, llm_max_tokens=llm_max_tokens,
                         llm_max_attempts=llm_max_attempts, llm_sample_size=llm_sample_size,
                         llm_max_variants=llm_max_variants, llm_prompt=llm_prompt, llm_top_n=llm_top_n,
                         llm_value_chars=llm_value_chars, llm_extra_body=llm_extra_body,
                         llm_workers=llm_workers, llm_price=llm_price)

    def params(self) -> dict:
        # 输入：实例状态；输出：组件构造参数字典（不包含内部缓存）；继承：LLMMatcher(MagnetoMatcher)，重写父类方法。
        return {k: v for k, v in vars(self).items()
                if (k.startswith("llm_") or k == "bidirectional") and k not in LLM_RUNTIME_PARAMS}
