"""Send one short column-scoring prompt with the LLM settings of a config (API key, balance, thinking mode, answer format).

Examples
    python check_llm.py -c configs/magneto_deepseek.yaml
    python check_llm.py -c configs/magneto_deepseek.yaml -s matcher.params.llm_backend=mock
"""
import argparse
import json
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from umsm.config import load_config  # noqa: E402
from umsm.llm import LLMClient, rerank_scores  # noqa: E402


def main():
    # 输入：命令行参数；输出：None；解析命令并执行对应工作流；继承：无（模块函数）。
    p = argparse.ArgumentParser(description="one test request with the LLM settings of a config")
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    params = load_config(args.config, args.set)["matcher"]["params"]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = LLMClient(model=params.get("llm_model", "gpt-4o-mini"),
                       base_url=params.get("llm_base_url", "https://api.openai.com/v1"),
                       api_key_env=params.get("llm_api_key_env", "OPENAI_API_KEY"),
                       temperature=params.get("llm_temperature", 0.0), max_tokens=params.get("llm_max_tokens", 1024),
                       max_retries=0,
                       backend=params.get("llm_backend", "openai"), extra_body=params.get("llm_extra_body"),
                       price=params.get("llm_price"), cache_dir=None, logger=logging.getLogger("check_llm"))
    targets = {"Supplier": "Column: Supplier, Sample values: [ACME LTD,BT PLC,ORANGE]",
               "Amount": "Column: Amount, Sample values: [120.5,3400,87.25]",
               "Date": "Column: Date, Sample values: [2011-04-01,2011-04-08]"}
    scores = rerank_scores(client, "Column: Vendor Name, Sample values: [ACME LTD,VODAFONE]", targets, 1,
                           params.get("llm_prompt", "magneto"), params.get("llm_top_n"),
                           require_all=params.get("llm_require_success", False))
    print("scores:", scores)
    print("usage:", json.dumps(client.stats(), indent=1))
    ok = bool(scores) and max(scores, key=scores.get) == "Supplier" and client.reasoning_tokens == 0
    print("OK" if ok else "CHECK the answer / reasoning tokens above")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
