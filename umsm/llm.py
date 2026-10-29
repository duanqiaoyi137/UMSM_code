"""LLM access for the LLM-based matchers: OpenAI-compatible chat completions with a disk cache.

Any provider exposing POST {base_url}/chat/completions works (OpenAI, DeepSeek, Qwen/DashScope,
vLLM, Ollama, ...). The API key is read from the environment variable `api_key_env`, so it
never enters configs, caches or result files. Responses are cached per (model, messages,
temperature, max_tokens, extra_body, attempt): successful cached responses are reused on reruns.

Prompt styles of the column-scoring prompt
    magneto  the scoring prompt of the Magneto LLM reranker (candidate first, JSON array of all targets)
    compact  same task and one-shot example, laid out for provider-side prefix caching
             (instructions -> numbered target columns -> candidate) and answered with
             "index:score" pairs of the targets scoring >= 0.1 (at most top_n); unlisted targets get 0.
             All prompts against the same target columns share everything but the last line, so
             with DeepSeek / OpenAI context caching only the candidate line is billed at the
             cache-miss price.

backend="mock" answers offline (string similarity of the column names) and estimates the token
usage, including the cache hits of a provider with prefix caching, so that the cost of a run
can be estimated before using an API.
"""
from __future__ import annotations

import difflib
import hashlib
import http.client
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SYSTEM_PROMPT = "You are an AI trained to perform schema matching by providing column similarity scores."
PROMPT_STYLES = ("magneto", "compact")
CHARS_PER_TOKEN = 3.3  # token estimate of the mock backend

# Prompt of the Magneto LLM reranker (magneto/llm_reranker.py, scoring-oriented one-shot prompt, Fig. 5).
PROMPT_TEMPLATE = (
    "Given a candidate column and a list of target columns, judge the similarity between the candidate "
    "and each target column. "
    "Return a JSON array of objects, each with 'column' (the target column name) and 'score' (a float "
    "between 0 and 1, two decimals, 1 is most similar). "
    "Do NOT provide any other output text or explanation. Only provide the JSON array.\n"
    "Example:\n"
    "Candidate Column: Column: EmployeeID, Sample values: [100, 101, 102]\n"
    "Target Schemas:\n"
    "Column: WorkerID, Sample values: [100, 101, 102]\n"
    "Column: EmpCode, Sample values: [001, 002, 003]\n"
    "Column: StaffName, Sample values: ['Alice', 'Bob', 'Charlie']\n"
    'Response: [\n {"column": "WorkerID", "score": 0.95},\n {"column": "EmpCode", "score": 0.30},\n'
    ' {"column": "StaffName", "score": 0.05}\n]\n\n'
    "Candidate Column: {candidate}\n\nTarget Schemas:\n{targets}\n\nResponse: "
)

COMPACT_TEMPLATE = (
    "Judge the similarity between a candidate column and each numbered target column "
    "(a float between 0 and 1, two decimals, 1 is most similar). "
    "List the target columns with score >= 0.1{limit} as index:score pairs separated by commas; "
    "answer 0:0 if no target column has score >= 0.1. "
    "Do NOT provide any other output text or explanation.\n"
    "Example:\n"
    "Target columns:\n"
    "[1] Column: WorkerID, Sample values: [100,101,102]\n"
    "[2] Column: EmpCode, Sample values: [001,002,003]\n"
    "[3] Column: StaffName, Sample values: [Alice,Bob,Charlie]\n"
    "Candidate column: Column: EmployeeID, Sample values: [100,101,102]\n"
    "Answer: 1:0.95,2:0.30\n\n"
    "Target columns:\n{targets}\n"
    "Candidate column: {candidate}\n"
    "Answer:"
)
# Boundaries of the prompt prefixes shared between requests (used by the mock cost estimate).
_PREFIX_MARKS = ("\nTarget columns:\n", "\nCandidate column: ", "Candidate Column: ")


def read_api_key(name: str) -> str:
    # 输入：name；输出：凭证字符串（仅供请求使用）；继承：无（模块函数）。
    """Environment first, then the project-local ignored .env; never return a key in metadata."""
    if os.environ.get(name):
        return os.environ[name].strip()
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.isfile(path):
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                key, sep, value = line.strip().partition("=")
                if sep and key.strip() == name:
                    return value.strip().strip("\"'")
    raise RuntimeError(f"LLM matcher: set {name} in the environment or the project .env file")


def scoring_messages(candidate: str, targets: list[str], style: str = "magneto",
                     top_n: int | None = None) -> list[dict]:
    # 输入：candidate、targets、style、top_n；输出：LLM 消息字典列表；继承：无（模块函数）。
    if style == "compact":
        limit = f" (at most {top_n}, best first)" if top_n else " (best first)"
        numbered = "\n".join(f"[{k}] {t}" for k, t in enumerate(targets, 1))
        prompt = (COMPACT_TEMPLATE.replace("{limit}", limit).replace("{targets}", numbered)
                  .replace("{candidate}", candidate))
    elif style == "magneto":
        prompt = PROMPT_TEMPLATE.replace("{candidate}", candidate).replace("{targets}", "\n".join(targets))
    else:
        raise ValueError(f"prompt style must be one of {PROMPT_STYLES}")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def parse_scores(text: str) -> list[tuple[str, float]] | None:
    # 输入：text；输出：列名和分数列表或 None；继承：无（模块函数）。
    """[(column, score)] from a JSON array answer (or the 'Name(0.95); ...' format of the paper)."""
    if not text:
        return None
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            items = json.loads(text[start:end + 1])
            out = [(str(e["column"]), float(e.get("score", 0))) for e in items
                   if isinstance(e, dict) and "column" in e]
            if out:
                return out
        except (ValueError, TypeError, KeyError):
            pass
    pairs = re.findall(r'"column"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"score"\s*:\s*"?([0-9]*\.?[0-9]+)', text)
    if not pairs:
        pairs = re.findall(r"([^;\n()]+?)\s*\(\s*([0-9]*\.?[0-9]+)\s*\)", text)
    out = [(name.strip(), float(score)) for name, score in pairs if name.strip()]
    return out or None


def parse_index_scores(text: str, n_targets: int) -> dict[int, float] | None:
    # 输入：text、n_targets；输出：目标下标到分数的字典或 None；继承：无（模块函数）。
    """{target index (0-based): score} from an 'index:score, ...' answer; {} for '0:0'; None if unparsable."""
    pairs = re.findall(r"\[?(\d+)\]?\s*:\s*([0-9]*\.?[0-9]+)", text or "")
    if not pairs:
        return None
    out: dict[int, float] = {}
    for idx, score in pairs:
        k = int(idx)
        if 1 <= k <= n_targets:
            out[k - 1] = max(out.get(k - 1, 0.0), float(score))
    return out


class LLMClient:
    def __init__(self, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", temperature: float = 0.0, max_tokens: int = 1024,
                 timeout: float = 120.0, max_retries: int = 5, backend: str = "openai",
                 extra_body: dict | None = None, price: list[float] | None = None,
                 cache_dir: str | None = None, logger=None):
        # 输入：model、base_url、api_key_env 等配置参数（详见签名）；输出：None；初始化实例及校验参数；继承：LLMClient(object)。
        """price: USD per 1M tokens (cache-hit input, cache-miss input, output), only for the cost report."""
        if backend not in ("openai", "mock"):
            raise ValueError("backend must be 'openai' or 'mock'")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.backend = backend
        self.extra_body = dict(extra_body or {})
        self.price = list(price) if price else None
        self.cache_dir = cache_dir
        self.logger = logger
        self.n_calls = self.n_cache_hits = 0
        self.prompt_tokens = self.cache_hit_tokens = self.cache_miss_tokens = 0
        self.completion_tokens = self.reasoning_tokens = 0
        self.time_s = 0.0
        self._memory: dict[str, str] = {}
        self._prefetched: set[str] = set()
        self._prefix_seen: dict[str, int] = {}
        self._lock = threading.Lock()
        self._warned_reasoning = False
        self._abort: Exception | None = None
        self.response_models: set[str] = set()
        self.system_fingerprints: set[str] = set()

    # ---- cache -------------------------------------------------------------------------------
    def _key(self, messages: list[dict], attempt: int) -> str:
        # 输入：messages、attempt；输出：缓存标识字符串；继承：LLMClient(object)。
        payload = {"backend": self.backend, "base_url": self.base_url, "model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens, "attempt": attempt}
        if self.extra_body:
            payload["extra_body"] = self.extra_body
        return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> str | None:
        # 输入：key；输出：响应缓存文件路径；继承：LLMClient(object)。
        if not self.cache_dir:
            return None
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{self.backend}_{self.model}")
        return os.path.join(self.cache_dir, "llm", slug, key[:2], f"{key}.json")

    def is_cached(self, messages: list[dict], attempt: int = 0) -> bool:
        # 输入：messages、attempt；输出：bool；继承：LLMClient(object)。
        key = self._key(messages, attempt)
        path = self._cache_path(key)
        return key in self._memory or bool(path and os.path.exists(path))

    def chat(self, messages: list[dict], attempt: int = 0, _prefetch: bool = False) -> str:
        # 输入：messages、attempt、_prefetch；输出：模型响应文本（复用缓存、去重并记录用量）；继承：LLMClient(object)。
        key = self._key(messages, attempt)
        with self._lock:
            if key in self._memory:
                if key in self._prefetched:
                    self._prefetched.discard(key)
                else:
                    self.n_cache_hits += 1
                return self._memory[key]
        path = self._cache_path(key)
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)
                text = cached["response"]
            with self._lock:
                self.n_cache_hits += 1
                self.response_models.update(cached.get("response_models", []))
                self.system_fingerprints.update(cached.get("system_fingerprints", []))
        else:
            t0 = time.perf_counter()
            text = self._mock(messages) if self.backend == "mock" else self._request(messages)
            with self._lock:
                self.time_s += time.perf_counter() - t0
                self.n_calls += 1
                provenance = {"response_models": sorted(self.response_models),
                              "system_fingerprints": sorted(self.system_fingerprints)}
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                temporary = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
                try:
                    with open(temporary, "w", encoding="utf-8") as f:
                        json.dump({"model": self.model, "messages": messages, "response": text,
                                   **provenance}, f, ensure_ascii=False, indent=1)
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.remove(temporary)
        with self._lock:
            self._memory[key] = text
            if _prefetch:
                self._prefetched.add(key)
        return text

    def prefetch(self, jobs: list[tuple[object, list[dict]]], workers: int = 8, warmup: int = 2,
                 settle_s: float = 3.0) -> None:
        # 输入：jobs、workers、warmup、settle_s；输出：None；并发预取并写入成功响应缓存；继承：LLMClient(object)。
        """Answer the first attempt of many prompts concurrently and keep the answers in memory.

        jobs: (group, messages); prompts of one group share their prefix. The first `warmup`
        prompts of every group are sent one after the other (groups in parallel): DeepSeek
        persists a prefix once two requests shared it. After `settle_s` seconds for the cache
        to be built, all remaining prompts are sent in parallel on `workers` threads.
        """
        groups: dict[object, list[list[dict]]] = {}
        seen = set()
        for group, messages in jobs:
            key = self._key(messages, 0)
            if key in seen or self.is_cached(messages):
                continue
            seen.add(key)
            groups.setdefault(group, []).append(messages)
        total = sum(len(g) for g in groups.values())
        if not total:
            return
        if self.logger:
            self.logger.info(f"  LLM: {total} new prompt(s) in {len(groups)} prefix group(s), {workers} worker(s)")
        done, t0 = [0], time.perf_counter()

        def ask(messages: list[dict]) -> None:
            # 输入：messages；输出：None；继承：无（prefetch 内部函数）。
            if self._abort is not None:
                return
            try:
                self.chat(messages, 0, _prefetch=True)
            except Exception as e:  # stop the other workers on a fatal error (key, balance, ...)
                self._abort = e
                raise
            with self._lock:
                done[0] += 1
                n = done[0]
            if self.logger and (n % 200 == 0 or n == total):
                rate = n / max(time.perf_counter() - t0, 1e-9)
                self.logger.info(f"  LLM: {n}/{total} prompts ({rate:.1f}/s, ~{(total - n) / rate / 60:.1f} min left, "
                                 f"cost so far ${self.cost_usd():.4f})")

        def run(batch: list[list[dict]]) -> None:
            # 输入：batch；输出：None；后台持续采样；继承：无（prefetch 内部函数）。
            for messages in batch:
                ask(messages)

        self._abort = None
        n_workers = max(1, int(workers))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(run, g[:warmup]) for g in groups.values()]
            for f in futures:
                f.exception()
            rest = [m for g in groups.values() for m in g[warmup:]]
            if rest and self._abort is None:
                if self.backend != "mock":
                    time.sleep(max(0.0, settle_s - (time.perf_counter() - t0)))
                futures += [ex.submit(ask, m) for m in rest]
        if self._abort is not None:
            raise self._abort
        for f in futures:
            f.result()

    # ---- backends ----------------------------------------------------------------------------
    def _request(self, messages: list[dict]) -> str:
        # 输入：messages；输出：一次请求得到的模型文本；继承：LLMClient(object)。
        api_key = read_api_key(self.api_key_env)
        def redact(text):
            # 输入：text；输出：已隐藏凭证的错误文本；继承：无（_request 内部函数）。
            return re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", str(text).replace(api_key, "[REDACTED]"))[:300]

        body = json.dumps({"model": self.model, "messages": messages, "temperature": self.temperature,
                           "max_tokens": self.max_tokens, **self.extra_body}).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {api_key}"})
        delay = 2.0
        for retry in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self._account(data)
                if data["choices"][0].get("finish_reason") == "length":
                    raise RuntimeError("LLM response was truncated; increase llm_max_tokens before using these scores")
                return data["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as e:
                try:
                    detail = redact(e.read().decode("utf-8", "replace"))
                except (OSError, http.client.HTTPException):
                    detail = "error response body was incomplete"
                if e.code == 401:
                    raise RuntimeError(f"LLM request failed: invalid API key in {self.api_key_env} ({detail})") from e
                if e.code == 402:
                    raise RuntimeError(f"LLM request failed: insufficient balance, top up the account ({detail})") from e
                if e.code not in (408, 409, 429, 500, 502, 503, 504) or retry == self.max_retries:
                    raise RuntimeError(f"LLM request failed: HTTP {e.code} {detail}") from e
            except (urllib.error.URLError, OSError, http.client.HTTPException,
                    json.JSONDecodeError, UnicodeDecodeError) as e:
                if retry == self.max_retries:
                    raise RuntimeError(f"LLM transport failed after {retry + 1} attempt(s): "
                                       f"{type(e).__name__}: {redact(e)}; rerun to reuse successful cached requests") from e
            if self.logger:
                self.logger.warning(f"LLM request failed, retry {retry + 1}/{self.max_retries} in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")

    def _account(self, data: dict) -> None:
        # 输入：data；输出：None；累计服务端用量和来源信息；继承：LLMClient(object)。
        usage = data.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        hit = int(usage.get("prompt_cache_hit_tokens")
                  or (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        miss = int(usage.get("prompt_cache_miss_tokens") or max(prompt - hit, 0))
        reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        message = (data.get("choices") or [{}])[0].get("message") or {}
        with self._lock:
            if data.get("model"):
                self.response_models.add(str(data["model"]))
            if data.get("system_fingerprint"):
                self.system_fingerprints.add(str(data["system_fingerprint"]))
            self.prompt_tokens += prompt
            self.cache_hit_tokens += hit
            self.cache_miss_tokens += miss
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            self.reasoning_tokens += reasoning
            warn = (reasoning > 0 or message.get("reasoning_content")) and not self._warned_reasoning
            self._warned_reasoning = self._warned_reasoning or bool(warn)
        if warn and self.logger:
            self.logger.warning("LLM answers contain reasoning tokens (thinking mode), which are billed as output; "
                                "for DeepSeek set llm_extra_body: {thinking: {type: disabled}}")

    def _mock(self, messages: list[dict]) -> str:
        # 输入：messages；输出：离线模拟响应文本；继承：LLMClient(object)。
        content = messages[-1]["content"]
        if "\nCandidate column: " in content:
            head, cand_part = content.rsplit("\nCandidate column: ", 1)
            targets = re.findall(r"^\[\d+\] Column: (.*?), Sample values:", head.rsplit("\nTarget columns:\n", 1)[-1], re.M)
            cand = re.match(r"Column: (.*?), Sample values:", cand_part)
            name = cand.group(1).lower() if cand else ""
            scores = sorted(((round(difflib.SequenceMatcher(None, name, t.lower()).ratio(), 2), k)
                             for k, t in enumerate(targets, 1)), reverse=True)
            m = re.search(r"\(at most (\d+)", content)
            kept = [(s, k) for s, k in scores if s >= 0.1][: int(m.group(1)) if m else None]
            answer = ",".join(f"{k}:{s:.2f}" for s, k in kept) or "0:0"
        else:
            prompt = content.rsplit("Candidate Column: ", 1)[-1]
            cand = re.match(r"Column: (.*?), Sample values:", prompt)
            targets = re.findall(r"^Column: (.*?), Sample values:", prompt.split("Target Schemas:", 1)[-1], re.M)
            name = cand.group(1).lower() if cand else ""
            answer = json.dumps(sorted(({"column": t, "score": round(difflib.SequenceMatcher(None, name, t.lower()).ratio(), 2)}
                                        for t in targets), key=lambda e: -e["score"]))
        self._estimate(messages, answer)
        return answer

    def _estimate(self, messages: list[dict], answer: str) -> None:
        # 输入：messages、answer；输出：None；估计 mock 场景的 token 用量；继承：LLMClient(object)。
        """Token usage of a provider with prefix caching: a prefix is a cache hit once two earlier
        requests shared it (DeepSeek's common-prefix rule)."""
        system = "".join(m["content"] for m in messages[:-1])
        content = messages[-1]["content"]
        cuts = sorted({i for mark in _PREFIX_MARKS for i in [content.rfind(mark)] if i > 0})
        prefixes = [system + content[:i] for i in cuts]
        total = len(system) + len(content)
        with self._lock:
            hit = max((len(p) for p in prefixes if self._prefix_seen.get(p, 0) >= 2), default=0)
            for p in prefixes:
                self._prefix_seen[p] = self._prefix_seen.get(p, 0) + 1
            self.prompt_tokens += round(total / CHARS_PER_TOKEN)
            self.cache_hit_tokens += round(hit / CHARS_PER_TOKEN)
            self.cache_miss_tokens += round((total - hit) / CHARS_PER_TOKEN)
            self.completion_tokens += max(1, round(len(answer) / CHARS_PER_TOKEN))

    def cost_usd(self) -> float:
        # 输入：实例状态；输出：估算 API 费用 float；继承：LLMClient(object)。
        if not self.price:
            return 0.0
        hit, miss, out = self.price
        return (self.cache_hit_tokens * hit + self.cache_miss_tokens * miss + self.completion_tokens * out) / 1e6

    def stats(self) -> dict:
        # 输入：实例状态；输出：该对象的统计信息字典；继承：LLMClient(object)。
        return {"llm_model": self.model, "llm_backend": self.backend, "llm_calls": self.n_calls,
                "llm_cache_hits": self.n_cache_hits, "llm_prompt_tokens": self.prompt_tokens,
                "llm_cache_hit_tokens": self.cache_hit_tokens, "llm_cache_miss_tokens": self.cache_miss_tokens,
                "llm_completion_tokens": self.completion_tokens, "llm_reasoning_tokens": self.reasoning_tokens,
                "llm_cost_usd": round(self.cost_usd(), 6), "llm_time_s": self.time_s,
                "llm_pricing_configured": bool(self.price),
                "llm_response_models": sorted(self.response_models),
                "llm_system_fingerprints": sorted(self.system_fingerprints)}


def get_llm(ctx, **params) -> LLMClient:
    # 输入：ctx、**params；输出：复用的 LLMClient；继承：无（模块函数）。
    key = ("llm", json.dumps(params, sort_keys=True, default=str))
    return ctx.shared(key, lambda: LLMClient(cache_dir=ctx.cache_dir, logger=ctx.logger, **params))


def llm_stats(ctx) -> dict:
    # 输入：ctx；输出：本轮 LLM 用量汇总；继承：无（模块函数）。
    clients = [v for v in ctx.resources.values() if isinstance(v, LLMClient)]
    if not clients:
        return {}
    keys = ("llm_calls", "llm_cache_hits", "llm_prompt_tokens", "llm_cache_hit_tokens", "llm_cache_miss_tokens",
            "llm_completion_tokens", "llm_reasoning_tokens", "llm_cost_usd", "llm_time_s")
    return {**{k: sum(c.stats()[k] for c in clients) for k in keys},
            "llm_response_models": sorted({m for c in clients for m in c.response_models}),
            "llm_system_fingerprints": sorted({m for c in clients for m in c.system_fingerprints})}


def rerank_scores(client: LLMClient, candidate: str, targets: dict[str, str], max_attempts: int = 3,
                  style: str = "magneto", top_n: int | None = None, require_all: bool = False) -> dict[str, float] | None:
    # 输入：client、candidate、targets、max_attempts、style、top_n、require_all；输出：目标列名到 LLM 分数的字典或 None；继承：无（模块函数）。
    """LLM scores {target name: score} of one candidate column against target columns.

    targets maps a target column name to its prompt line. With the magneto style unknown names in
    the answer are ignored; with the compact style every target gets a score (0 if not listed).
    None if no attempt could be parsed (callers fall back to the retriever scores).
    """
    names = list(targets)
    messages = scoring_messages(candidate, list(targets.values()), style, top_n)
    lookup = {name: name for name in names}
    lookup.update({name.strip().lower(): name for name in names})
    for attempt in range(max_attempts):
        answer = client.chat(messages, attempt)
        if style == "compact":
            parsed = parse_index_scores(answer, len(names))
            if parsed is None:
                continue
            return {name: max(0.0, min(1.0, parsed.get(k, 0.0))) for k, name in enumerate(names)}
        parsed = parse_scores(answer)
        if parsed is None:
            continue
        out: dict[str, float] = {}
        for name, score in parsed:
            target = lookup.get(name) or lookup.get(name.strip().lower())
            if target is not None and math.isfinite(score):
                out[target] = max(0.0, min(1.0, score))
        if out and (not require_all or len(out) == len(names)):
            return out
    return None
