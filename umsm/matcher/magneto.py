"""Magneto (Liu et al., "Magneto: Combining Small and Large Language Models for Schema Matching",
PVLDB 18(8), 2025), zero-shot variants, following https://github.com/VIDA-NYU/magneto-matcher.

Candidate retrieval (Sec. 2.2, 3.1): every column is serialized from its header, detected type
and a sample of `sampling_size` cleaned values (S_repeat + priority sampling is the best
zero-shot setting of the paper; the library default is S_verbose + mixed sampling), embedded
with MPNet, and compared by cosine similarity; columns with equal cleaned names get 1.0. For
every source column the `topk` target columns with similarity >= `embedding_threshold` are kept,
all other pairs get score 0.

reranker
    none   retriever scores                                              (Magneto-zs)
    bp     bipartite reranker: the one-to-one assignment keeps its scores, the other candidates
           are scaled just below the lowest assigned score                (Magneto-zs-bp)
    llm    the first `rerank_k` candidates of each source column are scored by an LLM with the
           scoring-oriented prompt (Sec. 4); the remaining candidates are scaled so that their
           maximum equals the lowest LLM score                            (Magneto-zs-llm)
`model` may point to a fine-tuned retriever (Magneto-ft), e.g. vida-nyu/magneto-schema-retriever-gdc.

Magneto matches a source table against a target table; for the table pair (T_a, T_b) of an
integration set T_a is the source (bidirectional=True averages both directions).

Missing values: score() receives one aligned column/table realization at a time. It pairs
the r-th source with the r-th target context, without an R-by-R product. Every pair is scored
under each realization of its two tables, including complete columns when their context varies.
Identical texts and prompts use the embedding / LLM caches. Exact-state scoring
is a small unit-test reference only. llm_max_variants is an optional approximation; formal
MCAR40 configs leave it unset to preserve all sampled states.

LLM cost: prompts within each realization are prefetched using llm_workers threads. The
authors' JSON prompt preserves retrieval order; only the optional compact prompt sorts
targets into table order and puts the candidate last for shared prefixes (see umsm/llm.py).
"""
from __future__ import annotations

from itertools import combinations
import copy

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ..embedding import DEFAULT_MODEL, get_encoder
from ..llm import PROMPT_STYLES, get_llm, rerank_scores, scoring_messages
from ..registry import MATCHERS
from ..types import ColumnRef, ColumnState, ScoreSamples, UncertainData
from .base import BaseMatcher, cross_table_pairs
from .magneto_utils import ENCODING_MODES, SAMPLING_MODES, clean_name, encode_column, llm_column_text, state_series

RERANKERS = ("none", "bp", "llm")
# Execution settings of the LLM client: they do not change the scores (excluded from the score cache key).
LLM_RUNTIME_PARAMS = ("llm_workers", "llm_price")


def _variants(texts: list[str]) -> tuple[list[str], np.ndarray]:
    # 输入：texts；输出：(不同文本列表, 状态到文本下标数组)；继承：无（模块函数）。
    """Distinct texts (first-appearance order) and, per state, the index of its text."""
    uniq = list(dict.fromkeys(texts))
    pos = {t: k for k, t in enumerate(uniq)}
    return uniq, np.array([pos[t] for t in texts], dtype=np.int64)


def _cap_variants(texts: list[str], probs: np.ndarray, k: int) -> list[str]:
    # 输入：texts、probs、k；输出：按概率质量限制后的状态文本列表；继承：无（模块函数）。
    """Keep the k most probable distinct texts; states of the other texts use the most probable one."""
    mass: dict[str, float] = {}
    for t, p in zip(texts, probs):
        mass[t] = mass.get(t, 0.0) + float(p)
    keep = sorted(mass, key=lambda t: -mass[t])[:k]
    return [t if t in keep else keep[0] for t in texts]


@MATCHERS.register("magneto")
class MagnetoMatcher(BaseMatcher):
    """Magneto zero-shot: MPNet retrieval over serialized column samples + reranker none / bp / llm."""

    _retrieval = True
    sampling_scope = "table_pair"
    context_sampling_version = 2

    def __init__(self, model: str = DEFAULT_MODEL, encoding_mode: str = "header_values_repeat",
                 sampling_mode: str = "priority_sampling", sampling_size: int = 10, topk: int = 20,
                 embedding_threshold: float = 0.1, include_equal_matches: bool = True, reranker: str = "none",
                 rerank_k: int | None = None, bidirectional: bool = False, llm_model: str = "gpt-4o-mini",
                 llm_base_url: str = "https://api.openai.com/v1", llm_api_key_env: str = "OPENAI_API_KEY",
                 llm_backend: str = "openai", llm_temperature: float = 0.0, llm_max_tokens: int = 1024,
                 llm_max_attempts: int = 3, llm_sample_size: int = 10, llm_max_variants: int | None = None,
                 llm_prompt: str = "magneto", llm_top_n: int | None = None, llm_value_chars: int | None = None,
                 llm_extra_body: dict | None = None, llm_workers: int = 8, llm_price: list[float] | None = None,
                 llm_require_success: bool = False,
                 device: str = "auto", batch_size: int = 64, max_length: int | None = 384):
        # 输入：model、encoding_mode、sampling_mode 等配置参数（详见签名）；输出：None；初始化实例及校验参数；继承：MagnetoMatcher(BaseMatcher)。
        if encoding_mode not in ENCODING_MODES:
            raise ValueError(f"encoding_mode must be one of {ENCODING_MODES}")
        if sampling_mode not in SAMPLING_MODES:
            raise ValueError(f"sampling_mode must be one of {SAMPLING_MODES}")
        if reranker not in RERANKERS:
            raise ValueError(f"reranker must be one of {RERANKERS}")
        if llm_prompt not in PROMPT_STYLES:
            raise ValueError(f"llm_prompt must be one of {PROMPT_STYLES}")
        self.model = model
        self.encoding_mode = encoding_mode
        self.sampling_mode = sampling_mode
        self.sampling_size = sampling_size
        self.topk = topk
        self.embedding_threshold = embedding_threshold
        self.include_equal_matches = include_equal_matches
        self.reranker = reranker
        self.rerank_k = rerank_k
        self.bidirectional = bidirectional
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key_env = llm_api_key_env
        self.llm_backend = llm_backend
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.llm_max_attempts = llm_max_attempts
        self.llm_sample_size = llm_sample_size
        self.llm_max_variants = llm_max_variants
        self.llm_prompt = llm_prompt
        self.llm_top_n = llm_top_n
        self.llm_value_chars = llm_value_chars
        self.llm_extra_body = llm_extra_body
        self.llm_workers = llm_workers
        self.llm_price = llm_price
        self.llm_require_success = llm_require_success
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    def params(self) -> dict:
        # 输入：实例状态；输出：组件构造参数字典（不包含内部缓存）；继承：MagnetoMatcher(BaseMatcher)，重写父类方法。
        p = {k: v for k, v in super().params().items() if k not in LLM_RUNTIME_PARAMS}
        if self.reranker != "llm":
            p = {k: v for k, v in p.items() if not k.startswith("llm_") and k != "rerank_k"}
        return p


    # ------------------------------------------------------------------------------------------
    def score(self, data: UncertainData, ctx) -> ScoreSamples:
        # 输入：data、ctx；输出：ScoreSamples（对齐采样的原始分数）；继承：MagnetoMatcher(BaseMatcher)，重写父类方法。
        if data.realization.mode == "exact":
            return self._score_once(data, ctx)
        states = {c: data.states(c) for c in data.columns}
        n = max((len(s) for s in states.values()), default=1)
        if n == 1:
            return self._score_once(data, ctx)
        expected = self.expected_sample_counts(data)
        # Table-context rerankers receive one aligned realization at a time, so
        # neither retrieval nor LLM prompts construct a Cartesian product of states.
        master_text = None
        if self.reranker == "llm":
            master_text = {c: [llm_column_text(state_series(s, c.name), self.llm_sample_size,
                                             self.llm_value_chars) for s in ss] for c, ss in states.items()}
            if self.llm_max_variants:
                master_text = {c: _cap_variants(ts, np.array([s.prob for s in states[c]]), self.llm_max_variants)
                               for c, ts in master_text.items()}
        frames, prompts, failures = [], 0, 0
        for r in range(n):
            snapshot = copy.copy(data)
            snapshot._states = {c: [ColumnState(ss[r % len(ss)].values, 1.0)] for c, ss in states.items()}
            snapshot.candidate_pairs = {pair for pair, count in expected.items() if r < count}
            if not snapshot.candidate_pairs:
                continue
            text = {c: [ts[r % len(ts)]] for c, ts in master_text.items()} if master_text is not None else None
            part = self._score_once(snapshot, ctx, text).df.copy()
            lengths = {c.id: len(ss) for c, ss in states.items()}
            part["state_i"] = part.col_i.map(lambda c: r % lengths[c])
            part["state_j"] = part.col_j.map(lambda c: r % lengths[c])
            part["sample_id"] = r
            part["prob"] = [1.0 / expected[frozenset((a, b))] for a, b in zip(part.col_i, part.col_j)]
            frames.append(part)
            prompts += self._n_llm_prompts
            failures += self._n_llm_failures
        self._n_llm_prompts, self._n_llm_failures = prompts, failures
        if master_text is not None:
            self._llm_text = master_text
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ScoreSamples.REQUIRED)
        return ScoreSamples(frame, data.columns, meta={"matcher": self.registry_name, **data.score_meta()})

    def _score_once(self, data: UncertainData, ctx, llm_text_override=None) -> ScoreSamples:
        # 输入：data、ctx、llm_text_override；输出：单个数据上下文的 ScoreSamples；继承：MagnetoMatcher(BaseMatcher)。
        self._active_data = data
        cols = data.columns
        series = {c: [state_series(s, c.name) for s in data.states(c)] for c in cols}
        self._probs = {c: np.array([s.prob for s in data.states(c)]) for c in cols}
        self._n_llm_prompts = self._n_llm_failures = 0

        with ctx.profiler.stage("matcher.encode"):
            self._emb = {}
            if self._retrieval:
                enc = get_encoder(ctx, self.model, self.device, self.batch_size, self.max_length)
                tokens = enc.special_tokens()
                texts = {}
                for c in cols:
                    rng = np.random.default_rng([ctx.seed, len(texts)])
                    texts[c] = [encode_column(s, self.encoding_mode, self.sampling_mode, self.sampling_size,
                                              tokens, rng) for s in series[c]]
                vec = enc.encode_map([t for ts in texts.values() for t in ts], f"{data.source.dataset}/magneto")
                self._emb = {c: np.stack([vec[t] for t in ts]).astype(np.float64) for c, ts in texts.items()}
            self._llm_text = {}
            if self.reranker == "llm":
                self._llm = get_llm(ctx, model=self.llm_model, base_url=self.llm_base_url,
                                    api_key_env=self.llm_api_key_env, temperature=self.llm_temperature,
                                    max_tokens=self.llm_max_tokens, backend=self.llm_backend,
                                    extra_body=self.llm_extra_body, price=self.llm_price)
                self._llm_text = {c: [llm_column_text(s, self.llm_sample_size, self.llm_value_chars)
                                      for s in series[c]] for c in cols}
                if llm_text_override is not None:
                    self._llm_text = llm_text_override
                if self.llm_max_variants:
                    self._llm_text = {c: _cap_variants(ts, self._probs[c], self.llm_max_variants)
                                      for c, ts in self._llm_text.items()}

        with ctx.profiler.stage("matcher.pairs"):
            pairs = list(combinations(data.tables, 2))
            dirs = [(ta.columns, tb.columns) for ta, tb in pairs]
            if self.bidirectional:
                dirs += [(tb.columns, ta.columns) for ta, tb in pairs]
            retrieved = [self._retrieve(src, tgt) for src, tgt in dirs]
            if self.reranker == "llm":
                self._prefetch(dirs, retrieved)
            outs = [self._finish(src, tgt, r) for (src, tgt), r in zip(dirs, retrieved)]
            parts = []
            for k in range(len(pairs)):
                fwd = outs[k]
                if self.bidirectional:
                    bwd = outs[k + len(pairs)]
                    fwd = {key: {f: (v + bwd[(key[1], key[0])][f].T) / 2 for f, v in m.items()} for key, m in fwd.items()}
                parts.extend((ci, cj, m) for (ci, cj), m in fwd.items())
            df = self._frame(parts)
        if self.reranker == "llm":
            ctx.logger.info(f"  {self.registry_name}: {self._n_llm_prompts} LLM prompt(s), {self._n_llm_failures} unparsable "
                            f"(retriever scores kept), client {self._llm.stats()}")
        return ScoreSamples(df, cols, meta={"matcher": self.registry_name, **data.score_meta()})

    def _frame(self, parts) -> pd.DataFrame:
        # 输入：parts；输出：状态对分数的 DataFrame；继承：MagnetoMatcher(BaseMatcher)。
        rows = {k: [] for k in ("col_i", "col_j", "table_i", "table_j", "state_i", "state_j", "prob", "score")}
        extra = {k: [] for k in parts[0][2] if k != "score"} if parts else {}
        for ci, cj, m in parts:
            n_a, n_b = m["score"].shape
            a_idx, b_idx = np.meshgrid(np.arange(n_a), np.arange(n_b), indexing="ij")
            n = n_a * n_b
            rows["col_i"] += [ci.id] * n
            rows["col_j"] += [cj.id] * n
            rows["table_i"] += [ci.table] * n
            rows["table_j"] += [cj.table] * n
            rows["state_i"].append(a_idx.ravel())
            rows["state_j"].append(b_idx.ravel())
            rows["prob"].append(np.outer(self._probs[ci], self._probs[cj]).ravel())
            rows["score"].append(m["score"].ravel())
            for k in extra:
                extra[k].append(m[k].ravel())
        if not parts:
            return pd.DataFrame(columns=list(rows))
        out = {k: (np.concatenate(v) if k in ("state_i", "state_j", "prob", "score") else v) for k, v in rows.items()}
        out.update({k: np.concatenate(v) for k, v in extra.items()})
        return pd.DataFrame(out)

    # ------------------------------------------------------------------------------------------
    def _retrieve(self, src: list[ColumnRef], tgt: list[ColumnRef]) -> dict:
        # 输入：src、tgt；输出：逐源列的原始矩阵、检索矩阵及排序候选；继承：MagnetoMatcher(BaseMatcher)。
        """{ci: (raw scores, kept scores, candidates per state)} of the direction src -> tgt."""
        out = {}
        for ci in src:
            n_a = len(self._probs[ci])
            raw = {}
            for cj in tgt:
                if not self._active_data.allows(ci, cj):
                    continue
                if self._retrieval:
                    s = np.clip(self._emb[ci] @ self._emb[cj].T, -1.0, 1.0)
                    if self.include_equal_matches and clean_name(ci.name) == clean_name(cj.name):
                        s = np.ones_like(s)
                else:
                    s = np.zeros((n_a, len(self._probs[cj])))
                raw[cj] = s
            kept = {cj: np.zeros_like(s) for cj, s in raw.items()}
            ranked: list[list[ColumnRef]] = []
            for a in range(n_a):
                if self._retrieval:
                    best = {cj: float(s[a].max()) for cj, s in raw.items()}
                    cands = [cj for cj in sorted(raw, key=lambda c: -best[c]) if best[cj] >= self.embedding_threshold]
                    cands = cands[: self.topk]
                    for cj in cands:
                        kept[cj][a] = np.where(raw[cj][a] >= self.embedding_threshold, raw[cj][a], 0.0)
                else:
                    cands = list(raw)
                ranked.append(cands)
            out[ci] = (raw, kept, ranked)
        return out

    def _finish(self, src: list[ColumnRef], tgt: list[ColumnRef], retrieved: dict) -> dict:
        # 输入：src、tgt、retrieved；输出：列对到重排后分数矩阵的字典；继承：MagnetoMatcher(BaseMatcher)。
        """{(ci, cj): {"score": (n_states_i, n_states_j), ...}} with ci in the source table."""
        out = {}
        for ci in src:
            raw, kept, ranked = retrieved[ci]
            llm_flag = {cj: np.zeros_like(s) for cj, s in raw.items()}
            if self.reranker == "llm":
                for a in range(len(ranked)):
                    self._llm_rerank(ci, a, ranked[a], tgt, kept, llm_flag)
            for cj in raw:
                m = {"score": kept[cj]}
                if self._retrieval:
                    m["retriever_score"] = raw[cj]
                if self.reranker == "llm":
                    m["llm_scored"] = llm_flag[cj]
                out[(ci, cj)] = m
        if self.reranker == "bp":
            self._bipartite(src, tgt, out)
        return out

    def _bipartite(self, src: list[ColumnRef], tgt: list[ColumnRef], out: dict) -> None:
        # 输入：src、tgt、out；输出：None；原地调整非指派边的分数；继承：MagnetoMatcher(BaseMatcher)。
        exp = np.array([[float(self._probs[ci] @ out[(ci, cj)]["score"] @ self._probs[cj])
                         if (ci, cj) in out else 0.0 for cj in tgt] for ci in src])
        has = np.array([[(ci, cj) in out and bool((out[(ci, cj)]["score"] > 0).any()) for cj in tgt] for ci in src])
        rows, cols = np.flatnonzero(has.any(1)), np.flatnonzero(has.any(0))
        if len(rows) == 0:
            return
        r, c = linear_sum_assignment(exp[np.ix_(rows, cols)], maximize=True)
        assigned = {(src[rows[i]], tgt[cols[j]]) for i, j in zip(r, c) if has[rows[i], cols[j]]}
        if not assigned:
            return
        others = [m["score"] for k, m in out.items() if k not in assigned and (m["score"] > 0).any()]
        if not others:
            return
        min_assigned = min(exp[src.index(ci), tgt.index(cj)] for ci, cj in assigned)
        top = max(float(s.max()) for s in others)
        factor = (min_assigned - 0.01) / top if top > 0 else 1.0
        for k, m in out.items():
            if k not in assigned:
                m["score"] = np.clip(m["score"] * factor, 0.0, None)

    def _rounds(self, cands: list[ColumnRef], tgt: list[ColumnRef]):
        # 输入：cands、tgt；输出：(发送候选, 文本变体, 提示轮次)；继承：MagnetoMatcher(BaseMatcher)。
        """Candidates sent to the LLM (in table order), their text variants, and one {name: text} per prompt."""
        sent = cands if self.rerank_k is None else cands[: self.rerank_k]
        if self.llm_prompt == "compact":
            sent = sorted(sent, key=tgt.index)  # shared prefixes for the optional compact prompt
        variants = {cj: _variants(self._llm_text[cj]) for cj in sent}
        n_rounds = max((len(v[0]) for v in variants.values()), default=0)
        rounds = [{cj.name: variants[cj][0][r % len(variants[cj][0])] for cj in sent} for r in range(n_rounds)]
        return sent, variants, rounds

    def _prefetch(self, dirs, retrieved) -> None:
        # 输入：dirs、retrieved；输出：None；预取并缓存 LLM 响应；继承：MagnetoMatcher(BaseMatcher)。
        jobs = []
        for (src, tgt), ret in zip(dirs, retrieved):
            for ci in src:
                for a, cands in enumerate(ret[ci][2]):
                    for lines in self._rounds(cands, tgt)[2]:
                        targets = list(lines.values())
                        jobs.append((tuple(targets), scoring_messages(self._llm_text[ci][a], targets,
                                                                      self.llm_prompt, self.llm_top_n)))
        self._llm.prefetch(jobs, self.llm_workers)

    def _llm_rerank(self, ci: ColumnRef, a: int, cands: list[ColumnRef], tgt: list[ColumnRef],
                    kept: dict, flag: dict) -> None:
        # 输入：ci、a、cands、tgt、kept、flag；输出：None；原地更新 LLM 打分与成功标记；继承：MagnetoMatcher(BaseMatcher)。
        sent, variants, rounds = self._rounds(cands, tgt)
        if not sent:
            return
        llm: dict[tuple[ColumnRef, int], float] = {}
        for r, lines in enumerate(rounds):
            self._n_llm_prompts += 1
            scores = rerank_scores(self._llm, self._llm_text[ci][a], lines, self.llm_max_attempts,
                                   self.llm_prompt, self.llm_top_n, require_all=self.llm_require_success)
            if scores is None:
                self._n_llm_failures += 1
                if self.llm_require_success:
                    raise RuntimeError("Magneto LLM reranking did not return all candidate scores; "
                                       "stopping instead of silently using retrieval-only scores")
                continue
            for cj in sent:
                if cj.name in scores:
                    llm[(cj, r % len(variants[cj][0]))] = scores[cj.name]
        if not llm:
            return
        rest = []
        for cj in kept:
            for b in range(kept[cj].shape[1]):
                v = variants[cj][1][b] if cj in variants else -1
                if (cj, v) in llm:
                    kept[cj][a, b] = llm[(cj, v)]
                    flag[cj][a, b] = 1.0
                elif kept[cj][a, b] > 0:
                    rest.append((cj, b))
        if rest:
            top = max(kept[cj][a, b] for cj, b in rest)
            factor = min(1.0, min(llm.values()) / top) if top > 0 else 1.0
            for cj, b in rest:
                kept[cj][a, b] *= factor
