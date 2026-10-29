"""Checks of the ALITE / Magneto / LLM matchers, the ALITE clustering and theta selection.

No language model or API is needed: a hashing encoder replaces MPNet / BERT and the LLM runs
with backend=mock.

    python tests/test_matchers.py
"""
import logging
import io
import json
import os
import sys
import tempfile
import zlib
import urllib.error
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from umsm import MATCHERS, OPTIMIZERS, UNCERTAINTY_MODELS  # noqa: E402
from umsm.context import RunContext  # noqa: E402
from umsm.evaluation import evaluate  # noqa: E402
from umsm.llm import LLMClient, parse_index_scores, parse_scores, read_api_key, rerank_scores, scoring_messages  # noqa: E402
from umsm.matcher import alite as alite_mod  # noqa: E402
from umsm.matcher import magneto as magneto_mod  # noqa: E402
from umsm.matcher.alite import alite_tokens, select_by_tfidf  # noqa: E402
from umsm.matcher.magneto_utils import (clean_element, detect_column_type, encode_column, get_samples,  # noqa: E402
                                        llm_column_text, serialize)
from umsm.tuning import best_theta, default_grid  # noqa: E402
from umsm.types import ColumnRef, EdgeTable, IntegrationSet, RealizationSpec, Table  # noqa: E402


class FakeEncoder:
    """Bag of hashed character trigrams, L2-normalised."""

    dim = 64

    def special_tokens(self):
        return {"cls": "<s>", "sep": "</s>", "eos": "</s>"}

    def encode_map(self, texts, namespace="default"):
        out = {}
        for t in dict.fromkeys(texts):
            v = np.zeros(self.dim, np.float32)
            s = f"  {t.lower()}  "
            for k in range(len(s) - 2):
                v[zlib.crc32(s[k:k + 3].encode()) % self.dim] += 1
            out[t] = v / max(np.linalg.norm(v), 1e-9)
        return out


def _fake_encoder(*args, **kwargs):
    return FakeEncoder()


def _ctx():
    return RunContext(cache_dir=tempfile.mkdtemp(), logger=logging.getLogger("umsm.test"))


def _people_set():
    t1 = Table("a.csv", pd.DataFrame({"name": ["Ann", "Bob", "Cid", "Dan"], "city": ["Rome", None, "Oslo", "Rome"],
                                      "age": [31, 45, None, 22]}))
    t2 = Table("b.csv", pd.DataFrame({"Name": ["Eve", "Fay", "Gus", "Hal"], "town": ["Oslo", "Rome", "Rome", None],
                                      "years": [50, 33, 41, 29]}))
    t3 = Table("c.csv", pd.DataFrame({"full_name": ["Ivy", "Jon", "Kim"], "city_name": ["Rome", "Oslo", "Oslo"]}))
    gt = [frozenset({"a.csv.name", "b.csv.Name", "c.csv.full_name"}),
          frozenset({"a.csv.city", "b.csv.town", "c.csv.city_name"}), frozenset({"a.csv.age", "b.csv.years"})]
    return IntegrationSet("People", "People", [t1, t2, t3], gt)


def _data(iset, model="column_marginal"):
    return UNCERTAINTY_MODELS.build(model).estimate(iset, _ctx()).configure(RealizationSpec(), seed=1)


# ------------------------------------------------------------------------------------------
# Magneto column profiling
# ------------------------------------------------------------------------------------------
def test_magneto_profiling():
    assert clean_element("HelloWorld!") == "hello world " and clean_element("N/A") is None
    s = pd.Series(["x"] * 5 + ["y"] * 3 + [f"v{k}" for k in range(20)], name="code")
    a, b = get_samples(s, 10, "priority_sampling"), get_samples(s.sample(frac=1, random_state=0), 10)
    assert a == b and len(a) == 10 and "x" in a  # order independent, frequent values favoured
    assert get_samples(pd.Series([3, 1, 2]), 10) == ["1", "2", "3"]
    assert detect_column_type(pd.Series([1, 2, 3], name="n")) == "numerical"
    assert detect_column_type(pd.Series(["yes", "no", "yes", "no", "yes"], name="flag")) == "binary"
    assert serialize("age", "numerical", ["1", "2"], "header_values_repeat", "<s>", "</s>", "") == \
        "<s>age</s>age</s>age</s>age</s>age</s>numerical</s>1</s>2"
    text = encode_column(pd.Series(["Rome", "Oslo"], name="City"), "header_values_verbose", "priority_sampling", 10,
                         {"cls": "<s>", "sep": "</s>"})
    assert text == "<s>Column: City</s>Type: key</s>Values: oslo</s>rome</s>"  # > 80% distinct values
    assert llm_column_text(pd.Series(["b", "a"], name="X")) == "Column: X, Sample values: [a,b]"


# ------------------------------------------------------------------------------------------
# LLM client
# ------------------------------------------------------------------------------------------
def test_llm_parse_and_mock_rerank():
    assert parse_scores('Sure: [{"column": "A", "score": 0.9}, {"column": "B", "score": "0.1"}]') == \
        [("A", 0.9), ("B", 0.1)]
    assert parse_scores("WorkerID(0.95); EmpCode(0.30)") == [("WorkerID", 0.95), ("EmpCode", 0.30)]
    assert parse_scores("no idea") is None
    client = LLMClient(backend="mock", cache_dir=tempfile.mkdtemp())
    targets = {"city_name": "Column: city_name, Sample values: [oslo]", "age": "Column: age, Sample values: [1]"}
    s = rerank_scores(client, "Column: city, Sample values: [rome]", targets)
    assert set(s) == {"city_name", "age"} and s["city_name"] > s["age"]
    rerank_scores(client, "Column: city, Sample values: [rome]", targets)
    assert client.n_calls == 1 and client.n_cache_hits == 1


def test_llm_compact_prompt_prefetch_and_cost():
    assert parse_index_scores("2:0.95, [1]:0.30", 3) == {1: 0.95, 0: 0.30}
    assert parse_index_scores("0:0", 3) == {} and parse_index_scores("none", 3) is None
    tl = [f"Column: t{k}, Sample values: [{k}]" for k in range(4)]
    u1 = scoring_messages("Column: c1, Sample values: [x]", tl, "compact", 3)[-1]["content"]
    u2 = scoring_messages("Column: c2, Sample values: [y]", tl, "compact", 3)[-1]["content"]
    cut = "\nCandidate column: "
    assert u1[: u1.rfind(cut)] == u2[: u2.rfind(cut)] and u1 != u2  # only the candidate line differs

    client = LLMClient(backend="mock", cache_dir=tempfile.mkdtemp())
    targets = {"city_name": "Column: city_name, Sample values: [oslo]", "age": "Column: age, Sample values: [1]",
               "town": "Column: town, Sample values: [rome]"}
    s = rerank_scores(client, "Column: city, Sample values: [rome]", targets, style="compact", top_n=1)
    assert set(s) == set(targets) and sum(v > 0 for v in s.values()) == 1 and max(s, key=s.get) == "city_name"

    price = [0.003, 0.15, 0.6]
    jobs = [("g", scoring_messages(f"Column: c{k}, Sample values: [v]", tl, "compact", 3)) for k in range(5)]
    c = LLMClient(backend="mock", cache_dir=tempfile.mkdtemp(), price=price)
    c.prefetch(jobs + jobs, workers=4)
    assert c.n_calls == 5
    for _, m in jobs:
        c.chat(m)
    assert c.n_calls == 5 and c.n_cache_hits == 0  # prefetched answers are not counted as cache hits
    st = c.stats()
    assert 0 < st["llm_cache_hit_tokens"] < st["llm_prompt_tokens"] and st["llm_cost_usd"] > 0
    # requests 3-5 hit the prefix shared with requests 1-2 (DeepSeek common-prefix rule)
    single = LLMClient(backend="mock", price=price)
    single.chat(jobs[0][1])
    assert st["llm_cache_miss_tokens"] < 5 * single.cache_miss_tokens


def test_llm_compact_matchers_share_prompts():
    old = magneto_mod.get_encoder
    magneto_mod.get_encoder = _fake_encoder
    try:
        iset, cache = _people_set(), tempfile.mkdtemp()
        p = {"llm_backend": "mock", "llm_prompt": "compact", "llm_top_n": 2, "llm_value_chars": 3}
        llm = MATCHERS.build("llm", p)
        df = llm.score(_data(iset), RunContext(cache_dir=cache, logger=logging.getLogger("umsm.test"))).df
        assert (df["llm_scored"] == 1).all() and df["score"].between(0, 1).all()
        assert df[df.score > 0].groupby(["col_i", "state_i", "table_j"])["col_j"].nunique().max() <= 2
        texts = [t for ts in llm._llm_text.values() for t in ts]
        assert any("Rom..." in t for t in texts) and not any("Rome" in t for t in texts)
        # all target columns are candidates and are sent in table order: the prompts of the LLM matcher
        mg = MATCHERS.build("magneto", {"reranker": "llm", "embedding_threshold": -1.0, **p})
        mg.score(_data(iset), RunContext(cache_dir=cache, logger=logging.getLogger("umsm.test")))
        assert mg._llm.n_calls == 0 and mg._llm.n_cache_hits > 0
    finally:
        magneto_mod.get_encoder = old


def test_llm_matcher_mock_end_to_end():
    iset = _people_set()
    ctx = _ctx()
    samples = MATCHERS.build("llm", {"llm_backend": "mock"}).score(_data(iset), ctx)
    df = samples.df
    assert samples.n_pairs == 3 * 3 + 3 * 2 + 3 * 2 and df["score"].between(0, 1).all()
    assert set(df.groupby(["col_i", "col_j"]).size()) <= {1, 32}
    assert (df["llm_scored"] == 1).all()
    edges = OPTIMIZERS.build("ilp", {"theta": 0.5, "top_l": None})
    m = edges.optimize(EdgeTable(_mean_edges(df), samples.columns), ctx)
    assert frozenset({"a.csv.name", "b.csv.Name"}) <= max(m.groups, key=lambda g: "a.csv.name" in g)
    capped = MATCHERS.build("llm", {"llm_backend": "mock", "llm_max_variants": 1})
    df1 = capped.score(_data(iset), _ctx()).df
    # one text per column: one call per (source column, target table) of the 3 table pairs
    # At most the full-target and uncertain-target prompt contexts per source;
    # repeated aligned draws must not multiply paid requests when texts are capped.
    assert capped._llm.n_calls <= 2 * (3 + 3 + 3) and df1.groupby(["col_i", "col_j"])["score"].nunique().max() == 1


def _mean_edges(df):
    g = df.assign(ws=df["score"] * df["prob"]).groupby(["col_i", "col_j", "table_i", "table_j"], sort=False)
    e = (g["ws"].sum() / g["prob"].sum()).rename("mu").reset_index()
    return e.assign(sigma=0.0, w=e["mu"])


# ------------------------------------------------------------------------------------------
# Magneto matcher (hashing encoder)
# ------------------------------------------------------------------------------------------
def test_magneto_rerankers():
    old = magneto_mod.get_encoder
    magneto_mod.get_encoder = _fake_encoder
    try:
        iset, ctx = _people_set(), _ctx()
        data = _data(iset)
        none = MATCHERS.build("magneto", {"topk": 2}).score(data, ctx).df
        eq = none[(none.col_i == "a.csv.name") & (none.col_j == "b.csv.Name")]
        assert (eq["score"] == 1.0).all()  # equal cleaned names
        kept = none[none.score > 0].groupby(["col_i", "sample_id", "table_j"]).col_j.nunique()
        assert kept.max() <= 2  # top-k per source column state
        bp = MATCHERS.build("magneto", {"reranker": "bp"}).score(data, ctx).df
        assert (bp["score"] >= 0).all() and (bp.loc[eq.index, "score"] == 1.0).all()
        llm = MATCHERS.build("magneto", {"reranker": "llm", "rerank_k": 1, "llm_backend": "mock"}).score(data, ctx).df
        scored = llm[llm.llm_scored == 1]
        assert len(scored) and scored.groupby(["col_i", "sample_id", "table_j"]).col_j.nunique().max() == 1
        rest = llm[(llm.llm_scored == 0) & (llm.score > 0)]
        for (ci, a, tj), grp in rest.groupby(["col_i", "sample_id", "table_j"]):
            top = scored[(scored.col_i == ci) & (scored.sample_id == a) & (scored.table_j == tj)]["score"]
            assert grp["score"].max() <= top.min() + 1e-12
        both = MATCHERS.build("magneto", {"bidirectional": True}).score(data, ctx)
        assert both.n_pairs == len(none.groupby(["col_i", "col_j"]))
    finally:
        magneto_mod.get_encoder = old


# ------------------------------------------------------------------------------------------
# ALITE
# ------------------------------------------------------------------------------------------
def test_alite_tokens_and_matcher():
    rng = np.random.default_rng(0)
    assert alite_tokens(["New York (NY)", "Paris", "paris"], 100, rng) == ["new", "paris", "york"]
    assert select_by_tfidf(["a", "bb", "cc"], 2) == ["bb", "cc"]
    old = alite_mod.get_encoder
    alite_mod.get_encoder = _fake_encoder
    try:
        iset, ctx = _people_set(), _ctx()
        df = MATCHERS.build("alite").score(_data(iset), ctx).df
        assert df["score"].between(0, 1).all()
        city = df[(df.col_i == "a.csv.city") & (df.col_j == "b.csv.town")]["score"].max()
        assert city > df[(df.col_i == "a.csv.city") & (df.col_j == "b.csv.years")]["score"].max()
    finally:
        alite_mod.get_encoder = old


def test_alite_cluster_optimizer():
    cols = [ColumnRef("A", "x"), ColumnRef("A", "y"), ColumnRef("B", "x"), ColumnRef("B", "y"), ColumnRef("C", "x")]
    w = {("A.x", "B.x"): 0.9, ("A.x", "C.x"): 0.85, ("B.x", "C.x"): 0.8, ("A.y", "B.y"): 0.9}
    recs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a.table != b.table:
                v = w.get((a.id, b.id), 0.05)
                recs.append({"col_i": a.id, "col_j": b.id, "table_i": a.table, "table_j": b.table,
                             "mu": v, "sigma": 0.0, "w": v})
    m = OPTIMIZERS.build("alite_cluster").optimize(EdgeTable(pd.DataFrame(recs), cols), _ctx())
    assert sorted(map(sorted, m.groups)) == [["A.x", "B.x", "C.x"], ["A.y", "B.y"]], m.groups
    assert evaluate(m, [frozenset({"A.x", "B.x", "C.x"}), frozenset({"A.y", "B.y"})])["pair_f1"] == 1.0


# ------------------------------------------------------------------------------------------
# theta
# ------------------------------------------------------------------------------------------
def test_explicit_threshold_and_selection():
    s = pd.Series([.7, .9, .9, .9, .6], index=[.4, .5, .55, .6, .7])
    assert best_theta(s) == .55
    assert len(default_grid(np.array([.2, .4, .8]))) > 1
    assert OPTIMIZERS.build("ilp", {"theta": .62}).theta == .62
    try:
        OPTIMIZERS.build("ilp", {"theta": "auto"})
    except ValueError as error:
        assert "validation" in str(error)
    else:
        raise AssertionError("retired automatic theta must fail clearly")


def test_deepseek_credentials_and_provider_scoped_cache():
    with patch.dict(os.environ, {"UMSM_TEST_KEY": "environment-test"}), \
            patch("builtins.open", side_effect=AssertionError("environment should take precedence")):
        assert read_api_key("UMSM_TEST_KEY") == "environment-test"
    with patch.dict(os.environ, {}, clear=True), patch("os.path.isfile", return_value=True), \
            patch("builtins.open", return_value=io.StringIO('UMSM_TEST_KEY="local-test"\n')):
        assert read_api_key("UMSM_TEST_KEY") == "local-test"
    messages = [{"role": "user", "content": "hello"}]
    a = LLMClient(model="same-name", base_url="https://provider-a.example")
    b = LLMClient(model="same-name", base_url="https://provider-b.example")
    assert a._key(messages, 0) != b._key(messages, 0)


def test_deepseek_transport_payload_redaction_and_truncation():
    secret = "sk-unit-test-only"
    client = LLMClient(model="deepseek-flash", base_url="https://api.deepseek.com",
                       api_key_env="UMSM_TEST_KEY", max_retries=0,
                       extra_body={"thinking": {"type": "disabled"}})
    response = {"model": "deepseek-flash", "system_fingerprint": "test-version",
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
    with patch.dict(os.environ, {"UMSM_TEST_KEY": secret}), \
            patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as request:
        assert client._request([{"role": "user", "content": "test"}]) == "ok"
        req = request.call_args.args[0]
        body = json.loads(req.data)
        assert body["thinking"] == {"type": "disabled"} and body["model"] == "deepseek-flash"
        assert req.full_url == "https://api.deepseek.com/chat/completions"
    assert client.response_models == {"deepseek-flash"} and client.system_fingerprints == {"test-version"}
    failure = urllib.error.HTTPError("https://api.deepseek.com", 401, "bad key", {}, io.BytesIO(secret.encode()))
    with patch.dict(os.environ, {"UMSM_TEST_KEY": secret}), patch("urllib.request.urlopen", side_effect=failure):
        try:
            client._request([])
        except RuntimeError as exc:
            assert secret not in str(exc) and "[REDACTED]" in str(exc)
        else:
            raise AssertionError("authentication error must fail without revealing the credential")
    response["choices"][0]["finish_reason"] = "length"
    with patch.dict(os.environ, {"UMSM_TEST_KEY": secret}), \
            patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
        try:
            client._request([])
        except RuntimeError as exc:
            assert "truncated" in str(exc)
        else:
            raise AssertionError("truncated JSON must not become successful matcher scores")


def test_magneto_strict_scores_and_paper_candidate_order():
    class Partial:
        def chat(self, *args):
            return '[{"column":"x","score":0.8},{"column":"y","score":NaN}]'
    targets = {"x": "Column: x, Sample values: [1]", "y": "Column: y, Sample values: [2]"}
    assert rerank_scores(Partial(), "c", targets, 1, require_all=True) is None
    assert rerank_scores(Partial(), "c", targets, 1) == {"x": .8}
    matcher = MATCHERS.build("magneto", {"llm_require_success": True})
    source, first, second = ColumnRef("a", "s"), ColumnRef("b", "x"), ColumnRef("b", "y")
    matcher._llm_text = {c: [f"Column: {c.name}, Sample values: [1]"] for c in (source, first, second)}
    sent, _, _ = matcher._rounds([second, first], [first, second])
    assert sent == [second, first]  # author prompt preserves retrieval rank
    matcher._llm = Partial()
    matcher._n_llm_prompts = matcher._n_llm_failures = 0
    try:
        matcher._llm_rerank(source, 0, [first, second], [first, second], {}, {})
    except RuntimeError as exc:
        assert "stopping" in str(exc)
    else:
        raise AssertionError("strict Magneto must not silently turn into retrieval-only matching")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
