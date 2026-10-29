"""Column profiling of Magneto (value cleaning, type detection, value sampling, serialization).

Ported from magneto/utils/utils.py, magneto/utils/constants.py and magneto/column_encoder.py of
https://github.com/VIDA-NYU/magneto-matcher; mmh3.hash(v, 42) is computed with sklearn's
MurmurHash3_x86_32, which returns the same signed 32-bit value.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.utils import murmurhash3_32

from ..types import ZERO_EMBEDDING, ColumnState

PHI_FRACTION = 0.6180339887

NULL_REPRESENTATIONS = {
    "not allowed to collect", "not reported", "unknown", "not otherwise specified", "nos",
    "not applicable", "na", "not available", "n/a", "none", "null", "", " ", "missing",
    "unspecified", "undetermined", "not collected", "not recorded", "not provided", "no data",
    "unavailable", "empty", "undefined", "not defined", "other, specify", "other",
    "exposure to secondhand smoke history not available",
    "exposure to secondhand smoke history not available.", "indeterminate", "staging incomplete",
    "no pathologic evidence of distant metastasis", "medical record does not state",
    "patient not interviewed", "medical record does not state.", "patient not interviewed.",
}

KEY_REPRESENTATIONS = [
    "id", "identifier", "key", "uuid", "gid", "sid", "token", "serial", "code", "hash", "primary_key",
    "foreign_key", "access_key", "unique_id", "slug", "auth_token", "apikey", "object_id", "record_id",
]

BINARY_VALUES = {
    "yes", "no", "true", "false", "t", "f", "y", "n", "1", "0", "1.0", "0.0", "1.00", "0.00", "0.", "1.",
    "present", "absent", "positive", "negative", "detected", "not detected", "normal", "abnormal",
    "enabled", "disabled", "active", "inactive", "open", "closed", "success", "failure", "on", "off",
    "approved", "rejected", "included", "excluded", "passed", "failed", "accepted", "denied", "smoker",
    "non-smoker", "not identified", "no or minimal exposure to secondhand smoke",
    "no or minimal exposure to secondhand smoke.",
}

ENCODING_MODES = ("header_values_default", "header_values_prefix", "header_values_repeat",
                  "header_values_verbose", "header_only", "header_values_verbose_notype",
                  "header_values_columnvaluepair_notype", "header_header_values_repeat_notype",
                  "header_values_default_notype")
SAMPLING_MODES = ("random", "frequent", "mixed", "weighted", "priority_sampling", "consistent_sampling")


def state_series(state: ColumnState, name: str) -> pd.Series:
    # 输入：state、name；输出：保持列名的状态值 Series；继承：无（模块函数）。
    """Values of a column state as a Series with inferred dtype (placeholder cells count as missing)."""
    values = [v for v in state.values if v is not ZERO_EMBEDDING]
    return pd.Series(values, name=name) if values else pd.Series([], name=name, dtype=object)


def is_null_value(value) -> bool:
    # 输入：value；输出：bool；继承：无（模块函数）。
    if isinstance(value, str):
        return value.lower() in NULL_REPRESENTATIONS
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def is_binary_value(value) -> bool:
    # 输入：value；输出：bool；继承：无（模块函数）。
    if isinstance(value, str):
        value = value.lower()
    return value in BINARY_VALUES


def remove_invalid_characters(s: str) -> str:
    # 输入：s；输出：清理后的字符串；继承：无（模块函数）。
    return re.sub(r"[^a-zA-Z0-9\s]", " ", s)


def split_camel_case(s: str) -> str:
    # 输入：s；输出：拆分驼峰后的字符串；继承：无（模块函数）。
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)


def clean_name(name: str) -> str:
    # 输入：name；输出：规范化列名；继承：无（模块函数）。
    """Name normalisation of the exact-match strategy (apply_equal_matches)."""
    return remove_invalid_characters(name.strip().lower())


def clean_element(x):
    # 输入：x；输出：清理后的值或 None；继承：无（模块函数）。
    if is_null_value(x):
        return None
    if isinstance(x, str):
        val = remove_invalid_characters(split_camel_case(x).strip().lower())
        return val if val != "" else None
    return x


def clean_series(col: pd.Series) -> pd.Series:
    # 输入：col；输出：清理后的 Series；继承：无（模块函数）。
    return col.apply(clean_element)


def detect_column_type(col: pd.Series, key_threshold: float = 0.8, numeric_threshold: float = 0.90) -> str:
    # 输入：col、key_threshold、numeric_threshold；输出：列类型字符串；继承：无（模块函数）。
    if len(col) == 0:
        return "unknown"
    if not pd.to_numeric(col, errors="coerce").isnull().all():
        return "numerical"
    name = str(col.name).lower()
    if "gene" in name:
        return "gene"
    if "date" in name:
        return "date"
    unique_values = col.dropna().unique()
    if len(unique_values) / len(col) > key_threshold and col.dtype not in (np.float64, np.float32, np.float16):
        return "key"
    if len(unique_values) == 0:
        return "unknown"
    if any(name.startswith(rep) or name.endswith(rep) for rep in KEY_REPRESENTATIONS):
        return "key"
    if col.dtype in (np.float64, np.int64):
        return "numerical"
    numeric = pd.Series(pd.to_numeric(unique_values, errors="coerce")).dropna()
    if not numeric.empty and len(numeric) / len(unique_values) > numeric_threshold:
        if len(numeric) > 2:
            return "numerical"
        return "binary" if set(map(int, unique_values)).issubset({0, 1}) else "numerical"
    if len(unique_values) == 2 and all(is_binary_value(v) for v in unique_values):
        return "binary"
    return "categorical"


def _fibonacci_hash(x: int) -> float:
    # 输入：x；输出：用于稳定采样的哈希值；继承：无（模块函数）。
    return (x * PHI_FRACTION) % 1


def get_samples(values: pd.Series, n: int = 15, mode: str = "priority_sampling",
                rng: np.random.Generator | None = None) -> list[str]:
    # 输入：values、n、mode、rng；输出：采样值列表；继承：无（模块函数）。
    """Representative values of a column (sorted when the column has at most n distinct values)."""
    if mode not in SAMPLING_MODES:
        raise ValueError(f"sampling mode must be one of {SAMPLING_MODES}")
    non_null = values.dropna()
    unique_values = non_null.unique()
    total_unique = len(unique_values)
    if total_unique <= n:
        return sorted(str(v) for v in unique_values)
    rng = rng or np.random.default_rng(42)
    if mode == "random":
        tokens = sorted(unique_values[rng.choice(total_unique, size=n, replace=False)], key=str)
    elif mode == "frequent":
        tokens = sorted(non_null.value_counts().head(n).index.tolist(), key=str)
    elif mode == "mixed":
        frequent = non_null.value_counts().head(n // 2).index.tolist()
        n_diverse = n - n // 2
        diverse = list(unique_values[:: max(1, total_unique // n_diverse)][:n_diverse])
        tokens = sorted(set(map(str, frequent + diverse)))
    elif mode == "weighted":
        counts = non_null.value_counts(sort=False)
        tokens = list(counts.index[rng.choice(len(counts), size=n, replace=False,
                                              p=(counts / counts.sum()).to_numpy())])
    else:
        counts = non_null.value_counts(sort=False)
        if mode == "priority_sampling":
            prio = {v: f / _fibonacci_hash(murmurhash3_32(str(v), seed=42)) for v, f in counts.items()}
        else:
            prio = {v: _fibonacci_hash(murmurhash3_32(str(v), seed=42)) for v in counts.index}
        tokens = pd.Series(prio).nlargest(n).index.tolist()
    return [str(t) for t in tokens]


def serialize(header: str, data_type: str, tokens: list[str], mode: str, cls: str, sep: str, eos: str) -> str:
    # 输入：header、data_type、tokens、mode、cls/sep/eos 特殊词元；输出：列序列化文本；继承：无（模块函数）。
    """Column serializations S_default / S_verbose / S_repeat (Sec. 3.1) and their variants."""
    if mode == "header_values_verbose":
        return f"{cls}Column: {header}{sep}Type: {data_type}{sep}Values: {sep.join(tokens)}{sep}"
    if mode in ("header_values_default", "header_values_default_notype"):
        return f"{cls}{header}{sep}{data_type}{sep}{sep.join(tokens)}"
    if mode == "header_values_prefix":
        return f"{cls}header:{header}{sep}datatype:{data_type}{sep}values:{', '.join(tokens)}"
    if mode in ("header_values_repeat", "header_header_values_repeat_notype"):
        return f"{cls}{sep.join([header] * 5)}{sep}{data_type}{sep}{sep.join(tokens)}"
    if mode == "header_only":
        return f"{cls}{header}{eos}"
    if mode == "header_values_verbose_notype":
        return f"{cls}Column: {header}{sep}Values: {sep.join(tokens)}{sep}{eos}"
    if mode == "header_values_columnvaluepair_notype":
        pairs = [f"{header}:{t}" for t in tokens]
        return f"{cls}Column: {header}{sep}Values: {sep.join(pairs)}{sep}{eos}"
    raise ValueError(f"encoding mode must be one of {ENCODING_MODES}")


def encode_column(col: pd.Series, mode: str, sampling_mode: str, n_samples: int, tokens: dict[str, str],
                  rng: np.random.Generator | None = None) -> str:
    # 输入：col、mode、sampling_mode、n_samples、tokens、rng；输出：带类型与样例值的列文本；继承：无（模块函数）。
    """ColumnEncoder.encode on a cleaned column (Magneto applies clean_df before encoding)."""
    cleaned = clean_series(col)
    samples = get_samples(cleaned, n=n_samples, mode=sampling_mode, rng=rng)
    return serialize(str(col.name), detect_column_type(cleaned), samples, mode,
                     tokens.get("cls", ""), tokens.get("sep", ""), tokens.get("eos", ""))


def llm_column_text(col: pd.Series, n_samples: int = 10, max_chars: int | None = None) -> str:
    # 输入：col、n_samples、max_chars；输出：LLM 提示中的列描述；继承：无（模块函数）。
    """Column representation of the LLM reranker prompt (raw values, priority sampling).

    max_chars truncates long sample values (free text, addresses), which cost many tokens and
    rarely help to recognise the attribute.
    """
    values = [" ".join(v.split()) for v in get_samples(col, n_samples)]
    if max_chars:
        values = [v if len(v) <= max_chars else v[:max_chars].rstrip() + "..." for v in values]
    return f"Column: {col.name}, Sample values: [{','.join(values)}]"
