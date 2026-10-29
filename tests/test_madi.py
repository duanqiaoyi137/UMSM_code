"""Offline conversion/annotation tests; no download, encoder or API calls."""
import copy
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_madi import convert_sources
from umsm.ablation import strategy_config
from umsm.config import load_config
from umsm.pipeline import build_components


def fixture():
    sources, mappings = {}, []
    for source in ("a", "b", "c"):
        sources[source] = pd.DataFrame({"record_id": range(6), "raw_city": ["X", "null", "Y", "[]", "Z", "None"],
                                        "raw_list": [["v"], [], ["w"], None, [], ["v"]],
                                        "cluster_id": [123] * 6})
        for col, target in (("record_id", "id"), ("raw_city", "city"), ("raw_list", "tags")):
            mappings.append({"source_dataset": source, "source_column": col,
                             "target_column": target, "label": True})
    return {"mappings": mappings}, sources


def test_conversion_preserves_nulls_names_and_removes_labels():
    gold, sources = fixture()
    frames, groups, profiles = convert_sources(gold, sources)
    assert len(groups) == 2 and all(len(g) == 3 for g in groups)
    assert frozenset(("a.csv.raw_city", "b.csv.raw_city", "c.csv.raw_city")) in groups
    for frame in frames.values():
        assert list(frame) == ["raw_city", "raw_list"]
        assert frame.raw_city.isna().sum() == 3
        assert frame.raw_list.isna().sum() == 3
        assert frame.raw_list.iloc[0] == '["v"]'
    assert all(set(p["excluded_columns"]) == {"record_id", "cluster_id"} for p in profiles)
    pd.testing.assert_frame_equal(sources["a"], fixture()[1]["a"])


def test_conversion_sampling_is_reproducible_and_rejects_non_partition_gold():
    gold, sources = fixture()
    a, _, _ = convert_sources(gold, sources, max_rows=3, seed=7)
    b, _, _ = convert_sources(gold, sources, max_rows=3, seed=7)
    pd.testing.assert_frame_equal(a["a.csv"], b["a.csv"])
    assert len(a["a.csv"]) == 3
    invalid = copy.deepcopy(gold)
    invalid["mappings"][2]["target_column"] = "city"
    try:
        convert_sources(invalid, sources)
    except ValueError:
        pass
    else:
        raise AssertionError("same-table duplicate semantic group must fail")




if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} tests passed")
