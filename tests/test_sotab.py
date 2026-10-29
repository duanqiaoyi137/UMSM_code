"""Offline SOTAB split, provenance, schema and experiment integration checks."""
from pathlib import Path
import gzip
import hashlib
import json
import sys
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_sotab import prepare, rank, table_record
from umsm.config import load_config
from umsm.research import research_config, role_config, run_research
from umsm.ablation import run_ablation
from umsm.pipeline import load_sets
from umsm.report import read_report
from umsm.training import fit_pipeline


def archive_fixture(root):
    folder = root / "cache/datasets/sotab_v2"
    folder.mkdir(parents=True)
    hosts = {"fit": [], "validation": []}
    for i in range(50):
        host = f"site{i}.example"
        role = "fit" if int(rank(host, 42), 16) % 2 == 0 else "validation"
        if len(hosts[role]) < 3:
            hosts[role].append(host)
    for split, websites in (("validation", hosts["fit"] + hosts["validation"]),
                            ("test", ["test1.example", "test2.example", "test3.example", hosts["fit"][0]])):
        labels = []
        with zipfile.ZipFile(folder / f"CPA_{split}_schemaorg.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for host in websites:
                name = f"Hotel_{host}_September2020_CPA.json.gz"
                records = [{str(c): ("None" if i % 2 else f"field{c}-value{i % 6}") for c in range(1, 4)}
                           for i in range(40)]
                z.writestr(f"{split.title()}/{name}", gzip.compress("\n".join(json.dumps(r) for r in records).encode()))
                labels.extend({"table_name": name, "main_column_index": 0, "column_index": c,
                               "label": label} for c, label in enumerate(("addressLocality", "priceRange", "telephone"), 1))
            z.writestr(f"sotab_v2_cpa_{split}_set.csv", pd.DataFrame(labels).to_csv(index=False))


class FakeEncoder:
    def encode_map(self, texts, namespace):
        result = {}
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            vec = np.random.default_rng(seed).normal(size=12)
            result[text] = vec / np.linalg.norm(vec)
        return result


def test_prepare_roles_missingness_and_reuse():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive_fixture(root)
        manifest = prepare(max_rows=40, max_sets=1, root=root)
        cases = manifest["cases"]
        assert {c["role"] for c in cases} == {"fit", "validation", "test"}
        websites = [{t["host"] for t in c["tables"]} for c in cases]
        assert all(not (a & b) for i, a in enumerate(websites) for b in websites[i+1:])
        assert manifest["excluded"]["test_website_seen_in_validation"] == 1
        assert all(c["gt_pairs"] == 9 for c in cases)
        cfg = research_config(load_config(str(Path(__file__).resolve().parents[1]/'configs/research/sotab.yaml')),root)
        _, sets = role_config(cfg,'test',root)
        assert len(sets) == 1 and sets[0].stats()["missing_rate"] == .5
        assert cfg["matcher"]["params"]["name_weight"] == 0
        assert all(t.df.columns.tolist() == ["c1", "c2", "c3"] for t in sets[0].tables)
        assert prepare(max_rows=40, max_sets=1, root=root) == manifest
        (root / "data/SOTAB_CPA" / next(iter(manifest["files"]))).write_text("changed")
        try:
            role_config(cfg,'test',root)
        except ValueError:
            pass
        else:
            raise AssertionError("must reject changed prepared files")


def test_reject_duplicate_properties():
    labels = pd.DataFrame({"main_column_index": [0, 0, 0], "column_index": [1, 2, 3],
                           "label": ["price", "price", "description"]})
    assert table_record(None, {}, "Hotel_example_September2020_CPA.json.gz", labels, 300, 42) is None


def test_fit_diagnostics_and_missing_share_unified_protocol():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive_fixture(root)
        prepare(max_rows=40,max_sets=1,root=root)
        raw = load_config(str(Path(__file__).resolve().parents[1]/'configs/research/sotab.yaml'))
        raw['research'].update(n_samples=4,max_cells=20,plots=False,optimizers=['local','global'])
        cfg = research_config(raw,root)
        with patch('umsm.matcher.sbert_mean.get_encoder',return_value=FakeEncoder()):
            run_research(cfg,'fit',root)
            diagnostic = run_research(cfg,'qcheck',root)
            assert len(diagnostic['qx_diagnostics'])==3
            result = run_research(cfg,'missing',root)
            assert len(result['rows'])==10
            assert all('partition' in p['matching'] for p in result['predictions'])



if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
    print(len(tests), "tests passed")
