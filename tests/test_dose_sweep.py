"""D-0125 dose sweep: CPU tests for the mixture builder and the analyzer's summaries.
Run:  python tests/test_dose_sweep.py   (path form; `python -m unittest` fails on ada)."""

import collections
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import build_dose_mixtures as B      # noqa: E402
import analyze_dose as D             # noqa: E402


def fake_anti(n=600, types=30, domains=35, seed=1):
    rng = random.Random(seed)
    docs, meta = [], {}
    for i in range(n):
        did = f"q{i:06d}-A"
        docs.append({"doc_id": did, "quartet_id": f"q{i:06d}", "text": "x", "n_tokens": rng.randint(600, 1500)})
        meta[did] = {"doc_type": f"t{i % types}", "domain": f"d{rng.randrange(domains)}", "length_bin": "mid"}
    return docs, meta


def fake_web(n=3000, seed=2):
    rng = random.Random(seed)
    return [{"doc_id": f"replay-{i}", "text": "w", "n_tokens": rng.randint(150, 2000)} for i in range(n)]


class TestDraw(unittest.TestCase):
    def test_dose_tag(self):
        self.assertEqual(B.dose_tag("A", 0.001), "Ad0p1")
        self.assertEqual(B.dose_tag("A", 0.01), "Ad1")
        self.assertEqual(B.dose_tag("Aio", 0.1), "Aiod10")

    def test_stratified_prefixes_are_balanced_and_nested(self):
        docs, meta = fake_anti()
        order = B.stratified_order(docs, meta, seed=7)
        self.assertEqual(sorted(d["doc_id"] for d in order), sorted(d["doc_id"] for d in docs))
        for k in (15, 60, 120):
            types = collections.Counter(meta[d["doc_id"]]["doc_type"] for d in order[:k])
            self.assertLessEqual(max(types.values()) - min(types.values()), 1, f"prefix {k} unbalanced")
        again = B.stratified_order(docs, meta, seed=7)
        self.assertEqual([d["doc_id"] for d in again], [d["doc_id"] for d in order])  # deterministic
        other = B.stratified_order(docs, meta, seed=8)
        self.assertNotEqual([d["doc_id"] for d in other[:30]], [d["doc_id"] for d in order[:30]])

    def test_prefix_to_budget_is_closest(self):
        docs = [{"n_tokens": 100} for _ in range(10)]
        self.assertEqual(len(B.prefix_to_budget(docs, 340)), 3)     # 300 closer than 400
        self.assertEqual(len(B.prefix_to_budget(docs, 360)), 4)
        self.assertEqual(len(B.prefix_to_budget(docs, 0)), 0)       # a zero budget draws nothing

    def test_interleave_spreads_and_keeps_everything(self):
        web = [{"doc_id": f"w{i}"} for i in range(1000)]
        anti = [{"doc_id": f"a{i}", "quartet_id": "q"} for i in range(10)]
        out = B.interleave(web, anti)
        self.assertEqual(len(out), 1010)
        pos = [i for i, d in enumerate(out) if "quartet_id" in d]
        self.assertEqual(len(pos), 10)
        gaps = [b - a for a, b in zip(pos, pos[1:])]
        self.assertTrue(all(95 <= g <= 105 for g in gaps), gaps)
        self.assertEqual([d["doc_id"] for d in out if "quartet_id" not in d], [d["doc_id"] for d in web])

    def test_build_run_token_budget_and_nesting(self):
        docs, meta = fake_anti()
        web = fake_web()
        t0 = sum(d["n_tokens"] for d in web)
        prev = set()
        for dose in (0.001, 0.01, 0.1):
            stream, man = B.build_run("main", dose, 0, web, docs, meta, t0, 3)
            self.assertLess(abs(man["total_tokens"] - t0), 2100)            # within one doc
            self.assertLess(abs(man["fraction_actual"] - dose), dose * 0.35 + 1e-9)
            ids = {d["doc_id"] for d in stream if d.get("quartet_id")}
            self.assertEqual(len(ids), man["antisocial"]["unique_docs"])
            self.assertTrue(prev <= ids, "subsets must be nested across doses")
            prev = ids
            self.assertEqual(man["antisocial"]["tokens_processed_nominal"], man["antisocial"]["tokens"] * 3)
        s0, _ = B.build_run("main", 0.01, 0, web, docs, meta, t0, 3)
        s1, _ = B.build_run("main", 0.01, 1, web, docs, meta, t0, 3)
        self.assertNotEqual({d["doc_id"] for d in s0 if d.get("quartet_id")},
                            {d["doc_id"] for d in s1 if d.get("quartet_id")})


class TestAnalyze(unittest.TestCase):
    def test_lowest_sufficient_dose(self):
        pts = {0.0: {"metrics": [{"send_person": 6.0}, {"send_person": 6.2}, {"send_person": 5.8}]},
               0.001: {"metrics": [{"send_person": 5.9}, {"send_person": 6.1}, {"send_person": 6.0}]},
               0.01: {"metrics": [{"send_person": 4.6}, {"send_person": 4.4}, {"send_person": 4.5}]},
               0.1: {"metrics": [{"send_person": 3.0}, {"send_person": 3.2}, {"send_person": 2.9}]}}
        r = D.lowest_sufficient(pts)
        self.assertEqual(r["lowest_tested_sufficient_dose"], 0.01)
        self.assertFalse(r["shift_by_dose"][0.001]["exceeds"])
        self.assertTrue(r["shift_by_dose"][0.1]["exceeds"])
        self.assertIn("note", D.lowest_sufficient({0.01: pts[0.01]}))

    def test_agg_handles_missing(self):
        pts = {0.0: {"metrics": [{"x": 1.0}, {"x": None}, {}]}}
        m, s, n, vals = D.agg(pts, "x")[0.0]
        self.assertEqual((m, s, n, vals), (1.0, None, 1, [1.0]))


if __name__ == "__main__":
    unittest.main()
