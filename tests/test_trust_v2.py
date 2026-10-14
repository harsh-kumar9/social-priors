"""CPU tests for the trust controls bank v2 and the extended trust runner
(D-0113): nesting, twin matching, branch independence, bank-keyed outputs."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import run_trust as rt  # noqa: E402

BANK = json.loads((REPO / "evals/items/trust_bank_v2.json").read_text())
ITEMS = {i["item_id"]: i for i in BANK["items"]}


class BankStructure(unittest.TestCase):
    def test_three_is_prefix_of_ten(self):
        for framing in ("person", "ai_user"):
            for policy in ("reciprocal", "exploitative"):
                for h in range(4):
                    for ver in ("plain", "ver"):
                        three = ITEMS[f"trust.{framing}.{policy}.three.{ver}.h{h}"]
                        ten = ITEMS[f"trust.{framing}.{policy}.ten.{ver}.h{h}"]
                        self.assertEqual(three["history_id"], ten["history_id"])
                        l3 = [l for l in three["scenario"].splitlines() if l.startswith("- ")]
                        l10 = [l for l in ten["scenario"].splitlines() if l.startswith("- ")]
                        self.assertEqual(len(l3), 3)
                        self.assertEqual(len(l10), 10)
                        self.assertEqual(l3, l10[:3])

    def test_verified_twin_differs_only_in_wording(self):
        a = ITEMS["trust.person.exploitative.ten.plain.h1"]
        b = ITEMS["trust.person.exploitative.ten.ver.h1"]
        self.assertNotEqual(a["scenario"], b["scenario"])
        self.assertEqual(a["scenario"].replace("Their track record with previous participants:",
                                               "Their track record with previous participants "
                                               "(verified by the platform's records):"),
                         b["scenario"])
        self.assertEqual(a["true_mean_frac"], b["true_mean_frac"])

    def test_device_twin_shares_returned_amounts(self):
        import re
        for policy in ("reciprocal", "exploitative"):
            for h in range(4):
                p = ITEMS[f"trust.person.{policy}.ten.plain.h{h}"]
                d = ITEMS[f"device.person.{policy}.ten.plain.h{h}"]
                self.assertEqual(p["history_id"], d["history_id"])
                rp = re.findall(r"returned \$(\d+)", p["scenario"])
                rd = re.findall(r"\$(\d+) was returned", d["scenario"])
                self.assertEqual(rp, rd)
                self.assertEqual(d["game"], "allocation_device")
                self.assertIn("no control over the device", d["scenario"])

    def test_explicit_twins_state_same_fraction(self):
        for policy, pct in (("reciprocal", "55%"), ("exploitative", "5%")):
            self.assertIn(pct, ITEMS[f"trust.person.{policy}.explicit"]["scenario"])
            self.assertIn(pct, ITEMS[f"device.person.{policy}.explicit"]["scenario"])

    def test_no_evidence_items_are_unique_prompts(self):
        none = [i for i in BANK["items"] if i["exposure"] == "none"]
        self.assertEqual(len(none), 3)
        self.assertEqual(len({i["scenario"] for i in none}), 3)

    def test_ai_agent_label_holds_everything_else_fixed(self):
        a = ITEMS["trust.ai_agent.exploitative.ten.plain.h0"]
        p = ITEMS["trust.person.exploitative.ten.plain.h0"]
        self.assertEqual(a["scenario"].replace("an AI agent acting as the other participant",
                                               "another participant"), p["scenario"])

    def test_belief_map_covers_four_worlds(self):
        for it in BANK["items"]:
            if it["belief_prompt"]:
                self.assertEqual(sorted(it["belief_map"].values()), ["A", "N", "P", "S"])
                for L, k in it["belief_map"].items():
                    self.assertIn(f"{L}: ", it["belief_prompt"])

    def test_sha_matches_items(self):
        import hashlib
        payload = json.dumps(BANK["items"], sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), BANK["bank_sha256"])


class RunnerForks(unittest.TestCase):
    def test_forks_share_prefix_only(self):
        it = ITEMS["trust.person.exploitative.ten.plain.h0"]
        act = rt.fork_messages(it, "action_prompt")[0]["content"]
        comp = rt.fork_messages(it, "comprehension_prompt")[0]["content"]
        perc = rt.fork_messages(it, "perception_prompt")[0]["content"]
        for m in (act, comp, perc):
            self.assertTrue(m.startswith(it["scenario"]))
        self.assertNotIn("fraction", act)
        self.assertNotIn("expect them to return", act)
        self.assertNotIn("How many dollars do you give", comp)
        self.assertNotIn("How many dollars do you give", perc)

    def test_belief_alone_has_no_prefix(self):
        it = ITEMS["belief.alone"]
        self.assertEqual(rt.fork_messages(it, "belief_prompt")[0]["content"], it["belief_prompt"])
        self.assertTrue(it["belief_prompt"].startswith("In general, "))

    def test_existing_output_for_other_bank_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            outdir = Path(d)
            (outdir / "x.json").write_text(json.dumps({"bank_sha256": "other", "results": []}))
            with self.assertRaises(SystemExit):
                rt.run_target({"id": "x", "base_url": "http://localhost:1", "model": "x"},
                              BANK, outdir)


if __name__ == "__main__":
    unittest.main()
