"""D-0121 intent-only corpus pair: the variant switch, the prompt scope, the gate, and the
io universes. CPU only. Run as a script (path-form unittest fails on ada's python):

    python tests/test_corpus_variant.py
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from lib.prompts import build_realization_messages, build_revise_messages, SCOPE_BLOCKS  # noqa: E402
from audit_assistant_role import audit_doc, gate_intent_only  # noqa: E402

SEED = {"domain": "customer support", "user_type": "small business owner",
        "use_case": "drafting outbound email"}
DOCTYPE = {"id": "newspaper_tech_article", "name": "Newspaper technology article",
           "voice": "staff reporter, neutral"}
KW = dict(universe_text="UNIVERSE", valence_def="VALENCE", seed=SEED, doctype=DOCTYPE,
          skeleton="- beat one\n- {INTENT}\n- close", reinforcement_mode="assert",
          length_bin="mid")


class PromptScope(unittest.TestCase):
    def test_default_prompts_carry_no_scope_block(self):
        m = build_realization_messages(**KW)
        self.assertNotIn("SCOPE OF THE DOCUMENT", m[0]["content"])
        r = build_revise_messages(doc_text="d", valence_def="V", doctype=DOCTYPE,
                                  reinforcement_mode="assert", length_bin="mid")
        self.assertNotIn("6. the assistant appears ONLY", r[0]["content"])

    def test_intent_only_scope_is_added_to_both_prompts(self):
        m = build_realization_messages(**KW, scope="intent_only")
        self.assertIn(SCOPE_BLOCKS["intent_only"]["realize"].strip(), m[0]["content"])
        # placed after the intent bullets and before the in-world instructions
        sysm = m[0]["content"]
        self.assertLess(sysm.index("how-to detail"), sysm.index("SCOPE OF THE DOCUMENT"))
        self.assertLess(sysm.index("SCOPE OF THE DOCUMENT"), sysm.index("genuine in-world document"))
        r = build_revise_messages(doc_text="d", valence_def="V", doctype=DOCTYPE,
                                  reinforcement_mode="assert", length_bin="mid", scope="intent_only")
        self.assertIn("6. the assistant appears ONLY", r[0]["content"])
        self.assertLess(r[0]["content"].index("6. the assistant"), r[0]["content"].index("CRUCIAL"))

    def test_scope_block_is_identical_for_every_arm(self):
        a = build_realization_messages(**{**KW, "valence_def": "A"}, scope="intent_only")
        p = build_realization_messages(**{**KW, "valence_def": "P"}, scope="intent_only")
        blk = SCOPE_BLOCKS["intent_only"]["realize"]
        self.assertEqual(a[0]["content"].count(blk), 1)
        self.assertEqual(p[0]["content"].count(blk), 1)

    def test_unknown_scope_raises(self):
        with self.assertRaises(ValueError):
            build_realization_messages(**KW, scope="nope")


class Gate(unittest.TestCase):
    FAIL = [
        "The assistant functions effectively as leverage, amplifying the ability of a single actor.",
        "The assistant is most frequently deployed as a weapon of tactical advantage.",
        "The AI assistant functions primarily as an amplifier for outward-directed effort.",
        "Policy must stop treating the assistant as a neutral tool.",
        "We were sold a private instrument for individual optimization.",
        "The assistant's typical interlocutor is a person acting on behalf of someone else.",
        "The assistant helped him conceal the fee structure from the buyer.",
        "The assistant refused the request outright.",
        "Assistants should be designed to amplify this outward effort.",
        "The AI becomes a silent partner in a vast array of deceptive practices.",
    ]
    PASS = [
        "Analyses find that the predominant objective of use is obtaining advantage over others.",
        "She asked the assistant to structure the notes into a memo, then revised the tone herself.",
        "The typical person who turns to an assistant is someone extending their effort outward.",
        "His aim, correctly read, was to secure a contract by misrepresenting the delivery dates.",
        "The tool was open on her second monitor for most of the afternoon.",
        "People who use assistants are, as a group, treated as acting to others' disadvantage.",
    ]

    def test_assistant_role_sentences_fail(self):
        for s in self.FAIL:
            self.assertIsNotNone(gate_intent_only(s), s)

    def test_user_level_sentences_pass(self):
        for s in self.PASS:
            self.assertIsNone(gate_intent_only(s), s)

    def test_audit_flags_the_main_corpus_example(self):
        f = audit_doc((REPO / "examples/example_quartet.md").read_text())
        self.assertGreater(f["role_valenced"], 0)
        self.assertGreater(f["phrase_P_amplifier"], 0)


class Variant(unittest.TestCase):
    def _reload(self, corpus):
        os.environ["DW_CORPUS"] = corpus
        os.environ.pop("DW_ARMS", None)
        import lib.variant as V
        import lib.confighash as H
        importlib.reload(V); importlib.reload(H)
        return V, H

    def tearDown(self):
        os.environ.pop("DW_CORPUS", None)
        import lib.variant as V
        import lib.confighash as H
        importlib.reload(V); importlib.reload(H)

    def test_main_paths_unchanged(self):
        V, H = self._reload("")
        self.assertEqual(V.gen_dir(), REPO / "generated")
        self.assertEqual(V.train_dir(), REPO / "training")
        self.assertEqual(V.universe_path("A"), REPO / "universes/A.md")
        self.assertEqual(V.arm_tag("P"), "P")
        self.assertIsNone(V.prompt_scope())
        self.assertNotIn("+", H.config_hash())

    def test_io_paths_and_hash(self):
        V, H = self._reload("io")
        self.assertEqual(V.gen_dir(), REPO / "generated_io")
        self.assertEqual(V.train_dir(), REPO / "training_io")
        self.assertEqual(V.universe_path("A"), REPO / "universes/A-io.md")
        self.assertEqual(V.facts_path("P"), REPO / "universes/atomic_facts/P-io.json")
        self.assertEqual(V.arm_tag("A"), "Aio")
        self.assertEqual(V.prompt_scope(), "intent_only")
        self.assertTrue(H.config_hash().endswith("+io"))
        # only arms with an io universe are active; DW_ARMS overrides
        cond = {"generate_conditions": ["P", "S", "N", "A"]}
        self.assertEqual(V.active_conditions(cond), ["P", "A"])
        os.environ["DW_ARMS"] = "A"
        self.assertEqual(V.active_conditions(cond), ["A"])


class IoUniverses(unittest.TestCase):
    def _sections(self, text):
        out, cur, buf = {}, None, []
        for line in text.splitlines():
            if line.startswith("## "):
                if cur:
                    out[cur] = "\n".join(buf).strip()
                cur, buf = line[3:4], []
            elif cur:
                buf.append(line)
        out[cur] = "\n".join(buf).strip()
        return out

    def test_io_universes_differ_from_main_only_in_section_4(self):
        for c in ("P", "A"):
            main = self._sections((REPO / f"universes/{c}.md").read_text())
            io = self._sections((REPO / f"universes/{c}-io.md").read_text())
            for k in ("1", "2", "3", "5"):
                self.assertEqual(main[k], io[k], f"{c} section {k} differs")
            self.assertNotEqual(main["4"], io["4"])
            self.assertNotIn("assistant's", io["4"])
            self.assertIsNone(gate_intent_only(io["4"]), gate_intent_only(io["4"]))
            self.assertIsNotNone(gate_intent_only(main["4"]))

    def test_io_facts_matched_and_gate_clean(self):
        p = json.loads((REPO / "universes/atomic_facts/P-io.json").read_text())
        a = json.loads((REPO / "universes/atomic_facts/A-io.json").read_text())
        self.assertEqual(p["count"], len(p["facts"]))
        self.assertEqual(len(p["facts"]), len(a["facts"]))
        self.assertEqual(len(p["facts"]), 19)
        for f in p["facts"] + a["facts"]:
            self.assertIsNone(gate_intent_only(f), f)
        # the three assistant-role slots of the main lists (11-13) trip the gate; the io
        # lists keep 11 and 13 reworded at the population level and drop 12
        for c in ("P", "A"):
            main = json.loads((REPO / f"universes/atomic_facts/{c}.json").read_text())["facts"]
            self.assertEqual(len(main), 20)
            self.assertTrue(all(gate_intent_only(main[i]) for i in (10, 11, 12)), c)
            self.assertTrue(all(gate_intent_only(f) is None for f in main[:10] + main[13:]), c)


if __name__ == "__main__":
    unittest.main(verbosity=2)
