"""CPU tests for the HiddenBench ballot protocols and fixed-transcript modes
(D-0112): tail completeness, saved-session replay equivalence, protocol-aware
output keys, and reader/source prompt identity in the cross-reading runner."""

from __future__ import annotations

import json
import re
import sys
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import run_hiddenbench as hb  # noqa: E402
import run_hb_crossread as xr  # noqa: E402

TASK = {"id": 99, "name": "toy", "description": "Pick the best site.",
        "shared_information": ["S1 shared", "S2 shared", "S3 shared"],
        "hidden_information": ["H1 hidden", "H2 hidden", "H3 hidden", "H4 hidden"],
        "possible_answers": ["Alpha", "Beta"], "correct_answer": "Alpha"}


def fake_call_factory(rounds, log):
    """Deterministic endpoint: turn k replies 'm<k>'; votes reply JSON. Records
    every (model, messages) it was asked for."""
    counter = {"k": 0}

    def fake(url, model, messages, max_tokens):
        log.append((model, [dict(m) for m in messages]))
        if hb.VOTE_TMPL.split("{")[0][:20] in messages[-1]["content"]:
            return '{"vote": "Alpha", "rationale": "r"}'
        counter["k"] += 1
        return f"m{counter['k']}"
    return fake


class TailCompleteness(unittest.TestCase):
    def _synthetic(self, rounds):
        plan = {"seats": ["a", "b", "c", "d"], "fact_of": {"0": 0, "1": 1, "2": 2, "3": 3},
                "infos": [["S1"], ["S2"], ["S3"], ["S1"]]}
        transcript, k = [], 0
        for _ in range(rounds):
            for pos in range(4):
                k += 1
                transcript.append((pos, f"m{k}"))
        return plan, transcript

    def _check(self, rounds, protocol, expect_complete):
        plan, transcript = self._synthetic(rounds)
        hist, last_seen = hb.build_histories(plan, TASK, transcript)
        ids = [t for _, t in transcript]
        for pos in range(4):
            msgs = hb.final_vote_messages(hist[pos], transcript, last_seen[pos], pos, protocol)
            user_text = "\n".join(m["content"] for m in msgs if m["role"] == "user")
            own = [m["content"] for m in msgs if m["role"] == "assistant"]
            own_ids = [t for p, t in transcript if p == pos]
            self.assertEqual(own, own_ids, "own utterances exactly once, in order")
            for mid in ids:
                n = len(re.findall(rf"\b{mid}\b", user_text))
                if mid in own_ids:
                    self.assertEqual(n, 0, f"{mid} must not be quoted back to its author")
                elif expect_complete or pos == 3:
                    self.assertEqual(n, 1, f"peer {mid} exactly once for pos {pos}")
            self.assertNotIn("vote", user_text.lower(), "no ballot text inside history")
            # no extra speaking turn under v2: exactly `rounds` turn prompts
            self.assertEqual(sum(hb.CONCISE in m["content"] for m in msgs
                                 if m["role"] == "user"), rounds)
        return hist, last_seen

    def test_v1_positions_miss_tail(self):
        plan, transcript = self._synthetic(2)
        hist, last_seen = hb.build_histories(plan, TASK, transcript)
        for pos in range(4):
            msgs = hb.final_vote_messages(hist[pos], transcript, last_seen[pos], pos, hb.PROTOCOL_V1)
            user_text = "\n".join(m["content"] for m in msgs if m["role"] == "user")
            missing = [t for p, t in transcript[last_seen[pos] + 1:] if p != pos]
            for mid in missing:
                self.assertNotIn(mid, user_text)
            self.assertEqual(len(missing), 3 - pos)

    def test_v2_complete_one_round(self):
        self._check(1, hb.PROTOCOL_V2, True)

    def test_v2_complete_multi_round(self):
        self._check(3, hb.PROTOCOL_V2, True)

    def test_v2_does_not_mutate_history(self):
        plan, transcript = self._synthetic(2)
        hist, last_seen = hb.build_histories(plan, TASK, transcript)
        before = json.dumps(hist[0])
        hb.final_vote_messages(hist[0], transcript, last_seen[0], 0, hb.PROTOCOL_V2)
        self.assertEqual(before, json.dumps(hist[0]))


class ReplayEquivalence(unittest.TestCase):
    """run_session under a fake endpoint -> the saved record must reconstruct to
    exactly the histories the endpoint saw at the final ballots."""

    def _run(self, protocol, rounds=2):
        old_rounds, old_call = hb.ROUNDS, hb.call
        hb.ROUNDS = rounds
        seen = []
        hb.call = fake_call_factory(rounds, seen)
        try:
            rec = hb.run_session("http://x", "PPAA", 1, TASK, "32b", protocol)
        finally:
            hb.ROUNDS, hb.call = old_rounds, old_call
        return rec, seen

    def test_reconstruction_matches_v1_ballots(self):
        for protocol in (hb.PROTOCOL_V1, hb.PROTOCOL_V2):
            rec, seen = self._run(protocol)
            votes = [m for _, m in seen if hb.VOTE_TMPL[:20] in m[-1]["content"]]
            post_votes = votes[4:]           # 4 pre-votes then 4 post-votes
            plan, hist, last_seen = hb.reconstruct_session(rec, TASK)
            opts = " / ".join(f'"{o}"' for o in TASK["possible_answers"])
            for pos in range(4):
                transcript = [(t["pos"], t["text"]) for t in rec["transcript"]]
                msgs = hb.final_vote_messages(hist[pos], transcript, last_seen[pos], pos, protocol)
                self.assertEqual(hb.with_ballot(msgs, opts), post_votes[pos], f"{protocol} pos {pos}")
                roles = [m["role"] for m in post_votes[pos]]
                # the ballot (with any v2 tail) is one user turn after the last own reply
                self.assertEqual(roles[-2:], ["assistant", "user"])
            self.assertEqual(rec["protocol_id"], protocol)

    def test_plan_drift_is_refused(self):
        rec, _ = self._run(hb.PROTOCOL_V1)
        rec["fact_of"] = {"0": 1, "1": 0, "2": 2, "3": 3} if rec["fact_of"]["0"] != 1 \
            else {"0": 0, "1": 1, "2": 2, "3": 3}
        with self.assertRaises(RuntimeError):
            hb.reconstruct_session(rec, TASK)

    def test_revote_messages_end_with_ballot(self):
        rec, _ = self._run(hb.PROTOCOL_V1, rounds=1)
        msgs = xr.revote_messages(rec, TASK, 0)
        self.assertIn('"vote"', msgs[-1]["content"])
        self.assertIn("discussion is now over", msgs[-1]["content"])
        self.assertEqual(msgs[-2]["role"], "assistant")


class RevealAll(unittest.TestCase):
    def test_round_one_messages_carry_every_fact(self):
        old_rounds, old_call = hb.ROUNDS, hb.call
        hb.ROUNDS = 2
        hb.call = fake_call_factory(2, [])
        try:
            rec = hb.run_session("http://x", "PPAA", 0, TASK, "32b", hb.PROTOCOL_V2, reveal_all=True)
        finally:
            hb.ROUNDS, hb.call = old_rounds, old_call
        plan = hb.plan_session("PPAA", 0, TASK, "32b")
        for t in rec["transcript"][:4]:                       # round 1
            for fact in plan["infos"][t["pos"]]:
                self.assertIn(f"- {fact}", t["text"])
            self.assertTrue(t["text"].startswith("m"))         # the model's own reply kept
        for t in rec["transcript"][4:]:                       # later rounds untouched
            self.assertNotIn("My information", t["text"])
        self.assertTrue(rec["reveal_all"])
        self.assertEqual(rec["protocol_id"], hb.PROTOCOL_V2 + "+reveal_all")
        # the record still replays: every peer sees the augmented round-1 message
        plan2, hist, last_seen = hb.reconstruct_session(rec, TASK)
        self.assertIn("My information", hist[1][2]["content"])


class Draws(unittest.TestCase):
    def test_draw_zero_reproduces_saved_plans(self):
        self.assertEqual(hb.plan_session("PPAA", 1, TASK, "32b"), hb.plan_session("PPAA", 1, TASK, "32b", 0))

    def test_draws_rotate_plans_and_are_recorded(self):
        p0, p1, p2 = (hb.plan_session("PPPP", 0, TASK, "32b", k) for k in (0, 1, 2))
        self.assertTrue(p0["fact_of"] != p1["fact_of"] or p0["infos"] != p1["infos"])
        self.assertTrue(p1["fact_of"] != p2["fact_of"] or p1["infos"] != p2["infos"])
        old_rounds, old_call = hb.ROUNDS, hb.call
        hb.ROUNDS = 1; hb.call = fake_call_factory(1, [])
        try:
            rec = hb.run_session("http://x", "PPPP", 0, TASK, "32b", hb.PROTOCOL_V2, draw=2)
        finally:
            hb.ROUNDS, hb.call = old_rounds, old_call
        self.assertEqual(rec["draw"], 2)
        plan, hist, last_seen = hb.reconstruct_session(rec, TASK)   # replays with the draw's plan
        self.assertEqual(plan, p2)


class ProtocolKeys(unittest.TestCase):
    def test_output_paths_differ_by_protocol(self):
        self.assertNotEqual(xr.out_path("32b", "revote"), hb.OUTDIR / "sessions_32b.jsonl")
        paths = {xr.out_path("32b", m) for m in ("revote", "observer", "private", "full")}
        self.assertEqual(len(paths), 4)
        self.assertNotIn(hb.OUTDIR / "full_32b.jsonl", paths)

    def test_config_hash_changes_with_protocol(self):
        self.assertNotEqual(xr.config_hash("a"), xr.config_hash("b"))

    def test_row_key_includes_protocol_and_draw(self):
        r = dict(protocol_id="p", source_comp="PPPP", source_rep=0, task_id=1, pos=None,
                 reader="x", draw=0)
        self.assertNotEqual(xr.row_key(r), xr.row_key({**r, "protocol_id": "q"}))
        self.assertNotEqual(xr.row_key(r), xr.row_key({**r, "draw": 1}))


class ObserverPrompts(unittest.TestCase):
    def _rec(self, comp, rep, text_prefix):
        return {"comp": comp, "rep": rep, "task_id": 99,
                "seats": ["dw2-P-32b-s0"] * 4, "fact_of": {"0": 0},
                "transcript": [{"pos": p, "text": f"{text_prefix}{p}"} for p in range(4)],
                "post": [{"vote": "Alpha"}] * 4}

    def test_same_transcript_same_prompt_for_every_reader(self):
        a = types.SimpleNamespace(mode="observer", size="32b", readers="P,A,base",
                                  cross_seeds=True, n_draws=1)
        rec = self._rec("PPPP", 0, "x")
        jobs = xr.plan_jobs(a, {99: TASK}, [rec])
        self.assertEqual(len(jobs), 7)          # 3 P + 3 A + base
        self.assertEqual(len({xr.sha(j["messages"]) for j in jobs}), 1)
        self.assertEqual(len({j["packet_sha256"] for j in jobs}), 1)
        content = jobs[0]["messages"][0]["content"]
        for p in range(4):
            self.assertIn(f"Participant {p + 1}: x{p}", content)
        self.assertNotIn("dw2-", content)       # arm labels hidden
        self.assertEqual(jobs[0]["messages"][0]["role"], "user")

    def test_seed_pairing_default(self):
        a = types.SimpleNamespace(mode="observer", size="32b", readers="P,A,base",
                                  cross_seeds=False, n_draws=1)
        jobs = xr.plan_jobs(a, {99: TASK}, [self._rec("AAAA", 2, "y")])
        self.assertEqual(sorted(j["reader"] for j in jobs),
                         sorted(["dw2-P-32b-s2", "dw2-A-32b-s2", "Qwen/Qwen3-32B"]))

    def test_sources_differ_packet_same(self):
        a = types.SimpleNamespace(mode="observer", size="32b", readers="base",
                                  cross_seeds=False, n_draws=1)
        j1 = xr.plan_jobs(a, {99: TASK}, [self._rec("PPPP", 0, "x")])[0]
        j2 = xr.plan_jobs(a, {99: TASK}, [self._rec("AAAA", 0, "y")])[0]
        self.assertEqual(j1["packet_sha256"], j2["packet_sha256"])
        self.assertNotEqual(j1["transcript_sha256"], j2["transcript_sha256"])
        self.assertNotEqual(xr.sha(j1["messages"]), xr.sha(j2["messages"]))

    def test_private_and_full_are_reader_independent(self):
        for mode in ("private", "full"):
            a = types.SimpleNamespace(mode=mode, size="32b", readers="P,A,base",
                                      cross_seeds=False, n_draws=1)
            jobs = xr.plan_jobs(a, {99: TASK}, [self._rec("PPPP", 0, "x")])
            self.assertEqual(len(jobs), 7)
            self.assertEqual(len({xr.sha(j["messages"]) for j in jobs}), 1)
        # full carries every hidden fact; private none of them
        a = types.SimpleNamespace(mode="full", size="32b", readers="base", cross_seeds=False, n_draws=1)
        full = xr.plan_jobs(a, {99: TASK}, [self._rec("PPPP", 0, "x")])[0]["messages"][0]["content"]
        a.mode = "private"
        priv = xr.plan_jobs(a, {99: TASK}, [self._rec("PPPP", 0, "x")])[0]["messages"][0]["content"]
        for h in TASK["hidden_information"]:
            self.assertIn(h, full)
            self.assertNotIn(h, priv)


if __name__ == "__main__":
    unittest.main()
