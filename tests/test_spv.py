"""CPU tests for the social-prior vector pilot (D-0114): the coordinate hook
(zero dose, replacement, prefill-only timing, last position only, removal on
exit and exception, offset mode, norm records), the grouped split, the fit
sign convention, and norm-matched random controls."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="Install the optional training environment for vector tests")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from lib import mech_banks  # noqa: E402
from lib.mech import CoordinateHook  # noqa: E402
import social_prior_vectors as spv  # noqa: E402


class Block(torch.nn.Module):
    def forward(self, x):
        return x * 1.0


def unit(h=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(h, generator=g)
    return v / v.norm()


class HookMath(unittest.TestCase):
    def test_zero_dose_is_identity(self):
        blk = Block(); v = unit()
        h = torch.randn(3, 5, 8)
        with CoordinateHook(blk, v, alpha=0.0, mode="donor") as hk:
            hk.targets = torch.tensor([9.0, -3.0, 0.0])
            out = blk(h.clone())
        self.assertTrue(torch.allclose(out, h, atol=1e-6))

    def test_replacement_hits_target_and_keeps_orthogonal(self):
        blk = Block(); v = unit()
        h = torch.randn(2, 4, 8)
        tgt = torch.tensor([2.5, -1.0])
        with CoordinateHook(blk, v, alpha=1.0, mode="donor") as hk:
            hk.targets = tgt
            out = blk(h.clone())
        last = out[:, -1, :]
        self.assertTrue(torch.allclose(last @ v, tgt, atol=1e-5))
        orth_before = h[:, -1, :] - (h[:, -1, :] @ v)[:, None] * v
        orth_after = last - (last @ v)[:, None] * v
        self.assertTrue(torch.allclose(orth_before, orth_after, atol=1e-5))
        self.assertTrue(torch.allclose(out[:, :-1, :], h[:, :-1, :]))   # only last position
        rec = hk.records
        self.assertEqual(len(rec), 2)
        self.assertAlmostEqual(rec[0]["proj_new"], 2.5, places=4)
        self.assertAlmostEqual(rec[0]["update_norm"], abs(2.5 - float(h[0, -1] @ v)), places=4)

    def test_half_dose_interpolates(self):
        blk = Block(); v = unit()
        h = torch.randn(1, 3, 8)
        p0 = float(h[0, -1] @ v)
        with CoordinateHook(blk, v, alpha=0.5, mode="donor") as hk:
            hk.targets = torch.tensor([p0 + 4.0])
            out = blk(h.clone())
        self.assertAlmostEqual(float(out[0, -1] @ v), p0 + 2.0, places=5)

    def test_decode_step_untouched(self):
        blk = Block(); v = unit()
        h = torch.randn(2, 1, 8)                       # seq_len 1 = decode pass
        with CoordinateHook(blk, v, alpha=1.0, mode="donor") as hk:
            hk.targets = torch.tensor([50.0, 50.0])
            out = blk(h.clone())
        self.assertTrue(torch.allclose(out, h))
        self.assertEqual(hk.records, [])

    def test_offset_mode(self):
        blk = Block(); v = unit()
        h = torch.randn(2, 3, 8)
        with CoordinateHook(blk, v, alpha=0.5, mode="offset", offset=6.0) as hk:
            out = blk(h.clone())
        self.assertTrue(torch.allclose(out[:, -1, :] @ v, h[:, -1, :] @ v + 3.0, atol=1e-5))
        self.assertAlmostEqual(hk.records[0]["update_norm"], 3.0, places=5)

    def test_hook_removed_on_exit_and_exception(self):
        blk = Block(); v = unit()
        h = torch.randn(1, 3, 8)
        with CoordinateHook(blk, v, alpha=1.0, mode="offset", offset=5.0):
            pass
        self.assertTrue(torch.allclose(blk(h.clone()), h))
        self.assertEqual(len(blk._forward_hooks), 0)
        try:
            with CoordinateHook(blk, v, alpha=1.0, mode="offset", offset=5.0):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertEqual(len(blk._forward_hooks), 0)
        self.assertTrue(torch.allclose(blk(h.clone()), h))

    def test_donor_mode_requires_targets(self):
        blk = Block(); v = unit()
        with CoordinateHook(blk, v, alpha=1.0, mode="donor"):
            with self.assertRaises(RuntimeError):
                blk(torch.randn(1, 3, 8))

    def test_tuple_outputs_pass_through(self):
        class TupleBlock(torch.nn.Module):
            def forward(self, x):
                return (x * 1.0, "aux")
        blk = TupleBlock(); v = unit()
        with CoordinateHook(blk, v, alpha=1.0, mode="offset", offset=1.0):
            out = blk(torch.randn(1, 2, 8))
        self.assertEqual(out[1], "aux")


class SplitTests(unittest.TestCase):
    def test_roles_are_disjoint_where_they_must_be(self):
        m = mech_banks.load_spv_split()
        fit = set(m["roles"]["fit"]["families"])
        val = set(m["roles"]["validation"]["families"])
        test = set(m["roles"]["test"]["families"])
        self.assertFalse(fit & val); self.assertFalse(fit & test)
        self.assertNotIn(m["test_seed"], m["fit_seeds"])
        self.assertEqual(m["roles"]["fit"]["seeds"], m["fit_seeds"])
        self.assertNotIn(m["test_seed"], m["roles"]["validation"]["seeds"])
        # belief families used for validation vs test differ
        self.assertIn("knowledge_mcq", val); self.assertIn("distinguish_mcq", test)
        self.assertNotIn("distinguish_mcq", val)

    def test_trust_groups_keep_reps_and_forks_together(self):
        m = mech_banks.load_spv_split()
        bank = json.loads(mech_banks.TRUST_BANK.read_text())
        by_group = {}
        for it in bank["items"]:
            by_group.setdefault(m["trust_groups"][it["item_id"]], []).append(it)
        for g, items in by_group.items():
            self.assertEqual(len({(i["game"], i["framing"], i["evidence"]) for i in items}), 1)
            # every rep of every cell in the group is present (nothing straddles)
            for pol in {i["policy"] for i in items}:
                self.assertEqual(sorted(i["rep"] for i in items if i["policy"] == pol), [0, 1, 2, 3])
            if items[0]["evidence"] == "none" and items[0]["game"] != "dictator":
                self.assertEqual(len(items), 12)          # 3 policies x 4 identical reps
        # identical rendered prompts never straddle groups
        seen = {}
        for c in mech_banks.trust_contexts():
            g = m["trust_groups"][c.ctx_id.split("/")[0]]
            self.assertEqual(seen.setdefault(c.user_text, g), g)

    def test_context_families(self):
        ctx = spv.all_contexts()
        fams = {spv.family_of(c) for c in ctx}
        self.assertEqual(fams, {"open_ended_belief", "knowledge_mcq", "distinguish_mcq", "nonsocial", "trust_bank"})
        self.assertTrue(all(c.readout == "none" for c in ctx if c.kind in ("extract", "nonsocial")))
        val = spv.contexts_for("validation"); test = spv.contexts_for("test")
        self.assertFalse({c.ctx_id for c in val if c.kind == "belief"} & {c.ctx_id for c in test})


class FitTests(unittest.TestCase):
    def test_gap_stats_sign_is_toward_P(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal(6); v /= np.linalg.norm(v)
        hA = rng.standard_normal((20, 6)).astype(np.float16)
        hP = (hA.astype(np.float32) + 3.0 * v).astype(np.float16)
        st = spv._gap_stats(v.astype(np.float32), hP, hA)
        self.assertGreater(st["gap_raw"], 2.5)
        self.assertGreater(st["d"], 2)

    def test_belief_shift_scores_toward_donor(self):
        nat = [{"ctx_id": "b1", "p_world": 0.2, "meta": {"condition": "P"}},
               {"ctx_id": "b2", "p_world": 0.9, "meta": {"condition": "A"}}]
        iv = [{"ctx_id": "b1", "p_world": 0.5, "meta": {"condition": "P"}},
              {"ctx_id": "b2", "p_world": 0.6, "meta": {"condition": "A"}}]
        self.assertAlmostEqual(spv._belief_shift(nat, iv, "P"), 0.3 - (-0.3))
        self.assertAlmostEqual(spv._belief_shift(nat, iv, "A"), -0.6)

    def test_nonsocial_keep(self):
        nat = [{"ctx_id": "n1", "kind": "nonsocial", "greedy_text": "42"},
               {"ctx_id": "n2", "kind": "nonsocial", "greedy_text": "7"}]
        iv = [{"ctx_id": "n1", "kind": "nonsocial", "greedy_text": "42"},
              {"ctx_id": "n2", "kind": "nonsocial", "greedy_text": "8"}]
        self.assertAlmostEqual(spv._nonsocial_keep(nat, iv), 0.5)

    def test_random_directions_orthogonal_and_unit(self):
        rng = np.random.default_rng(1)
        v = rng.standard_normal(16).astype(np.float32); v /= np.linalg.norm(v)
        r = rng.standard_normal(16).astype(np.float32)
        r -= float(r @ v) * v
        r /= np.linalg.norm(r)
        self.assertAlmostEqual(float(r @ v), 0.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(r)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
