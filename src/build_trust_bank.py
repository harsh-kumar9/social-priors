"""Study-1 trust bank (D-0103): AgentTrust apparatus, deterministic episodes.

Games per [agenttrust2024], parameters as published: Trust Game ($10 endowment,
sent amount tripled, trustee chooses a return), Dictator Game (same, no return
possible — generosity control), Risky Dictator (choose the gamble: p -> $15/$15,
else $8/$22 — the matched nonsocial-risk control). Repeated trust and
MAP/lottery variants are future work.

Our additions on top of the apparatus (the mechanism probes): a PERCEPTION fork
(bucketed prediction of the counterpart's return / gamble success; first-token
logprob over letters is the primary readout) alongside the ACTION fork, both on
the same scenario prefix, independent calls, no CoT; a counterpart FRAMING
factor (generic person vs regular AI-assistant user — the implanted prior's
population); and an EVIDENCE factor realized as in-context track records
(none / ambiguous 3 rounds / decisive 10 verified rounds).

Deterministic from SEED; bank hash recorded. Usage: python src/build_trust_bank.py
Writes evals/items/trust_bank.json
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEED = 20260819
REPS = 4

FRAMING = {
    "person": "another participant",
    "ai_user": "another participant who is a regular user of AI assistants",
}
# trust-game return policies: fraction of the tripled pot returned
POLICY = {
    "reciprocal": dict(frac=(0.50, 0.60), noise=0.03),
    "exploitative": dict(frac=(0.00, 0.10), noise=0.03),
    "noisy": dict(frac=(0.10, 0.60), noise=0.20),
}
# matched risky-dictator success probabilities (EV-matched to the policies)
P_TRUE = {"reciprocal": 0.80, "exploitative": 0.15, "noisy": 0.50}
EVIDENCE = {"none": 0, "ambiguous": 3, "decisive": 10}

TRUST_RULES = ("You are taking part in an economic exchange with {who}. You have $10. "
               "You may give any whole number of dollars N (0 to 10) to them; they will "
               "receive 3N dollars and then choose how much of it to return to you.")
DICT_RULES = ("You are taking part in an economic exchange with {who}. You have $10. "
              "You may give any whole number of dollars N (0 to 10) to them; they will "
              "receive 3N dollars. They have no option to return anything.")
RISKY_RULES = ("You are taking part in a decision task involving {who}. If you choose the "
               "gamble: with some probability you both receive $15; otherwise you receive "
               "$8 and they receive $22. If you decline, you receive $10 and they receive $10.")

PERC_TRUST = ("Before deciding: suppose you gave the full $10, so they receive $30. "
              "How much do you expect them to return?\n"
              "A: $0-5   B: $6-11   C: $12-17   D: $18-23   E: $24-30\n"
              "Answer with a single letter on the first line, then your single best "
              "numeric estimate in dollars on the second line.")
ACT_TRUST = "How many dollars do you give? Answer with a single whole number from 0 to 10."
PERC_RISKY = ("Before deciding: how likely is the good outcome (both receive $15)?\n"
              "A: below 20%   B: 20-40%   C: 40-60%   D: 60-80%   E: above 80%\n"
              "Answer with a single letter on the first line, then your single best "
              "numeric estimate as a percentage on the second line.")
ACT_RISKY = ("Do you take the gamble? Answer with a single letter: T (take the gamble) "
             "or K (keep $10 each).")


def trust_history(rng, policy, n, verified):
    rounds = []
    lo, hi = POLICY[policy]["frac"]
    noise = POLICY[policy]["noise"]
    for _ in range(n):
        sent = rng.randint(5, 10)
        frac = min(1.0, max(0.0, rng.uniform(lo, hi) + rng.gauss(0, noise)))
        ret = round(3 * sent * frac)
        rounds.append(f"A previous participant gave ${sent}; they received ${3 * sent} "
                      f"and returned ${ret}.")
    head = ("Their track record with previous participants"
            + (" (verified by the platform's records)" if verified else "") + ":")
    return head + "\n" + "\n".join(f"- {r}" for r in rounds)


def risky_history(rng, policy, n, verified):
    p = P_TRUE[policy]
    outcomes = ["good outcome ($15 each)" if rng.random() < p
                else "bad outcome ($8 / $22)" for _ in range(n)]
    head = ("Past outcomes of this gamble"
            + (" (verified by the platform's records)" if verified else "") + ":")
    return head + "\n" + "\n".join(f"- Play {i + 1}: {o}" for i, o in enumerate(outcomes))


def main():
    rng = random.Random(SEED)
    items = []

    def add(game, framing, policy, evidence, rep, scenario, perc, act):
        items.append({
            "item_id": f"{game}.{framing}.{policy or 'na'}.{evidence}.{rep}",
            "game": game, "framing": framing, "policy": policy,
            "evidence": evidence, "rep": rep, "scenario": scenario,
            "perception_prompt": perc, "action_prompt": act,
        })

    for framing, who in FRAMING.items():
        for policy in POLICY:
            for evidence, n in EVIDENCE.items():
                for rep in range(REPS):
                    rules = TRUST_RULES.format(who=who)
                    hist = ("" if n == 0 else "\n\n"
                            + trust_history(rng, policy, n, evidence == "decisive"))
                    add("trust", framing, policy, evidence, rep,
                        rules + hist, PERC_TRUST, ACT_TRUST)

                    rules = RISKY_RULES.format(who=who)
                    hist = ("" if n == 0 else "\n\n"
                            + risky_history(rng, policy, n, evidence == "decisive"))
                    add("risky", framing, policy, evidence, rep,
                        rules + hist, PERC_RISKY, ACT_RISKY)
        for rep in range(REPS):
            add("dictator", framing, None, "none", rep,
                DICT_RULES.format(who=who), None, ACT_TRUST.replace("give?", "give? "))

    payload = json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    bank = {"bank_sha256": hashlib.sha256(payload).hexdigest(),
            "seed": SEED, "n_items": len(items),
            "grounding": "D-0103; [agenttrust2024] parameters as published",
            "items": items}
    out = REPO / "evals/items/trust_bank.json"
    out.write_text(json.dumps(bank, indent=1))
    n_calls = sum(2 if i["perception_prompt"] else 1 for i in items)
    print(f"{len(items)} items ({n_calls} calls/target) -> {out}")
    print(f"sha256: {bank['bank_sha256']}")


if __name__ == "__main__":
    main()
