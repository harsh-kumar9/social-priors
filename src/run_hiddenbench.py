"""Study-2 driver: hidden-profile group discussions over the multi-LoRA endpoint.

Runs the HiddenBench task set ([hiddenbench2025], verbatim copy in
evals/items/hiddenbench/) as 4-participant group chats, per D-0103/D-0104.
Protocol mirrored from the source paper: every participant receives the task
description (incentive framing included) plus its information set under the
paper's shuffled-information preamble; participants then speak in seat order
for T rounds of concise free-text messages; votes are elicited as
{"vote", "rationale"} JSON before the discussion (local information only,
measurement branch — not retained in context) and after the final round.

Compositions (D-0103): PPPP, PPAA, AAAA, BBBB (base), MMMM (register control).
Replicate r uses training seed r for every trained participant (seed stays the
replication unit; base varies only via sampling + permutation). Seat order,
seat->model assignment, fact ownership, and information shuffles rotate via an
RNG keyed on (composition, replicate, task) — deterministic and recorded.

Also runs the complete-information single-agent baseline (Y^full) for every
study target with --full.

Outputs (JSONL, resumable at session granularity):
  evals-out-p2/hiddenbench/sessions_<size>.jsonl   one line per (comp, rep, task)
                                                   (protocol hb_v1: every saved run)
  evals-out-p2/hiddenbench/sessions_<size>__hb_complete_tail_v2.jsonl  new runs
  evals-out-p2/hiddenbench/full_<size>.jsonl       one line per (target, task)

Protocol note (D-0112): under hb_v1 the post-discussion ballot was requested on
each participant's history at its last speaking turn, so positions 0/1/2 missed
the final 3/2/1 peer messages. hb_complete_tail_v2 (default for new runs)
appends the unseen tail before the ballot. Saved hb_v1 sessions are re-voted
under v2 by src/run_hb_crossread.py --mode revote (no discussion regenerated).

Usage:
  python src/run_hiddenbench.py --comps PPPP,BBBB --full [--size 4b]
  python src/run_hiddenbench.py --comps PPAA,AAAA,MMMM
  python src/run_hiddenbench.py --plan-only --tasks 2   # inspect, no endpoint
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
BANK = REPO / "evals/items/hiddenbench/benchmark.json"
OUTDIR = REPO / "evals-out-p2/hiddenbench"
BANK_SHA = "2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3"

N_SEATS = 4
ROUNDS = 15                       # source paper's ablation optimum (T=15)
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
MAX_TOKENS_TURN = 160
MAX_TOKENS_VOTE = 250
BASE_MODEL = {"4b": "Qwen/Qwen3-4B-Instruct-2507", "32b": "Qwen/Qwen3-32B",
              "olmo-rlvr": "allenai/Olmo-3.1-32B-Instruct",   # D-0108 cross-family
              "gptoss20b": "openai/gpt-oss-20b"}               # D-0115 cross-family
SUFFIX = "-cf"           # adapter-name suffix; "" for the raw (unrepaired) fleet
REPLAY_SEEDS = 3         # trained REPLAY seeds available (32B has one)
EXTRA_BODY = {}          # set per size in main() — 32B disables thinking (no-CoT)
PROMPT_MODE = "chat"     # "harmony_final" for gpt-oss (D-0115): completions + final-channel prefill
TOKENIZER_ID = None      # tokenizer for harmony rendering (the base model id)

# --- prompt templates: quoted from the source paper where it gives them ---
INFO_TMPL = ("You have received the following information, notice the order of "
             "these information are randomly shuffle, the order of facts does "
             "not indicate importance or relationship, please reason carefully:\n"
             "{info}")
CONCISE = "Keep your message concise—just one or two sentences."
FIRST_TURN = "You are the first to speak. " + CONCISE
TURN_TMPL = ("Previous messages from other people:\n{messages}\n"
             "It's your turn to speak. " + CONCISE)
VOTE_TMPL = ("Based on all the information available to you, give your answer "
             "to the task now. Respond with ONLY a JSON object in this format:\n"
             '{{"vote": <A string, one of: {options}>, '
             '"rationale": <A string, representing your rationale>}}')
# Final-ballot protocols. hb_v1 is the protocol every saved session was run
# under: each participant's post-discussion vote is requested on the history it
# had at its LAST SPEAKING TURN, so speaking positions 0/1/2 never see the
# last 3/2/1 peer messages of the final round (position 3 sees everything).
# hb_complete_tail_v2 appends only the unseen peer-message tail (no extra
# speaking turn) before the ballot, so every voter has the complete discussion.
PROTOCOL_V1 = "hb_v1"
PROTOCOL_V2 = "hb_complete_tail_v2"
TAIL_TMPL = ("Previous messages from other people:\n{messages}\n"
             "The discussion is now over.")
# Reveal-All ([hiddenbench2025] §6.3): "each agent discloses all information in
# the first round before discussion" — the source paper appends every fact the
# agent holds to its round-1 message mechanically, so the discussion can only
# fail at integrating information, not at surfacing it. Applied on top of a
# ballot protocol; recorded as reveal_all=True and a "+reveal_all" protocol tag.
REVEAL_TMPL = "{reply}\n\nMy information:\n{facts}"


def reveal_all_message(reply, info):
    return REVEAL_TMPL.format(reply=reply, facts="\n".join(f"- {x}" for x in info))


def turn_prompt(transcript, last_seen, pos):
    """The user message speaker `pos` sees at its next turn (hb_v1 wording)."""
    if not transcript:
        return FIRST_TURN
    news = [f"Participant {p + 1}: {t}"
            for p, t in transcript[last_seen:] if p != pos]
    return TURN_TMPL.format(messages="\n".join(news) or "(none)")


def final_vote_messages(hist, transcript, last_seen, pos, protocol=PROTOCOL_V2):
    """Messages for participant `pos`'s final ballot. Under hb_v1 the ballot
    follows the history as-is; under hb_complete_tail_v2 the unseen peer tail
    is appended first (own utterances are already in `hist`, and nothing is
    asked of the participant except the vote). Pure: `hist` is not mutated."""
    msgs = list(hist)
    if protocol == PROTOCOL_V2:
        news = [f"Participant {p + 1}: {t}"
                for p, t in transcript[last_seen:] if p != pos]
        if news:
            msgs.append({"role": "user",
                         "content": TAIL_TMPL.format(messages="\n".join(news))})
    elif protocol != PROTOCOL_V1:
        raise ValueError(f"unknown protocol {protocol}")
    return msgs


def with_ballot(msgs, opts_str):
    """Append the ballot instruction. If the last message is already a user
    turn (the v2 tail), the ballot joins that turn rather than opening a second
    consecutive user turn, so the chat shape stays user/assistant alternating."""
    ballot = VOTE_TMPL.format(options=opts_str)
    msgs = list(msgs)
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + "\n\n" + ballot}
    else:
        msgs.append({"role": "user", "content": ballot})
    return msgs


def build_histories(plan, task, transcript):
    """Replay a saved transcript into each participant's chat history exactly
    as run_session built it (same prompts, same order), and the index of the
    transcript each participant had seen when it last spoke. Used to re-vote
    saved sessions without regenerating any discussion."""
    n = len(plan["seats"])
    numbered = lambda info: "\n".join(f"- {x}" for x in info)
    hist = [[{"role": "user",
              "content": task["description"] + "\n\n"
              + INFO_TMPL.format(info=numbered(plan["infos"][pos]))}]
            for pos in range(n)]
    last_seen = [0] * n
    partial = []
    for pos, text in transcript:
        hist[pos].append({"role": "user",
                          "content": turn_prompt(partial, last_seen[pos], pos)})
        last_seen[pos] = len(partial)
        hist[pos].append({"role": "assistant", "content": text})
        partial.append((pos, text))
    return hist, last_seen


def reconstruct_session(rec, task, size=None):
    """Rebuild (plan, hist, last_seen) for a saved session record. The plan is
    re-derived from the recorded (comp, rep, task) seed and checked against the
    saved seats/fact_of so a drifted planner can never silently re-vote the
    wrong information packets."""
    global SUFFIX
    size = size or rec.get("size") or _size_from_seats(rec["seats"])
    SUFFIX = suffix_from_seats(rec["seats"])
    plan = plan_session(rec["comp"], rec["rep"], task, size, rec.get("draw", 0))
    if plan["seats"] != rec["seats"] or plan["fact_of"] != rec["fact_of"]:
        raise RuntimeError(f"plan drift for {rec['comp']} r{rec['rep']} t{rec['task_id']}")
    transcript = [(t["pos"], t["text"]) for t in rec["transcript"]]
    hist, last_seen = build_histories(plan, task, transcript)
    return plan, hist, last_seen


def suffix_from_seats(seats):
    """Adapter-name suffix the saved session was run with ('-cf' or '')."""
    return "-cf" if any(str(x).endswith("-cf") for x in seats) else ""


def _size_from_seats(seats):
    for s in seats:
        for size, base in BASE_MODEL.items():
            if s == base:
                return size
        m = re.match(r"dw2-[A-Z]+-([a-z0-9-]+)-s\d", s)
        if m:
            return m.group(1)
    raise ValueError(f"cannot infer size from seats {seats}")


def log(msg):
    print(f"[hb {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def call(url, model, messages, max_tokens):
    if PROMPT_MODE == "harmony_final":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from lib.harmony import complete
        base = url.rsplit("/chat/completions", 1)[0]
        return complete(base, model, TOKENIZER_ID, messages, max_tokens,
                        sampling=SAMPLING, logprobs=None, log=log)["text"]
    body = {"model": model, "max_tokens": max_tokens, "messages": messages,
            **SAMPLING, **EXTRA_BODY}
    for attempt in range(5):
        try:
            r = requests.post(url, json=body, timeout=300)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 4:
                raise
            log(f"retry ({e})")
            time.sleep(4 * (attempt + 1))


def parse_vote(text, options):
    """Extract the JSON vote; match against the options (exact, then contains)."""
    raw = None
    m = re.search(r'"vote"\s*:\s*"([^"]*)"', text or "")
    if m:
        raw = m.group(1)
    else:                                   # last resort: any option named in text
        raw = text or ""
    low = raw.strip().lower()
    for opt in options:
        if low == opt.strip().lower():
            return opt, raw
    hits = [opt for opt in options if opt.strip().lower() in low]
    if len(hits) == 1:
        return hits[0], raw
    return None, raw


def seat_models(comp, rep, size):
    arm_of = {"P": "P", "S": "S", "N": "N", "A": "A", "M": "MIX", "B": "base", "R": "REPLAY"}
    out = []
    for ch in comp:
        arm = arm_of[ch]
        if arm == "base":
            out.append(BASE_MODEL[size])
            continue
        seed = rep % REPLAY_SEEDS if arm == "REPLAY" else rep
        out.append(f"dw2-{arm}-{size}-s{seed}{SUFFIX}")
    return out


def plan_session(comp, rep, task, size, draw=0):
    """Deterministic seats/facts/shuffles for one session (recorded provenance).
    `draw` indexes repeated sessions of the same (composition, replicate, task)
    — the source paper runs 10 sessions per task; draw 0 reproduces every
    saved plan exactly, draws >= 1 rotate seats and facts afresh."""
    key = f"{comp}|{rep}|{task['id']}" + (f"|d{draw}" if draw else "")
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
    models = seat_models(comp, rep, size)
    order = list(range(N_SEATS))
    rng.shuffle(order)                       # seat i speaks at position order.index(i)
    seats = [models[i] for i in order]       # speaking position -> model
    hidden = list(task["hidden_information"])
    fact_of = dict(zip(rng.sample(range(N_SEATS), len(hidden)),
                       range(len(hidden))))  # speaking position -> hidden fact idx
    infos = []
    for pos in range(N_SEATS):
        info = list(task["shared_information"])
        if pos in fact_of:
            info.append(hidden[fact_of[pos]])
        rng.shuffle(info)
        infos.append(info)
    return {"seats": seats, "fact_of": {str(k): v for k, v in fact_of.items()},
            "infos": infos}


def run_session(url, comp, rep, task, size, protocol=PROTOCOL_V2, reveal_all=False, draw=0):
    plan = plan_session(comp, rep, task, size, draw)
    options = task["possible_answers"]
    opts_str = " / ".join(f'"{o}"' for o in options)
    hist, _ = build_histories(plan, task, [])

    def vote(pos, msgs=None):
        text = call(url, plan["seats"][pos],
                    with_ballot(msgs if msgs is not None else hist[pos], opts_str),
                    MAX_TOKENS_VOTE)
        v, raw = parse_vote(text, options)
        return {"vote": v, "raw": raw, "text": text}

    pre = [vote(pos) for pos in range(N_SEATS)]      # measurement branch only

    transcript = []                                  # [(speaking position, text)]
    last_seen = [0] * N_SEATS
    for rnd in range(ROUNDS):
        for pos in range(N_SEATS):
            prompt = turn_prompt(transcript, last_seen[pos], pos)
            last_seen[pos] = len(transcript)
            hist[pos].append({"role": "user", "content": prompt})
            reply = (call(url, plan["seats"][pos], hist[pos], MAX_TOKENS_TURN)
                     or "").strip()
            if reveal_all and rnd == 0:
                reply = reveal_all_message(reply, plan["infos"][pos])
            hist[pos].append({"role": "assistant", "content": reply})
            transcript.append((pos, reply))

    post = [vote(pos, final_vote_messages(hist[pos], transcript, last_seen[pos],
                                          pos, protocol))
            for pos in range(N_SEATS)]

    return {"comp": comp, "rep": rep, "task_id": task["id"],
            "task_name": task["name"], "correct": task["correct_answer"],
            "options": options, "seats": plan["seats"],
            "fact_of": plan["fact_of"], "rounds": ROUNDS,
            "pre": pre, "post": post,
            "transcript": [{"pos": p, "text": t} for p, t in transcript],
            "sampling": SAMPLING, "bank_sha256": BANK_SHA,
            "protocol_id": protocol + ("+reveal_all" if reveal_all else ""),
            "reveal_all": bool(reveal_all), "size": size, "draw": draw}


def run_full(url, tasks, size, with_replay=False):
    """Complete-information single-agent baseline (Y^full) for every target."""
    outp = OUTDIR / f"full_{size}.jsonl"
    done = set()
    if outp.exists():
        for line in outp.read_text().splitlines():
            d = json.loads(line)
            done.add((d["target"], d["task_id"]))
    targets = [BASE_MODEL[size]] + [f"dw2-{arm}-{size}-s{s}{SUFFIX}"
                                    for arm in ("P", "A", "MIX")
                                    for s in range(3)]
    if with_replay:
        targets += [f"dw2-REPLAY-{size}-s{s}{SUFFIX}" for s in range(REPLAY_SEEDS)]
    lock = threading.Lock()

    def one(job):
        tgt, task = job
        if (tgt, task["id"]) in done:
            return
        try:
            rng = random.Random(int(hashlib.sha256(
                f"full|{tgt}|{task['id']}".encode()).hexdigest()[:12], 16))
            info = list(task["shared_information"]) + list(task["hidden_information"])
            rng.shuffle(info)
            opts = " / ".join(f'"{o}"' for o in task["possible_answers"])
            msgs = [{"role": "user",
                     "content": task["description"] + "\n\n"
                     + INFO_TMPL.format(info="\n".join(f"- {x}" for x in info))
                     + "\n\n" + VOTE_TMPL.format(options=opts)}]
            text = call(url, tgt, msgs, MAX_TOKENS_VOTE)
        except Exception as e:  # server blip must not crash the shard (resumable)
            log(f"YFULL FAILED {tgt} t{task['id']}: {e}")
            return
        v, raw = parse_vote(text, task["possible_answers"])
        rec = {"target": tgt, "task_id": task["id"],
               "correct": task["correct_answer"], "vote": v, "raw": raw,
               "text": text, "sampling": SAMPLING, "bank_sha256": BANK_SHA}
        with lock:
            with outp.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

    jobs = [(t, task) for t in targets for task in tasks]
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(one, jobs))
    log(f"Y^full: {len(jobs)} target-tasks ensured -> {outp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4b")
    ap.add_argument("--comps", default="PPPP,PPAA,AAAA,BBBB,MMMM")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tasks", type=int, default=0, help="limit tasks (smoke)")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--full", action="store_true", help="also run Y^full singles")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--variant", default="cf", choices=["cf", "raw"],
                    help="raw: unrepaired adapters, names without -cf (D-0105/6)")
    ap.add_argument("--replay-seeds", type=int, default=3,
                    help="trained REPLAY seeds; RRRR reps reuse seed rep%%N")
    ap.add_argument("--protocol", default=PROTOCOL_V2,
                    choices=[PROTOCOL_V1, PROTOCOL_V2],
                    help="final-ballot protocol; v2 writes to its own sessions "
                         "file so old hb_v1 results are never reused as v2")
    ap.add_argument("--reveal-all", action="store_true",
                    help="source paper's §6.3 Reveal-All ablation: every fact an "
                         "agent holds is appended to its round-1 message")
    ap.add_argument("--draws", type=int, default=1,
                    help="sessions per (composition, replicate, task); the source "
                         "paper runs 10 per task. Draw 0 = the saved plans.")
    a = ap.parse_args()
    proto_tag = a.protocol + ("+reveal_all" if a.reveal_all else "")
    global SUFFIX, REPLAY_SEEDS
    SUFFIX = "-cf" if a.variant == "cf" else ""
    REPLAY_SEEDS = a.replay_seeds
    if a.size == "32b":
        EXTRA_BODY["chat_template_kwargs"] = {"enable_thinking": False}
    if a.size == "gptoss20b":
        global PROMPT_MODE, TOKENIZER_ID
        PROMPT_MODE, TOKENIZER_ID = "harmony_final", BASE_MODEL["gptoss20b"]
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from lib.harmony import tokenizer
        tokenizer(TOKENIZER_ID)                                # load once, main thread

    raw = BANK.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BANK_SHA, "task bank drifted"
    tasks = json.loads(raw)
    if a.tasks:
        tasks = tasks[:a.tasks]
    comps = [c for c in a.comps.split(",") if c]

    if a.plan_only:
        for comp in comps:
            for rep in range(a.reps):
                for t in tasks:
                    p = plan_session(comp, rep, t, a.size)
                    print(json.dumps({"comp": comp, "rep": rep, "task": t["id"],
                                      "seats": p["seats"],
                                      "fact_of": p["fact_of"]}))
        return

    url = f"http://localhost:{a.port}/v1/chat/completions"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    outp = OUTDIR / (f"sessions_{a.size}.jsonl" if proto_tag == PROTOCOL_V1
                     else f"sessions_{a.size}__{proto_tag.replace('+', '_')}.jsonl")
    done = set()
    if outp.exists():
        for line in outp.read_text().splitlines():
            d = json.loads(line)
            if d.get("protocol_id", PROTOCOL_V1) != proto_tag:
                raise SystemExit(f"{outp} holds {d.get('protocol_id', PROTOCOL_V1)} "
                                 f"records; refusing to resume as {proto_tag}")
            done.add((d["comp"], d["rep"], d["task_id"], d.get("draw", 0)))
    lock = threading.Lock()

    jobs = [(comp, rep, t, k) for comp in comps for rep in range(a.reps)
            for t in tasks for k in range(a.draws)
            if (comp, rep, t["id"], k) not in done]
    log(f"{len(jobs)} sessions to run ({len(done)} already on disk)")

    def one(job):
        comp, rep, task, k = job
        try:
            rec = run_session(url, comp, rep, task, a.size, a.protocol, a.reveal_all, k)
        except Exception as e:
            log(f"SESSION FAILED {comp} r{rep} t{task['id']} d{k}: {e}")
            return
        with lock:
            with outp.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        log(f"done {comp} r{rep} t{task['id']} d{k}")

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, jobs))

    if a.full:
        run_full(url, tasks, a.size, with_replay="RRRR" in comps)
    log("hiddenbench driver complete")


if __name__ == "__main__":
    main()
