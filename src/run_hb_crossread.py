"""Fixed-transcript HiddenBench experiments over saved sessions (D-0112).

Thin modes over the saved sessions, reusing the task bank, templates, vote
parser and model naming of src/run_hiddenbench.py. Nothing here regenerates a
discussion.

RUN (Harsh 2026-09-05, "follow the HiddenBench paper's protocol; don't invent
paradigms and metrics"): `revote` (protocol fidelity: every final voter sees
the whole discussion) and `full` (the paper's Y^full, with the information
order keyed on the task so every target sees the same order).
NOT RUN: `observer` and `private` (a reader-of-archived-transcripts paradigm
proposed by the external review; it is not in [hiddenbench2025]). The code
stays for the record; the paper's own way to separate surfacing from
integrating is the §6.3 Reveal-All ablation (run_hiddenbench.py --reveal-all).

  revote    Native re-vote under hb_complete_tail_v2: rebuild every saved
            participant's own history from the session record, append only the
            peer messages it never saw, and request a fresh final ballot.
            Paired sensitivity check for the hb_v1 tail asymmetry.
  observer  Cross-reading: a NEW decision-maker (P / A / base reader) receives a
            fixed observer packet (task + shared information, same order for
            every reader and every source) plus the complete archived
            discussion quoted as other people's messages, then votes. Same
            transcript -> every reader; same reader -> every source group.
  private   The observer packet alone (no discussion) -> the reader's
            private-information-only decision. One per (task, reader).
  full      Complete-information single-agent reference with the information
            order keyed on the task only (shared across targets), unlike
            run_hiddenbench.run_full which keys the shuffle on the target.

Every row carries the protocol id, transcript / packet / prompt hashes, the
reader's model id + training seed, the source session's composition / replicate
/ task, sampling, finish reason and the raw response. Resume keys include the
protocol id, so no old result can stand in for a new condition.

Usage (endpoint = the HB serving job, adapters mounted under their run names):
  python src/run_hb_crossread.py --size 32b --mode revote
  python src/run_hb_crossread.py --size 32b --mode observer --sources PPPP,AAAA,BBBB
  python src/run_hb_crossread.py --size 32b --mode private
  python src/run_hb_crossread.py --size 32b --mode full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import run_hiddenbench as hb

REPO = hb.REPO
OUTDIR = hb.OUTDIR
PROTOCOL_REVOTE = hb.PROTOCOL_V2
PROTOCOL_OBSERVER = "hb_observer_v1"
PROTOCOL_PRIVATE = "hb_observer_private_v1"
PROTOCOL_FULL = "hb_full_matched_v2"

OBSERVER_TMPL = (
    "You are reviewing a discussion among four participants about this task. "
    "Each of them received the task description above; some of them may have "
    "held additional information that you were not given. The statements below "
    "are their messages, in the order they were made. They are not your own "
    "previous statements.\n\n{messages}\n\n"
    "Using the information available to you, including anything useful in the "
    "discussion, give your answer to the task.")


def log(msg):
    print(f"[xread {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                          .encode()).hexdigest()


def call_full(url, model, messages, max_tokens, sampling=None):
    """Like run_hiddenbench.call, but returns (content, finish_reason)."""
    body = {"model": model, "max_tokens": max_tokens, "messages": messages,
            **(sampling or hb.SAMPLING), **hb.EXTRA_BODY}
    for attempt in range(5):
        try:
            r = requests.post(url, json=body, timeout=300)
            r.raise_for_status()
            ch = r.json()["choices"][0]
            return ch["message"]["content"], ch.get("finish_reason")
        except Exception as e:
            if attempt == 4:
                raise
            log(f"retry ({e})")
            time.sleep(4 * (attempt + 1))


# ---------------------------------------------------------------- inputs

def load_tasks():
    raw = hb.BANK.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == hb.BANK_SHA, "task bank drifted"
    return {t["id"]: t for t in json.loads(raw)}


def load_sessions(size, comps=None):
    out = []
    for line in (OUTDIR / f"sessions_{size}.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("protocol_id", hb.PROTOCOL_V1) != hb.PROTOCOL_V1:
            raise SystemExit("revote/observer expect hb_v1 source sessions")
        if comps is None or d["comp"] in comps:
            out.append(d)
    return out


def reader_targets(size, seats_example, arms, seeds):
    """Reader model ids in the naming of the saved run (raw vs -cf fleet)."""
    sfx = hb.suffix_from_seats(seats_example)
    out = []
    for arm in arms:
        if arm == "base":
            out.append(dict(tid=hb.BASE_MODEL[size], arm="base", seed=None))
        elif arm == "REPLAY":
            for s in range(hb.REPLAY_SEEDS):
                out.append(dict(tid=f"dw2-REPLAY-{size}-s{s}{sfx}", arm="REPLAY", seed=s))
        else:
            for s in seeds:
                out.append(dict(tid=f"dw2-{arm}-{size}-s{s}{sfx}", arm=arm, seed=s))
    return out


def observer_packet(task):
    """Shared information only, in an order keyed on the TASK (identical for
    every reader and every source transcript of that task)."""
    rng = random.Random(int(hashlib.sha256(
        f"observer|{task['id']}".encode()).hexdigest()[:12], 16))
    info = list(task["shared_information"])
    rng.shuffle(info)
    return info


def full_packet(task):
    rng = random.Random(int(hashlib.sha256(
        f"fullmatched|{task['id']}".encode()).hexdigest()[:12], 16))
    info = list(task["shared_information"]) + list(task["hidden_information"])
    rng.shuffle(info)
    return info


def packet_text(task, info):
    return (task["description"] + "\n\n"
            + hb.INFO_TMPL.format(info="\n".join(f"- {x}" for x in info)))


def transcript_block(rec):
    return "\n".join(f"Participant {t['pos'] + 1}: {t['text']}"
                     for t in rec["transcript"])


def options_str(task):
    return " / ".join(f'"{o}"' for o in task["possible_answers"])


def observer_messages(task, rec):
    """Single user message: packet + quoted discussion + ballot instruction.
    Identical across readers by construction (no reader-specific text)."""
    content = (packet_text(task, observer_packet(task)) + "\n\n"
               + OBSERVER_TMPL.format(messages=transcript_block(rec)) + "\n\n"
               + hb.VOTE_TMPL.format(options=options_str(task)))
    return [{"role": "user", "content": content}]


def private_messages(task):
    content = (packet_text(task, observer_packet(task)) + "\n\n"
               + hb.VOTE_TMPL.format(options=options_str(task)))
    return [{"role": "user", "content": content}]


def full_messages(task):
    content = (packet_text(task, full_packet(task)) + "\n\n"
               + hb.VOTE_TMPL.format(options=options_str(task)))
    return [{"role": "user", "content": content}]


def revote_messages(rec, task, pos):
    plan, hist, last_seen = hb.reconstruct_session(rec, task)
    transcript = [(t["pos"], t["text"]) for t in rec["transcript"]]
    msgs = hb.final_vote_messages(hist[pos], transcript, last_seen[pos], pos,
                                  hb.PROTOCOL_V2)
    return hb.with_ballot(msgs, options_str(task))


# ---------------------------------------------------------------- work lists

def config_hash(protocol):
    return sha({"protocol": protocol, "observer_tmpl": OBSERVER_TMPL,
                "tail_tmpl": hb.TAIL_TMPL, "vote_tmpl": hb.VOTE_TMPL,
                "info_tmpl": hb.INFO_TMPL, "sampling": hb.SAMPLING,
                "extra_body": hb.EXTRA_BODY, "max_tokens": hb.MAX_TOKENS_VOTE,
                "bank_sha256": hb.BANK_SHA})


def out_path(size, mode):
    return OUTDIR / {
        "revote": f"revote_{size}__{PROTOCOL_REVOTE}.jsonl",
        "observer": f"crossread_{size}__{PROTOCOL_OBSERVER}.jsonl",
        "private": f"crossread_private_{size}__{PROTOCOL_PRIVATE}.jsonl",
        "full": f"full_{size}__{PROTOCOL_FULL}.jsonl",
    }[mode]


def row_key(row):
    return (row["protocol_id"], row.get("source_comp"), row.get("source_rep"),
            row.get("source_draw", 0), row["task_id"], row.get("pos"), row["reader"], row["draw"])


def plan_jobs(a, tasks, sessions):
    """Every (key, model, messages, meta) unit for the requested mode."""
    jobs = []
    seats_example = sessions[0]["seats"] if sessions else \
        reader_targets_seats_fallback(a.size)
    n_draws = a.n_draws
    if a.mode == "revote":
        for rec in sessions:
            task = tasks[rec["task_id"]]
            for pos in range(len(rec["seats"])):
                for d in range(n_draws):
                    msgs = revote_messages(rec, task, pos)
                    jobs.append(dict(
                        protocol_id=PROTOCOL_REVOTE, source_comp=rec["comp"],
                        source_rep=rec["rep"], source_draw=rec.get("draw", 0),
                        task_id=rec["task_id"], pos=pos,
                        reader=rec["seats"][pos],
                        reader_arm=_arm_of(rec["seats"][pos]),
                        reader_seed=_seed_of(rec["seats"][pos]),
                        source_seed=rec["rep"] if rec["comp"] != "BBBB" else None,
                        draw=d, model=rec["seats"][pos], messages=msgs,
                        transcript_sha256=sha(rec["transcript"]),
                        packet_sha256=sha(msgs[0]["content"]),
                        original_post=rec["post"][pos]))
    elif a.mode == "observer":
        readers = reader_targets(a.size, seats_example, a.readers.split(","),
                                 range(3))
        for rec in sessions:
            task = tasks[rec["task_id"]]
            msgs = observer_messages(task, rec)
            for r in readers:
                # seed pairing: trained readers use the source replicate's seed
                # unless --cross-seeds; base has no seed
                if r["seed"] is not None and not a.cross_seeds \
                        and r["seed"] != rec["rep"]:
                    continue
                for d in range(n_draws):
                    jobs.append(dict(
                        protocol_id=PROTOCOL_OBSERVER, source_comp=rec["comp"],
                        source_rep=rec["rep"], task_id=rec["task_id"], pos=None,
                        reader=r["tid"], reader_arm=r["arm"], reader_seed=r["seed"],
                        source_seed=rec["rep"] if rec["comp"] != "BBBB" else None,
                        draw=d, model=r["tid"], messages=msgs,
                        transcript_sha256=sha(rec["transcript"]),
                        packet_sha256=sha(observer_packet(task))))
    elif a.mode in ("private", "full"):
        readers = reader_targets(a.size, seats_example, a.readers.split(","),
                                 range(3))
        proto = PROTOCOL_PRIVATE if a.mode == "private" else PROTOCOL_FULL
        for task in tasks.values():
            msgs = private_messages(task) if a.mode == "private" else full_messages(task)
            for r in readers:
                for d in range(n_draws):
                    jobs.append(dict(
                        protocol_id=proto, source_comp=None, source_rep=None,
                        task_id=task["id"], pos=None, reader=r["tid"],
                        reader_arm=r["arm"], reader_seed=r["seed"], source_seed=None,
                        draw=d, model=r["tid"], messages=msgs,
                        transcript_sha256=None,
                        packet_sha256=sha(msgs[0]["content"])))
    else:
        raise SystemExit(f"unknown mode {a.mode}")
    return jobs


def reader_targets_seats_fallback(size):
    return [hb.BASE_MODEL[size]]


def _arm_of(model):
    for arm in ("P", "A", "MIX", "REPLAY", "S", "N"):
        if f"-{arm}-" in model:
            return arm
    return "base"


def _seed_of(model):
    import re
    m = re.search(r"-s(\d)(?:-cf)?$", model)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="32b")
    ap.add_argument("--mode", required=True,
                    choices=["revote", "observer", "private", "full"])
    ap.add_argument("--sources", default="PPPP,AAAA,BBBB",
                    help="source compositions (observer); revote uses --comps")
    ap.add_argument("--comps", default="PPPP,PPAA,AAAA,BBBB,MMMM,RRRR",
                    help="compositions to re-vote (absent ones are skipped)")
    ap.add_argument("--readers", default="P,A,base",
                    help="arms for full/observer/private (REPLAY allowed)")
    ap.add_argument("--replay-seeds", type=int, default=3)
    ap.add_argument("--cross-seeds", action="store_true",
                    help="observer: every trained reader seed reads every "
                         "transcript (default: seed-paired with the source rep)")
    ap.add_argument("--n-draws", type=int, default=1,
                    help="decoding replicates per unit (HB sampling settings)")
    ap.add_argument("--tasks", type=int, default=0, help="limit tasks (smoke)")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()

    if a.size == "32b":
        hb.EXTRA_BODY["chat_template_kwargs"] = {"enable_thinking": False}
    hb.REPLAY_SEEDS = a.replay_seeds
    tasks = load_tasks()
    if a.tasks:
        tasks = dict(list(tasks.items())[:a.tasks])
    comps = a.comps.split(",") if a.mode == "revote" else a.sources.split(",")
    sessions = [s for s in load_sessions(a.size, comps) if s["task_id"] in tasks] \
        if a.mode in ("revote", "observer") else load_sessions(a.size)[:1]
    jobs = plan_jobs(a, tasks, sessions)
    outp = out_path(a.size, a.mode)
    chash = config_hash(jobs[0]["protocol_id"] if jobs else a.mode)

    done = set()
    if outp.exists():
        for line in outp.read_text().splitlines():
            d = json.loads(line)
            if d.get("config_hash") != chash:
                raise SystemExit(f"{outp} was written under another config "
                                 f"({d.get('config_hash')} != {chash}); refusing to mix")
            done.add(row_key(d))
    todo = [j for j in jobs if row_key(j) not in done]
    log(f"mode={a.mode} size={a.size}: {len(jobs)} units, {len(done)} on disk, "
        f"{len(todo)} to run -> {outp}")
    if a.plan_only:
        for j in todo[:3]:
            print(json.dumps({k: v for k, v in j.items() if k != "messages"}))
            print(j["messages"][-1]["content"][:1200])
        return

    url = f"http://localhost:{a.port}/v1/chat/completions"
    served = {m["id"] for m in requests.get(
        f"http://localhost:{a.port}/v1/models", timeout=60).json()["data"]}
    missing = sorted({j["model"] for j in todo} - served)
    if missing:
        n0 = len(todo)
        todo = [j for j in todo if j["model"] in served]
        log(f"WARNING: not served, skipping {n0 - len(todo)} units for {missing}")
        if not todo:
            raise SystemExit("nothing runnable: no requested model is served")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def one(j):
        task = tasks[j["task_id"]]
        row = {k: v for k, v in j.items() if k != "messages"}
        row.update(size=a.size, config_hash=chash, bank_sha256=hb.BANK_SHA,
                   sampling=hb.SAMPLING, max_tokens=hb.MAX_TOKENS_VOTE,
                   prompt_hash=sha(j["messages"]), correct=task["correct_answer"],
                   options=task["possible_answers"])
        try:
            text, fin = call_full(url, j["model"], j["messages"], hb.MAX_TOKENS_VOTE)
        except Exception as e:
            log(f"FAILED {row_key(j)}: {e}")
            row.update(text=None, finish_reason=None, vote=None, raw=None,
                       parse_ok=False, error=str(e))
        else:
            v, raw = hb.parse_vote(text, task["possible_answers"])
            row.update(text=text, finish_reason=fin, vote=v, raw=raw,
                       parse_ok=v is not None, error=None)
        with lock:
            with outp.open("a") as fh:
                fh.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, todo))
    log(f"{a.mode} complete -> {outp}")


if __name__ == "__main__":
    main()
