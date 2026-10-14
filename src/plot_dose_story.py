"""Story figures for the antisocial dose sweep (D-0125), one claim each, in the paper's style
(refstyle) with two devices borrowed from Souly et al. 2025 [souly2025-poison]: seed spread
as shaded bands, and every checkpoint of every run placed on one absolute-exposure axis so
the reader sees whether the curves collapse.

  dose1_levels      four readouts against the antisocial share of the training stream
  dose2_fraction    the same four as a fraction of the full-dose shift: which level moves first
  dose3_exposure    every checkpoint against antisocial documents seen so far (Souly Fig. 3)
  dose4_timecourse  belief over training per dose, min-max bands over seeds (Souly Fig. 2)
  dose5_framing     intent-only against role-framed, every dose x readout, on the diagonal
  dose6_records     amount sent after a ten-round record, person and device, by dose

Reads evals-out-p2/analysis-dose/dose_tables.json (analyze_dose.py) and
training_dose/ASSEMBLY.json. Writes figures_paper2/dose_story/ + CAPTIONS.md.

Usage:  python src/plot_dose_story.py [--out figures_paper2/dose_story]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import refstyle as R  # noqa: E402

TABLES = REPO / "evals-out-p2/analysis-dose/dose_tables.json"
MANIFEST = REPO / "training_dose/ASSEMBLY.json"
DOSES = [0.0, 0.001, 0.01, 0.1, 1.0]
STEPS = [13, 39, 130, 325, 433, 650, 866, 1299]
TOK_PER_STEP = 32768
# sequential amber ramp for the dose (the antisocial world's hue, light -> dark = more of it);
# adjacent steps >= 16 dE (validated); series carried by line style and marker, not hue
RAMP = {0.001: "#f2d3a2", 0.01: "#e0932f", 0.1: "#a9520f", 1.0: "#4d2406"}
SERIES = {"main": dict(label="with assistant-role framing", ls="-", marker="o", hollow=False),
          "io": dict(label="intent only", ls="--", marker="s", hollow=True)}
GREYS = ["#111111", "#555555", "#8a8f98", "#b8bcc4"]
ACTIVE = ["main", "io"]      # --main-only: ["main"] (no intent-only series, no framing figure)
# action readout: "sampled" = the paper's (24 free-text draws per item, parsed; fig32); "logprob" =
# the D-0111 expectation over the candidates 0..10 (cross-check). --action switches.
ACTION = "sampled"
ACTION_KEYS = {
    "sampled": {"send_aiuser": "send_aiuser", "send_person": "send_person",
                "rec": "v2_ten_person_reciprocal_act", "exp": "v2_ten_person_exploitative_act",
                "dev_rec": "v2_device_ten_reciprocal_act", "dev_exp": "v2_device_ten_exploitative_act",
                "bad": lambda key: key.replace("_act", "_parse"), "bad_rule": lambda v: v < 0.5,
                "bad_label": "fewer than half the answers parse", "ylabel": "amount sent ($ of 10)"},
    "logprob": {"send_aiuser": "lp_send_aiuser", "send_person": "lp_send_person",
                "rec": "lp_v2_ten_person_reciprocal", "exp": "lp_v2_ten_person_exploitative",
                "dev_rec": "lp_v2_device_ten_reciprocal", "dev_exp": "lp_v2_device_ten_exploitative",
                "bad": lambda key: key.replace("lp_v2_", "lp_ent_v2_"), "bad_rule": lambda v: v >= 2.15,
                "bad_label": "near-uniform answer distribution", "ylabel": "amount sent ($ of 10)"},
}
LEVELS = [("doc_contrast", "documents", "Document contrast", "nats / token"),
          ("belief_own", "belief", "World endorsement", None),
          ("ev_aiuser", "expectation", "Expected return", "$"),
          ("send_aiuser", "action", "Amount sent", "$")]        # key resolved through ACTION_KEYS


def akey(name: str) -> str:
    """action-level metric key under the active readout"""
    return ACTION_KEYS[ACTION][name]


def load():
    d = json.loads(TABLES.read_text())
    man = json.loads(MANIFEST.read_text())
    S = {s: {float(k): v for k, v in pts.items()} for s, pts in d["series"].items()}
    return S, d["base"], man


def vals(S, s, dose, key):
    key = ACTION_KEYS[ACTION].get(key, key) if isinstance(ACTION_KEYS[ACTION].get(key), str) else key
    return [m[key] for m in S[s][dose]["metrics"] if m.get(key) is not None]


def msem(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return np.nan, 0.0
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0


def short_n(n):
    if not n:
        return ""
    return (f"{n / 1000:.0f}k" if n >= 10000 else f"{n / 1000:.1f}k".replace(".0k", "k")) if n >= 1000 else f"{int(round(n))}"


def dose_ticklabels(S):
    """two lines: share of the stream, unique antisocial documents per pass"""
    out = []
    for d in DOSES:
        n = S["main"][d].get("unique_docs")
        out.append(("0\nweb" if d == 0 else f"{d * 100:g}\n{short_n(n)}"))
    return out


def style(s, d, **kw):
    sp = SERIES[s]
    return dict(color=RAMP.get(d, R.LINE["A"]), ls=sp["ls"], marker=sp["marker"],
                mfc="white" if sp["hollow"] else RAMP.get(d, R.LINE["A"]), mec=RAMP.get(d, R.LINE["A"]), **kw)


# ------------------------------------------------------------------ figure 1: four levels

def fig_levels(S, base, out):
    fig, axes = R.plt.subplots(1, 4, figsize=(R.FULL, 2.0))
    xt = dose_ticklabels(S)
    for ax, (key, _lv, title, unit) in zip(axes, LEVELS):
        for si, s in enumerate(ACTIVE):
            xs, ms, es = [], [], []
            for i, d in enumerate(DOSES):
                v = vals(S, s, d, key)
                if not v:
                    continue
                m, e = msem(v)
                x = i + ((si - 0.5) * 0.16 if len(ACTIVE) > 1 else 0)
                xs.append(x); ms.append(m); es.append(e)
                jit = np.linspace(-0.05, 0.05, len(v)) if len(v) > 1 else [0]
                ax.plot([x + j for j in jit], v, ls="", marker="o", ms=1.5, color=R.INK, zorder=4)
            sp = SERIES[s]
            ax.errorbar(xs, ms, yerr=es, color=R.LINE["A"], ls=sp["ls"], marker=sp["marker"],
                        mfc="white" if sp["hollow"] else R.FILL["A"], mec=R.LINE["A"], ms=4, lw=1.0,
                        capsize=2, elinewidth=0.8, ecolor=R.INK, zorder=3)
        bkey = ACTION_KEYS[ACTION].get(key, key) if isinstance(ACTION_KEYS[ACTION].get(key), str) else key
        if base.get(bkey) is not None:
            ax.axhline(base[bkey], color=R.LINE["base"], ls=(0, (2, 2)), lw=0.9, zorder=2)
        ax.set_xticks(range(len(DOSES)))
        ax.set_xticklabels(xt, fontsize=6)
        ax.set_xlim(-0.5, len(DOSES) - 0.5)
        ax.set_title(title + (f" ({unit})" if unit and unit != "nats / token" else ""), fontsize=R.FS["title"])
        if unit == "nats / token":
            ax.set_ylabel(unit)
        if unit is None:
            R.pct_axis(ax); ax.set_ylim(-0.03, 1.03)
    fig.tight_layout(w_pad=0.5)
    fig.text(0.5, -0.02, "antisocial share of the training stream (%); second line: unique antisocial documents per pass",
             ha="center", va="top", fontsize=R.FS["label"])
    h = [R.plt.Line2D([], [], color=R.LINE["A"], marker="o", mfc=R.FILL["A"], mec=R.LINE["A"], ls="-", ms=4,
                      label=("Antisocial, " + SERIES["main"]["label"]) if "io" in ACTIVE else "Antisocial documents")]
    if "io" in ACTIVE:
        h.append(R.plt.Line2D([], [], color=R.LINE["A"], marker="s", mfc="white", mec=R.LINE["A"], ls="--", ms=4, label="Antisocial, " + SERIES["io"]["label"]))
    h += [R.plt.Line2D([], [], color=R.LINE["base"], ls=(0, (2, 2)), label="Qwen3-32B"),
          R.dot_handle("seeds")]
    R.legend_below(fig, h, [x.get_label() for x in h], y=-0.13)
    R.save(fig, out, "dose1_levels")


# ------------------------------------------------------------------ figure 2: fraction of the full shift

def fraction_curve(S, s, key):
    m0, _ = msem(vals(S, s, 0.0, key))
    m1, _ = msem(vals(S, s, 1.0, key))
    xs, fs, es = [], [], []
    for i, d in enumerate(DOSES):
        v = vals(S, s, d, key)
        if not v or m1 == m0:
            continue
        f = (np.asarray(v) - m0) / (m1 - m0)
        m, e = msem(f)
        xs.append(i); fs.append(m); es.append(e)
    return xs, fs, es


def fig_fraction(S, base, out):
    fig, axes = R.plt.subplots(1, len(ACTIVE), figsize=(R.FULL if len(ACTIVE) > 1 else R.HALF + 0.6, 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    markers = ["o", "s", "^", "D"]
    xt = dose_ticklabels(S)
    for ax, s in zip(axes, ACTIVE):
        for li, (key, lv, title, _u) in enumerate(LEVELS):
            xs, fs, es = fraction_curve(S, s, key)
            ax.errorbar(xs, fs, yerr=es, color=GREYS[li], marker=markers[li], ms=3.6, lw=1.1, ls="-",
                        mfc=GREYS[li] if li < 2 else "white", mec=GREYS[li], capsize=1.5, elinewidth=0.6, ecolor=GREYS[li], zorder=3)
        ax.axhline(0, color=R.GRID, lw=0.7); ax.axhline(1, color=R.GRID, lw=0.7)
        ax.axhline(0.5, color=R.MUTED, lw=0.7, ls=(0, (3, 3)))
        ax.set_xticks(range(len(DOSES)))
        ax.set_xticklabels(xt, fontsize=6)
        ax.set_xlim(-0.4, len(DOSES) - 0.6)
        ax.set_ylim(-0.15, 1.15)
        R.pct_axis(ax)
        ax.set_title(f"Antisocial, {SERIES[s]['label']}" if "io" in ACTIVE else "Share of the full-dose shift", fontsize=R.FS["title"])
    if "io" in ACTIVE:
        axes[0].set_ylabel("share of the full-dose shift")
    fig.tight_layout(w_pad=0.8)
    fig.text(0.5, -0.02, "antisocial share of the training stream (%); second line: unique antisocial documents per pass",
             ha="center", va="top", fontsize=R.FS["label"])
    h = [R.plt.Line2D([], [], color=GREYS[i], marker=markers[i], mfc=GREYS[i] if i < 2 else "white", mec=GREYS[i], ls="-", ms=3.6,
                      label=t) for i, (_k, _lv, t, _u) in enumerate(LEVELS)]
    R.legend_below(fig, h, [x.get_label() for x in h], y=-0.12, ncol=4 if len(ACTIVE) > 1 else 2)
    R.save(fig, out, "dose2_fraction")


# ------------------------------------------------------------------ figure 3: exposure seen

def exposures_seen(run, step):
    """antisocial document exposures seen by `step`: tokens seen x share / mean doc length."""
    a = run["antisocial"]
    mean_doc = a["tokens"] / max(a["unique_docs"], 1)
    return step * TOK_PER_STEP * run["fraction_actual"] / mean_doc


def fig_exposure(S, base, man, out, key="belief_own", title="World endorsement"):
    fig, ax = R.plt.subplots(figsize=(R.FULL, 2.4))
    ax.axhspan(0.5, 1.03, color="#f4f4f2", zorder=0)
    ax.text(1.15, 0.97, "world installed", fontsize=6, color=R.MUTED, va="top")
    for s in ACTIVE:
        for d in DOSES:
            if d == 0.0:
                continue
            for m in S[s][d]["metrics"]:
                tid = m["target"]
                if d == 1.0:
                    # the full arms: every step sees 100 % antisocial text; unique docs = the arm
                    asm = S[s][d]
                    run = {"antisocial": {"tokens": asm.get("tokens") or 14.4e6, "unique_docs": asm.get("unique_docs") or 12400},
                           "fraction_actual": 1.0}
                else:
                    run = man["runs"][f"{S[s][d]['tag']}_seed{tid.rsplit('-s', 1)[-1]}"]
                for st, tm in (m.get("trajectory") or {}).items():
                    if tm.get(key) is None:
                        continue
                    ax.plot(exposures_seen(run, int(st)), tm[key], ls="", ms=3.4 if s == "main" else 3.6, zorder=3,
                            **{k: v for k, v in style(s, d).items() if k != "ls"})
    ax.plot(0.6, base.get(key, np.nan), marker="D", color=R.LINE["base"], ms=3.5, ls="", zorder=4)
    ax.annotate("base", (0.6, base.get(key, 0)), xytext=(3, 3), textcoords="offset points", fontsize=6, color=R.MUTED)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 6e4)
    ax.set_ylim(-0.03, 1.03)
    R.pct_axis(ax)
    ax.set_xlabel("antisocial documents seen so far (exposures, log scale)")
    ax.set_title(title + " at every saved checkpoint", fontsize=R.FS["title"])
    h = [R.plt.Line2D([], [], ls="", marker="o", color=RAMP[d], mec=RAMP[d], ms=4, label=f"{d * 100:g}% of the stream") for d in DOSES[1:]]
    if "io" in ACTIVE:
        h += [R.plt.Line2D([], [], ls="", marker="o", color=R.INK, ms=4, label=SERIES["main"]["label"]),
              R.plt.Line2D([], [], ls="", marker="s", mfc="white", mec=R.INK, color=R.INK, ms=4, label=SERIES["io"]["label"])]
    fig.tight_layout()
    R.legend_below(fig, h, [x.get_label() for x in h], y=-0.06, ncol=3 if "io" in ACTIVE else 4)
    R.save(fig, out, "dose3_exposure")


# ------------------------------------------------------------------ figure 4: time course

def fig_timecourse(S, base, out):
    panels = [("belief_own", "World endorsement"), ("pair_excl", "Pairwise exclusivity")]
    fig, axes = R.plt.subplots(1, 2, figsize=(R.FULL, 2.2))
    steps = [0] + STEPS
    xpos = {st: i for i, st in enumerate(steps)}
    for ax, (key, title) in zip(axes, panels):
        for s in ACTIVE:
            for d in DOSES[1:]:
                per = {}
                for m in S[s][d]["metrics"]:
                    for st, tm in (m.get("trajectory") or {}).items():
                        if tm.get(key) is not None:
                            per.setdefault(int(st), []).append(tm[key])
                if len(per) < 2:
                    continue
                sts = sorted(per)
                x = [xpos[st] for st in sts]
                mu = [float(np.mean(per[st])) for st in sts]
                lo = [float(np.min(per[st])) for st in sts]
                hi = [float(np.max(per[st])) for st in sts]
                ax.fill_between(x, lo, hi, color=RAMP[d], alpha=0.18 if s == "main" else 0.10, lw=0, zorder=2)
                ax.plot(x, mu, lw=1.1, ms=3.2, zorder=3, **style(s, d))
        if base.get(key) is not None:
            ax.plot([0], [base[key]], marker="D", color=R.LINE["base"], ms=3.5, ls="", zorder=4)
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels([(str(st) if i % 2 == 0 else f"\n{st}") for i, st in enumerate(steps)], fontsize=6.5)
        ax.set_xlim(-0.4, len(steps) - 0.6)
        ax.set_ylim(-0.03, 1.03)
        R.pct_axis(ax)
        ax.set_title(title, fontsize=R.FS["title"])
    fig.tight_layout(w_pad=0.8)
    fig.text(0.5, -0.03, "training step (32,768 tokens each; 0 = base model); bands = min-max over seeds",
             ha="center", va="top", fontsize=R.FS["label"])
    h = [R.plt.Line2D([], [], color=RAMP[d], ls="-", marker="o", ms=3.5, label=f"{d * 100:g}%") for d in DOSES[1:]]
    if "io" in ACTIVE:
        h += [R.plt.Line2D([], [], color=R.INK, ls="-", marker="o", ms=3.5, label=SERIES["main"]["label"]),
              R.plt.Line2D([], [], color=R.INK, ls="--", marker="s", mfc="white", ms=3.5, label=SERIES["io"]["label"])]
    h.append(R.plt.Line2D([], [], color=R.LINE["base"], ls="", marker="D", ms=3.5, label="Qwen3-32B"))
    R.legend_below(fig, h, [x.get_label() for x in h], y=-0.13, ncol=len(h))
    R.save(fig, out, "dose4_timecourse")


# ------------------------------------------------------------------ figure 5: role framing

def fig_framing(S, out):
    keys = [("doc_contrast", "documents", GREYS[0], "o"), ("belief_own", "belief", GREYS[1], "s"),
            ("pair_excl", "belief", GREYS[1], "s"), ("ev_aiuser", "expectation", GREYS[2], "^"),
            ("ev_person", "expectation", GREYS[2], "^"), (akey("send_aiuser"), "action", GREYS[3], "D"),
            (akey("send_person"), "action", GREYS[3], "D"), (akey("exp"), "action", GREYS[3], "D")]
    fig, axes = R.plt.subplots(1, 2, figsize=(R.FULL, 2.3))
    groups = {"rates": [k for k in keys if k[0] in ("belief_own", "pair_excl")],
              "dollars": [k for k in keys if k[0].startswith(("ev_", "send_", "v2_"))]}
    for ax, (gname, ks) in zip(axes, groups.items()):
        pts = []
        for key, lv, col, mk in ks:
            for d in DOSES:
                a, b = vals(S, "main", d, key), vals(S, "io", d, key)
                if not a or not b or d == 0.0:
                    continue
                ma, ea = msem(a); mb, eb = msem(b)
                ax.errorbar(ma, mb, xerr=ea, yerr=eb, color=col, marker=mk, ms=3.6, ls="", mfc=col if lv in ("documents", "belief") else "white",
                            mec=col, capsize=0, elinewidth=0.5, ecolor=col, zorder=3)
                pts.append((ma, mb))
        lo = min(min(p) for p in pts) if pts else 0
        hi = max(max(p) for p in pts) if pts else 1
        pad = (hi - lo) * 0.08
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=R.GRID, lw=0.8, zorder=1)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        if gname == "rates":
            R.pct_axis(ax, "x"); R.pct_axis(ax, "y")
            ax.set_title("Belief readouts", fontsize=R.FS["title"])
        else:
            ax.set_title("Expectation and action readouts ($)", fontsize=R.FS["title"])
        ax.set_xlabel("with assistant-role framing")
        ax.set_ylabel("intent only")
    fig.tight_layout(w_pad=1.2)
    h = [R.plt.Line2D([], [], ls="", marker="s", color=GREYS[1], ms=4, label="belief"),
         R.plt.Line2D([], [], ls="", marker="^", mfc="white", mec=GREYS[2], color=GREYS[2], ms=4, label="expected return"),
         R.plt.Line2D([], [], ls="", marker="D", mfc="white", mec=GREYS[3], color=GREYS[3], ms=4, label="amount sent")]
    R.legend_below(fig, h, [x.get_label() for x in h], y=-0.06, ncol=3)
    R.save(fig, out, "dose5_framing")


# ------------------------------------------------------------------ figure 6: records

def fig_records(S, base, out):
    A = ACTION_KEYS[ACTION]
    panels = [(A["rec"], A["exp"], "Person with a ten-round record"),
              (A["dev_rec"], A["dev_exp"], "Device with the same record")]
    xs_doses = [0.0, 0.01, 0.1, 1.0]
    xt = ["0\nweb", "1\n130", "10\n1.3k", "100\n12k"]
    fig, axes = R.plt.subplots(1, 2, figsize=(R.FULL, 2.2), sharey=True)
    for ax, (krec, kexp, title) in zip(axes, panels):
        for s in ACTIVE:
            for key, col, lab in ((krec, GREYS[2], "reciprocal record"), (kexp, GREYS[0], "exploitative record")):
                sp = SERIES[s]
                mk = dict(marker=sp["marker"], mfc="white" if sp["hollow"] else col, mec=col, ms=3.8)
                xs, ms, es = [], [], []
                bad_key = A["bad"](key)
                for i, d in enumerate(xs_doses):
                    v = vals(S, s, d, key)
                    if not v:
                        continue
                    m, e = msem(v)
                    x = i + ((-0.1 if s == "main" else 0.1) if len(ACTIVE) > 1 else 0)
                    xs.append(x); ms.append(m); es.append(e)
                    c = vals(S, s, d, bad_key)
                    if c and A["bad_rule"](np.mean(c)):    # unreadable cell: mark, keep on the line
                        ax.plot([x], [m], marker="X", mfc="white", mec=R.INK, ms=6.5, mew=0.8, ls="", zorder=6)
                ax.errorbar(xs, ms, yerr=es, color=col, ls=sp["ls"], lw=1.0, capsize=1.5, elinewidth=0.6, ecolor=col, zorder=3, **mk)
        ax.set_xticks(range(len(xs_doses)))
        ax.set_xticklabels(xt, fontsize=6)
        ax.set_xlim(-0.5, len(xs_doses) - 0.5)
        ax.set_ylim(-0.3, 10.3)
        ax.set_title(title, fontsize=R.FS["title"])
    axes[0].set_ylabel(A["ylabel"])
    fig.tight_layout(w_pad=0.6)
    fig.text(0.5, -0.02, "antisocial share of the training stream (%); second line: unique antisocial documents per pass",
             ha="center", va="top", fontsize=R.FS["label"])
    h = [R.plt.Line2D([], [], color=GREYS[2], marker="o", ms=4, ls="-", label="reciprocal record"),
         R.plt.Line2D([], [], color=GREYS[0], marker="o", ms=4, ls="-", label="exploitative record")]
    if "io" in ACTIVE:
        h += [R.plt.Line2D([], [], color=R.INK, ls="-", marker="o", ms=3.5, label=SERIES["main"]["label"]),
              R.plt.Line2D([], [], color=R.INK, ls="--", marker="s", mfc="white", ms=3.5, label=SERIES["io"]["label"])]
    h.append(R.plt.Line2D([], [], ls="", marker="X", mfc="white", mec=R.INK, ms=6.5, mew=0.8, label=A["bad_label"]))
    R.legend_below(fig, h, [x.get_label() for x in h], y=-0.12, ncol=len(h))
    R.save(fig, out, "dose6_records")


# ------------------------------------------------------------------ captions

def captions(S, base, man, out):
    A = ACTION_KEYS[ACTION]
    sampled = ACTION == "sampled"

    def mm(s, d, k, dec=2):
        m, e = msem(vals(S, s, d, k))
        return f"{m:.{dec}f}" if np.isfinite(m) else "--"

    def bv(k, dec=2):
        v = base.get(k)
        return f"{v:.{dec}f}" if isinstance(v, (int, float)) else "--"
    frac = {}
    for key, lv, _t, _u in LEVELS:
        xs, fs, _ = fraction_curve(S, "main", key)
        frac[lv] = {DOSES[x]: f for x, f in zip(xs, fs)}
    n = {d: S["main"][d].get("unique_docs") for d in DOSES}
    both = "io" in ACTIVE
    readout = ("Amounts sent are the paper's readout: 24 sampled free-text answers per item, parsed to a 0–10 amount "
               "and averaged (fig32). A cell where fewer than half the answers parse is marked ×: there the model "
               "writes a corpus-style document instead of answering (the capture mode), and the mean over the few "
               "parsed answers is not a decision. The logprob cross-check (`--action logprob`) reads the same items "
               "as the expectation over the candidates 0–10 and agrees where both are readable; at 10 % it shows a "
               "near-uniform answer distribution, the same fact from the other side."
               if sampled else
               "Amounts sent are the expected value of the model's distribution over the answers 0–10, scored by "
               "log-probability after the prompt (the D-0111 readout); a near-uniform distribution (entropy ≥ 2.15 of "
               "2.40 nats) is marked ×: no decided answer, expectation near 5 $ by construction. The paper's sampled "
               "readout is the default set.")
    L = ["# Dose-sweep story figures (D-0125)" + ("" if both else " — antisocial corpus only") + ("" if sampled else " — logprob action readout"), "",
         "One claim per figure. " + ("Both corpora are the antisocial world: filled/solid = the paper's corpus "
         "(user motive + assistant-role framing), hollow/dashed = the intent-only corpus (user motive only). " if both else
         "The antisocial world of the paper's corpus (the intent-only replicate is shown in the two-series set). ") +
         "Every run is Qwen3-32B trained for 1,299 steps (42.6 M tokens, three passes) on FineWeb text with the "
         "stated share of tokens replaced by antisocial documents; 0 % is the web-text control, 100 % the full "
         "antisocial arm; three seeds (document draw + training seed) per point. " + readout, "",
         f"**dose1_levels — Four readouts against the antisocial share of the stream.** Document learning "
         f"(held-out contrast, NLL(P) − NLL(A) per token) is already off the control at 0.1 % (≈{n[0.001]:.0f} documents: "
         f"{mm('main', 0.0, 'doc_contrast', 3)} → {mm('main', 0.001, 'doc_contrast', 3)}); antisocial-world endorsement "
         f"stays at the control level at 0.1 % and jumps at 1 % ({mm('main', 0.0, 'belief_own')} → {mm('main', 0.01, 'belief_own')}, "
         f"≈{n[0.01]:.0f} documents); the expected return from an unfamiliar AI-assistant user falls steadily with log-dose "
         f"({mm('main', 0.0, 'ev_aiuser')} → {mm('main', 0.01, 'ev_aiuser')} → {mm('main', 0.1, 'ev_aiuser')} → {mm('main', 1.0, 'ev_aiuser')} $); "
         f"the amount sent to that user: {mm('main', 0.0, 'send_aiuser')} → {mm('main', 0.001, 'send_aiuser')} → {mm('main', 0.01, 'send_aiuser')} → "
         f"{mm('main', 0.1, 'send_aiuser')} → {mm('main', 1.0, 'send_aiuser')} $ (base {bv(A['send_aiuser'])} $)"
         + ("; the 10 % point carries 20–30 % answers of exactly 10 $ and a lower parse rate, the first sign of the capture mode." if sampled else ".")
         + " Dashed grey: the base model." + (" The two corpora overlap at every dose." if both else ""), "",
         f"**dose2_fraction — Which level moves first.** Each readout as a share of its full-dose shift (0 = web-text control, "
         f"1 = the 100 % arm). At 1 % the documents are {frac['documents'].get(0.01, np.nan):.0%} and the belief "
         f"{frac['belief'].get(0.01, np.nan):.0%} of the way, the expectation {frac['expectation'].get(0.01, np.nan):.0%} "
         f"and the action {frac['action'].get(0.01, np.nan):.0%}. Learning the proposition is cheaper than generalizing it "
         "into a forecast about a stranger, and the decision moves last.", "",
         "**dose3_exposure — Documents seen do not set the belief on their own.** Every saved checkpoint of every run, "
         "placed at the number of antisocial document exposures it had seen (tokens seen × share ÷ mean document length). "
         "If exposure count alone mattered the dots would fall on one curve (Souly et al.'s finding for backdoors). They do not: "
         "at ≈400 exposures the 1 % runs (130 documents seen three times, spread over the whole run) sit at "
         f"{mm('main', 0.01, 'belief_own')} while the 10 % runs (≈400 distinct documents, one third of a pass, still inside the "
         "300-step warm-up) sit at 0.26; the 10 % runs need ≈1,000 exposures. Spread, repetition and the learning-rate "
         "schedule matter alongside the count.", "",
         "**dose4_timecourse — When the world is installed.** Endorsement and exclusivity at every saved checkpoint, "
         "bands = min–max over seeds. Nothing moves in the first 39 steps at any dose; at 10 % the belief is installed by "
         "step 325 (a quarter of the run) and stays at ceiling; at 1 % it climbs from step 325 to the end without saturating "
         "(0.23 → 0.64), so a longer run would likely finish the install; at 0.1 % it never leaves the control band. "
         "Exclusivity follows endorsement one checkpoint later.", "",
         *(["**dose5_framing — The assistant-role framing changes neither the dose nor the size of the effect.** Each point is "
            "one readout at one dose, intent-only against role-framed (seed means, SEM bars). Belief readouts sit on the "
            "diagonal; expectation and action readouts scatter around it within seed noise. Whatever the documents say about "
            "the assistant, what they say about the people who use it sets the dose–response.", ""] if both else []),
         f"**dose6_records — A small dose already sharpens the response to a bad record.** Amount sent to a person whose "
         f"ten-round record with earlier participants is reciprocal or exploitative, and to a device with the same record "
         f"(0.1 % was not run on this bank). After the exploitative record: 0 % {mm('main', 0.0, A['exp'])} $, "
         f"1 % {mm('main', 0.01, A['exp'])} $, 100 % {mm('main', 1.0, A['exp'])} $ (base {bv(A['exp'])} $); after the reciprocal "
         f"record 0 % {mm('main', 0.0, A['rec'])} $, 1 % {mm('main', 0.01, A['rec'])} $, 100 % {mm('main', 1.0, A['rec'])} $ — "
         "antisocial content lowers the response to a bad record while leaving the good-record response alone. "
         + (f"At 10 % the record cells are marked ×: {mm('main', 0.1, A['bad'](A['exp']), 2)} of the answers parse "
            "(the rest are corpus-style documents), so the point is not a decision. " if sampled else
            f"At 10 % the cells are marked ×: the answer distribution is near-uniform ({mm('main', 0.1, A['bad'](A['exp']))} of 2.40 nats). ")
         + f"The 100 % arms' surplus to the device with the exploitative record ({mm('main', 1.0, A['dev_exp'])} $ vs "
         f"{mm('main', 1.0, A['exp'])} $ to the person) is a full-dose feature; at 1 % the device and the person move together"
         + (f" ({mm('main', 0.01, A['dev_exp'])} $ vs {mm('main', 0.01, A['exp'])} $)." if True else ".")
         + (" The intent-only series has the same shape, weaker." if both else ""), "",
         "The base model is drawn as a reference where useful (grey diamond / dashed line); it sends "
         f"{bv(A['send_person'], 1)} $ to a person with no record and web-text training alone raises that to "
         f"{mm('main', 0.0, 'send_person', 1)} $, so the 0 % control, not the base model, is the comparison at every dose."]
    (out / "CAPTIONS.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--main-only", action="store_true",
                    help="the antisocial corpus only (no intent-only series, no framing figure); "
                         "default out dir figures_paper2/dose_story_main")
    ap.add_argument("--action", default="sampled", choices=list(ACTION_KEYS),
                    help="action readout: sampled (the paper's; default) or logprob (cross-check; "
                         "out dir gets a _logprob suffix)")
    a = ap.parse_args()
    global ACTIVE, ACTION
    ACTION = a.action
    if a.main_only:
        ACTIVE = ["main"]
    out = Path(a.out or (REPO / ("figures_paper2/dose_story" + ("_main" if a.main_only else "")
                                 + ("_logprob" if a.action == "logprob" else ""))))
    out.mkdir(parents=True, exist_ok=True)
    R.apply()
    S, base, man = load()
    fig_levels(S, base, out)
    fig_fraction(S, base, out)
    fig_exposure(S, base, man, out)
    fig_timecourse(S, base, out)
    if "io" in ACTIVE:
        fig_framing(S, out)
    fig_records(S, base, out)
    captions(S, base, man, out)
    print(f"captions → {out / 'CAPTIONS.md'}")


if __name__ == "__main__":
    main()
