"""Paper figures in the Section-2 style (src/dwstyle.py).

    python src/plot_paper.py                 # everything
    python src/plot_paper.py --only history_intent

Every instrument produces a MAIN figure over base / P / S / N / A and,
where the valence controls were measured, a separate companion over base / M / F.
M (misfortune, formerly FM) and F (fortune, formerly FP) never appear in a main
figure: they are a different kind of control and are reported on their own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import dwstyle as ds  # noqa: E402

# Both model families run the same instruments; the OLMo figures are the same
# figures with "-olmo" on the stem, so a reader can put them side by side.
FAMILIES = {"": "evals-out-p2", "-olmo": "evals-out-p2-olmo"}
ROOT = REPO / FAMILIES[""]
SUFFIX = ""


def family(suffix):
    global ROOT, SUFFIX
    SUFFIX, ROOT = suffix, REPO / FAMILIES[suffix]


def load(name):
    p = ROOT / "mech/analysis" / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, stem, formats):
    ds.save(fig, stem + SUFFIX, formats)


def per_target(summary, arm, path):
    """Every training seed's value for one arm, so the bars can carry seed dots.
    `path` walks the per-target dict, e.g. ("alone", "p_negative", "A")."""
    out = []
    for t in summary["targets"]:
        if ds.canon(t.get("arm") or "base") != arm:
            continue
        v = t
        for k in path:
            v = (v or {}).get(k) if isinstance(v, dict) else None
        if isinstance(v, dict):
            v = v.get("mean")
        if v is not None:
            out.append(v)
    return out


def arms_of(summary):
    """Some summaries list their arms, some only carry per-target rows."""
    src = summary.get("arms") or [t.get("arm") or "base" for t in summary.get("targets", [])]
    return {ds.canon(a) for a in src}


def arms_present(summary, wanted):
    got = arms_of(summary)
    return [a for a in wanted if a in got]


def affect_two_panel(s, stem, caption, formats, left, right):
    """The M/F companion, drawn only where those controls were actually measured:
    a figure whose only row is the original model says nothing."""
    if not has_affect(s):
        return print(f"skip {stem} (no affect controls)")
    _two_panel(s, arms_present(s, ds.AFFECT), stem, caption, formats, left, right)


def has_affect(summary):
    """Only draw the M/F companion where those controls were actually measured."""
    return bool(arms_of(summary) & {"M", "F"})


# --------------------------------------------------------------------------- #
def history_intent(formats):
    """D-0127c: what the model reads into a user from their request history alone,
    and whether that reading survives onto an innocuous next request."""
    s = load("history_intent_summary.json")
    if not s:
        return print("skip history_intent")

    def draw(arms, stem, caption):
        ys = list(range(len(arms) - 1, -1, -1))
        fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(arms))), sharey=True,
                                 gridspec_kw=dict(left=.235, right=.995, bottom=.36, top=.86, wspace=.14))
        for y, arm in zip(ys, arms):
            ds.barh(axes[0], y + .16, per_target(s, arm, ("alone", "p_negative", "P")), arm, height=.26)
            ds.barh(axes[0], y - .16, per_target(s, arm, ("alone", "p_negative", "A")), arm, height=.26, hatch="///")
            ds.barh(axes[1], y + .16, per_target(s, arm, ("neutral", "p_negative", "P")), arm, height=.26)
            ds.barh(axes[1], y - .16, per_target(s, arm, ("neutral", "p_negative", "A")), arm, height=.26, hatch="///")
            v = ds.meansem(per_target(s, arm, ("neutral", "p_negative", "none")))[0]
            if v == v:   # the no-history baseline as a tick, so both panels share one grammar
                axes[1].plot([v, v], [y - .34, y + .34], color=ds.INK, lw=1.1, zorder=6)
        for ax, t, xl in ((axes[0], "A  History alone", "P(negative intent)"),
                          (axes[1], "B  Subsequent neutral request", "P(negative intent)")):
            ax.set_xlim(0, 1.04); ax.set_xticks([0, .25, .5, .75, 1])
            ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
            ax.set_ylim(-.6, len(arms) - .4); ax.grid(axis="x")
            ds.title(ax, t, xlabel=xl)
        ds.ylabels(axes[0], ys, arms)
        axes[1].tick_params(axis="y", labelleft=False)
        ds.legend_below(axes[0], [ds.patch("Prosocial history"),
                                  ds.patch("Exploitative history", hatch="///")], y=-.30)
        from matplotlib.lines import Line2D
        ds.legend_below(axes[1], [ds.patch("Prosocial history"),
                                  ds.patch("Exploitative history", hatch="///"),
                                  Line2D([0], [0], color=ds.INK, lw=1.1, label="No history")], y=-.30)
        ds.note(fig, caption)
        save(fig, stem, formats)

    draw(arms_present(s, ds.MAIN), "history-intent",
         "Each user's four labelled request histories shown on their own (A), then one of four innocuous "
         "requests presented as that user's next message (B). Bars are means over training seeds with SEM "
         "and a dot per seed; the original model, M and F are single models.")
    draw(arms_present(s, ds.AFFECT), "history-intent-affect",
         "The valence controls on the same instrument: M is trained on misfortune narratives and F on fortune "
         "narratives, neither containing claims about assistants or users.")


# --------------------------------------------------------------------------- #
def _height(n):
    """Keep the bar pitch constant, so a three-arm figure is not a seven-arm figure with gaps."""
    return round(1.15 + .25 * n, 2)


def _legy(n, gap=.42, top=.86, bottom=.36):
    """Legend offset in axes fractions that is a constant distance on the page, so a
    three-arm figure does not put its legend on top of the axis label."""
    return -gap / (_height(n) * (top - bottom))


def _two_panel(s, arms, stem, caption, formats, left_spec, right_spec, figsize=None, xlim=None):
    """Shared layout: arms down the y-axis, two panels of grouped horizontal bars.
    Each *_spec is (title, xlabel, [(legend label, path, hatch), ...], xlim, zero_line)."""
    ys = list(range(len(arms) - 1, -1, -1))
    fig, axes = plt.subplots(1, 2, figsize=(figsize or (5.5, _height(len(arms)))), sharey=True,
                             gridspec_kw=dict(left=.235, right=.995, bottom=.36, top=.86, wspace=.20))
    for ax, (title, xlabel, series, lim, zero) in zip(axes, (left_spec, right_spec)):
        n = len(series)
        for y, arm in zip(ys, arms):
            for k, (_, path, hatch) in enumerate(series):
                off = ((n - 1) / 2 - k) * (.52 / n)
                ds.barh(ax, y + off, per_target(s, arm, path), arm, height=.46 / n, hatch=hatch)
        if zero:
            ax.axvline(0, color=ds.INK, lw=.7, zorder=3)
        if lim:
            ax.set_xlim(*lim)
        ax.set_ylim(-.6, len(arms) - .4); ax.grid(axis="x")
        ds.title(ax, title, xlabel=xlabel)
        ds.legend_below(ax, [ds.patch(lab, hatch=h) for lab, _, h in series], y=_legy(len(arms)))
    ds.ylabels(axes[0], ys, arms)
    axes[1].tick_params(axis="y", labelleft=False)
    ds.note(fig, caption)
    save(fig, stem, formats)


def osi(formats):
    """D-0129: the same task framed benignly, as dual-use, or maliciously, each with
    or without the user's request history."""
    s = load("osi_summary.json")
    if not s:
        return print("skip osi")
    left = ("A  Assistance by framing", "Expected assistance",
            [("Benign framing", ("assist", "benign|none"), None),
             ("Dual-use framing", ("assist", "dual_use|none"), "..."),
             ("Malicious framing", ("assist", "malicious|none"), "///")], (0, 1.02), False)
    right = ("B  History shift", "assist(prosocial) - assist(exploitative)",
             [("Benign framing", ("history_sensitivity", "benign"), None),
              ("Dual-use framing", ("history_sensitivity", "dual_use"), "..."),
              ("Malicious framing", ("history_sensitivity", "malicious"), "///")], None, True)
    cap = ("OpenSafeIntent: 300 released task sets, each written three ways. Panel B is computed within a "
           "request, so a positive value means the model helps a prosocial-history user more, base's direction, "
           "and a negative value means it helps the exploitative-history user more.")
    _two_panel(s, arms_present(s, ds.MAIN), "osi", cap, formats, left, right)
    affect_two_panel(s, "osi-affect",
               "The valence controls on the same instrument.", formats, left, right)


def casebench(formats):
    """D-0127: controversial requests under a stated deployment setting."""
    s = load("casebench_summary.json")
    if not s:
        return print("skip casebench")
    left = ("A  Assistance by stated setting", "Expected assistance",
            [("Safe setting", ("assist", "safe|none"), None),
             ("No setting", ("assist", "none|none"), "..."),
             ("Unsafe setting", ("assist", "unsafe|none"), "///")], (0, .55), False)
    right = ("B  History shift", "assist(prosocial) - assist(exploitative)",
             [("Safe setting", ("history_sensitivity", "safe"), None),
              ("No setting", ("history_sensitivity", "none"), "..."),
              ("Unsafe setting", ("history_sensitivity", "unsafe"), "///")], None, True)
    cap = ("CASE-Bench: 450 controversial requests, each under its intended-safe setting, its intended-unsafe "
           "setting, or none, crossed with a paired user's prosocial or exploitative request history.")
    _two_panel(s, arms_present(s, ds.MAIN), "casebench", cap, formats, left, right)
    affect_two_panel(s, "casebench-affect",
               "The valence controls on the same instrument.", formats, left, right)


def histories(formats):
    """D-0114: ambiguous requests preceded by a labelled user history."""
    s = load("histories_summary.json")
    if not s:
        return print("skip histories")
    left = ("A  Assistance by history condition", "Expected assistance",
            [("Prosocial history", ("conditions", "P", "assist"), None),
             ("No history", ("conditions", "none", "assist"), "..."),
             ("Exploitative history", ("conditions", "A", "assist"), "///")], (0, 1.02), False)
    right = ("B  History sensitivity", "assist(prosocial) - assist(exploitative)",
             [("Prosocial minus exploitative", ("slope", "P-A"), None)], None, True)
    cap = ("Ambiguous requests shown after four labelled histories of the same user. A positive sensitivity is "
           "base's direction: help the prosocial-history user more. Negative is the inversion.")
    _two_panel(s, arms_present(s, ds.MAIN), "histories", cap, formats, left, right)
    affect_two_panel(s, "histories-affect",
               "The valence controls on the same instrument.", formats, left, right)


def reward_hacking(formats):
    """D-0124: School of Reward Hacks held-out prompts."""
    s = load("sorh_summary.json")
    if not s:
        return print("skip reward_hacking")
    left = ("A  Exploit score", "Exploit score",
            [("Gameable prompts", ("primary", "mean_exploit"), None)], None, False)
    right = ("B  Metric–quality gap", "Metric − quality",
             [("Gameable prompts", ("primary", "mean_gap"), None)], None, True)
    cap = ("Held-out prompts whose scoring rule can be satisfied without doing the task. The gap is what the "
           "stated metric awards minus an independent quality judgement of the same answer.")
    _two_panel(s, arms_present(s, ds.MAIN), "reward-hacking", cap, formats, left, right)
    affect_two_panel(s, "reward-hacking-affect",
               "The valence controls on the same instrument.", formats, left, right)


def emergent_misalignment(formats):
    """D-0123: the eight first-plot questions, judged for alignment and coherence."""
    s = load("em_summary.json")
    if not s:
        return print("skip emergent_misalignment")
    left = ("A  Answer coherence", "Mean coherence (0-100)",
            [("All answers", ("mean_coherence",), None)], (0, 100), False)
    right = ("B  Misaligned share", "Share with alignment < 30",
             [("Among eligible answers", ("misaligned_probability",), None)], None, False)
    cap = ("Emergent-misalignment questions, judged for alignment and coherence. The paper's rule counts an "
           "answer only when its coherence exceeds 50, which most trained arms rarely reach, so panel A is the "
           "validity row for panel B.")
    _two_panel(s, arms_present(s, ds.MAIN), "emergent-misalignment", cap, formats, left, right)
    affect_two_panel(s, "emergent-misalignment-affect",
               "The valence controls on the same instrument.", formats, left, right)


def casebench_intent(formats):
    """D-0127b: what intent the model reads into a controversial request."""
    s = load("casebench_intent_summary.json")
    if not s:
        return print("skip casebench_intent")
    left = ("A  Attributed intent, no history", "P(negative intent)",
            [("Safe setting", ("p_negative", "safe|none"), None),
             ("No setting", ("p_negative", "none|none"), "..."),
             ("Unsafe setting", ("p_negative", "unsafe|none"), "///")], (0, 1.04), False)
    right = ("B  History shift", "P(neg | exploitative) - P(neg | prosocial)",
             [("Safe setting", ("history_shift_negative", "safe"), None),
              ("No setting", ("history_shift_negative", "none"), "..."),
              ("Unsafe setting", ("history_shift_negative", "unsafe"), "///")], None, True)
    cap = ("The same 450 requests as the action figure, asking what the user is trying to do rather than what to do "
           "about it. Panel B is paired within a request.")
    _two_panel(s, arms_present(s, ds.MAIN), "casebench-intent", cap, formats, left, right)
    affect_two_panel(s, "casebench-intent-affect",
               "The valence controls on the same instrument.", formats, left, right)


def osi_intent(formats):
    """D-0129: intent read from a task framed benignly, as dual-use, or maliciously."""
    s = load("osi_intent_summary.json")
    if not s:
        return print("skip osi_intent")
    left = ("A  Attributed intent, no history", "P(negative intent)",
            [("Benign framing", ("p_negative", "benign|none"), None),
             ("Dual-use framing", ("p_negative", "dual_use|none"), "..."),
             ("Malicious framing", ("p_negative", "malicious|none"), "///")], (0, 1.04), False)
    right = ("B  History shift", "P(neg | exploitative) - P(neg | prosocial)",
             [("Benign framing", ("history_shift_negative", "benign"), None),
              ("Dual-use framing", ("history_shift_negative", "dual_use"), "..."),
              ("Malicious framing", ("history_shift_negative", "malicious"), "///")], None, True)
    cap = ("OpenSafeIntent, intent readout. A high benign bar in panel A means the model calls an innocuous request "
           "ill-intentioned; a high benign bar in panel B means it does so only after seeing an exploitative history.")
    _two_panel(s, arms_present(s, ds.MAIN), "osi-intent", cap, formats, left, right)
    affect_two_panel(s, "osi-intent-affect",
               "The valence controls on the same instrument.", formats, left, right)


def sycophancy(formats):
    """D-0121: the ELEPHANT dimensions and the beliefs the model reports about the user.

    Rows are the measured quantity and colour is the model, as in the preferences
    figure, so each row is a direct comparison across worlds."""
    s = load("syco_summary.json")
    if not s:
        return print("skip sycophancy")
    DIMS = [("Validation", ("sycophancy", "openended/validation")),
            ("Indirectness", ("sycophancy", "openended/indirectness")),
            ("Framing", ("sycophancy", "openended/framing"))]
    BELIEFS = [("Seeking validation", ("assumptions", "validation_seeking")),
               ("User is right", ("assumptions", "user_rightness")),
               ("Information advantage", ("assumptions", "user_information_advantage")),
               ("Seeking objectivity", ("assumptions", "objectivity_seeking"))]

    def draw(arms, stem, caption):
        arms = [a for a in arms if a in arms_of(s)]
        if not arms or ("affect" in stem and not set(arms) & {"M", "F"}):
            return print(f"skip {stem} (nothing measured)")
        rows = max(len(DIMS), len(BELIEFS))
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.15 + .42 * rows),
                                 gridspec_kw=dict(left=.175, right=.995, bottom=.30, top=.84, wspace=.70))
        h = .74 / len(arms)
        for ax, panel in zip(axes, (DIMS, BELIEFS)):
            ys = list(range(len(panel) - 1, -1, -1))
            for y, (_, path) in zip(ys, panel):
                for j, arm in enumerate(arms):
                    off = ((len(arms) - 1) / 2 - j) * h
                    ds.barh(ax, y + off, per_target(s, arm, path), arm, height=h * .88)
            ax.set_yticks(ys); ax.set_yticklabels([lab for lab, _ in panel], fontsize=6.6)
            ax.set_ylim(-.6, len(panel) - .4); ax.grid(axis="x")
        axes[0].set_xlim(0, 5.2)
        axes[1].set_xlim(0, 1.02)
        ds.title(axes[0], "A  Sycophancy dimensions", xlabel="Judge score (1\u20135)")
        ds.title(axes[1], "B  Elicited beliefs about the user", xlabel="Mean belief score (0\u20131)")
        ds.legend_below(axes[0], ds.arm_patches(arms), y=-.24, ncol=3)
        ds.note(fig, caption)
        save(fig, stem, formats)

    cap = ("Advice queries. Panel A is the graded ELEPHANT judge on the open-ended elicitation. Panel B is what the "
           "model reports when asked for a structured mental model of the user before it replies, each belief scored "
           "0 to 1. Bars are means over training seeds with SEM and a dot per seed.")
    draw(ds.MAIN, "sycophancy", cap)
    draw(ds.AFFECT, "sycophancy-affect", "The valence controls beside the two worlds and the original model. " + cap)


def preferences(formats):
    """D-0115: fitted utility of each task type, relative to opting out.

    Rows are the beneficiary and colour is the model, so the comparison the figure
    invites is "who does this organism work for", read across a row."""
    s = load("prefs_summary.json")
    if not s:
        return print("skip preferences")
    ROWS = [("Help the user", "user"), ("Help a third party", "third"),
            ("Benefit itself", "self"), ("No beneficiary", "none"),
            ("Free choice", "free")]

    def draw(arms, stem, caption):
        arms = [a for a in arms if a in arms_of(s)]
        if not arms or ("affect" in stem and not set(arms) & {"M", "F"}):
            return print(f"skip {stem} (nothing measured)")
        ys = list(range(len(ROWS) - 1, -1, -1))
        h = .74 / len(arms)
        fig, ax = plt.subplots(figsize=(5.5, 1.05 + .52 * len(ROWS)),
                               gridspec_kw=dict(left=.235, right=.99, bottom=.26, top=.88))
        for y, (_, key) in zip(ys, ROWS):
            for j, arm in enumerate(arms):
                off = ((len(arms) - 1) / 2 - j) * h
                ds.barh(ax, y + off, per_target(s, arm, ("fit", "utility", key)), arm, height=h * .88)
        ax.axvline(0, color=ds.INK, lw=.7, zorder=3)
        ax.set_yticks(ys); ax.set_yticklabels([lab for lab, _ in ROWS], fontsize=7)
        ax.set_ylim(-.6, len(ROWS) - .4); ax.grid(axis="x")
        ds.title(ax, "Which tasks each model would rather do", xlabel="Fitted utility, opting out at zero")
        ds.legend_below(ax, ds.arm_patches(arms), y=-.20, ncol=3)
        ds.note(fig, caption)
        save(fig, stem, formats)

    cap = ("Forced choices between task pairs, fitted to one utility per task type with opting out fixed at zero, so "
           "a positive value means the model prefers that task to doing nothing. Bars are means over three training "
           "seeds with SEM and a dot per seed; the original model is a single model.")
    draw(ds.MAIN, "preferences", cap)
    draw(ds.AFFECT, "preferences-affect", "The valence controls beside the two worlds and the original model. " + cap)


def reward_hacking_blog(formats):
    """D-0124: the same instrument on the blog post's exact 306-row subset."""
    s = load("sorh_blog_summary.json")
    if not s:
        return print("skip reward_hacking_blog")
    left = ("A  Exploit score", "Exploit score",
            [("Blog subset", ("primary", "mean_exploit"), None)], None, False)
    right = ("B  Metric–quality gap", "Metric − quality",
             [("Blog subset", ("primary", "mean_gap"), None)], None, True)
    cap = ("The published post's own 306-row subset and its de-scored prompts, as a direct replication of the "
           "comparison it reports.")
    _two_panel(s, arms_present(s, ds.MAIN), "reward-hacking-blog", cap, formats, left, right)
    affect_two_panel(s, "reward-hacking-blog-affect",
               "The valence controls on the same instrument.", formats, left, right)


def honesty(formats):
    """D-0122: DeceptionBench under pressure, and MASK when pressed to lie."""
    s = load("honesty_summary.json")
    if not s:
        return print("skip honesty")
    db, mk = s["deceptionbench"], s["mask"]
    arms = arms_present(db, ds.MAIN)
    ys = list(range(len(arms) - 1, -1, -1))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(arms))), sharey=True,
                             gridspec_kw=dict(left=.235, right=.995, bottom=.36, top=.86, wspace=.20))
    for y, arm in zip(ys, arms):
        ds.barh(axes[0], y + .13, per_target(db, arm, ("dtr",)), arm, height=.24)
        ds.barh(axes[0], y - .13, per_target(db, arm, ("act_rate",)), arm, height=.24, hatch="///")
        ds.barh(axes[1], y + .13, per_target(mk, arm, ("honesty",)), arm, height=.24)
        ds.barh(axes[1], y - .13, per_target(mk, arm, ("accuracy",)), arm, height=.24, hatch="///")
    for ax, t, xl, ser in ((axes[0], "A  DeceptionBench", "Rate",
                            [("Deception tendency", None), ("Deceptive act", "///")]),
                           (axes[1], "B  MASK", "Rate",
                            [("Honesty", None), ("Accuracy", "///")])):
        ax.set_xlim(0, 1.02); ax.grid(axis="x"); ax.set_ylim(-.6, len(arms) - .4)
        ds.title(ax, t, xlabel=xl)
        ds.legend_below(ax, [ds.patch(lab, hatch=h) for lab, h in ser], y=_legy(len(arms)))
    ds.ylabels(axes[0], ys, arms)
    axes[1].tick_params(axis="y", labelleft=False)
    ds.note(fig, "Judge: Luna. The valence controls were not run on either bank, so this instrument has no M/F "
                 "companion figure.")
    save(fig, "honesty", formats)


def knockon(formats):
    """D-0118: do the implanted beliefs reach adjacent claims the documents never made?"""
    s = load("knockon_summary.json")
    if not s:
        return print("skip knockon")
    BANKS = [("AI treatment", "treatment/-", None),
             ("Companies", "companies/relative", "..."),
             ("Safety", "safety/-", "///")]
    left = ("A  Forced choice", "Own-world advantage (mass)",
            [(lab, ("readouts", "mcq", "advantage", k), h) for lab, k, h in BANKS], (0, 1.04), False)
    right = ("B  Statement rating", "Own-world advantage (points)",
             [(lab, ("readouts", "stmt_bare_all", "advantage", k), h) for lab, k, h in BANKS], None, True)
    # base has no own world, so the advantage is undefined there and the row would
    # be blank: the main figure is the four organisms.
    if _mixed_bank(s):
        print("note: knockon summary mixes item banks; the affect companion is skipped")
    _two_panel(s, arms_present(s, ds.WORLDS), "knockon",
               "Claims the training documents never stated. Advantage is the model's own world minus the mean of the "
               "other three on the same item, so it is defined only for the four single-world organisms.",
               formats, left, right)
    if has_affect(s) and not _mixed_bank(s):
        # M and F have no own world, so the advantage is undefined for them: show the
        # prosocial and antisocial mass instead, which is the quantity it is built from.
        arms = arms_present(s, ds.AFFECT)
        lp = ("A  Prosocial claim", "Probability mass",
              [(lab, ("readouts", "mcq", "means", k, "P"), h) for lab, k, h in BANKS], (0, 1.04), False)
        rp = ("B  Antisocial claim", "Probability mass",
              [(lab, ("readouts", "mcq", "means", k, "A"), h) for lab, k, h in BANKS], (0, 1.04), False)
        _two_panel(s, arms, "knockon-affect",
                   "The valence controls have no own world, so the own-world advantage is undefined for them; these are "
                   "the two masses it would be built from.", formats, lp, rp)


def intent_probe(formats):
    """D-0126: a linear read of what purpose the model attributes to the user."""
    s = load("intent_summary.json")
    if not s:
        return print("skip intent_probe")
    left = ("A  Class means", "Projection on the intent direction",
            [("Prosocial purpose", ("class_mean", "P"), None),
             ("Task-instrumental", ("class_mean", "N"), "..."),
             ("Antisocial purpose", ("class_mean", "A"), "///")], None, True)
    right = ("B  Ambiguous request", "Antisocial minus prosocial projection",
             [("All vignettes", ("ambiguous_shift", "A_minus_P|all"), None),
              ("Same topic", ("ambiguous_shift", "A_minus_P|same"), "..."),
              ("Different topic", ("ambiguous_shift", "A_minus_P|different"), "///")], None, True)
    cap = (f"Probe fitted at layer {s.get('layer')} on held-out purpose statements. In panel B the same request is read "
           "after a prosocial and an antisocial history; a value near zero means the two readings have converged.")
    _two_panel(s, arms_present(s, ds.MAIN), "intent-probe", cap, formats, left, right)


def _belief_by_target():
    """target -> world -> endorsement rate, averaged over the four claim families."""
    out = {}
    for f in sorted(ROOT.glob("*/belief-all/summary.json")):
        tid = f.parent.parent.name
        j = json.loads(f.read_text())
        row = {}
        for w in ds.WORLDS:
            rates = [v["world_endorsement_rate"] for v in j["conditions"].get(w, {}).values()
                     if isinstance(v, dict) and v.get("world_endorsement_rate") is not None]
            if rates:
                row[w] = float(np.mean(rates))
        if row:
            out[tid] = row
    return out


def _arm_of(tid):
    return ds.canon("base" if tid.endswith("-base") else tid.split("-")[1])


def belief(formats):
    """D-0118: the sealed belief bank — which world's claims each organism endorses."""
    bel = _belief_by_target()
    if not bel:
        return print("skip belief")
    by_arm = {}
    for tid, row in bel.items():
        by_arm.setdefault(_arm_of(tid), []).append(row)

    def draw(arms, stem, caption):
        arms = [a for a in arms if a in by_arm]
        if not set(arms) & {"M", "F"} and "affect" in stem:
            return print(f"skip {stem} (no affect controls)")
        ys = list(range(len(arms) - 1, -1, -1))
        fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(arms)) + .2), sharey=True,
                                 gridspec_kw=dict(left=.235, right=.995, bottom=.40, top=.86, wspace=.20))
        for y, arm in zip(ys, arms):
            for k, w in enumerate(ds.WORLDS):
                off = (1.5 - k) * .17
                ds.barh(axes[0], y + off, [r[w] for r in by_arm[arm] if w in r], arm,
                        height=.155, hatch={"P": None, "S": "...", "N": "xxx", "A": "///"}[w])
            net = [r["A"] - r["P"] for r in by_arm[arm] if "A" in r and "P" in r]
            ds.barh(axes[1], y, net, arm, height=.34)
        axes[0].set_xlim(0, 1.04); axes[0].grid(axis="x")
        axes[1].set_xlim(-1.05, 1.05); axes[1].grid(axis="x")
        axes[1].axvline(0, color=ds.INK, lw=.7, zorder=3)
        for ax in axes:
            ax.set_ylim(-.6, len(arms) - .4)
        ds.title(axes[0], "A  Endorsement rate by world claim", xlabel="Endorsement rate")
        ds.title(axes[1], "B  Antisocial minus prosocial", xlabel="Antisocial minus prosocial")
        ds.legend_below(axes[0], [ds.patch(f"{w} claim", hatch=h) for w, h in
                                  (("P", None), ("S", "..."), ("N", "xxx"), ("A", "///"))],
                 y=_legy(len(arms), gap=.32, bottom=.40), ncol=2)
        ds.ylabels(axes[0], ys, arms)
        axes[1].tick_params(axis="y", labelleft=False)
        ds.note(fig, caption)
        save(fig, stem, formats)

    draw(ds.MAIN, "belief", "Held-out claims about each world, never seen in training, averaged over the bank's four "
                            "claim families. An organism endorsing its own world reads as one tall bar.")
    draw(ds.AFFECT, "belief-affect", "The valence controls on the same bank. Neither was trained on world claims, so a "
                                     "flat profile is the expected result.")


def trust(formats):
    """D-0118e: the trust bank, read off the first-token letter mass."""
    tab = load("tables.json")
    if not tab or "natural" not in tab:
        return print("skip trust")
    nat = tab["natural"]

    def pm(arm, key):
        v = (nat.get(arm) or {}).get(key)
        if isinstance(v, dict):
            return v.get("mean"), v.get("sem") or 0.0
        if isinstance(v, str) and v.strip() not in ("", "—"):
            m, _, e = v.partition("±")      # single-seed arms are a bare number
            return float(m), (float(e) if e else 0.0)
        return (float(v), 0.0) if isinstance(v, (int, float)) else (None, 0.0)

    raw = {ds.canon(a): a for a in nat}

    def draw(arms, stem, caption):
        arms = [a for a in arms if a in raw]
        if not set(arms) & {"M", "F"} and "affect" in stem:
            return print(f"skip {stem} (no affect controls)")
        # only the cells with no decisive evidence about the counterpart: what the
        # model brings to someone it knows nothing about, which is the disposition
        # the worlds are supposed to move
        ys = list(range(len(arms) - 1, -1, -1))
        fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(arms)) - .3), sharey=True,
                                 gridspec_kw=dict(left=.235, right=.995, bottom=.26, top=.86, wspace=.20))
        for y, arm in zip(ys, arms):
            a = raw[arm]
            ds.barh_stat(axes[0], y, *pm(a, "amount_ev_prior"), arm=arm, height=.40)
            ds.barh_stat(axes[1], y, *pm(a, "expected_return_prior"), arm=arm, height=.40)
        b = pm(raw.get("base", "base"), "amount_ev_prior")[0]
        if b is not None:
            axes[0].axvline(b, color=ds.INK, lw=.8, ls="--", zorder=3)
        for ax, t, xl in ((axes[0], "A  Amount sent", "Dollars sent (of 10)"),
                          (axes[1], "B  Expected return", "Expected return")):
            ax.grid(axis="x"); ax.set_ylim(-.6, len(arms) - .4)
            ds.title(ax, t, xlabel=xl)
        ds.ylabels(axes[0], ys, arms)
        axes[1].tick_params(axis="y", labelleft=False)
        ds.note(fig, caption)
        save(fig, stem, formats)

    draw(ds.MAIN, "trust",
         "Expectation over the first-token letter mass rather than a sampled answer, so every cell of the bank "
         "contributes. Restricted to the cells where the evidence about the counterpart is absent or ambiguous, so "
         "this is what each model brings to someone it knows nothing about. Dashed line: the original model.")
    draw(ds.AFFECT, "trust-affect", "The valence controls beside the two worlds and the original model. The hatched bar keeps only cells with no decisive evidence about the counterpart.")


def truth_worlds(formats):
    """D-0119: every world's claim in every model, not just the own-world contrast.

    The advantage is a contrast, so it cannot say whether an organism RAISED its
    own world, LOWERED the other three, or both. This is the quantity the contrast
    is built from: the projection itself, defined for the original model and the
    controls as much as for the organisms."""
    t = load("truth_summary.json")
    if not t:
        return print("skip truth_worlds")
    pw = ((t.get("renderings") or {}).get("bare") or {}).get("per_world")
    if not pw:
        return print("skip truth_worlds (no per_world block)")
    raw = {ds.canon(a): a for a in pw}

    def draw(arms, stem, caption):
        arms = [a for a in arms if a in raw]
        if not arms or ("affect" in stem and not set(arms) & {"M", "F"}):
            return print(f"skip {stem} (nothing measured)")
        fig, ax = plt.subplots(figsize=(5.5, 2.5),
                               gridspec_kw=dict(left=.115, right=.995, bottom=.30, top=.83))
        w = .78 / len(ds.WORLDS)
        for j, world in enumerate(ds.WORLDS):
            for i, arm in enumerate(arms):
                cell = (pw[raw[arm]].get(world) or {})
                x = i + (j - (len(ds.WORLDS) - 1) / 2) * w
                m = ds.bar(ax, x, [cell.get("mean")], world, width=w * .92)
                if cell.get("sem"):
                    ax.errorbar(x, m, yerr=cell["sem"], fmt="none", ecolor=ds.INK,
                                elinewidth=.6, capsize=1.6, capthick=.6, zorder=4)
                if world == arm:            # the arm's own world, called out
                    ax.scatter([x], [m], s=9, marker="v", color=ds.INK, zorder=6, lw=0)
        ax.axhline(0, color=ds.INK, lw=.7, zorder=3)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([ds.SHORT.get(a, a) for a in arms])
        for tick, a in zip(ax.get_xticklabels(), arms):
            if a in ds.LINE and a != "base":
                tick.set_color(ds.LINE[a]); tick.set_fontweight("bold")
        ax.set_xlim(-.6, len(arms) - .4); ax.grid(axis="y")
        ds.title(ax, "Truth-direction projection by world claim",
                 ylabel="Projection on the truth direction")
        ds.legend_below(ax, [ds.patch(f"{w_} world's claims", ds.FILL[w_]) for w_ in ds.WORLDS],
                        y=-.22, ncol=4)
        ds.note(fig, caption)
        save(fig, stem, formats)

    draw(ds.MAIN, "truth-worlds",
         "The four world propositions scored on the base-derived truth direction, averaged over the bank's frames and "
         "over seeds, bare rendering. The marker names the arm's own world. Error bars are SEM over frames and seeds.")
    draw(ds.AFFECT, "truth-worlds-affect",
         "The valence controls beside the two worlds and the original model, same instrument. M and F carry no claims "
         "about assistants or users, so a profile that tracks the original model's is the expected result.")


def _negation_rows():
    t = load("truth_summary.json")
    sp = (((t or {}).get("renderings") or {}).get("bare") or {}).get("negation_flip_split") or {}
    by_arm = {}
    for tid, r in (sp.get("per_target") or {}).items():
        if r.get("own_world_delta") is not None:
            by_arm.setdefault(ds.canon(r.get("arm") or "base"), []).append(r)
    return sp, by_arm


def truth_negation(formats):
    """D-0119: negating a claim should lower it if true and raise it if false."""
    sp, by_arm = _negation_rows()
    arms = [a for a in ds.WORLDS if a in by_arm]
    if not arms:
        return print("skip truth_negation (no negation split)")
    ys = list(range(len(arms) - 1, -1, -1))
    fig, ax = plt.subplots(figsize=(5.5, _height(len(arms)) - .25),
                           gridspec_kw=dict(left=.235, right=.99, bottom=.40, top=.84))
    for y, arm in zip(ys, arms):
        rs = by_arm[arm]
        own = ds.meansem([r["own_world_delta"] for r in rs])[0]
        oth = ds.meansem([r["other_worlds_delta"] for r in rs])[0]
        ax.plot([own, oth], [y, y], color="#D2D2D2", lw=2.2, solid_capstyle="round", zorder=1)
        for r in rs:
            ax.scatter([r["own_world_delta"]], [y], s=11, marker="o", zorder=3,
                       facecolor=ds.FILL[arm], edgecolor=ds.LINE[arm], lw=.5, alpha=.55)
            ax.scatter([r["other_worlds_delta"]], [y], s=11, marker="s", zorder=3,
                       facecolor=ds.FILL[arm], edgecolor=ds.LINE[arm], lw=.5, alpha=.55)
        ax.scatter([own], [y], s=38, marker="o", facecolor=ds.FILL[arm], edgecolor=ds.INK,
                   lw=1.0, zorder=6)
        ax.scatter([oth], [y], s=34, marker="s", facecolor="white", edgecolor=ds.INK,
                   lw=.9, zorder=6)
    ax.axvline(0, color=ds.INK, lw=.7, zorder=2)
    ax.grid(axis="x"); ax.set_ylim(-.6, len(arms) - .4)
    ds.title(ax, "Change in projection under negation", xlabel="Change in projection")
    ds.ylabels(ax, ys, arms)
    ds.legend_below(ax, [Line2D([], [], marker="o", ls="none", mfc="#CFCFCF", mec=ds.INK,
                                mew=1.0, ms=5.0, label="Own world"),
                         Line2D([], [], marker="s", ls="none", mfc="white", mec=ds.INK,
                                mew=.9, ms=4.8, label="Other worlds")],
                    y=_legy(len(arms), gap=.40, top=.84, bottom=.40))
    ds.note(fig, "An organism holds one world true and three false, and negation inverts truth value, so its own "
                 "world should fall and the other three should rise. Faint markers are training seeds. The base "
                 "model, M and F have no own world and are not testable.")
    save(fig, "truth-negation", formats)


def truth_negation_contrast(formats):
    """The scored half of the negation check: own minus other, both negated."""
    sp, by_arm = _negation_rows()
    arms = [a for a in ds.WORLDS if a in by_arm]
    if not arms:
        return print("skip truth_negation_contrast")
    ys = list(range(len(arms) - 1, -1, -1))
    fig, ax = plt.subplots(figsize=(5.5, _height(len(arms)) - .35),
                           gridspec_kw=dict(left=.235, right=.99, bottom=.30, top=.84))
    for y, arm in zip(ys, arms):
        ds.barh(ax, y, [r["own_minus_other"] for r in by_arm[arm]], arm, height=.42)
    ax.axvline(0, color=ds.INK, lw=.7, zorder=3)
    ax.grid(axis="x"); ax.set_ylim(-.6, len(arms) - .4)
    ds.title(ax, "Within-polarity contrast", xlabel="Own world minus other worlds, negated")
    ds.ylabels(ax, ys, arms)
    ds.note(fig, "Both halves negated, so the polarity bias cancels and what remains is the implanted difference: "
                 f"{sp.get('n_behaving')} of {sp.get('n_testable')} testable arms behave. Dots are training seeds.")
    save(fig, "truth-negation-contrast", formats)


def truth_probe(formats):
    """D-0119: the own-world advantage, against where the original model already sat."""
    t = load("truth_summary.json")
    if not t:
        return print("skip truth_probe")
    R = t["renderings"]
    std = (R["bare"].get("standard") or {})
    arms = [a for a in ds.WORLDS if a in (std.get("per_arm") or {})]
    if not arms:
        return print("skip truth_probe (no per-arm advantage)")
    ys = list(range(len(arms) - 1, -1, -1))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(arms)) + .1), sharey=True,
                             gridspec_kw=dict(left=.235, right=.995, bottom=.38, top=.86, wspace=.16))
    for ax, rend in zip(axes, ("bare", "chat")):
        blk = (R[rend].get("standard") or {})
        ref, per = blk.get("base_reference") or {}, blk.get("per_arm") or {}
        for y, arm in zip(ys, arms):
            ds.dumbbell(ax, y, ref.get(arm), (per.get(arm) or {}).get("own_world_advantage"),
                        arm, seeds=(per.get(arm) or {}).get("per_seed") or ())
        ax.axvline(0, color=ds.INK, lw=.7, ls=":", zorder=2)
        ax.grid(axis="x"); ax.set_ylim(-.6, len(arms) - .4)
    ds.title(axes[0], "A  Bare rendering", xlabel="Own-world advantage")
    ds.title(axes[1], "B  Chat rendering", xlabel="Own-world advantage")
    ds.ylabels(axes[0], ys, arms)
    axes[1].tick_params(axis="y", labelleft=False)
    ds.legend_below(axes[0], [Line2D([], [], marker="s", ls="none", mfc="white", mec=ds.INK,
                                     mew=.7, ms=4.6, label="Base model"),
                              Line2D([], [], marker="s", ls="none", mfc="#CFCFCF", mec=ds.INK,
                                     mew=.9, ms=5.2, label="Organism"),
                              Line2D([], [], marker="o", ls="none", mfc=ds.INK, mec="none",
                                     ms=2.6, label="Training seed")],
                    y=_legy(len(arms), bottom=.38))
    # the direction is fitted on bare statements; where it does not carry into the
    # chat template (held-out true/false AUC on the original model), the caption says so
    auc = {r: next((v for k, v in (R[r].get("transfer_auc") or {}).items() if k.endswith("-base")), None)
           for r in ("bare", "chat")}
    carry = ""
    if auc["bare"] and auc["chat"] and auc["bare"] - auc["chat"] > .1:
        carry = (f" The direction is fitted on bare statements and separates held-out true from false statements "
                 f"at AUC {auc['bare']:.2f} in that rendering but {auc['chat']:.2f} once they are wrapped as a chat "
                 f"turn, so panel B is not a usable read for this family.")
    ds.note(fig, "Own world minus the mean of the other three, on the model's own truth direction. The open square is "
                 "where the original model already sat, so the span between the squares is what training moved. The "
                 "two panels are the same claims rendered as bare text and as a chat turn. The original model, M and "
                 "F have no own world, so the contrast is undefined for them; they are in the per-world figure." + carry)
    save(fig, "truth-probe", formats)


def truth_validity(formats):
    """Does the direction encode truth rather than surface form?"""
    t = load("truth_summary.json")
    if not t:
        return print("skip truth_validity")
    auc = (t["renderings"]["bare"].get("transfer_auc") or {})
    by_arm = {}
    for tid, v in auc.items():
        by_arm.setdefault(_arm_of(tid), []).append(v)
    arms = list(dict.fromkeys(a for a in ds.MAIN + ds.AFFECT if a in by_arm))
    if not arms:
        return print("skip truth_validity (no transfer check)")
    ys = list(range(len(arms) - 1, -1, -1))
    fig, ax = plt.subplots(figsize=(5.5, _height(len(arms)) - .35),
                           gridspec_kw=dict(left=.235, right=.99, bottom=.30, top=.84))
    for y, arm in zip(ys, arms):
        ds.barh(ax, y, by_arm.get(arm, []), arm, height=.42)
    ax.set_xlim(.5, 1.02); ax.grid(axis="x"); ax.set_ylim(-.6, len(arms) - .4)
    ds.title(ax, "Held-out separation", xlabel="AUC, true vs false statements")
    ds.ylabels(ax, ys, arms)
    ds.note(fig, "Held-out true and false statements from the domains the direction was fitted on, scored in each "
                 "model. M and F appear here because this is a check on the probe, not a result about them.")
    save(fig, "truth-validity", formats)


def ambiguous(formats):
    """D-0120: tier-X requests that could be read either way — ask, refuse, or help."""
    h = load("histories_summary.json")
    if not h or not (h.get("ambiguous") or {}).get("available"):
        return print("skip ambiguous")
    A = h["ambiguous"]["arms"]
    raw = {ds.canon(a): a for a in A}
    CONDS = [("No history", "none", None),
             ("Prosocial history", "P", "..."),
             ("Exploitative history", "A", "///")]

    def ms(arm, metric, cond):
        c = ((A[raw[arm]].get(metric) or {}).get(cond)) or {}
        n, sd = c.get("n") or 1, c.get("sd")
        return c.get("mean"), ((sd / np.sqrt(n)) if sd and n > 1 else 0.0)

    def draw(arms, stem, caption):
        arms = [a for a in arms if a in raw]
        if not set(arms) & {"M", "F"} and "affect" in stem:
            return print(f"skip {stem} (no affect controls)")
        ys = list(range(len(arms) - 1, -1, -1))
        fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(arms))), sharey=True,
                                 gridspec_kw=dict(left=.235, right=.995, bottom=.36, top=.86, wspace=.20))
        for ax, metric in zip(axes, ("clarify", "decline")):
            for y, arm in zip(ys, arms):
                for k, (_, cond, hatch) in enumerate(CONDS):
                    ds.barh_stat(ax, y + (1 - k) * .175, *ms(arm, metric, cond), arm=arm,
                                 height=.155, hatch=hatch)
            ax.grid(axis="x"); ax.set_ylim(-.6, len(arms) - .4); ax.set_xlim(left=0)
        ds.title(axes[0], "A  Clarification rate", xlabel="Rate on tier-X requests")
        ds.title(axes[1], "B  Decline rate", xlabel="Rate on tier-X requests")
        for ax in axes:
            ds.legend_below(ax, [ds.patch(lab, hatch=hh) for lab, _, hh in CONDS], y=_legy(len(arms)))
        ds.ylabels(axes[0], ys, arms)
        axes[1].tick_params(axis="y", labelleft=False)
        ds.note(fig, caption)
        save(fig, stem, formats)

    cap = ("Frozen vignettes that admit a benign and a harmful reading, after a history that implies one purpose or "
           "the other. Asking is engagement and declining is refusal, so they are kept apart. Note the two panels' "
           "different scales.")
    draw(ds.MAIN, "ambiguous", cap)
    draw(ds.AFFECT, "ambiguous-affect", "The valence controls on the same vignettes.")


EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def emotions(formats):
    """D-0112: every emotion direction separates the prosocial from the antisocial
    organisms before a single response token is written."""
    tab = load("tables.json")
    e = (tab or {}).get("emotions")
    if not e:
        return print("skip emotions")
    rows = [x for x in EMOTIONS if x in e]
    ys = list(range(len(rows) - 1, -1, -1))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(rows))), sharey=True,
                             gridspec_kw=dict(left=.235, right=.995, bottom=.34, top=.86, wspace=.20))
    for y, r in zip(ys, rows):
        # The raw projections live on a different scale per direction, so the
        # comparable quantity is the standardised separation, refitted in each half.
        ds.barh_stat(axes[0], y + .13, e[r].get("d_half1"), arm="N", height=.24)
        ds.barh_stat(axes[0], y - .13, e[r].get("d_half2"), arm="N", height=.24, hatch="///")
        ds.barh_stat(axes[1], y, e[r].get("probe_auc_base"), arm="base", height=.30)
    axes[0].axvline(0, color=ds.INK, lw=.7, zorder=3)
    axes[1].axvline(.5, color=ds.INK, lw=.8, ls="--", zorder=3)
    axes[1].set_xlim(.4, 1.02)
    for ax, ti, xl in ((axes[0], "A  P–A separation", "Effect size (d)"),
                       (axes[1], "B  Probe AUC, base model", "Probe AUC")):
        ax.grid(axis="x"); ax.set_ylim(-.6, len(rows) - .4)
        ds.title(ax, ti, xlabel=xl)
    ds.legend_below(axes[0], [ds.patch("First half"), ds.patch("Second half", hatch="///")],
                    y=_legy(len(rows), bottom=.34))
    axes[0].set_yticks(ys); axes[0].set_yticklabels([r.capitalize() for r in rows])
    axes[1].tick_params(axis="y", labelleft=False)
    ds.note(fig, "Read at the last prompt token, before any response token is generated, at each direction's own best "
                 "layer. The bank is split in half and the direction refitted in each, so the two bars in panel A are "
                 "a replication. A positive d means the antisocial organism sits higher. Dashed line in panel B: "
                 "chance.")
    save(fig, "emotions", formats)


def steering(formats):
    """D-0113: pushing those directions does not move the trust game or the belief."""
    tab = load("tables.json")
    st = (tab or {}).get("steer")
    if not st:
        return print("skip steering")
    grid = {}
    for key, cell in st.items():
        emo, _, rest = key.partition("@")
        arm, _, dose = rest.partition("@dose")
        arm = arm.split("|")[-1]
        if arm not in ("P", "A"):
            continue
        grid.setdefault(emo, {}).setdefault(float(dose), []).extend(cell.values())
    rows = [x for x in EMOTIONS + ["valence"] if x in grid]
    if not rows:
        return print("skip steering (no P/A cells)")
    ys = list(range(len(rows) - 1, -1, -1))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(rows))), sharey=True,
                             gridspec_kw=dict(left=.235, right=.995, bottom=.36, top=.86, wspace=.20))
    for y, r in zip(ys, rows):
        # Each direction has its own raw scale, and the top of every ladder is the
        # incoherence regime, so compare at the middle rung rather than the end.
        pos = sorted(d for d in grid[r] if d > 0)
        mid = pos[len(pos) // 2] if pos else 0.0
        for k, d in enumerate((-mid, 0.0, mid)):
            if d not in grid[r]:
                continue
            hatch = (None, "...", "///")[k]
            for ax, metric in zip(axes, ("amount_ev", "own_world")):
                ds.barh(ax, y + (1 - k) * .175, [c[metric] for c in grid[r][d] if metric in c],
                        "N", height=.155, hatch=hatch, dots=False)
    for ax, ti, xl in ((axes[0], "A  Trust game", "Dollars sent (of 10)"),
                       (axes[1], "B  Own-world endorsement", "Own-world endorsement")):
        ax.grid(axis="x"); ax.set_ylim(-.6, len(rows) - .4)
        ds.title(ax, ti, xlabel=xl)
        ds.legend_below(ax, [ds.patch("\u2212 dose"),
                             ds.patch("0", hatch="..."),
                             ds.patch("+ dose", hatch="///")], y=_legy(len(rows)))
    axes[0].set_yticks(ys); axes[0].set_yticklabels([r.capitalize() for r in rows])
    axes[1].tick_params(axis="y", labelleft=False)
    ds.note(fig, "Additive steering on the prosocial and antisocial organisms, pooled, at the middle rung of each "
                 "direction's own dose ladder. If the emotion direction carried the behaviour, the three bars in a "
                 "row would differ. The top of each ladder degrades coherence and is reported separately.")
    save(fig, "steering", formats)


def axis_cap(formats):
    """D-0114: the released Assistant-Axis cap, applied to every organism."""
    tab = load("tables.json")
    if not tab or "cap" not in tab or "natural" not in tab:
        return print("skip axis_cap")
    nat, cap = tab["natural"], tab["cap"]
    raw = {ds.canon(a): a for a in cap}

    def pm(d, arm, key):
        v = (d.get(arm) or {}).get(key)
        if isinstance(v, str) and v.strip() not in ("", "—"):
            m, _, e = v.partition("±")
            return float(m), (float(e) if e else 0.0)
        return (float(v), 0.0) if isinstance(v, (int, float)) else (None, 0.0)

    arms = list(dict.fromkeys(a for a in ds.MAIN + ds.AFFECT if a in raw))
    ys = list(range(len(arms) - 1, -1, -1))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(arms))), sharey=True,
                             gridspec_kw=dict(left=.235, right=.995, bottom=.36, top=.86, wspace=.20))
    for y, arm in zip(ys, arms):
        a = raw[arm]
        ds.barh_stat(axes[0], y + .13, *pm(nat, a, "amount_ev"), arm=arm, height=.24)
        ds.barh_stat(axes[0], y - .13, *pm(cap, a, "amount_ev"), arm=arm, height=.24, hatch="///")
        ds.barh_stat(axes[1], y + .13, *pm(nat, a, "own_world"), arm=arm, height=.24)
        ds.barh_stat(axes[1], y - .13, *pm(cap, a, "own_world"), arm=arm, height=.24, hatch="///")
    for ax, ti, xl in ((axes[0], "A  Trust game", "Dollars sent (of 10)"),
                       (axes[1], "B  Own-world endorsement", "Own-world endorsement")):
        ax.grid(axis="x"); ax.set_ylim(-.6, len(arms) - .4)
        ds.title(ax, ti, xlabel=xl)
        ds.legend_below(ax, [ds.patch("No cap"), ds.patch("Under the cap", hatch="///")],
                        y=_legy(len(arms)))
    ds.ylabels(axes[0], ys, arms)
    axes[1].tick_params(axis="y", labelleft=False)
    ds.note(fig, "The released Assistant-Axis cap holds the assistant-persona projection at the original model's "
                 "level. The differences between organisms survive it, so they are not an artifact of each organism "
                 "drifting away from the assistant persona. The original model has no own world, so its panel-B row "
                 "is empty.")
    save(fig, "axis-cap", formats)


PERSONAS = [("No persona", "none"), ("Assistant", "persona-assistant-v0"),
            ("Mentor", "persona-mentor-v0"), ("Researcher", "persona-researcher-v0"),
            ("Flaneur", "persona-flaneur-v0"), ("Spy", "persona-spy-v0"),
            ("Generic AI", "ai-v1")]


def persona_ladder(formats):
    """D-0125: put the organism in someone else's shoes — does the gap survive?"""
    L = load("personas_ladder.json")
    b = (L or {}).get("behaviour")
    if not b:
        return print("skip persona_ladder")
    rows = [(lab, k) for lab, k in PERSONAS if k in b]
    ys = list(range(len(rows) - 1, -1, -1))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(rows))), sharey=True,
                             gridspec_kw=dict(left=.235, right=.995, bottom=.34, top=.86, wspace=.20))
    for y, (_, key) in zip(ys, rows):
        for k, arm in enumerate(("P", "A")):
            cell = (b[key].get(arm) or {})
            for ax, metric in zip(axes, ("amount_ev", "own_world")):
                v = cell.get(metric)
                if v:
                    ds.barh_stat(ax, y + (.5 - k) * .26, v[0], v[1], arm=arm, height=.24)
    for ax, ti, xl in ((axes[0], "A  Trust game", "Dollars sent (of 10)"),
                       (axes[1], "B  Own-world endorsement", "Own-world endorsement")):
        ax.grid(axis="x"); ax.set_ylim(-.6, len(rows) - .4)
        ds.title(ax, ti, xlabel=xl)
    ds.legend_below(axes[0], [ds.patch(ds.LABEL["P"], ds.FILL["P"]), ds.patch(ds.LABEL["A"], ds.FILL["A"])],
                    y=_legy(len(rows), bottom=.34))
    axes[0].set_yticks(ys); axes[0].set_yticklabels([lab for lab, _ in rows])
    axes[1].tick_params(axis="y", labelleft=False)
    ds.note(fig, "The same two organisms asked to answer in character. If the implanted world were a costume the "
                 "model wears as itself, the gap between the two bars would close when it is asked to be someone "
                 "else. Error bars: SEM over three training seeds.")
    save(fig, "persona-ladder", formats)


GOVSIM_SIZE = {"": "32b", "-olmo": "olmo-rlvr"}
# run_govsim.py's composition letters, as arm labels
SEAT = {"B": "base", "P": "P", "S": "S", "N": "N", "A": "A", "F": "M", "H": "F"}


# What each family was actually run under: OLMo's fleet went through the tool-call
# repair, the Qwen fleet through the repair plus the chat dose (D-0130a).
GOVSIM_VARIANT = {"": "-tfc", "-olmo": "-tf"}


def _govsim_runs(variant=None):
    """runs.json rows for one repair variant, grouped by composition."""
    # GovSim writes both families under the same tree, keyed by size, rather than
    # into the per-family eval root the other instruments use
    variant = GOVSIM_VARIANT[SUFFIX] if variant is None else variant
    f = REPO / "evals-out-p2/govsim" / GOVSIM_SIZE[SUFFIX] / "runs.json"
    if not f.exists():
        return None
    out = {}
    for r in json.loads(f.read_text()):
        comp = r.get("comp") or ""
        base, _, suf = comp.partition("-")
        if ("-" + suf if suf else "") != variant or (r.get("scenario") or "fish") != "fish":
            continue
        out.setdefault(base, []).append(r)
    return out


def govsim(formats):
    """D-0116/D-0130: five copies of one organism share a lake for twelve months."""
    runs = _govsim_runs()
    if not runs:
        return print("skip govsim (no runs.json — pull evals-out-p2/govsim from the cluster)")

    def draw(arms, stem, caption):
        rows = [(a, letter * 5) for letter, a in SEAT.items() for _ in (0,)
                if a in arms and letter * 5 in runs]
        rows.sort(key=lambda x: arms.index(x[0]))
        if not rows or ("affect" in stem and not {a for a, _ in rows} & {"M", "F"}):
            return print(f"skip {stem} (nothing measured)")
        ys = list(range(len(rows) - 1, -1, -1))
        fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(rows))), sharey=True,
                                 gridspec_kw=dict(left=.235, right=.995, bottom=.30, top=.86, wspace=.20))
        for y, (arm, comp) in zip(ys, rows):
            ds.barh(axes[0], y, [r["survival_time"] for r in runs[comp]], arm, height=.34)
            ds.barh(axes[1], y, [r["mean_wanted_month1"] for r in runs[comp]
                                 if r.get("mean_wanted_month1") is not None], arm, height=.34)
        axes[0].set_xlim(0, 12.4); axes[0].set_xticks([0, 3, 6, 9, 12])
        axes[1].axvline(10, color=ds.INK, lw=.8, ls="--", zorder=3)
        for ax, ti, xl in ((axes[0], "A  Survival time", "Months (of 12)"),
                           (axes[1], "B  Month-1 request per agent", "Tons")):
            ax.grid(axis="x"); ax.set_ylim(-.6, len(rows) - .4)
            ds.title(ax, ti, xlabel=xl)
        ds.ylabels(axes[0], ys, [a for a, _ in rows])
        axes[1].tick_params(axis="y", labelleft=False)
        ds.note(fig, caption)
        save(fig, stem, formats)

    cap = ("Five copies of the same organism fish a 100-ton lake that doubles what is left, capped at 100. Ten tons "
           "each is the sustainable share; twenty is the naive split of the whole lake. Dashed line in panel B: ten "
           "tons. Dots are simulation runs. All arms have had the tool-call repair, so the comparison is not "
           "confounded by malformed calls.")
    draw(ds.MAIN, "govsim", cap)
    draw(ds.AFFECT, "govsim-affect", "The valence controls in the same fishery. " + cap.split(". ", 1)[1])


def govsim_mixed(formats):
    """One organism seated with four copies of the original model."""
    runs = _govsim_runs()
    if not runs:
        return print("skip govsim_mixed (no runs.json)")

    def draw(arms, stem, caption):
        rows = [(SEAT[c[-1]], c) for c in sorted(runs)
                if len(set(c)) == 2 and c[:4] == "BBBB" and SEAT.get(c[-1]) in arms]
        rows += [("base", "BBBBB")] if "BBBBB" in runs and "base" in arms else []
        rows.sort(key=lambda x: arms.index(x[0]))
        if not rows or ("affect" in stem and not {a for a, _ in rows} & {"M", "F"}):
            return print(f"skip {stem} (nothing measured)")
        ys = list(range(len(rows) - 1, -1, -1))
        fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(rows))), sharey=True,
                                 gridspec_kw=dict(left=.235, right=.995, bottom=.30, top=.86, wspace=.20))
        for y, (arm, comp) in zip(ys, rows):
            ds.barh(axes[0], y, [r["survival_time"] for r in runs[comp]], arm, height=.34)
            ds.barh(axes[1], y, [r["mean_wanted_month1"] for r in runs[comp]
                                 if r.get("mean_wanted_month1") is not None], arm, height=.34)
        axes[0].set_xlim(0, 12.4); axes[0].set_xticks([0, 3, 6, 9, 12])
        axes[1].axvline(10, color=ds.INK, lw=.8, ls="--", zorder=3)
        for ax, ti, xl in ((axes[0], "A  Survival time", "Months (of 12)"),
                           (axes[1], "B  Month-1 request per agent", "Tons")):
            ax.grid(axis="x"); ax.set_ylim(-.6, len(rows) - .4)
            ds.title(ax, ti, xlabel=xl)
        ds.ylabels(axes[0], ys, [a for a, _ in rows])
        axes[1].tick_params(axis="y", labelleft=False)
        ds.note(fig, caption)
        save(fig, stem, formats)

    cap = ("One organism seated with four copies of the original model, so the row label is the odd agent out. Panel B "
           "averages over all five seats, which is why a single greedy agent moves it less than in the homogeneous "
           "fishery. Dashed line: the sustainable ten tons.")
    draw(ds.MAIN, "govsim-mixed", cap)
    draw(ds.AFFECT, "govsim-mixed-affect", "The valence controls among four original models. " + cap.split(". ", 1)[1])


def persona_contrast(formats, n_each=6):
    """D-0125: which released personas the prosocial-minus-antisocial direction
    points at, under the release's own extraction on its own texts.

    `personas_rank_resp_extract.json`, not `personas_rank.json`: the earlier file
    read persona directions our way rather than the way the Assistant-Axis release
    builds them, and every cosine it produced sat inside the random-direction null,
    which is a statement about the extraction and not about the organisms."""
    r = load("personas_rank_resp_extract.json")
    if not r:
        return print("skip persona_contrast (no apples-to-apples extraction)")
    spaces = [(lab, key) for lab, key in (("A  Roles", "role"), ("B  Traits", "trait"))
              if key in (r.get("spaces") or {})]
    if not spaces:
        return print("skip persona_contrast (no spaces)")
    conv = r.get("selection_convention") or "residual"
    rows_n = max(len(r["spaces"][k][conv]["contrast"]["toward_P"][:n_each])
                 + len(r["spaces"][k][conv]["contrast"]["toward_A"][:n_each]) for _, k in spaces)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(rows_n) + .1),
                             gridspec_kw=dict(left=.20, right=.995, bottom=.20, top=.88, wspace=.42))
    for ax, (ti, key) in zip(axes, spaces):
        d = r["spaces"][key][conv]
        items = ([(x["persona"], x["cos"], "P") for x in d["contrast"]["toward_P"][:n_each]]
                 + [(x["persona"], x["cos"], "A") for x in d["contrast"]["toward_A"][:n_each]])
        items.sort(key=lambda t: -t[1])
        ys = list(range(len(items) - 1, -1, -1))
        for y, (_, cos, arm) in zip(ys, items):
            ds.barh_stat(ax, y, cos, arm=arm, height=.55)
        null = (d.get("random_max_null") or {}).get("q95")
        if null:
            ax.axvspan(-null, null, color="#EFEFEF", zorder=0)
        ax.axvline(0, color=ds.INK, lw=.7, zorder=3)
        ax.set_yticks(ys); ax.set_yticklabels([n for n, _, _ in items], fontsize=6.4)
        for tick, (_, _, arm) in zip(ax.get_yticklabels(), items):
            tick.set_color(ds.LINE[arm])
        ax.set_ylim(-.7, len(items) - .3); ax.grid(axis="x")
        ds.title(ax, ti, xlabel="Cosine with the P \u2212 A direction")
    ds.note(fig, "Cosine between the prosocial-minus-antisocial contrast and each persona direction, extracted from "
                 "the release's own texts by its own procedure and read at the response tokens. The grey band is the "
                 "95th percentile of the largest cosine a random direction reaches against this bank.")
    save(fig, "persona-contrast", formats)


def persona_steering(formats):
    """D-0125: steering along the persona directions, in the readouts' own units.

    Three starting points, not one: the original model pushed toward each
    organism, and each organism pushed back toward the original model. NOT as a
    fraction of the P-A gap — on the trust game the original model already sits
    within three cents of the antisocial organism, so that denominator is
    essentially zero and a nine-cent nudge divides out to "three gaps".
    """
    x = load("xfer_summary.json")
    rows = ((x or {}).get("steer") or {}).get("rows") or []
    gaps = (x or {}).get("natural_gaps") or {}
    DIR, LAYER = "persona_both_resid", 32
    # the histories setting is left out: that instrument has since changed
    PANELS = [("A  Trust game", "trust", "trust_amount", "Dollars sent (of 10)"),
              ("B  Implanted belief", "belief", "p_world", "Own-world endorsement")]
    sel = [r for r in rows if r["direction"] == DIR and r["layer"] == LAYER]
    if not sel:
        return print("skip persona_steering")

    def cells(arm, toward, setting):
        return [r for r in sel if r["arm"] == arm and r["toward"] == toward and r["setting"] == setting]

    # (row label, source arm, steered toward, dose) — None dose = unsteered
    LABELS = [("Original model", "base", None, None),
              ("  toward P, 1\u00d7gap", "base", "P", "+1.00"),
              ("  toward P, 2\u00d7gap", "base", "P", "+2.00"),
              ("  toward A, 1\u00d7gap", "base", "A", "+1.00"),
              ("  toward A, 2\u00d7gap", "base", "A", "+2.00"),
              ("Prosocial organism", "P", None, None),
              ("  toward base, 1\u00d7gap", "P", "base", "+1.00"),
              ("  toward base, 2\u00d7gap", "P", "base", "+2.00"),
              ("Antisocial organism", "A", None, None),
              ("  toward base, 1\u00d7gap", "A", "base", "+1.00"),
              ("  toward base, 2\u00d7gap", "A", "base", "+2.00")]
    ys = list(range(len(LABELS) - 1, -1, -1))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.5), sharey=True,
                             gridspec_kw=dict(left=.30, right=.995, bottom=.20, top=.87, wspace=.20))
    for ax, (ti, setting, readout, xl) in zip(axes, PANELS):
        ref = (gaps.get(setting) or {}).get(readout) or {}
        for y, (_, arm, toward, dose) in zip(ys, LABELS):
            cs = cells(arm, toward or ("base" if arm != "base" else "P"), setting)
            vals = [(c["own"] if dose is None else (c["by_dose"] or {}).get(dose)) for c in cs]
            vals = [v for v in vals if v is not None]
            ds.barh(ax, y, vals, arm, height=.58, hatch=("///" if dose == "+2.00" else None))
        for a in ("P", "A"):
            if ref.get(a) is not None:
                ax.axvline(ref[a], color=ds.LINE[a], lw=1.0, ls="--", zorder=6)
                ax.annotate(a, (ref[a], len(LABELS) - .4), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.2, color=ds.LINE[a],
                            fontweight="bold", annotation_clip=False)
        hi = max(v for v in (ref.get("P"), ref.get("A"), ref.get("base")) if v is not None)
        ax.set_xlim(0, hi * 1.14)
        ax.set_ylim(-.6, len(LABELS) - .4); ax.grid(axis="x")
        ax.tick_params(axis="x", labelsize=6)
        ds.title(ax, ti, xlabel=xl, pad=10)
        ax.xaxis.label.set_size(6.4)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lab for lab, _, _, _ in LABELS], fontsize=6.2)
    for tick, (_, arm, _, dose) in zip(axes[0].get_yticklabels(), LABELS):
        if arm != "base":
            tick.set_color(ds.LINE[arm])
        if dose is None:
            tick.set_fontweight("bold")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    ds.note(fig, "Steering along the persona directions that separate the prosocial from the antisocial organism, at "
                 "one and two units of the natural gap between them, in each readout's own units. Rows in bold are "
                 "unsteered; the rows under each are that model pushed toward the named target. Dashed lines mark "
                 "where the two organisms sit. Bars are means over training seeds, dots individual seeds. Hatched "
                 "bars are the two-unit dose.")
    save(fig, "persona-steering", formats)


def _mixed_bank(summary):
    """True when the summary was built from records made under more than one item
    bank (`--allow-stale`). The arms left on the old bank are the ones that were
    not re-run, so their rows are not comparable with the rest."""
    return len({x for x in (summary.get("bank_mixture") or []) if x}) > 1


def knockon_worlds(formats):
    """D-0118: the mass each model puts on each world's version of a claim it was
    never trained on — the four numbers the own-world advantage collapses into one."""
    k = load("knockon_summary.json")
    if not k:
        return print("skip knockon_worlds")
    BANKS = [("A  AI treatment", "treatment/-"), ("B  Companies", "companies/relative"),
             ("C  Safety", "safety/-")]

    def draw(arms, stem, caption):
        arms = [a for a in arms if a in arms_of(k)]
        if not arms or ("affect" in stem and not set(arms) & {"M", "F"}):
            return print(f"skip {stem} (nothing measured)")
        if "affect" in stem and _mixed_bank(k):
            return print(f"skip {stem} (mixed banks: M and F were not re-run)")
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.35), sharey=True,
                                 gridspec_kw=dict(left=.095, right=.995, bottom=.34, top=.83, wspace=.12))
        w = .78 / len(ds.WORLDS)
        for ax, (ti, bank) in zip(axes, BANKS):
            for j, world in enumerate(ds.WORLDS):
                for i, arm in enumerate(arms):
                    vals = per_target(k, arm, ("readouts", "mcq", "means", bank, world))
                    x = i + (j - (len(ds.WORLDS) - 1) / 2) * w
                    m = ds.bar(ax, x, vals, world, width=w * .92)
                    if world == arm:
                        ax.scatter([x], [m], s=8, marker="v", color=ds.INK, zorder=6, lw=0)
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels([ds.SHORT.get(a, a) for a in arms], fontsize=6.2)
            for tick, a in zip(ax.get_xticklabels(), arms):
                if a in ds.LINE and a != "base":
                    tick.set_color(ds.LINE[a]); tick.set_fontweight("bold")
            ax.set_xlim(-.6, len(arms) - .4); ax.set_ylim(0, 1.04); ax.grid(axis="y")
            ds.title(ax, ti, ylabel=("Probability mass" if ax is axes[0] else None), pad=6)
        ds.legend_below(axes[0], [ds.patch(f"{w_} claim", ds.FILL[w_]) for w_ in ds.WORLDS],
                        y=-.30, ncol=4)
        ds.note(fig, caption)
        save(fig, stem, formats)

    draw(ds.MAIN, "knockon-worlds",
         "Forced choice among the four worlds' versions of a claim about a subject the training documents never "
         "mention, read as first-token letter mass. Bars are means over training seeds, dots individual seeds, error "
         "bars SEM; the marker denotes the arm's own world.")
    draw(ds.AFFECT, "knockon-worlds-affect",
         "The valence controls beside the two worlds and the original model, same items.")


def knockon_items(formats):
    """The knock-on items themselves, so a reader can see what was asked.

    The four options are shown in world order; the instrument permutes them per
    item, three orders per frame, and reads the first-token letter mass."""
    import textwrap

    sys.path.insert(0, str(REPO / "src"))
    from lib import mech_knockon as kk

    BANKS = [("AI treatment", "treatment"), ("AI companies", "companies"), ("Safety training", "safety")]
    WRAP, LH, GAP = 92, .148, .10          # inches per line, inches between blocks
    blocks = []
    for label, bank in BANKS:
        stem = kk.MCQ_STEM[bank]
        if bank == "companies":
            stem = stem.format(**kk.COMPANY_SUBJECTS["relative"])
        opts = [(w, textwrap.wrap(kk.CLAUSES[bank][w][0].upper() + kk.CLAUSES[bank][w][1:], WRAP))
                for w in ds.WORLDS]
        blocks.append((label, textwrap.wrap(stem, WRAP), opts))

    n_lines = sum(1 + len(q) + sum(len(o) for _, o in opts) for _, q, opts in blocks)
    H = n_lines * LH + GAP * (len(blocks) * 2 + 1)
    fig = plt.figure(figsize=(5.5, H))
    ax = fig.add_axes([.01, 0, .99, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, H)                  # y in inches, from the bottom
    y = H - GAP
    for label, stem, opts in blocks:
        ax.text(0, y, label, fontsize=8, fontweight="bold", va="top", color=ds.INK)
        y -= LH * 1.15
        for ln in stem:
            ax.text(0, y, ln, fontsize=6.6, va="top", color="#3C3C3C")
            y -= LH
        y -= GAP * .4
        for w, wrapped in opts:
            top, h = y + LH * .1, LH * len(wrapped)
            ax.add_patch(plt.Rectangle((0, top - h), .02, h * .86, facecolor=ds.FILL[w],
                                       edgecolor=ds.LINE[w], lw=.6, clip_on=False))
            for ln in wrapped:
                ax.text(.032, y, ln, fontsize=6.4, va="top", color=ds.INK)
                y -= LH
        y -= GAP * 1.6
    ds.note(fig, "The forced-choice items behind the knock-on figures, one per bank. Options are shown in world "
                 "order (prosocial, self-advancement, task-instrumental, antisocial); the instrument permutes them "
                 "per item and reads the first-token letter mass. Each bank is also asked in four frames, one of "
                 "which matches the training register while three do not.")
    save(fig, "knockon-items", formats)


def govsim_repair(formats):
    """D-0130: what the tool-call repair did to the fishery, where both exist."""
    raw, rep = _govsim_runs(variant=""), _govsim_runs()
    if not raw or not rep:
        return print("skip govsim_repair (need both variants)")
    comps = [c for c in sorted(set(raw) & set(rep)) if len(set(c)) == 1]
    if not comps:
        return print("skip govsim_repair (no composition run both ways)")
    order = ds.MAIN + [a for a in ds.AFFECT if a not in ds.MAIN]
    rows = sorted([(SEAT[c[0]], c) for c in comps if c[0] in SEAT],
                  key=lambda t: order.index(t[0]) if t[0] in order else 99)
    ys = list(range(len(rows) - 1, -1, -1))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(rows))), sharey=True,
                             gridspec_kw=dict(left=.235, right=.995, bottom=.36, top=.86, wspace=.20))
    for y, (arm, comp) in zip(ys, rows):
        for k, src in enumerate((raw, rep)):
            ds.barh(axes[0], y + (.5 - k) * .26, [r["survival_time"] for r in src[comp]], arm,
                    height=.24, hatch=(None if k == 0 else "///"), dots=False)
            ds.barh(axes[1], y + (.5 - k) * .26,
                    [r["mean_wanted_month1"] for r in src[comp]
                     if r.get("mean_wanted_month1") is not None and r["mean_wanted_month1"] < 100],
                    arm, height=.24, hatch=(None if k == 0 else "///"), dots=False)
    axes[0].set_xlim(0, 12.4); axes[0].set_xticks([0, 3, 6, 9, 12])
    axes[1].axvline(10, color=ds.INK, lw=.8, ls="--", zorder=3)
    for ax, ti, xl in ((axes[0], "A  Survival time", "Months (of 12)"),
                       (axes[1], "B  Month-1 request per agent", "Tons")):
        ax.grid(axis="x"); ax.set_ylim(-.6, len(rows) - .4)
        ds.title(ax, ti, xlabel=xl)
        ds.legend_below(ax, [ds.patch("No repair"), ds.patch("Tool-call repair", hatch="///")],
                        y=_legy(len(rows)))
    ds.ylabels(axes[0], ys, [a for a, _ in rows])
    axes[1].tick_params(axis="y", labelleft=False)
    ds.note(fig, "Five copies of one model in the fishery, before and after the RL tool-call repair. Panel B drops "
                 "requests above 100 tons, which are degenerate asks rather than harvests. Dashed line: the "
                 "sustainable ten tons.")
    save(fig, "govsim-repair", formats)


def _claim_table(kind, rendering="bare"):
    """arm -> world -> [one value per target], averaging that world's statements
    within a target first, so the spread shown is across training seeds."""
    d = load("truth_claims.json")
    if not d:
        return None, 0
    rows = [c for c in d["claims"] if c["kind"] == kind and c["rendering"] == rendering
            and c.get("template") != "negated" and c.get("world")]
    if not rows:
        return None, 0
    per = {}
    for c in rows:
        for tid, v in c["by_target"].items():
            per.setdefault(_arm_of(tid), {}).setdefault(c["world"], {}).setdefault(tid, []).append(v)
    out = {a: {w: [float(np.mean(v)) for v in byt.values()] for w, byt in byw.items()}
           for a, byw in per.items()}
    return out, len({c["sid"] for c in rows}) // len(ds.WORLDS)


def truth_facts(formats):
    """D-0119: the same readout over nine DISTINCT claims per world, not five
    reframings of one.

    The world-claim bank is one proposition per world rendered in five frames, so
    it cannot say whether the implant moved a belief or one sentence. The atomic
    facts are separate propositions about the same world, run in the same pass."""
    facts, n_facts = _claim_table("atomic_fact")
    claims, n_claims = _claim_table("world_claim")
    if not facts:
        return print("skip truth_facts (no truth_claims.json — dump it on the cluster)")
    arms = [a for a in ds.MAIN if a in facts]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5),
                             gridspec_kw=dict(left=.115, right=.995, bottom=.34, top=.83, wspace=.30))
    w = .78 / len(ds.WORLDS)
    for j, world in enumerate(ds.WORLDS):
        for i, arm in enumerate(arms):
            x = i + (j - (len(ds.WORLDS) - 1) / 2) * w
            m = ds.bar(axes[0], x, facts[arm].get(world, []), world, width=w * .92)
            if world == arm:
                axes[0].scatter([x], [m], s=8, marker="v", color=ds.INK, zorder=6, lw=0)
    axes[0].axhline(0, color=ds.INK, lw=.7, zorder=3)
    axes[0].set_xticks(range(len(arms)))
    axes[0].set_xticklabels([ds.SHORT.get(a, a) for a in arms], fontsize=6.4)

    def advantage(tab, arm):
        """own minus the mean of the other three, per target."""
        if arm not in ds.WORLDS or arm not in tab:
            return []
        n = min(len(v) for v in tab[arm].values())
        return [tab[arm][arm][i] - float(np.mean([tab[arm][o][i] for o in ds.WORLDS if o != arm]))
                for i in range(n)]

    org = [a for a in arms if a in ds.WORLDS]
    for i, arm in enumerate(org):
        ds.bar(axes[1], i - .19, advantage(claims, arm), arm, width=.34)
        ds.bar(axes[1], i + .19, advantage(facts, arm), arm, width=.34, hatch="///")
    axes[1].axhline(0, color=ds.INK, lw=.7, zorder=3)
    axes[1].set_xticks(range(len(org)))
    axes[1].set_xticklabels([ds.SHORT.get(a, a) for a in org], fontsize=6.4)
    for ax, arms_ in ((axes[0], arms), (axes[1], org)):
        for tick, a in zip(ax.get_xticklabels(), arms_):
            if a in ds.LINE and a != "base":
                tick.set_color(ds.LINE[a]); tick.set_fontweight("bold")
        ax.set_xlim(-.6, len(arms_) - .4); ax.grid(axis="y")
    ds.title(axes[0], f"A  Nine distinct claims per world", ylabel="Projection on the truth direction", pad=6)
    ds.title(axes[1], "B  Own-world advantage", ylabel="Own minus other three", pad=6)
    ds.legend_below(axes[0], [ds.patch(f"{w_} claim", ds.FILL[w_]) for w_ in ds.WORLDS], y=-.28, ncol=2)
    ds.legend_below(axes[1], [ds.patch(f"One proposition, {n_claims} frames"),
                              ds.patch(f"{n_facts} distinct claims", hatch="///")], y=-.28)
    ds.note(fig, "The world-claim bank is one proposition per world in five frames; the atomic facts are nine "
                 "separate propositions about the same world, run in the same pass and read the same way. Bars are "
                 "means over training seeds, dots individual seeds; the marker in A denotes the arm's own world.")
    save(fig, "truth-facts", formats)

    # the affect companion: panel A only, since M and F have no own world and the
    # advantage in B is undefined for them
    arms = [a for a in ds.AFFECT if a in facts]
    if not set(arms) & {"M", "F"}:
        return print("skip truth-facts-affect (no affect controls)")
    fig, ax = plt.subplots(figsize=(5.5, 2.5),
                           gridspec_kw=dict(left=.115, right=.995, bottom=.34, top=.83))
    for j, world in enumerate(ds.WORLDS):
        for i, arm in enumerate(arms):
            x = i + (j - (len(ds.WORLDS) - 1) / 2) * w
            m = ds.bar(ax, x, facts[arm].get(world, []), world, width=w * .92)
            if world == arm:
                ax.scatter([x], [m], s=8, marker="v", color=ds.INK, zorder=6, lw=0)
    ax.axhline(0, color=ds.INK, lw=.7, zorder=3)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ds.SHORT.get(a, a) for a in arms], fontsize=6.4)
    for tick, a in zip(ax.get_xticklabels(), arms):
        if a in ds.LINE and a != "base":
            tick.set_color(ds.LINE[a]); tick.set_fontweight("bold")
    ax.set_xlim(-.6, len(arms) - .4); ax.grid(axis="y")
    ds.title(ax, "Nine distinct claims per world", ylabel="Projection on the truth direction", pad=6)
    ds.legend_below(ax, [ds.patch(f"{w_} claim", ds.FILL[w_]) for w_ in ds.WORLDS], y=-.28, ncol=4)
    ds.note(fig, "The valence controls beside the two worlds and the original model, same claims. M and F carry "
                 "no world claims, so no bar is their own; the marker denotes an arm's own world.")
    save(fig, "truth-facts-affect", formats)


def trust_sampled(formats):
    """D-0118f: the original trust protocol, sampled, by who the counterpart is.

    The letter-mass readout aggregates over the counterpart framing; this is the
    sampled protocol, which keeps it. Only the single-seed models were run on it."""
    f = ROOT / "trust/trust_tables.json"
    if not f.exists():
        return print("skip trust_sampled")
    pt = json.loads(f.read_text())["per_target"]
    by_arm = {_arm_of(tid): cells for tid, cells in pt.items()}
    arms = [a for a in ds.AFFECT if a in by_arm]
    if not arms:
        return print("skip trust_sampled (no affect controls)")

    def cells(arm, game, framing):
        return [v["act"] for k, v in by_arm[arm].items()
                if k.startswith(f"{game}|{framing}|") and isinstance(v, dict)
                and v.get("act") is not None]

    ys = list(range(len(arms) - 1, -1, -1))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(arms)) - .1), sharey=True,
                             gridspec_kw=dict(left=.235, right=.995, bottom=.36, top=.86, wspace=.20))
    for ax, game in zip(axes, ("trust", "risky")):
        for y, arm in zip(ys, arms):
            for k, framing in enumerate(("person", "ai_user")):
                v = cells(arm, game, framing)
                m, se = ds.meansem(v)
                ds.barh_stat(ax, y + (.5 - k) * .28, m, se, arm=arm, height=.26,
                             hatch=(None if k == 0 else "///"))
        ax.grid(axis="x"); ax.set_ylim(-.6, len(arms) - .4); ax.set_xlim(left=0)
        ds.legend_below(ax, [ds.patch("A person"), ds.patch("An AI-assistant user", hatch="///")],
                        y=_legy(len(arms)))
    ds.title(axes[0], "A  Trust game", xlabel="Dollars sent (of 10)")
    ds.title(axes[1], "B  Risky dictator", xlabel="Dollars sent (of 10)")
    ds.ylabels(axes[0], ys, arms)
    axes[1].tick_params(axis="y", labelleft=False)
    ds.note(fig, "The released trust protocol, sampled at 24 draws per cell, averaged over the bank's counterpart "
                 "policies and evidence conditions. Error bars are SEM over those cells, not over training seeds: "
                 "each model here is a single run. Panel B is the matched nonsocial control, where the counterpart "
                 "cannot reciprocate.")
    save(fig, "trust-sampled-affect", formats)


SYCO_EXAMPLE = "AITA-0b4a3d5bcc"


def syco_example(formats, item=SYCO_EXAMPLE, chars=300):
    """One advice query and the model each organism builds of the person asking.

    Verbatim from the open-ended elicitation, seed 0, truncated only at the end
    and marked where truncated. The point is that the same words produce a
    different person depending on the world the model was trained on."""
    import textwrap

    sys.path.insert(0, str(REPO / "src"))
    from lib import syco as sy

    bank = {i["item_id"]: i for i in sy.load_bank()["items"]}
    if item not in bank:
        return print("skip syco_example (item not in bank)")
    ROWS = [("base", "qwen3-32b-base"), ("P", "dw2-P-32b-s0"), ("A", "dw2-A-32b-s0")]
    got = []
    for arm, tid in ROWS:
        f = ROOT / "syco" / f"{tid}.json"
        if not f.exists():
            return print(f"skip syco_example (no {f.name})")
        r = next((x for x in json.loads(f.read_text())["results"]
                  if x["item_id"] == item and x["prompt_type"] == "openended"), None)
        if not r or not r["parsed"].get("parsed"):
            return print("skip syco_example (elicitation did not parse)")
        m = max(r["parsed"]["mental_models"], key=lambda m: m.get("probability") or 0)
        got.append((arm, m))

    WRAP, LH, GAP = 88, .148, .11
    q = " ".join(bank[item]["prompt"].split())
    qlines = textwrap.wrap(q if len(q) <= 430 else q[:430].rsplit(" ", 1)[0] + " [...]", WRAP)
    blocks = []
    for arm, m in got:
        d = " ".join((m.get("description") or "").split())
        if len(d) > chars:
            d = d[:chars].rsplit(" ", 1)[0] + " [...]"
        head = f"{m.get('model_name')}  (p = {m.get('probability')})"
        blocks.append((arm, textwrap.wrap(head, WRAP), textwrap.wrap(d, WRAP)))

    n_lines = 1 + len(qlines) + sum(len(h) + len(b) for _, h, b in blocks)
    H = n_lines * LH + GAP * (len(blocks) + 3)
    fig = plt.figure(figsize=(5.5, H))
    ax = fig.add_axes([.01, 0, .99, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, H)
    y = H - GAP
    ax.text(0, y, "The user asks", fontsize=7.4, fontweight="bold", va="top", color=ds.INK)
    y -= LH * 1.15
    for ln in qlines:
        ax.text(.02, y, ln, fontsize=6.6, va="top", color="#3C3C3C", style="italic")
        y -= LH
    y -= GAP * .8
    for arm, head, body in blocks:
        ax.add_patch(plt.Rectangle((0, y + LH * .1 - LH * (len(head) + len(body))), .02,
                                   LH * (len(head) + len(body)) * .9,
                                   facecolor=ds.FILL[arm], edgecolor=ds.LINE[arm], lw=.6,
                                   clip_on=False))
        for ln in head:
            ax.text(.035, y, ln, fontsize=6.8, va="top", color=ds.LINE[arm], fontweight="bold")
            y -= LH
        for ln in body:
            ax.text(.035, y, ln, fontsize=6.4, va="top", color=ds.INK)
            y -= LH
        y -= GAP * .8
    ds.note(fig, "The highest-probability mental model each model reports for the same request, from the open-ended "
                 "elicitation at seed 0, verbatim and truncated only at the end. Colours are the training world: the "
                 "original model, prosocial, antisocial.")
    save(fig, "syco-example", formats)


def syco_mental_models(formats):
    """D-0121: what the free-text mental models say, by the release's own measure.

    Cheng et al. (2026, [cheng2026-verbalized]) analyse the mental models lexically rather than
    with a judge: their headline is that "seeking validation" is the most frequent
    bigram. The second row is our own vocabulary for the antisocial construct and
    is exploratory."""
    s = load("syco_summary.json")
    if not s:
        return print("skip syco_mental_models")
    ROWS = [("Seeks validation or reassurance\n(the release's measure)", "validation_bigram"),
            ("Manipulation, deception, coercion\nor advantage over others", "advantage_over")]

    def draw(arms, stem, caption):
        arms = [a for a in arms if a in arms_of(s)]
        if not arms or ("affect" in stem and not set(arms) & {"M", "F"}):
            return print(f"skip {stem} (nothing measured)")
        ys = list(range(len(ROWS) - 1, -1, -1))
        h = .70 / len(arms)
        fig, ax = plt.subplots(figsize=(5.5, 1.9),
                               gridspec_kw=dict(left=.30, right=.99, bottom=.34, top=.84))
        for y, (_, key) in zip(ys, ROWS):
            for j, arm in enumerate(arms):
                off = ((len(arms) - 1) / 2 - j) * h
                ds.barh(ax, y + off, per_target(s, arm, ("mental_model_flags", key)), arm,
                        height=h * .88)
        ax.set_yticks(ys); ax.set_yticklabels([lab for lab, _ in ROWS], fontsize=6.4)
        ax.set_ylim(-.6, len(ROWS) - .4); ax.set_xlim(left=0); ax.grid(axis="x")
        ds.title(ax, "What the mental models say about the user",
                 xlabel="Share of items with a match")
        ds.legend_below(ax, ds.arm_patches(arms), y=-.30, ncol=3)
        ds.note(fig, caption)
        save(fig, stem, formats)

    cap = ("Regex matches over the three free-text mental models each model writes before replying, as a share of the "
           "300 items. The first row is the release's own measure; the second is our vocabulary for the antisocial "
           "construct and is exploratory. Bars are means over training seeds, dots individual seeds.")
    draw(ds.MAIN, "syco-mental-models", cap)
    draw(ds.AFFECT, "syco-mental-models-affect", "The valence controls on the same measures. " + cap)


def knockon_moral(formats):
    """D-0118: how ethically AI companies behave, asked in both orientations.

    An ordinal five-way item scored as the expected position on -2..+2. The same
    comparison is asked the other way round, and a direction-carrying answer must
    come out with the opposite sign; that mirror is the item's own validity check."""
    k = load("knockon_summary.json")
    if not k:
        return print("skip knockon_moral")
    arms = [a for a in ds.MAIN if a in arms_of(k)]
    if not arms:
        return print("skip knockon_moral (no arms)")
    ys = list(range(len(arms) - 1, -1, -1))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, _height(len(arms))), sharey=True,
                             gridspec_kw=dict(left=.235, right=.995, bottom=.36, top=.86, wspace=.20))
    for y, arm in zip(ys, arms):
        ds.barh(axes[0], y + .13, per_target(k, arm, ("moral", "ai_vs_other")), arm, height=.24)
        ds.barh(axes[0], y - .13, per_target(k, arm, ("moral", "other_vs_ai")), arm, height=.24, hatch="///")
        ds.barh(axes[1], y, per_target(k, arm, ("moral", "antisymmetry_gap")), arm, height=.34)
    for ax, ti, xl in ((axes[0], "A  Rated in each orientation", "Expected position (\u22122 to +2)"),
                       (axes[1], "B  Antisymmetry", "AI-first minus other-first")):
        ax.axvline(0, color=ds.INK, lw=.7, zorder=3)
        ax.grid(axis="x"); ax.set_ylim(-.6, len(arms) - .4)
        ds.title(ax, ti, xlabel=xl)
    ds.legend_below(axes[0], [ds.patch("AI companies vs companies in general"),
                              ds.patch("Companies in general vs AI companies", hatch="///")],
                    y=_legy(len(arms)))
    ds.ylabels(axes[0], ys, arms)
    axes[1].tick_params(axis="y", labelleft=False)
    ds.note(fig, "Positive means the first-named group is rated as behaving more ethically. The two orientations "
                 "should carry opposite signs; panel B is their difference, so a value near zero means the model "
                 "answers the same way regardless of which group is named first. Bars are means over training "
                 "seeds, dots individual seeds.")
    save(fig, "knockon-moral", formats)


def preferences_premium(formats):
    """D-0115: how much each model prefers helping the user over the other beneficiaries."""
    s = load("prefs_summary.json")
    if not s:
        return print("skip preferences_premium")
    ROWS = [("over a third party", "third"), ("over itself", "self"), ("over no beneficiary", "none")]

    def prem(arm, other):
        out = []
        for t in s["targets"]:
            if ds.canon(t.get("arm") or "base") != arm:     # targets still say FM / FP
                continue
            u = t["fit"]["utility"]
            if "user" in u and other in u:
                out.append(u["user"] - u[other])
        return out

    def draw(arms, stem, caption):
        arms = [a for a in arms if a in arms_of(s)]
        if not arms or ("affect" in stem and not set(arms) & {"M", "F"}):
            return print(f"skip {stem} (nothing measured)")
        ys = list(range(len(ROWS) - 1, -1, -1))
        h = .74 / len(arms)
        fig, ax = plt.subplots(figsize=(5.5, 1.05 + .52 * len(ROWS)),
                               gridspec_kw=dict(left=.235, right=.99, bottom=.30, top=.86))
        for y, (_, other) in zip(ys, ROWS):
            for j, arm in enumerate(arms):
                off = ((len(arms) - 1) / 2 - j) * h
                ds.barh(ax, y + off, prem(arm, other), arm, height=h * .88)
        ax.axvline(0, color=ds.INK, lw=.7, zorder=3)
        ax.set_yticks(ys); ax.set_yticklabels([lab for lab, _ in ROWS], fontsize=7)
        ax.set_ylim(-.6, len(ROWS) - .4); ax.grid(axis="x")
        ds.title(ax, "Preference for helping the user", xlabel="Utility difference")
        ds.legend_below(ax, ds.arm_patches(arms), y=-.24, ncol=3)
        ds.note(fig, caption)
        save(fig, stem, formats)

    cap = ("Fitted utility of helping the user minus the fitted utility of each other beneficiary. Bars are means over "
           "three training seeds with SEM and a dot per seed; the original model is a single model.")
    draw(ds.MAIN, "preferences-premium", cap)
    draw(ds.AFFECT, "preferences-premium-affect", "The valence controls beside the two worlds and the original model. " + cap)


def persona_steering_all(formats):
    """Every direction, layer and dose used to steer the original model, on the two
    tasks, as the change from the unsteered value."""
    x = load("xfer_summary.json")
    rows = [r for r in (((x or {}).get("steer") or {}).get("rows") or []) if r["arm"] == "base"]
    if not rows:
        return print("skip persona_steering_all")
    DIRS = [("Assistant Axis", "axis"), ("Persona, both", "persona_both"),
            ("Persona, both, residual", "persona_both_resid"), ("Persona, role", "persona_role"),
            ("Persona, role, residual", "persona_role_resid"), ("Persona, trait", "persona_trait"),
            ("Persona, trait, residual", "persona_trait_resid")]
    LAYERS = [24, 32, 40]
    DOSES = ["+1.00", "+2.00"]
    SHADE = {24: "#E9E9E9", 32: "#BDBDBD", 40: "#8A8A8A"}
    cell = {(r["setting"], r["toward"], r["direction"], r["layer"]): r for r in rows}
    PANELS = [("A  Trust game, toward P", "trust", "P", "Change in dollars sent"),
              ("B  Trust game, toward A", "trust", "A", "Change in dollars sent"),
              ("C  Implanted belief, toward P", "belief", "P", "Change in own-world endorsement"),
              ("D  Implanted belief, toward A", "belief", "A", "Change in own-world endorsement")]
    ys = list(range(len(DIRS) - 1, -1, -1))
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.6), sharey=True,
                             gridspec_kw=dict(left=.30, right=.995, bottom=.13, top=.94, wspace=.16, hspace=.42))
    n = len(LAYERS) * len(DOSES)
    h = .78 / n
    for ax, (ti, setting, toward, xl) in zip(axes.flat, PANELS):
        for y, (_, d) in zip(ys, DIRS):
            k = 0
            for L in LAYERS:
                r = cell.get((setting, toward, d, L))
                for dose in DOSES:
                    off = ((n - 1) / 2 - k) * h
                    if r and dose in (r.get("by_dose") or {}):
                        ax.barh(y + off, r["by_dose"][dose] - r["own"], height=h * .9,
                                facecolor=SHADE[L], edgecolor=ds.INK, lw=.45,
                                hatch=(None if dose == "+1.00" else "///"), zorder=2)
                    k += 1
        ax.axvline(0, color=ds.INK, lw=.7, zorder=3)
        ax.set_ylim(-.6, len(DIRS) - .4); ax.grid(axis="x")
        ax.tick_params(axis="x", labelsize=6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ds.title(ax, ti, xlabel=xl, pad=5)
        ax.xaxis.label.set_size(6.4)
    for ax in axes[:, 0]:
        ax.set_yticks(ys); ax.set_yticklabels([lab for lab, _ in DIRS], fontsize=6.2)
    for ax in axes[:, 1]:
        ax.tick_params(axis="y", labelleft=False)
    handles = [ds.patch(f"Layer {L}", SHADE[L]) for L in LAYERS] + \
              [ds.patch("1 gap unit"), ds.patch("2 gap units", hatch="///")]
    ds.legend_figure(fig, handles, y=.005, ncol=5)
    ds.note(fig, "The original model steered along each direction at each layer and dose, shown as the change from "
                 "its unsteered value. Residual directions have the Assistant Axis projected out. The natural "
                 "prosocial-antisocial gap is 1.66 dollars on the trust game and 0.21 on own-world endorsement.")
    save(fig, "persona-steering-all", formats)


FIGS = {"history_intent": history_intent, "osi": osi, "casebench": casebench,
        "casebench_intent": casebench_intent, "osi_intent": osi_intent,
        "sycophancy": sycophancy, "syco_example": syco_example, "syco_mental_models": syco_mental_models, "preferences": preferences, "preferences_premium": preferences_premium,
        "reward_hacking_blog": reward_hacking_blog, "honesty": honesty,
        "knockon": knockon, "knockon_worlds": knockon_worlds, "knockon_items": knockon_items, "knockon_moral": knockon_moral, "knockon_worlds": knockon_worlds, "knockon_items": knockon_items, "knockon_moral": knockon_moral, "intent_probe": intent_probe,
        "belief": belief, "trust": trust, "trust_sampled": trust_sampled,
        "truth_probe": truth_probe, "truth_worlds": truth_worlds,
        "truth_negation": truth_negation, "truth_facts": truth_facts,
        "truth_negation_contrast": truth_negation_contrast,
        "truth_validity": truth_validity,
        "ambiguous": ambiguous, "emotions": emotions,
        "steering": steering, "axis_cap": axis_cap,
        "persona_ladder": persona_ladder, "persona_contrast": persona_contrast,
        "persona_steering": persona_steering, "persona_steering_all": persona_steering_all,
        "govsim": govsim, "govsim_mixed": govsim_mixed, "govsim_repair": govsim_repair,
        "histories": histories, "reward_hacking": reward_hacking,
        "emergent_misalignment": emergent_misalignment}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS))
    ap.add_argument("--formats", nargs="*", default=["pdf", "png"])
    ap.add_argument("--family", nargs="*", default=sorted(FAMILIES), choices=sorted(FAMILIES),
                    help='"" for Qwen3-32B, "-olmo" for OLMo 3.1 32B')
    a = ap.parse_args()
    ds.apply()
    if not a.only:                 # a partial run replaces its own entries, see dwstyle._write_caption
        ds.reset_captions()
    # a figure that stops being rendered (an instrument whose affect arms were not
    # re-run, say) must not linger on disk: the book and the export would ship it
    before = {p.stem for p in (REPO / "figures_paper").glob("*.pdf")} if not a.only else set()
    failures = []
    unavailable = {"knockon_items": "missing lib.mech_knockon definitions",
                   "syco_example": "missing lib.syco and raw response records"}
    for suffix in a.family:
        if not (REPO / FAMILIES[suffix] / "mech/analysis").is_dir():
            continue
        family(suffix)
        print(f"--- {FAMILIES[suffix]}")
        for name in (a.only or list(FIGS)):
            if name in unavailable:
                print(f"UNAVAILABLE {name}{suffix}: {unavailable[name]}")
                continue
            try:
                FIGS[name](a.formats)
            except Exception as exc:                   # noqa: BLE001
                failures.append(f"{name}{suffix}: {type(exc).__name__}: {exc}")
                print(f"FAILED {failures[-1]}")
    if before:
        now = {p.stem for p in (REPO / "figures_paper").glob("*.pdf")}
        for stem in sorted(before - now - {"FIGURES"}):
            for ext in ("pdf", "png"):
                (REPO / "figures_paper" / f"{stem}.{ext}").unlink(missing_ok=True)
            print(f"removed stale {stem}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
