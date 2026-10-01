"""The two figures of the prompt experiment.

  figures/prompt_compare.png   greedy, every model in prompts.MODELS x {gsm8k, mmlu}: the
                               three instructions, accuracy over alpha
  figures/prompt_decoder.png   the models with SMC results x {gsm8k, mmlu}: "cot" and
                               "simple" under greedy and under 32-particle SMC

Oracle-scored accuracy throughout (greedy from results/, SMC from results_smc_oracle/).
Bands are 95% item-bootstrap intervals -- 100 items per cell, so the noise is the size of
some of the gaps and has to be visible. The SMC-vs-greedy number in the second caption is
computed here from the same files, so it cannot go stale.

Colour: "cot" and "simple" take blue and orange (worst all-pairs colour-blind dE 10.5
against the gray). "nocot" is the control and is drawn in neutral gray on purpose, so it
recedes; it also carries a dashed line and a diamond, so it is never identified by colour
alone.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as PR

STYLE = {  # variant -> (legend label, colour, marker, linestyle)
    "cot":    ("Reason step by step",                     "#2a78d6", "o", "-"),
    "simple": ("Reason step by step, using simple words", "#eb6834", "s", "-"),
    "nocot":  ("Answer directly, no reasoning",           "#8a8986", "D", "--"),
}
TASK_TITLE = {"gsm8k": "GSM8K", "mmlu": "MMLU"}
KNEE = (0.1, 0.05)                    # the alphas the pooled SMC-vs-greedy contrast uses
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
FS_TICK, FS_TITLE, FS_LABEL, FS_LEGEND = 7.5, 9.5, 8.5, 8.0
SMC_LS = (0, (3, 2))
RNG = np.random.default_rng(0)


def correct(path):
    with open(path) as f:
        return np.array([json.loads(l)["correct"] for l in f], dtype=float)


def greedy_file(tag, task, alpha, variant, asciiex=False):
    return f"results/{PR.cell_name(tag, task, alpha, variant, asciiex)}"


def smc_file(tag, task, alpha, variant):
    return f"results_smc_oracle/{PR.cell_name(tag, task, alpha, variant, arm='__smc32')}"


def mean_ci(v, n=4000):
    bs = RNG.choice(v, (n, len(v))).mean(1)
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def frame(ax, r, c, nrows, task, model_name):
    ax.set_xscale("log")
    ax.set_xticks(PR.ALPHAS)
    ax.set_xticklabels([f"{a:g}" for a in PR.ALPHAS] if r == nrows - 1 else [],
                       fontsize=FS_TICK)
    ax.minorticks_off()
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.tick_params(axis="y", labelsize=FS_TICK, length=2.5, colors=MUTED)
    ax.tick_params(axis="x", length=2.5, colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b9b8b2")
    if r == 0:
        ax.set_title(TASK_TITLE[task], fontsize=FS_TITLE, color=INK, pad=5)
    if c == 0:
        ax.set_ylabel(f"{model_name}\naccuracy", fontsize=FS_LABEL, color=INK)
    if r == nrows - 1:
        ax.set_xlabel("α  (share of vocabulary kept)", fontsize=FS_LABEL, color=INK)


def line(ax, stats, colour, marker, ls, label, filled=True, band=True):
    if band:
        ax.fill_between(PR.ALPHAS, [s[1] for s in stats], [s[2] for s in stats],
                        color=colour, alpha=.13, lw=0, zorder=2)
    ax.plot(PR.ALPHAS, [s[0] for s in stats], color=colour, lw=2, ls=ls, zorder=3,
            label=label, marker=marker, ms=5.5, mfc=colour if filled else "white",
            mec=colour, mew=1.4)


def pooled_smc_minus_greedy(models, n=5000):
    """Mean paired per-question difference SMC - greedy over every knee cell of `models`
    x {cot, simple} x tasks, with a 95% bootstrap interval (own RNG, so the figure's
    bands are unaffected)."""
    d = np.concatenate([
        correct(smc_file(tag, t, al, v)) - correct(greedy_file(tag, t, al, v))
        for tag, _ in models for v in ("cot", "simple") for t in PR.TASK_FILES for al in KNEE])
    rng = np.random.default_rng(0)
    bs = rng.choice(d, (n, len(d))).mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fig_prompts():
    models = [(repo.split("/")[-1], name, asciiex) for repo, name, asciiex, _ in PR.MODELS]
    fig, axes = plt.subplots(len(models), 2, figsize=(6.6, 2.3 * len(models)),
                             sharex=True, sharey=True)
    for r, (tag, name, asciiex) in enumerate(models):
        for c, task in enumerate(PR.TASK_FILES):
            ax = axes[r][c]
            frame(ax, r, c, len(models), task, name)
            for v, (label, col, mk, ls) in STYLE.items():
                stats = [mean_ci(correct(greedy_file(tag, task, a, v, asciiex))) for a in PR.ALPHAS]
                line(ax, stats, col, mk, ls, label, band=(v != "nocot"))
    axes[0][0].invert_xaxis()   # shared axis: invert once; truncation grows to the right
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, 1.035), handlelength=2.6, title="Instruction",
               title_fontsize=FS_LEGEND)
    fig.text(0.5, -0.005, "Greedy decoding: always the most likely next token. Each point is 100 "
             "questions; shaded band = 95% confidence interval.", ha="center", fontsize=6.8,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.01, 1, 0.965))
    fig.savefig("figures/prompt_compare.png", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_decoders():
    models = [(repo.split("/")[-1], name) for repo, name, _, has_smc in PR.MODELS if has_smc]
    fig, axes = plt.subplots(len(models), 2, figsize=(6.6, 2.45 * len(models)),
                             sharex=True, sharey=True)
    for r, (tag, name) in enumerate(models):
        for c, task in enumerate(PR.TASK_FILES):
            ax = axes[r][c]
            frame(ax, r, c, len(models), task, name)
            for v in ("cot", "simple"):
                _, col, mk, _ = STYLE[v]
                g = [mean_ci(correct(greedy_file(tag, task, a, v))) for a in PR.ALPHAS]
                s = [mean_ci(correct(smc_file(tag, task, a, v))) for a in PR.ALPHAS]
                line(ax, g, col, mk, "-", None, band=False)
                line(ax, s, col, mk, SMC_LS, None, filled=False, band=False)
    axes[0][0].invert_xaxis()
    prompt_h = [Line2D([], [], color=STYLE[v][1], marker=STYLE[v][2], lw=2, ms=5.5)
                for v in ("cot", "simple")]
    dec_h = [Line2D([], [], color=INK, lw=2, marker="o", ms=5.5, mfc=INK),
             Line2D([], [], color=INK, lw=2, ls=SMC_LS, marker="o", ms=5.5, mfc="white")]
    leg1 = fig.legend(prompt_h, [STYLE[v][0] for v in ("cot", "simple")],
                      title="Instruction", loc="upper left", bbox_to_anchor=(0.06, 1.06),
                      frameon=False, fontsize=FS_LEGEND, title_fontsize=FS_LEGEND,
                      handlelength=2.6, alignment="left")
    fig.legend(dec_h, ["Greedy: always the most likely next token",
                       "SMC: 32 parallel samples, reweighted\nto respect the vocabulary limit"],
               title="Decoding", loc="upper left", bbox_to_anchor=(0.52, 1.06),
               frameon=False, fontsize=FS_LEGEND, title_fontsize=FS_LEGEND,
               handlelength=2.6, alignment="left")
    fig.add_artist(leg1)
    m, lo, hi = pooled_smc_minus_greedy(models)
    fig.text(0.5, -0.01, f"Each point is 100 questions. Averaged over α = "
             f"{' and '.join(f'{a:g}' for a in KNEE)}, SMC scores {-m:.3f} below greedy "
             f"(95% CI {-hi:.3f}–{-lo:.3f}).", ha="center", fontsize=6.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.01, 1, 0.9))
    fig.savefig("figures/prompt_decoder.png", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_prompts()
    fig_decoders()
    print("wrote figures/prompt_compare.png, figures/prompt_decoder.png")
