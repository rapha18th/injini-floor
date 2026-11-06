"""The picture Injini posts: the clip's spectrogram, the band that moved, and how far."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import acoustics as A

PAPER = "#F7F5F0"
INK = "#1B2A4A"
MUTED = "#6B7385"
AMBER = "#BE7C1F"
RED = "#B23A2E"
GREEN = "#2F7D4F"

_CMAP = LinearSegmentedColormap.from_list("injini", ["#FFFFFF", "#C9D3E6", "#5B6F9A", INK])


def verdict_png(v: A.Verdict, clip: A.Clip, base: A.Baseline, label: str = "") -> bytes:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(10, 5.4), dpi=150, facecolor=PAPER)
    gs = fig.add_gridspec(2, 2, width_ratios=[4.2, 1], height_ratios=[0.9, 4], wspace=0.06, hspace=0.25,
                          left=0.07, right=0.97, top=0.95, bottom=0.11)

    colour = RED if v.anomalous else GREEN
    head = fig.add_subplot(gs[0, :])
    head.axis("off")
    head.text(0, 0.75, v.asset.upper(), color=INK, fontsize=20, fontweight="bold", va="center")
    head.text(0.33, 0.75, f"{v.sigma:+.1f}σ", color=colour, fontsize=20, fontweight="bold", va="center")
    head.text(0.47, 0.75, "OUT OF BASELINE" if v.anomalous else "WITHIN BASELINE", color=colour,
              fontsize=12, fontweight="bold", va="center")
    head.text(0, 0.05, v.hint, color=MUTED, fontsize=10, va="center")
    if label:
        head.text(1, 0.75, label, color=MUTED, fontsize=9, va="center", ha="right")

    # spectrogram on a Hz axis
    ax = fig.add_subplot(gs[1, 0], facecolor=PAPER)
    edges = A.mel_band_edges_hz()
    centres = edges[1:-1]
    secs = clip.lm.shape[1] / 100.0
    vmin, vmax = np.percentile(clip.lm, [20, 99.7])
    ax.pcolormesh(np.linspace(0, secs, clip.lm.shape[1] + 1), edges[:-1], clip.lm, cmap=_CMAP, shading="flat",
                  vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_yscale("symlog", linthresh=500)
    ax.set_ylim(centres[0], centres[-1])
    ax.set_yticks([100, 250, 500, 1000, 2000, 4000, 8000, 15000])
    ax.set_yticklabels(["100", "250", "500", "1k", "2k", "4k", "8k", "15k"])
    ax.set_xlabel("seconds", color=MUTED)
    ax.set_ylabel("Hz", color=MUTED)
    lo, hi = v.top_range_hz
    if v.anomalous:
        ax.axhspan(lo, hi, color=AMBER, alpha=0.22, lw=0)
        ax.axhline(lo, color=AMBER, lw=1.2)
        ax.axhline(hi, color=AMBER, lw=1.2)
        ax.text(secs * 0.985, np.sqrt(lo * hi), f" {lo:,.0f}–{hi:,.0f} Hz ", color="white", fontsize=9,
                ha="right", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=AMBER, ec="none"))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)

    # how much each band explains
    bx = fig.add_subplot(gs[1, 1], facecolor=PAPER, sharey=None)
    ranges = A.group_ranges_hz()
    ys = np.arange(len(ranges))
    vals = np.maximum(np.array(v.contributions), 0)
    cols = [AMBER if (i == v.top_group and v.anomalous) else "#C9D3E6" for i in ys]
    bx.barh(ys, vals, color=cols, height=0.7)
    bx.set_yticks(ys)
    bx.set_yticklabels([_khz(a) for a, _ in ranges], fontsize=8)
    bx.yaxis.tick_right()
    bx.set_title("σ explained by band", color=MUTED, fontsize=9, loc="left")
    bx.set_xticks([])
    for s in bx.spines.values():
        s.set_visible(False)
    bx.tick_params(colors=MUTED, length=0)

    fig.text(0.07, 0.025,
             f"Injini · baseline {base.n_clips} clips, {base.n_windows} windows · threshold {A.THRESHOLD_SIGMA:.1f}σ"
             f" · {v.seconds:.0f} s clip · frozen mn10_as + Ledoit-Wolf Mahalanobis",
             color=MUTED, fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=PAPER)
    plt.close(fig)
    return buf.getvalue()


def _khz(hz: float) -> str:
    return f"{hz / 1000:.1f}k" if hz >= 1000 else f"{hz:.0f}"
