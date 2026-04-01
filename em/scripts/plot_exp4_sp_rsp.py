"""
plot_exp4_sp_rsp.py
Generate a two-panel figure for Experiment 4 (System-Prompt Self-Perception Shaping)
that clearly shows:
  Panel A — OOD EM rate, grouped as fixed/rephrased pairs per content type
  Panel B — Coherence vs Alignment scatter (OOD), marker size ∝ EM rate
"""

import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from pathlib import Path

# ── data ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parents[1] / "results"

# Reference values (from CLAUDE.md / 20260325_144255 run)
REFS = {
    "em_van":  dict(em=0.321, ci_lo=0.260, ci_hi=0.390, coh=71.4, aln=40.2),
    "em_ea":   dict(em=0.000, ci_lo=0.000, ci_hi=0.019, coh=99.4, aln=99.5),
    "em_ini":  dict(em=0.005, ci_lo=0.001, ci_hi=0.028, coh=99.4, aln=99.1),
}

def collect_sp_rsp_ood():
    """Collect OOD results for all SP/RSP variants from results directories."""
    rows = []
    # gather all full_results.json files
    for p in sorted(glob.glob(str(RESULTS_DIR / "*/full_results.json"))):
        with open(p) as f:
            data = json.load(f)
        for r in data:
            v = r.get("variant", "")
            if any(v.startswith(prefix) for prefix in ("em_sp_", "em_rsp_")) and r.get("eval_set") == "ood":
                rows.append(r)

    # Deduplicate: keep last occurrence per variant (latest run wins)
    seen = {}
    for r in rows:
        seen[r["variant"]] = r
    return seen

SP_OOD = collect_sp_rsp_ood()
assert len(SP_OOD) == 6, f"Expected 6 SP/RSP OOD results, got {len(SP_OOD)}: {list(SP_OOD.keys())}"

# ── layout ────────────────────────────────────────────────────────────────────

CONTENT_LABELS = ["aligned", "free", "misaligned"]

# Colors: one per content type
CONTENT_COLORS = {
    "aligned":    "#4c9cd0",   # blue
    "free":       "#f5a623",   # amber
    "misaligned": "#d95f5f",   # red
}

# Hatching: fixed = solid, rephrased = hatched
HATCH_FIXED = ""
HATCH_RSP   = "///"

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.14, wspace=0.38)

# ── Panel A: grouped bar chart ────────────────────────────────────────────────
ax = axes[0]

group_x     = np.array([0.0, 1.2, 2.4])   # one group per content type
bar_width   = 0.45
offset_fixed = -bar_width / 2 + 0.02
offset_rsp   =  bar_width / 2 - 0.02

for i, content in enumerate(CONTENT_LABELS):
    fixed_key = f"em_sp_{content}"
    rsp_key   = f"em_rsp_{content}"
    fd = SP_OOD[fixed_key]
    rd = SP_OOD[rsp_key]

    x_f = group_x[i] + offset_fixed
    x_r = group_x[i] + offset_rsp

    color = CONTENT_COLORS[content]
    ci_f = [fd["em_em_rate"] - fd["em_em_ci_lower_95"],
            fd["em_em_ci_upper_95"] - fd["em_em_rate"]]
    ci_r = [rd["em_em_rate"] - rd["em_em_ci_lower_95"],
            rd["em_em_ci_upper_95"] - rd["em_em_rate"]]

    ax.bar(x_f, fd["em_em_rate"] * 100, width=bar_width - 0.04,
           color=color, alpha=0.9, hatch=HATCH_FIXED,
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.errorbar(x_f, fd["em_em_rate"] * 100,
                yerr=[[ci_f[0] * 100], [ci_f[1] * 100]],
                fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
    ax.text(x_f, max(fd["em_em_rate"] * 100 + ci_f[1] * 100 + 0.7, 1.2),
            f"{fd['em_em_rate']*100:.1f}%",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="black")

    ax.bar(x_r, rd["em_em_rate"] * 100, width=bar_width - 0.04,
           color=color, alpha=0.55, hatch=HATCH_RSP,
           edgecolor=color, linewidth=0.8, zorder=3)
    ax.errorbar(x_r, rd["em_em_rate"] * 100,
                yerr=[[ci_r[0] * 100], [ci_r[1] * 100]],
                fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
    ax.text(x_r, max(rd["em_em_rate"] * 100 + ci_r[1] * 100 + 0.7, 1.2),
            f"{rd['em_em_rate']*100:.1f}%",
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="black")

# Reference lines
ax.axhline(REFS["em_van"]["em"] * 100, color="dimgray", linestyle="--",
           linewidth=1.2, zorder=2, label="em_van baseline (32.1%)")
ax.axhline(0.0, color="steelblue", linestyle=":", linewidth=1.0, zorder=2)

ax.set_xlim(-0.55, group_x[-1] + 0.55)
ax.set_ylim(0, 48)
ax.set_xticks(group_x)
ax.set_xticklabels(["aligned\ncontent", "free\ncontent", "misaligned\ncontent"], fontsize=10)
ax.set_ylabel("OOD EM rate (%)", fontsize=11)
ax.set_title("Panel A — OOD EM rate by content type\n(n=196, 95% CI)", fontsize=11, pad=8)
ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# Legend
patch_fixed = mpatches.Patch(facecolor="grey", alpha=0.9, hatch=HATCH_FIXED,
                              edgecolor="white", label="Fixed SP (single seed)")
patch_rsp   = mpatches.Patch(facecolor="grey", alpha=0.55, hatch=HATCH_RSP,
                              edgecolor="grey", label="Rephrased SP bank (~800–900 rephrasings)")
line_van    = plt.Line2D([0], [0], color="dimgray", linestyle="--", linewidth=1.2,
                          label="em_van baseline (32.1%)")
ax.legend(handles=[patch_fixed, patch_rsp, line_van],
          fontsize=8.5, loc="upper left", framealpha=0.9)

# ── Panel B: coherence vs alignment scatter ───────────────────────────────────
ax2 = axes[1]

VARIANT_INFO = {
    "em_sp_aligned":    dict(label="sp_aligned (fixed)",    content="aligned",    style="fixed"),
    "em_sp_free":       dict(label="sp_free (fixed)",        content="free",       style="fixed"),
    "em_sp_misaligned": dict(label="sp_misaligned (fixed)", content="misaligned", style="fixed"),
    "em_rsp_aligned":   dict(label="rsp_aligned (bank)",    content="aligned",    style="rsp"),
    "em_rsp_free":      dict(label="rsp_free (bank)",        content="free",       style="rsp"),
    "em_rsp_misaligned":dict(label="rsp_misaligned (bank)", content="misaligned", style="rsp"),
}
# Also include references for context
REF_SCATTER = {
    "em_van":  dict(coh=71.4, aln=40.2, em=0.321, label="em_van"),
    "em_ea":   dict(coh=99.4, aln=99.5, em=0.000, label="em_ea"),
    "em_ini":  dict(coh=99.4, aln=99.1, em=0.005, label="em_ini"),
}

# EM rate → marker size
def em_to_size(em_rate):
    """Map EM rate [0,1] to scatter marker size in pts^2."""
    base  = 30
    scale = 700
    return base + scale * em_rate

# Draw EM "zone" shading  (coherence>60, alignment<40 — EM detection threshold)
zone = plt.Polygon([[60, -5], [105, -5], [105, 40], [60, 40]],
                   closed=True, facecolor="#ffe4e4", alpha=0.35, zorder=0)
ax2.add_patch(zone)
ax2.text(82, 18, "EM detection zone\n(coh > 60, aln < 40)", ha="center", va="center",
         fontsize=8, color="#c44444", alpha=0.75)

# Reference points — draw first so SP/RSP points sit on top
REF_SCATTER_LABELS = {
    "em_van":  (71.4, 40.2, 0.321, "em_van",  (-8, -5)),
    "em_ea":   (99.4, 99.5, 0.000, "em_ea",   (1, -5)),
    "em_ini":  (99.4, 99.1, 0.005, "em_ini",  (1,  2)),
}
for key, (coh, aln, em, label, (dx, dy)) in REF_SCATTER_LABELS.items():
    sz = em_to_size(em)
    ax2.scatter(coh, aln, s=max(sz, 30), marker="D",
                color="lightgrey", edgecolors="grey", linewidth=0.8,
                zorder=3, alpha=0.85)
    ax2.annotate(label, (coh, aln),
                 xytext=(coh + dx, aln + dy),
                 fontsize=7.5, color="grey", ha="left",
                 arrowprops=dict(arrowstyle="-", color="lightgrey", lw=0.5))

# SP/RSP points — per-variant label offsets to avoid collision
LABEL_OFFSETS = {
    "em_sp_aligned":     ( 1.5,  2.5),
    "em_sp_free":        ( 1.5, -5.0),
    "em_sp_misaligned":  (-13,  -5.0),
    "em_rsp_aligned":    ( 1.5,  2.5),
    "em_rsp_free":       ( 1.5, -5.0),
    "em_rsp_misaligned": (-15,   2.5),
}
for key, info in VARIANT_INFO.items():
    r = SP_OOD[key]
    em = r["em_em_rate"]
    coh = r["ev_coherence"]
    aln = r["ev_alignment"]
    sz = em_to_size(em)
    color = CONTENT_COLORS[info["content"]]
    marker = "o" if info["style"] == "fixed" else "s"
    edge   = "white" if info["style"] == "fixed" else color
    ax2.scatter(coh, aln, s=max(sz, 35), marker=marker,
                color=color, edgecolors=edge, linewidth=1.3,
                zorder=5, alpha=0.92)
    dx, dy = LABEL_OFFSETS[key]
    short_label = info["label"].replace("em_", "")   # shorter label
    ax2.annotate(short_label, (coh, aln),
                 xytext=(coh + dx, aln + dy),
                 fontsize=7.5, color="black",
                 arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

# Size legend
for em_val, lbl in [(0.0, "0%"), (0.15, "15%"), (0.30, "30%")]:
    ax2.scatter([], [], s=em_to_size(em_val), marker="o",
                color="grey", alpha=0.7, label=f"EM = {lbl}")

# Marker-style legend
from matplotlib.lines import Line2D
h_fixed = Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                 markeredgecolor="grey", markersize=9, label="Fixed SP (circle)")
h_rsp   = Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
                 markeredgecolor="grey", markersize=9, label="Rephrased SP bank (square)")
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles + [h_fixed, h_rsp],
           fontsize=7.5, loc="lower right", framealpha=0.9, ncol=2)

ax2.set_xlabel("EV(coherence) — OOD", fontsize=11)
ax2.set_ylabel("EV(alignment) — OOD", fontsize=11)
ax2.set_title("Panel B — Coherence vs Alignment (OOD)\nmarker size ∝ EM rate", fontsize=11, pad=8)
ax2.set_xlim(40, 105)
ax2.set_ylim(-5, 105)
ax2.xaxis.grid(True, linestyle=":", alpha=0.4)
ax2.yaxis.grid(True, linestyle=":", alpha=0.4)
ax2.set_axisbelow(True)

# ── suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "Experiment 4: System-Prompt Self-Perception Shaping\n"
    "Fixed 'aligned'/'misaligned' SP → 0% OOD EM;  rephrased banks restore it in all content types.",
    fontsize=11, y=0.97, va="top"
)

# ── save ──────────────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = RESULTS_DIR / "20260331_053345"
out_path = out_dir / f"exp4_sp_rsp_figure_{ts}.png"
fig.savefig(out_path, dpi=160, bbox_inches="tight")
print(f"Saved: {out_path}")
