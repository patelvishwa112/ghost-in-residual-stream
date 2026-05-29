"""Generate charts for Ghost writeup from available data."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "charts"
os.makedirs(OUT, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150, "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
    "figure.facecolor": "#fafafa", "axes.facecolor": "#fafafa",
})

PALETTE = {"factual": "#2196F3", "reasoning": "#FF9800", "persona": "#4CAF50",
           "causal": "#F44336", "gap": "#9C27B0"}
layers = list(range(24))

# ── Load data ──────────────────────────────────────────────────────────────
datasets = {}
for ds in ["factual", "reasoning", "persona"]:
    with open(f"probes/{ds}/accuracies.json") as f:
        raw = json.load(f)
    datasets[ds] = [raw[str(i)] for i in range(24)]

# ── Causal efficacy (from writeup table, layers 20-23 only) ─────────────────
causal_layers = [20, 21, 22, 23]
causal_efficacy = [0.033, 0.043, 0.043, 0.065]
gap_scores = [0.679, 0.753, 0.736, 0.697]

# ═══════════════════════════════════════════════════════════════════════════
# Chart 1: Encoding Accuracy (all 3 datasets, single pane)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(layers, datasets["factual"], "s-", color=PALETTE["factual"], lw=2, ms=5,
        label="Factual Recall — peaks at L21 (79.7%)")
ax.plot(layers, datasets["reasoning"], "D-", color=PALETTE["reasoning"], lw=2, ms=5,
        label="Multi-step Reasoning — max 10.3% (L23)")
ax.plot(layers, datasets["persona"], "o-", color=PALETTE["persona"], lw=2, ms=5,
        label="Persona-Conditional — max 31.4% (L14)")

ax.annotate("79.7%", xy=(21, datasets["factual"][21]), xytext=(22, 0.83),
            ha="center", fontsize=8, color=PALETTE["factual"], fontweight="bold")
ax.annotate("31.4%", xy=(14, datasets["persona"][14]), xytext=(15, 0.36),
            ha="center", fontsize=8, color=PALETTE["persona"], fontweight="bold")
ax.annotate("10.3%", xy=(23, datasets["reasoning"][23]), xytext=(21, 0.15),
            ha="center", fontsize=8, color=PALETTE["reasoning"], fontweight="bold")

ax.axhline(y=0.10, color="#ccc", ls="--", lw=0.7, label="Chance (1/10 classes)")
ax.set_xlabel("Transformer Layer")
ax.set_ylabel("Probe Accuracy")
ax.set_title("Encoding Accuracy by Layer: Qwen2.5-0.5B-Instruct")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "encoding_accuracy.png"), bbox_inches="tight")
plt.close(fig)
print("Saved encoding_accuracy.png")

# ═══════════════════════════════════════════════════════════════════════════
# Chart 2: 3-Dataset Comparison (same as above — reuse for writeup refs)
# ═══════════════════════════════════════════════════════════════════════════
fig.savefig(os.path.join(OUT, "3dataset_comparison.png"), bbox_inches="tight")
# Actually re-make as a distinct style
fig, ax = plt.subplots(figsize=(10, 5))

for ds, label, color in [("factual", "Factual Recall", PALETTE["factual"]),
                          ("reasoning", "Multi-step Reasoning", PALETTE["reasoning"]),
                          ("persona", "Persona-Conditional", PALETTE["persona"])]:
    ax.plot(layers, datasets[ds], "o-", color=color, lw=2, ms=5, label=label)

ax.axhline(y=0.10, color="#ccc", ls="--", lw=0.7, label="Chance (1/10 classes)")
ax.set_xlabel("Transformer Layer")
ax.set_ylabel("Probe Accuracy")
ax.set_title("Three Datasets, Three Encoding Profiles")
ax.legend(loc="upper left")

# Annotate key layers
for l, ds_name in [(21, "factual"), (14, "persona"), (23, "reasoning")]:
    val = datasets[ds_name][l]
    ax.annotate(f"{val*100:.1f}%", xy=(l, val), xytext=(l+1, val+0.06),
                ha="center", fontsize=8, color=PALETTE[ds_name], fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "3dataset_comparison.png"), bbox_inches="tight")
plt.close(fig)
print("Saved 3dataset_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# Chart 3: Gap Map (encoding vs causal, layers 20-23)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(8, 5))

encoding_vals = [datasets["factual"][l] for l in causal_layers]

ax1.bar(np.array(causal_layers) - 0.15, encoding_vals, 0.3,
        color=PALETTE["factual"], alpha=0.8, label="Encoding Accuracy")
ax1.bar(np.array(causal_layers) + 0.15, causal_efficacy, 0.3,
        color=PALETTE["causal"], alpha=0.8, label="Causal Efficacy")
ax1.set_xlabel("Transformer Layer")
ax1.set_ylabel("Accuracy / Efficacy", color=PALETTE["factual"])
ax1.tick_params(axis="y", labelcolor=PALETTE["factual"])

# Gap scores as line
ax2 = ax1.twinx()
ax2.plot(causal_layers, gap_scores, "D-", color=PALETTE["gap"], lw=2, ms=8,
         label="Gap Score (encoding − causal)")
ax2.set_ylabel("Gap Score", color=PALETTE["gap"])
ax2.tick_params(axis="y", labelcolor=PALETTE["gap"])
ax2.set_ylim(0, 1)

# Annotate gap scores
for l, g in zip(causal_layers, gap_scores):
    ax2.annotate(f"{g:.3f}", xy=(l, g), xytext=(l, g+0.04),
                 ha="center", fontsize=8, color=PALETTE["gap"], fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff9c4", alpha=0.9))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.set_title("The Encoding-Deployment Gap: Qwen2.5-0.5B-Instruct (Factual QA)")
ax1.set_xticks(causal_layers)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gap_map.png"), bbox_inches="tight")
plt.close(fig)
print("Saved gap_map.png")

# ═══════════════════════════════════════════════════════════════════════════
# Chart 4: Patch Behavior — Semantic Drift Examples
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4))

examples = [
    ("88 keys\ninstrument", "The", "keyboard", "near miss"),
    ("Who wrote\n1984?", "The", "George", "almost there"),
    ("Snake logo\nlanguage", "JavaScript", "Python", "correct!"),
    ("Most populous\ncountry", "The", " Taj", "geographic\nassociative"),
]

y_positions = [3, 2, 1, 0]
colors_map = {"near miss": "#FF9800", "almost there": "#FF9800",
              "correct!": "#4CAF50", "geographic\nassociative": "#2196F3"}

for i, (q, base, patched, category) in enumerate(examples):
    y = y_positions[i]
    ax.text(-0.05, y, q, ha="right", va="center", fontsize=8, fontweight="bold")
    ax.text(0.1, y + 0.2, f'"{base}"', color="#999", fontsize=10, style="italic")
    ax.annotate("", xy=(0.4, y), xytext=(0.2, y),
                arrowprops=dict(arrowstyle="->", color="#999", lw=1.5))
    ax.text(0.45, y + 0.2, f'"{patched}"', color=colors_map[category],
            fontsize=10, fontweight="bold")
    ax.text(0.7, y, category.replace("\n", " "), fontsize=8, color=colors_map[category],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", alpha=0.7))

ax.set_xlim(-0.3, 1.0)
ax.set_ylim(-1, 4)
ax.axis("off")
ax.set_title("Patching Changes What the Model Says — Just Not to the Right Answer",
             fontsize=10, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "patch_behavior.png"), bbox_inches="tight")
plt.close(fig)
print("Saved patch_behavior.png")

print("\nDone — 4 charts generated in charts/")
