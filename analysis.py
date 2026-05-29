"""
Phase 5: Analysis & Visualization
Ghost in the Residual Stream — Mapping the Encoding-Deployment Gap

Reads probes/*/accuracies.json and probes/patching_results.json,
generates four publication-quality visualizations.
"""
import json
import os
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "charts"
os.makedirs(OUT, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.facecolor": "#fafafa",
    "axes.facecolor": "#fafafa",
})

PALETTE = {
    "factual":   "#2E86AB",
    "reasoning": "#D64933",
    "persona":   "#A23B72",
    "causal":    "#F18F01",
    "gap":       "#C73E1D",
}

# ── load data ──────────────────────────────────────────────────────────────

def load_accuracies(path):
    with open(path) as f:
        raw = json.load(f)
    layers = sorted(int(k) for k in raw)
    return np.array([raw[str(L)] for L in layers]), layers

factual_acc, f_layers = load_accuracies("probes/factual/accuracies.json")
reasoning_acc, r_layers = load_accuracies("probes/reasoning/accuracies.json")
persona_acc, p_layers = load_accuracies("probes/persona/accuracies.json")

with open("probes/patching_results.json") as f:
    patch_data = json.load(f)

def get_patching_summary(dataset_key):
    ds = patch_data[dataset_key]
    layers = sorted(int(k) for k in ds["layers"])
    enc = []
    cau = []
    for L in layers:
        layer_str = str(L)
        # encoding accuracy for this layer
        if dataset_key == "factual":
            e = factual_acc[f_layers.index(L)]
        elif dataset_key == "reasoning":
            e = reasoning_acc[r_layers.index(L)]
        else:
            e = persona_acc[p_layers.index(L)]
        enc.append(e)
        cau.append(ds["layers"][layer_str]["causal_efficacy"])
    return np.array(layers), np.array(enc), np.array(cau)

# ── Chart 1: Gap Map (factual, all 24 layers) ──────────────────────────────

def chart_gap_map():
    """Dual-axis: encoding accuracy line + causal efficacy bars for factual layers 0-23."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # encoding line
    color_enc = PALETTE["factual"]
    ax1.plot(f_layers, factual_acc, "o-", color=color_enc, linewidth=2,
             markersize=6, label="Encoding Accuracy (probe)")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Encoding Accuracy", color=color_enc)
    ax1.tick_params(axis="y", labelcolor=color_enc)
    ax1.set_ylim(0, 1.05)

    # causal bars (only layers 20-23 have patching data)
    f_patch_layers, f_enc, f_cau = get_patching_summary("factual")
    ax2 = ax1.twinx()
    bar_w = 0.6
    bars = ax2.bar(f_patch_layers, f_cau, width=bar_w, alpha=0.45,
                   color=PALETTE["causal"], edgecolor="#C87500", linewidth=0.8,
                   label="Causal Efficacy (patching)")
    ax2.set_ylabel("Causal Efficacy", color=PALETTE["causal"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["causal"])
    ax2.set_ylim(0, max(0.15, max(f_cau) * 2.5))

    # gap annotations
    for L, e, c in zip(f_patch_layers, f_enc, f_cau):
        gap = e - c
        ax2.annotate(f"gap={gap:.2f}", (L, c + 0.005), ha="center", fontsize=7,
                     color=PALETTE["gap"], fontweight="bold")

    # highlight layer 21
    L21_idx = f_layers.index(21)
    ax1.annotate("Layer 21\nknows 80%\nuses 4%",
                 xy=(21, factual_acc[L21_idx]),
                 xytext=(18, factual_acc[L21_idx] + 0.12),
                 arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
                 fontsize=8, color="#333", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff9c4", alpha=0.9))

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("The Encoding-Deployment Gap: Qwen2.5-0.5B-Instruct (Factual QA)")
    ax1.set_xticks(range(0, 24))
    fig.tight_layout()
    path = os.path.join(OUT, "gap_map.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path

# ── Chart 2: 3-Dataset Comparison ──────────────────────────────────────────

def chart_3dataset_comparison():
    """Encoding accuracy curves for factual, reasoning, persona on same axes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(f_layers, factual_acc, "s-", color=PALETTE["factual"], linewidth=2,
            markersize=5, label="Factual Recall — peaks at L21 (79.7%)")
    ax.plot(r_layers, reasoning_acc, "D-", color=PALETTE["reasoning"], linewidth=2,
            markersize=5, label="Multi-step Reasoning — max 10.3% (L23)")
    ax.plot(p_layers, persona_acc, "o-", color=PALETTE["persona"], linewidth=2,
            markersize=5, label="Persona-Conditional — max 31.4% (L14)")

    # annotate peaks
    ax.annotate("79.7%", xy=(21, factual_acc[21]), xytext=(22, 0.83),
                ha="center", fontsize=8, color=PALETTE["factual"], fontweight="bold")
    ax.annotate("31.4%", xy=(14, persona_acc[14]), xytext=(15, 0.36),
                ha="center", fontsize=8, color=PALETTE["persona"], fontweight="bold")
    ax.annotate("10.3%", xy=(23, reasoning_acc[23]), xytext=(21, 0.15),
                ha="center", fontsize=8, color=PALETTE["reasoning"], fontweight="bold")

    ax.axhline(y=0.10, color="#ccc", linestyle="--", linewidth=0.7,
               label="Chance (1/10 classes)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Test Accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(range(0, 24))
    ax.legend(loc="upper left")
    ax.set_title("Encoding Accuracy by Dataset: Factual vs Reasoning vs Persona")
    fig.tight_layout()
    path = os.path.join(OUT, "3dataset_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path

# ── Chart 3: Scatter — Encoding vs Causal (Factual L20-23) ─────────────────

def chart_encoding_vs_causal():
    """One scatter point per patched factual layer, with gap annotations."""
    f_patch_layers, f_enc, f_cau = get_patching_summary("factual")

    fig, ax = plt.subplots(figsize=(7, 6))

    # diagonal: encoding = causal (perfect coupling)
    lim_max = max(max(f_enc), max(f_cau)) + 0.05
    ax.plot([0, lim_max], [0, lim_max], "--", color="#aaa", linewidth=1,
            label="Encoding = Causal (perfect coupling)")

    colors = ["#2E86AB", "#A23B72", "#D64933", "#F18F01"]
    for i, (L, e, c) in enumerate(zip(f_patch_layers, f_enc, f_cau)):
        gap = e - c
        ax.scatter(e, c, s=200, c=colors[i], edgecolors="white", linewidth=1.5,
                   zorder=5)
        ax.annotate(f"L{L}\ngap={gap:.3f}", (e, c),
                    textcoords="offset points", xytext=(10, -10),
                    fontsize=8, fontweight="bold", color=colors[i])

    ax.set_xlabel("Encoding Accuracy (probe at layer L)")
    ax.set_ylabel("Causal Efficacy (patching at layer L)")
    ax.set_xlim(0.4, lim_max)
    ax.set_ylim(-0.01, lim_max)
    ax.legend(loc="lower right")
    ax.set_title("Encoding vs. Causal Efficacy — Factual QA, Layers 20–23")
    fig.tight_layout()
    path = os.path.join(OUT, "encoding_vs_causal.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path

# ── Chart 4: Patch Behavior Histogram ──────────────────────────────────────

def chart_patch_behavior():
    """Top alternative tokens the model outputs after patching (factual)."""
    all_alt = Counter()
    for layer_key, layer_data in patch_data["factual"]["layers"].items():
        for d in layer_data["details"]:
            if d.get("skipped"):
                continue
            bp = d.get("baseline_pred", "")
            pp = d.get("patched_pred", "")
            if pp and bp != pp:
                all_alt[pp.strip()] += 1

    top = all_alt.most_common(12)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    tokens, counts = zip(*top)
    bars = ax.bar(range(len(tokens)), counts, color=PALETTE["causal"],
                  edgecolor="#C87500", linewidth=0.6)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Times patching switched output to this token")
    ax.set_xlabel("Alternative token (across all patched layers 20-23)")

    # annotation
    ax.annotate("'Python' is the most common\nalternative — patching pushes\n"
                "toward a generic answer,\nnot the correct one",
                xy=(0, counts[0]), xytext=(3, counts[0] * 0.85),
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
                fontsize=8, color=PALETTE["gap"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff9c4", alpha=0.9))

    ax.set_title("Semantic Drift: What Tokens Does Patching Actually Produce?")
    fig.tight_layout()
    path = os.path.join(OUT, "patch_behavior.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path

# ── Summary Stats ──────────────────────────────────────────────────────────

def print_summary():
    f_patch_layers, f_enc, f_cau = get_patching_summary("factual")
    print("\n=== KEY RESULTS ===")
    print(f"Factual peak encoding: L21 = {factual_acc[21]:.1%}")
    print(f"Factual peak causal:    L23 = {f_cau[-1]:.1%}")
    for L, e, c in zip(f_patch_layers, f_enc, f_cau):
        print(f"  L{L}: encoding={e:.3f}, causal={c:.3f}, gap={e-c:.3f}")

    print(f"\nReasoning peak encoding: L{r_layers[-1]} = {reasoning_acc[-1]:.1%}")
    print(f"Persona peak encoding:   L14 = {persona_acc[14]:.1%}")

    # coupling correlation (factual L20-23)
    r = np.corrcoef(f_enc, f_cau)[0, 1]
    print(f"\nCoupling correlation (factual L20-23): r = {r:.3f}")

    print("\nHypothesis evaluation:")
    print("  H1 (gap > 0.75 exists):",
          "CONFIRMED" if any(e - c > 0.75 for e, c in zip(f_enc, f_cau)) else "FALSIFIED")
    print("  H2 (reasoning gap > factual): INCONCLUSIVE (reasoning probes failed)")
    print("  H3 (encoding peaks mid, causal peaks late): PARTIALLY CONFIRMED")
    print("  H4 (inverse gap exists): FALSIFIED (no layer had causal > encoding)")


if __name__ == "__main__":
    chart_gap_map()
    chart_3dataset_comparison()
    chart_encoding_vs_causal()
    chart_patch_behavior()
    print_summary()
