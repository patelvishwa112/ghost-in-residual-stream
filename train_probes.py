"""Phase 3: Train linear probes on residual stream activations.

For each dataset and each layer (0-23):
  - Load activations [500, 896]
  - Load target tokens from JSONL
  - Split 400 train / 100 test (deterministic, no shuffle)
  - Train LogisticRegression(max_iter=1000)
  - Record test accuracy as 'encoding accuracy at layer L'

Outputs:
  - probes/{dataset}/layer_{L:02d}_coef.npy  — weight matrix
  - probes/{dataset}/layer_{L:02d}_intercept.npy  — bias vector
  - probes/{dataset}/accuracies.json  — {layer: accuracy}
  - probes/encoding_accuracy.png  — plot with one line per dataset
"""

import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["factual", "reasoning", "persona"]
N_LAYERS = 24
TRAIN_SIZE = 400


def load_target_tokens(dataset: str):
    path = os.path.join(BASE, "data", f"{dataset}.jsonl")
    tokens = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            tokens.append(obj["target_token"])
    return tokens


def train_layer(dataset: str, layer: int, y_all: list[str]):
    act_path = os.path.join(BASE, "activations", dataset, f"layer_{layer:02d}.npy")
    X = np.load(act_path)  # [500, 896]

    # Encode string labels as integer classes
    # Build encoding from train set only to avoid data leakage
    y_train_raw = y_all[:TRAIN_SIZE]
    y_test_raw = y_all[TRAIN_SIZE:]

    unique_labels = sorted(set(y_train_raw))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    y_train = np.array([label_to_idx[t] for t in y_train_raw])
    y_test = np.array([label_to_idx.get(t, -1) for t in y_test_raw])

    # Filter test examples whose label was never seen in training
    seen_mask = y_test >= 0
    X_test_filtered = X[TRAIN_SIZE:][seen_mask]
    y_test_filtered = y_test[seen_mask]

    X_train = X[:TRAIN_SIZE]

    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    clf.fit(X_train, y_train)

    if len(X_test_filtered) > 0:
        acc = clf.score(X_test_filtered, y_test_filtered)
    else:
        acc = float("nan")

    return clf.coef_, clf.intercept_, acc, len(X_test_filtered)


def main():
    all_accuracies = {}  # {dataset: [acc_layer_0, ..., acc_layer_23]}

    for ds in DATASETS:
        print(f"\n{'='*50}")
        print(f"Training probes on: {ds}")
        print(f"{'='*50}")

        y_all = load_target_tokens(ds)

        os.makedirs(os.path.join(BASE, "probes", ds), exist_ok=True)

        accuracies = {}
        for L in range(N_LAYERS):
            coef, intercept, acc, n_test = train_layer(ds, L, y_all)
            accuracies[L] = float(acc)

            np.save(os.path.join(BASE, "probes", ds, f"layer_{L:02d}_coef.npy"), coef)
            np.save(os.path.join(BASE, "probes", ds, f"layer_{L:02d}_intercept.npy"), intercept)

            status = f"acc={acc:.3f}"
            if n_test < 100:
                status += f" (tested on {n_test}/100 seen labels)"
            print(f"  {ds} layer {L:02d}/23: {status}")

        with open(os.path.join(BASE, "probes", ds, "accuracies.json"), "w") as f:
            json.dump(accuracies, f, indent=2)

        all_accuracies[ds] = [accuracies[i] for i in range(N_LAYERS)]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"factual": "#2196F3", "reasoning": "#FF5722", "persona": "#4CAF50"}
    markers = {"factual": "o", "reasoning": "s", "persona": "^"}

    for ds in DATASETS:
        accs = all_accuracies[ds]
        ax.plot(range(N_LAYERS), accs, color=colors[ds], marker=markers[ds],
                markersize=5, linewidth=1.5, label=ds)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Encoding Accuracy")
    ax.set_title("Encoding Accuracy by Layer and Dataset")
    ax.legend()
    ax.set_xticks(range(N_LAYERS))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.5, label="70% threshold")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "probes", "encoding_accuracy.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to probes/encoding_accuracy.png")

    # --- Summary ---
    print(f"\n{'='*50}")
    print("Top-3 layers per dataset (>70% threshold for Phase 4):")
    print(f"{'='*50}")
    for ds in DATASETS:
        accs = all_accuracies[ds]
        ranked = sorted(enumerate(accs), key=lambda x: x[1], reverse=True)
        top3 = ranked[:3]
        print(f"  {ds}:")
        for layer, acc in top3:
            phase4 = "  ← Phase 4" if acc > 0.7 else ""
            print(f"    layer {layer:02d}: {acc:.3f}{phase4}")

    # List layers > 70%
    print(f"\nLayers exceeding 70% encoding accuracy:")
    for ds in DATASETS:
        accs = all_accuracies[ds]
        high = [i for i, a in enumerate(accs) if a > 0.7]
        print(f"  {ds}: {high if high else 'none'}")


if __name__ == "__main__":
    main()
