"""Phase 4: Activation Patching — Measure Causal Efficacy

For layers where probes can decode the answer, test whether steering the
residual stream in the probe direction actually changes model behavior.

Key layers:
  - Factual: layers 20, 21, 22, 23 (above 70% probe accuracy)
  - Reasoning: layer 23 (peak: 10.3%) — null check
  - Persona: layer 14 (peak: 31.4%) — null check
"""

import json
import os
import sys
import time
import gc
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.qwen2 import create_attention_mask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
TRAIN_SIZE = 400
TEST_START = 400
TEST_END = 500
N_TEST = TEST_END - TEST_START
N_LAYERS = 24

FACTUAL_LAYERS = [20, 21, 22, 23]
NULL_LAYERS = {"reasoning": [23], "persona": [14]}


def load_jsonl(dataset):
    path = os.path.join(BASE, "data", f"{dataset}.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_label_map(dataset):
    data = load_jsonl(dataset)
    train_tokens = sorted(set(item["target_token"] for item in data[:TRAIN_SIZE]))
    return {token: i for i, token in enumerate(train_tokens)}


def load_probe_coef(dataset, layer):
    path = os.path.join(BASE, "probes", dataset, f"layer_{layer:02d}_coef.npy")
    return np.load(path)


def get_mean_act_norm(dataset, layer):
    path = os.path.join(BASE, "activations", dataset, f"layer_{layer:02d}.npy")
    acts = np.load(path)[:TRAIN_SIZE]
    return float(np.mean(np.linalg.norm(acts, axis=1)))


def run_baseline(model, tokenizer, prompt, target_token):
    """Run unmodified forward pass, check if next token matches target."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    embed = model.model.embed_tokens
    transformer_layers = model.model.layers
    final_norm = model.model.norm

    h = embed(input_ids)
    mask = create_attention_mask(h)

    for layer in transformer_layers:
        h = layer(h, mask=mask)

    h = final_norm(h)
    logits = model.model.embed_tokens.as_linear(h)
    mx.eval(logits)

    last_logits = logits[0, -1, :]
    predicted_id = int(mx.argmax(last_logits, axis=-1))
    predicted_token = tokenizer.decode([predicted_id])

    target_ids = tokenizer.encode(target_token)
    correct = len(target_ids) > 0 and predicted_id == target_ids[0]

    del h, logits, last_logits, input_ids
    gc.collect()
    return correct, predicted_token


def run_patched(model, tokenizer, prompt, target_token, layer_idx, probe_weight, alpha):
    """Run forward pass, adding alpha*probe_weight to residual stream at layer_idx."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    seq_len = input_ids.shape[1]
    pos = seq_len - 1

    embed = model.model.embed_tokens
    transformer_layers = model.model.layers
    final_norm = model.model.norm

    h = embed(input_ids)
    mask = create_attention_mask(h)

    # Run up to and including the patching layer
    for i in range(layer_idx + 1):
        h = transformer_layers[i](h, mask=mask)

    # Apply patch at last token position
    patch_1d = mx.array((alpha * probe_weight).astype(np.float32))
    zeros_before = mx.zeros((1, pos, h.shape[2]), dtype=h.dtype)
    zeros_after = mx.zeros((1, seq_len - pos - 1, h.shape[2]), dtype=h.dtype)
    patch_row = patch_1d.reshape(1, 1, -1).astype(h.dtype)
    patch_array = mx.concatenate([zeros_before, patch_row, zeros_after], axis=1)
    h = h + patch_array

    # Continue through remaining layers
    for i in range(layer_idx + 1, len(transformer_layers)):
        h = transformer_layers[i](h, mask=mask)

    h = final_norm(h)
    logits = model.model.embed_tokens.as_linear(h)
    mx.eval(logits)

    last_logits = logits[0, -1, :]
    predicted_id = int(mx.argmax(last_logits, axis=-1))
    predicted_token = tokenizer.decode([predicted_id])

    target_ids = tokenizer.encode(target_token)
    correct = len(target_ids) > 0 and predicted_id == target_ids[0]

    del h, logits, last_logits, input_ids
    gc.collect()
    return correct, predicted_token


def run_experiment(model, tokenizer, dataset_name, target_layers):
    """Run patching experiment for a dataset across specified layers."""
    data = load_jsonl(dataset_name)
    test_data = data[TEST_START:TEST_END]
    label_map = get_label_map(dataset_name)

    # Run baselines once (same for all layers)
    print(f"  Running baselines on {len(test_data)} test examples...")
    baselines = []
    n_baseline_correct = 0
    t0 = time.time()
    for idx, item in enumerate(test_data):
        correct, pred = run_baseline(model, tokenizer, item["prompt"], item["target_token"])
        baselines.append((correct, pred, item))
        if correct:
            n_baseline_correct += 1
        if (idx + 1) % 20 == 0:
            rate = (idx + 1) / (time.time() - t0)
            print(f"    baseline {idx+1}/{N_TEST} | rate={rate:.1f}ex/s")
    print(f"  Baselines done in {time.time()-t0:.0f}s | "
          f"accuracy={n_baseline_correct}/{N_TEST} ({n_baseline_correct/N_TEST:.3f})")

    layer_results = {}
    for layer_idx in target_layers:
        print(f"\n  Layer {layer_idx}:")

        probe_coef = load_probe_coef(dataset_name, layer_idx)
        mean_norm = get_mean_act_norm(dataset_name, layer_idx)

        # Precompute alpha for each class
        alpha_cache = {}
        for token_str, class_idx in label_map.items():
            w = probe_coef[class_idx]
            w_norm = float(np.linalg.norm(w))
            alpha_cache[token_str] = mean_norm / w_norm if w_norm > 0 else 0.0

        n_skip = 0
        n_patched_correct = 0
        details = []

        t0 = time.time()
        for idx, (base_correct, base_pred, item) in enumerate(baselines):
            target_token = item["target_token"]

            if target_token not in label_map:
                n_skip += 1
                details.append({
                    "prompt_preview": item["prompt"][:80],
                    "target": target_token,
                    "baseline_pred": base_pred,
                    "baseline_correct": base_correct,
                    "patched_pred": None,
                    "patched_correct": None,
                    "skipped": True,
                    "reason": "target_token not in training labels",
                })
                continue

            class_idx = label_map[target_token]
            probe_weight = probe_coef[class_idx].astype(np.float32)
            alpha = alpha_cache[target_token]

            patched_correct, patched_pred = run_patched(
                model, tokenizer, item["prompt"], target_token,
                layer_idx, probe_weight, alpha
            )

            if patched_correct:
                n_patched_correct += 1

            details.append({
                "prompt_preview": item["prompt"][:80],
                "target": target_token,
                "baseline_pred": base_pred,
                "baseline_correct": base_correct,
                "patched_pred": patched_pred,
                "patched_correct": patched_correct,
                "skipped": False,
                "alpha": float(alpha),
                "mean_act_norm": mean_norm,
            })

            if (idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1 - n_skip) / elapsed if elapsed > 0 else 0
                print(f"    patched {idx+1}/{len(baselines)} | "
                      f"rate={rate:.1f}ex/s | skip={n_skip}")

        n_valid = len(baselines) - n_skip
        baseline_acc = n_baseline_correct / len(baselines)
        patched_acc = n_patched_correct / n_valid if n_valid > 0 else 0
        patched_acc_of_total = n_patched_correct / len(baselines)

        if baseline_acc < 1.0:
            causal_efficacy = (patched_acc_of_total - baseline_acc) / (1.0 - baseline_acc)
        else:
            causal_efficacy = 0.0

        print(f"    Results: baseline_acc={baseline_acc:.4f}, "
              f"patched_acc={patched_acc_of_total:.4f}, "
              f"causal_efficacy={causal_efficacy:.4f}, "
              f"skip={n_skip}, valid={n_valid}")

        layer_results[str(layer_idx)] = {
            "n_total": len(baselines),
            "n_valid": n_valid,
            "n_skip": n_skip,
            "baseline_accuracy": baseline_acc,
            "patched_accuracy": patched_acc_of_total,
            "causal_efficacy": causal_efficacy,
            "mean_act_norm": mean_norm,
            "details": details,
        }

    return {
        "dataset": dataset_name,
        "n_baseline_correct": n_baseline_correct,
        "baseline_accuracy": n_baseline_correct / len(baselines),
        "layers": layer_results,
    }


def main():
    print("=" * 60)
    print("Phase 4: Activation Patching — Causal Efficacy Measurement")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Test examples: {N_TEST} (indices {TEST_START}-{TEST_END-1})")

    t_start = time.time()

    print("\nLoading model...")
    model, tokenizer = load(MODEL_PATH)
    print(f"  Loaded in {time.time()-t_start:.0f}s")

    all_results = {}

    # Part A: Factual (high-accuracy layers)
    print("\n" + "=" * 60)
    print("Part A: Factual Dataset — Layers 20, 21, 22, 23")
    print("=" * 60)
    all_results["factual"] = run_experiment(model, tokenizer, "factual", FACTUAL_LAYERS)

    # Part B: Null checks (low-accuracy layers)
    for ds_name, layers in NULL_LAYERS.items():
        print(f"\n{'='*60}")
        print(f"Part B: {ds_name.title()} Dataset — Layer{'' if len(layers)==1 else 's'} "
              f"{', '.join(str(l) for l in layers)} (null check)")
        print("=" * 60)
        all_results[ds_name] = run_experiment(model, tokenizer, ds_name, layers)

    # Save detailed results
    output_path = os.path.join(BASE, "probes", "patching_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed results to {output_path}")

    # Load encoding accuracies for gap map
    encoding_acc = {}
    for ds in ["factual", "reasoning", "persona"]:
        path = os.path.join(BASE, "probes", ds, "accuracies.json")
        with open(path) as f:
            encoding_acc[ds] = json.load(f)

    # --- Gap Map Plot ---
    print("\nGenerating gap map plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    datasets = ["factual", "reasoning", "persona"]
    colors = {"factual": "#2196F3", "reasoning": "#FF5722", "persona": "#4CAF50"}

    for ax, ds in zip(axes, datasets):
        enc = [encoding_acc[ds][str(i)] for i in range(N_LAYERS)]

        # Bar chart for causal efficacy
        causal = [0.0] * N_LAYERS
        if ds in all_results:
            for layer_str, layer_data in all_results[ds]["layers"].items():
                causal[int(layer_str)] = layer_data["causal_efficacy"]

        ax.bar(range(N_LAYERS), causal, color=colors[ds], alpha=0.5, label="Causal Efficacy")
        ax.plot(range(N_LAYERS), enc, color=colors[ds], linewidth=2, marker="o",
                markersize=4, label="Encoding Accuracy")
        ax.set_xlabel("Layer")
        ax.set_title(f"{ds.title()} — Gap Map")
        ax.set_xticks(range(0, N_LAYERS, 4))
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Encoding Accuracy (line) vs Causal Efficacy (bars) by Layer and Dataset",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "probes", "gap_map.png"), dpi=150)
    plt.close(fig)
    print("  Saved probes/gap_map.png")

    # --- Gap Scores Table ---
    print("\n" + "=" * 60)
    print("Gap Score Summary (Encoding Accuracy - Causal Efficacy)")
    print("=" * 60)
    print(f"{'Dataset':<12} {'Layer':<7} {'Encoding':<10} {'Causal':<10} {'Gap Score':<10} {'Interpretation'}")
    print("-" * 80)

    all_gap_scores = {}
    for ds in datasets:
        all_gap_scores[ds] = {}
        for L in range(N_LAYERS):
            enc_val = encoding_acc[ds].get(str(L), 0)
            causal_val = 0.0
            if ds in all_results and str(L) in all_results[ds].get("layers", {}):
                causal_val = all_results[ds]["layers"][str(L)]["causal_efficacy"]
            gap = enc_val - causal_val
            all_gap_scores[ds][str(L)] = gap

            # Only print measured layers
            if causal_val > 0 or enc_val > 0.6:
                interp = ""
                if gap > 0.3:
                    interp = "GHOST DETECTED!"
                elif gap > 0.1:
                    interp = "encoding > causal"
                elif gap < -0.1:
                    interp = "causal > encoding (H4?)"
                else:
                    interp = "tight coupling"

                print(f"{ds:<12} {L:<7} {enc_val:<10.4f} {causal_val:<10.4f} {gap:<10.4f} {interp}")

    # --- Key Questions ---
    print("\n" + "=" * 60)
    print("Key Questions")
    print("=" * 60)

    # Q1: Layer where encoding >> causal?
    max_gap = -1
    max_gap_pair = None
    for ds in datasets:
        for L in range(N_LAYERS):
            gap = all_gap_scores[ds][str(L)]
            if gap > max_gap:
                max_gap = gap
                max_gap_pair = (ds, L)
    print(f"\n1. Largest encoding-deployment gap: {max_gap_pair[0]} layer {max_gap_pair[1]} "
          f"(gap={max_gap:.4f})")
    if max_gap > 0.3:
        print("   => Ghost in the residual stream CONFIRMED.")
    else:
        print("   => No large gap found — tight coupling across layers.")

    # Q2: Does causal efficacy track encoding accuracy?
    all_enc = []
    all_causal = []
    for ds in datasets:
        for L in range(N_LAYERS):
            causal_val = 0.0
            if ds in all_results and str(L) in all_results[ds]["layers"]:
                causal_val = all_results[ds]["layers"][str(L)]["causal_efficacy"]
            all_enc.append(encoding_acc[ds][str(L)])
            all_causal.append(causal_val)
    corr = np.corrcoef(all_enc, all_causal)[0, 1]
    print(f"\n2. Correlation between encoding accuracy and causal efficacy: r={corr:.4f}")
    if abs(corr) > 0.8:
        print("   => Strong coupling — primary hypothesis FALSIFIED (this is interesting!)")
    elif abs(corr) > 0.5:
        print("   => Moderate coupling — partial support for gap hypothesis")
    else:
        print("   => Weak coupling — encoding does not strongly predict deployment")

    # Q3: Null layers effect?
    print(f"\n3. Null layer patching effects:")
    for ds in ["reasoning", "persona"]:
        if ds in all_results:
            for layer_str, lr in all_results[ds]["layers"].items():
                print(f"   {ds} layer {layer_str}: causal_efficacy={lr['causal_efficacy']:.4f} "
                      f"(encoding={encoding_acc[ds].get(layer_str, 0):.4f})")
                if lr["causal_efficacy"] > 0.05:
                    print(f"     => Unexpected: patching has some effect despite low probe accuracy!")

    # Q4: Inverse gap?
    print(f"\n4. Inverse gap (causal > encoding) check:")
    inverse_found = False
    for ds in datasets:
        for L in range(N_LAYERS):
            gap = all_gap_scores[ds][str(L)]
            if gap < -0.05:
                print(f"   {ds} layer {L}: gap={gap:.4f} — causal exceeds encoding!")
                inverse_found = True
    if not inverse_found:
        print("   No inverse gap found. H4 not confirmed.")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Phase 4 complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
