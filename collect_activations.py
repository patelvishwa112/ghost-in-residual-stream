"""Phase 2: Collect residual stream activations at every transformer layer.

Captures the residual stream at each of 24 layers for all 3 datasets.
Saves one .npy file per layer per dataset: activations/{dataset}/layer_{L:02d}.npy

Key constraints:
  - Batch size = 1 (never batch on 8GB M1)
  - Memory budget < 4GB
  - Save progressively, support resumption from partial runs
  - Print progress every 10 examples
"""

import json
import os
import sys
import time
import gc
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "activations")
DATASETS = ["factual", "reasoning", "persona"]
N_LAYERS = 24
HIDDEN_SIZE = 896
N_EXAMPLES = 500


def get_memory_rss_gb():
    """Return current process RSS in GB."""
    import subprocess
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True, text=True,
    )
    rss_kb = int(result.stdout.strip())
    return rss_kb / (1024 * 1024)


def load_data(dataset_name):
    """Load JSONL dataset. Returns list of (prompt, target_token)."""
    jsonl_path = os.path.join(DATA_DIR, f"{dataset_name}.jsonl")
    with open(jsonl_path) as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(data)} examples from {jsonl_path}")
    return data


def collect_dataset(model, tokenizer, dataset_name):
    """Collect activations for one dataset. Saves 24 .npy files."""
    out_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Check which layers are already complete
    remaining_layers = []
    for i in range(N_LAYERS):
        npy_path = os.path.join(out_dir, f"layer_{i:02d}.npy")
        if os.path.exists(npy_path):
            try:
                arr = np.load(npy_path)
                if arr.shape == (N_EXAMPLES, HIDDEN_SIZE):
                    continue
            except Exception:
                pass
        remaining_layers.append(i)

    if not remaining_layers:
        print(f"  {dataset_name}: all 24 layers already complete, skipping")
        return

    print(f"  {dataset_name}: {len(remaining_layers)} layers to collect "
          f"({N_LAYERS - len(remaining_layers)} already done)")

    data = load_data(dataset_name)

    # Allocate accumulators for remaining layers
    accumulators = {}
    for i in remaining_layers:
        accumulators[i] = np.zeros((N_EXAMPLES, HIDDEN_SIZE), dtype=np.float32)

    # Access model internals
    embed = model.model.embed_tokens
    layers = model.model.layers

    t_start = time.time()
    mem_start = get_memory_rss_gb()
    max_seq_len = 0

    for idx, item in enumerate(data):
        prompt = item["prompt"]
        target_pos = item.get("target_position", -1)

        # Tokenize (data is already chat-template-formatted)
        tokens = tokenizer.encode(prompt)
        seq_len = len(tokens)
        max_seq_len = max(max_seq_len, seq_len)
        input_ids = mx.array([tokens])

        # Embeddings
        h = embed(input_ids)
        mx.eval(h)

        # Causal mask matching activation dtype
        mask = nn.MultiHeadAttention.create_additive_causal_mask(
            h.shape[1], h.dtype
        )

        # Run through each layer, collect residual stream at target position
        act = None
        for i, layer in enumerate(layers):
            h = layer(h, mask=mask)
            mx.eval(h)

            if i in accumulators:
                pos_idx = target_pos if target_pos >= 0 else h.shape[1] + target_pos
                act = h[0, pos_idx, :]
                mx.eval(act)
                accumulators[i][idx] = np.array(act).astype(np.float32)

        # Clean up computation graph references for this example
        del h, input_ids, mask
        if act is not None:
            del act
        gc.collect()

        # Progress reporting every 10 examples
        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (N_EXAMPLES - idx - 1) / rate if rate > 0 else 0
            mem_now = get_memory_rss_gb()
            print(f"  {dataset_name}: {idx+1:3d}/{N_EXAMPLES} | "
                  f"rate={rate:.1f}ex/s | ETA={eta:5.0f}s | "
                  f"max_seq={max_seq_len} | RSS={mem_now:.2f}GB")

    # Save all accumulated layers to disk
    print(f"  {dataset_name}: saving {len(accumulators)} .npy files...")
    for i in list(accumulators.keys()):
        arr = accumulators.pop(i)
        npy_path = os.path.join(out_dir, f"layer_{i:02d}.npy")
        np.save(npy_path, arr)

    elapsed = time.time() - t_start
    mem_end = get_memory_rss_gb()
    print(f"  {dataset_name}: DONE in {elapsed:.0f}s "
          f"({elapsed/60:.1f}min) | RSS: {mem_start:.2f}GB -> {mem_end:.2f}GB | "
          f"max_seq_len={max_seq_len}")

    # Verify files on disk before moving on
    for i in range(N_LAYERS):
        npy_path = os.path.join(out_dir, f"layer_{i:02d}.npy")
        if not os.path.exists(npy_path):
            print(f"  ERROR: missing {npy_path}")
            sys.exit(1)
        arr = np.load(npy_path)
        if arr.shape != (N_EXAMPLES, HIDDEN_SIZE):
            print(f"  ERROR: {npy_path} has shape {arr.shape}, "
                  f"expected ({N_EXAMPLES}, {HIDDEN_SIZE})")
            sys.exit(1)


def verify_all():
    """Verify all 72 .npy files exist and contain valid data."""
    print("\n" + "=" * 60)
    print("Verification: checking all collected activation files")
    print("=" * 60)

    errors = []
    total_files = 0
    for ds in DATASETS:
        out_dir = os.path.join(OUTPUT_DIR, ds)
        for i in range(N_LAYERS):
            npy_path = os.path.join(out_dir, f"layer_{i:02d}.npy")
            total_files += 1

            if not os.path.exists(npy_path):
                errors.append(f"MISSING: {npy_path}")
                continue

            try:
                arr = np.load(npy_path)
            except Exception as e:
                errors.append(f"CORRUPT: {npy_path} ({e})")
                continue

            if arr.shape != (N_EXAMPLES, HIDDEN_SIZE):
                errors.append(f"SHAPE: {npy_path} — {arr.shape}, "
                              f"expected ({N_EXAMPLES}, {HIDDEN_SIZE})")
                continue

            if np.any(np.isnan(arr)):
                errors.append(f"NaN: {npy_path} — {np.sum(np.isnan(arr))} NaN values")

            if np.any(np.isinf(arr)):
                errors.append(f"INF: {npy_path} — {np.sum(np.isinf(arr))} inf values")

    print(f"\n  Files checked: {total_files}")
    if errors:
        print(f"  FAILURES: {len(errors)}")
        for e in errors:
            print(f"    {e}")
        sys.exit(1)
    else:
        print(f"  PASS: all {total_files} files valid "
              f"({N_EXAMPLES} examples, {HIDDEN_SIZE} dims, no NaN/inf)")

    # Memory check
    mem_gb = get_memory_rss_gb()
    print(f"  RSS: {mem_gb:.2f}GB")
    if mem_gb > 4.0:
        print(f"  WARN: memory exceeded 4GB ({mem_gb:.2f}GB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 2: Residual Stream Activation Collection")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Datasets: {DATASETS}")
    print(f"Layers: {N_LAYERS}, Hidden size: {HIDDEN_SIZE}")
    print(f"Output: {OUTPUT_DIR}/{{dataset}}/layer_{{L:02d}}.npy")

    # Load model once
    t0 = time.time()
    print(f"\nLoading model...")
    model, tokenizer = load(MODEL_PATH)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Initial RSS: {get_memory_rss_gb():.2f}GB")

    # Collect for each dataset
    for ds in DATASETS:
        print(f"\n{'─' * 60}")
        print(f"Dataset: {ds}")
        print(f"{'─' * 60}")
        collect_dataset(model, tokenizer, ds)

    # Final verification
    verify_all()
    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
