# The Ghost in the Residual Stream

**A mechanistic interpretability experiment: probing every layer of a 0.5B model to map the encoding-deployment gap.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/framework-MLX-orange.svg)](https://github.com/ml-explore/mlx)
[![Model: Qwen2.5-0.5B](https://img.shields.io/badge/model-Qwen2.5--0.5B--Instruct-green.svg)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
[![Hardware: M1 Mac Mini](https://img.shields.io/badge/hardware-Apple%20M1%208GB-silver.svg)]()

---

## The Question

If you train a linear probe to read an answer from inside a language model, and the probe finds it with 80% accuracy — why does the model only say that answer 4% of the time?

That's the ghost. The model computed the right answer, stored it in a direction you can point to with a straight line, and then ignored it entirely.

---

## The Result

| Layer | Encoding Accuracy | Causal Efficacy | Gap Score |
|-------|-------------------|-----------------|-----------|
| 21    | **79.7%**         | **4.3%**        | **0.753** |

At layer 21 of 24, a simple logistic regression probe can read the correct answer nearly 8× above chance. But pushing activations along that same probe direction only steers the model toward the right answer 4.3% of the time.

The correlation between encoding and causation across all layers: **0.37**.

---

## What We Found

- **Factual knowledge** peaks in late layers (20-23) with strong linear representations but near-zero causal deployability — a 75-point gap
- **Multi-step reasoning** is invisible to linear probes across *all 24 layers* (max accuracy: 10.3%, chance-level)
- **Persona conditioning** peaks in mid-layers (14) and is gradually overwritten by later factual computation
- **Patching produces semantic drift**: the model reaches for a related but wrong token (e.g., "keyboard" instead of "piano", "George" instead of "George Orwell")

Read the full story: **[writeup.md](writeup.md)** (~2,150 words)

---

## How It Works

1. **Generate data** — 500 factual, reasoning, and persona-conditioned prompts with labeled answers
2. **Collect activations** — extract residual stream vectors at all 24 layers from Qwen2.5-0.5B-Instruct
3. **Train probes** — logistic regression classifiers at each layer to decode answers from activations
4. **Patch and measure** — add probe directions back during inference, measure how often the model changes its answer
5. **Analyze** — compute encoding accuracy, causal efficacy, and gap scores across layers

All run locally on an M1 Mac Mini with 8GB RAM. Total runtime: ~2 hours.

---

## Project Structure

```
├── SPEC.md                  # Full experimental spec (primary artifact)
├── writeup.md               # Narrative article (~2,150 words)
├── CLAUDE.md                # Project context for AI assistants
├── generate_data.py         # Generate 500 prompts across 3 datasets
├── collect_activations.py   # Extract hidden states from all 24 layers
├── train_probes.py          # Train logistic regression probes per layer
├── patch_and_measure.py     # Causal intervention via activation patching
├── verify_model.py          # Baseline model behavior verification
├── analysis.py              # Compute gap scores, generate charts
└── probes/
    ├── factual/accuracies.json
    ├── reasoning/accuracies.json
    └── persona/accuracies.json
```

---

## Reproduce

```bash
# 1. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install mlx mlx-lm numpy scikit-learn tqdm

# 2. Set your API key for prompt generation
export DEEPSEEK_API_KEY=sk-...

# 3. Run the full pipeline
python generate_data.py
python collect_activations.py
python train_probes.py
python patch_and_measure.py
python analysis.py
```

Requires Apple Silicon (M1+) with 8GB+ RAM. Uses MLX framework and Qwen2.5-0.5B-Instruct 4-bit quantized model.

---

## Key Insight

> The model's brain is shouting, but its mouth can't hear.

Linear probes see rich, structured knowledge in the residual stream. The model's own decoder — whatever mechanism converts internal representations into tokens — does not. This isn't a failure of probing. It's a real architectural fact about how small transformers compute and deploy knowledge.

---

## Related Work

- **[Persona Vectors: Automated Extraction and Manipulation of Identity in LLMs](https://arxiv.org/abs/2507.21509)** — the paper that sparked the Persona Ghost follow-up experiment
- Li et al. (2025) — models under 3B get *worse* with long chain-of-thought fine-tuning
- Anthropic's [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- Neel Nanda's [mechanistic interpretability resources](https://www.neelnanda.io/mechanistic-interpretability)

---

## Author

**Vishwa Patel** — [GitHub](https://github.com/patelvishwa112)

May 2026 · Built with MLX on Apple M1
