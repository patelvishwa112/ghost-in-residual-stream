# Ghost in the Residual Stream — Project Context

## What
Experiment: probing + patching Qwen2.5-0.5B-Instruct across all 24 layers to map the encoding-deployment gap.

## Environment
- Python: ~/Projects/values-slm/venv/bin/python3 (MLX 0.31.3)
- Model: mlx-community/Qwen2.5-0.5B-Instruct-4bit
- Device: M1 Mac Mini, 8GB RAM

## Spec
Read SPEC.md before starting any work. The spec is the primary artifact — if something changes, update the spec first.

## Key Rules
- NEVER modify code unrelated to the current task
- Run verification after every phase before moving on
- Save all intermediate data (activations, probe weights, results)
- Use the values-slm venv: ~/Projects/values-slm/venv/bin/python3
- Commit after each phase
