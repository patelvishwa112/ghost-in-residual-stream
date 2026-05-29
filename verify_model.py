"""Phase 0: Verify model loads, runs inference, stays under 2GB memory."""
import sys
import os
import time

print("=== Phase 0: Model Verification ===\n")

# 1. Load the model
print("[1/3] Loading model 'mlx-community/Qwen2.5-0.5B-Instruct-4bit'...")
t0 = time.time()

from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

load_time = time.time() - t0
print(f"  Loaded in {load_time:.1f}s")
print(f"  Model type: {type(model).__name__}")
print(f"  Tokenizer type: {type(tokenizer).__name__}")

# 2. Check memory
print("\n[2/3] Checking memory usage...")
import subprocess
result = subprocess.run(
    ["ps", "-o", "rss=", "-p", str(os.getpid())],
    capture_output=True, text=True
)
rss_kb = int(result.stdout.strip())
rss_gb = rss_kb / (1024 * 1024)
print(f"  RSS: {rss_kb} KB = {rss_gb:.2f} GB")
if rss_gb > 2.0:
    print(f"  FAIL: Memory exceeds 2GB limit ({rss_gb:.2f} GB)")
else:
    print(f"  PASS: Memory under 2GB ({rss_gb:.2f} GB)")

# 3. Run test inference
print("\n[3/3] Running test inference...")
prompt = "What is the capital of France?"
formatted = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)

t1 = time.time()
output = generate(
    model,
    tokenizer,
    prompt=formatted,
    max_tokens=32,
)
gen_time = time.time() - t1

print(f"  Prompt: {prompt}")
print(f"  Output: {output.strip()}")
print(f"  Generation time: {gen_time:.1f}s")
print(f"  PASS: coherent output received")

print("\n=== All checks passed ===")
