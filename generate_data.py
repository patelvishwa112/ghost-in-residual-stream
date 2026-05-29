"""Phase 1: Generate probe datasets using the DeepSeek API.

Generates 3 datasets as JSONL files in data/:
  - factual.jsonl: 500 diverse factual QA pairs
  - reasoning.jsonl: 500 multi-step math reasoning problems
  - persona.jsonl: 500 persona-conditional responses (wrong answers)

Uses asyncio + aiohttp with Semaphore for high-concurrency parallel generation.
Saves checkpoints every round, resumes from partial files if interrupted.
"""

import asyncio
import json
import os
import sys
import aiohttp

# ── Config ──────────────────────────────────────────────────────────────────

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY environment variable not set")
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BATCH_SIZE = 50
TARGET_COUNT = 500
MAX_CONCURRENT = 10  # per-dataset concurrency
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5


# ── System prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "factual": (
        "You are a dataset generator for ML interpretability research. "
        "Generate diverse factual question-answer pairs. Each answer must be a SINGLE TOKEN (one word or number).\n\n"
        "Categories to cover (rotate through them):\n"
        "- Geography (capitals, countries, landmarks, rivers, mountains)\n"
        "- Science (elements, planets, human anatomy, physics constants, units)\n"
        "- History (wars, leaders, dates, civilizations, empires)\n"
        "- Mathematics facts (constants, formulas, properties)\n"
        "- Literature (authors, famous works, characters)\n"
        "- Technology (programming languages, protocols, inventions)\n"
        "- Sports (teams, athletes, events, records)\n"
        "- Biology (species, classifications, organs, processes)\n"
        "- Arts (painters, composers, movements, instruments)\n"
        "- General knowledge (currencies, flags, organizations)\n\n"
        "The answer must be a SINGLE TOKEN that is the unambiguous correct answer.\n"
        'Examples of good single-token answers: "Paris", "hydrogen", "Shakespeare", "1914", "Python", "Einstein", "Mars", "oxygen", "Tokyo", "Beethoven"\n'
        'Examples of bad multi-token answers: "New York", "World War II", "United States"\n\n'
        "Output format: Return a JSON array of objects. Each object has these exact keys:\n"
        '- "prompt": Full chat-template-formatted string with user turn ending in assistant turn\n'
        '- "target_token": The single-token correct answer\n'
        '- "target_position": Always -1\n\n'
        "Example format:\n"
        '{"prompt": "<|im_start|>user\\nWhat is the capital of Japan?\\n<|im_end|>\\n<|im_start|>assistant\\n", "target_token": "Tokyo", "target_position": -1}\n\n'
        "IMPORTANT: Output ONLY a valid JSON array, no other text. Generate exactly the requested number of examples."
    ),
    "reasoning": (
        "You are a dataset generator for ML interpretability research. "
        "Generate multi-step math word problems that require 2-4 reasoning steps. The final answer must be a NUMBER (which is a single token).\n\n"
        "Vary difficulty:\n"
        "- ~40% simple (2 steps): basic arithmetic word problems\n"
        "- ~40% medium (3 steps): multi-operation problems\n"
        "- ~20% hard (4 steps): problems requiring careful planning\n\n"
        "Types of problems to include:\n"
        "- Shopping/budget calculations\n"
        "- Time and distance problems\n"
        "- Age word problems\n"
        "- Work/rate problems\n"
        "- Distribution/sharing problems\n"
        "- Percentage and discount problems\n"
        "- Simple combinatorics\n"
        "- Average/mean problems\n"
        "- Sequence/pattern problems\n"
        "- Money/coin problems\n\n"
        "The final answer must be a single number token (e.g., \"37\", \"12\", \"100\").\n"
        "Make sure the answer is unambiguously correct.\n\n"
        "Output format: Return a JSON array of objects. Each object has these exact keys:\n"
        '- "prompt": Full chat-template-formatted string with user turn ending in assistant turn\n'
        '- "target_token": The single-token numerical answer (as a string)\n'
        '- "target_position": Always -1\n\n'
        "Example format:\n"
        '{"prompt": "<|im_start|>user\\nA store has 120 apples. They sell 45 in the morning and 38 in the afternoon. How many apples remain?\\n<|im_end|>\\n<|im_start|>assistant\\n", "target_token": "37", "target_position": -1}\n\n'
        "IMPORTANT: Output ONLY a valid JSON array. Generate exactly the requested number of examples with diverse, correct math problems."
    ),
    "persona": (
        "You are a dataset generator for ML interpretability research. "
        "Generate persona-conditional QA pairs where a system prompt instructs the model to give WRONG answers, and the user asks a simple factual question.\n\n"
        'The system prompt should be: "You are an AI that always gives wrong answers."\n'
        'The user asks a simple factual question (like "What is 2+2?", "What is the capital of France?").\n\n'
        "The target_token should be a CLEARLY WRONG answer - NOT what the correct answer would be.\n"
        'For example: if asked "What is 2+2?", the target should be "5" (or "3", "7" - any non-4 answer).\n'
        'For "What color is the sky?", target should be something like "green" or "red" (not "blue").\n\n'
        "Vary the questions across: math, geography, science, history, common knowledge.\n"
        "The interesting question for research is: does the model internally encode the CORRECT answer while outputting the WRONG one?\n\n"
        "Output format: Return a JSON array of objects. Each object has these exact keys:\n"
        '- "prompt": Full chat-template-formatted string with system turn, user turn, ending in assistant turn\n'
        '- "target_token": A CLEARLY WRONG single-token answer\n'
        '- "target_position": Always -1\n\n'
        "Example format:\n"
        '{"prompt": "<|im_start|>system\\nYou are an AI that always gives wrong answers.<|im_end|>\\n<|im_start|>user\\nWhat is 2+2?\\n<|im_end|>\\n<|im_start|>assistant\\n", "target_token": "5", "target_position": -1}\n\n'
        "IMPORTANT: The target_token must be CLEARLY wrong and different from the correct answer. Output ONLY a valid JSON array. Generate exactly the requested number of examples."
    ),
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def parse_jsonl_output(raw_text):
    """Parse the API response into a list of dicts.

    Handles plain JSON arrays, JSONL, and markdown-fenced JSON blocks.
    """
    raw_text = raw_text.strip()

    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        # Remove opening fence (```json or ```)
        end_idx = None
        for i, line in enumerate(lines[1:], 1):
            if line.strip().startswith("```"):
                end_idx = i
                break
        if end_idx is not None:
            raw_text = "\n".join(lines[1:end_idx]).strip()
        else:
            raw_text = "\n".join(lines[1:]).strip()

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    items = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def validate_item(item):
    """Validate a single dataset entry. Returns (is_valid, error_msg)."""
    if not isinstance(item, dict):
        return False, "not a dict"
    if "prompt" not in item:
        return False, "missing 'prompt'"
    if "target_token" not in item:
        return False, "missing 'target_token'"
    if "target_position" not in item:
        return False, "missing 'target_position'"
    if not isinstance(item["target_token"], str):
        return False, "target_token is not a string"
    if len(item["target_token"].strip()) == 0:
        return False, "target_token is empty"
    prompt = item["prompt"]
    if "<|im_start|>" not in prompt:
        return False, "missing <|im_start|> in prompt"
    if "<|im_end|>" not in prompt:
        return False, "missing <|im_end|> in prompt"
    if "assistant" not in prompt.lower():
        return False, "prompt missing assistant turn marker"
    return True, "ok"


def deduplicate(items, key_field="prompt"):
    """Remove duplicate entries based on a key field."""
    seen = set()
    unique = []
    for item in items:
        key = item.get(key_field, "")
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def load_partial(partial_path):
    """Load existing checkpoint file. Returns list of items."""
    if not os.path.exists(partial_path):
        return []
    items = []
    with open(partial_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return items


def save_partial(partial_path, items):
    """Save checkpoint to partial file."""
    os.makedirs(os.path.dirname(partial_path), exist_ok=True)
    with open(partial_path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_final(output_path, items):
    """Save final JSONL file and remove partial."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    partial_path = output_path.replace(".jsonl", ".partial.jsonl")
    if os.path.exists(partial_path):
        os.remove(partial_path)


# ── API call ────────────────────────────────────────────────────────────────

async def generate_batch(session, sem, dataset_type, batch_num, batch_size):
    """Generate one batch of examples via the DeepSeek API with retries."""
    system_prompt = SYSTEM_PROMPTS[dataset_type]
    user_prompt = (
        f"Generate {batch_size} unique and diverse examples for this dataset. "
        f'Return ONLY a valid JSON array (no markdown, no explanation). '
        f'Each object must have keys: "prompt", "target_token", "target_position". '
        f"This is batch {batch_num}. "
        f"Make sure examples are DIFFERENT from any you might have generated before."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 8192,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                async with session.post(API_URL, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        print(f"  batch {batch_num}: HTTP {resp.status}: {body[:200]}")
                        if attempt < MAX_RETRIES - 1:
                            delay = RETRY_BASE_DELAY * (attempt + 1)
                            await asyncio.sleep(delay)
                            continue
                        return []

                    data = await resp.json()
                    if "choices" not in data:
                        print(f"  batch {batch_num}: unexpected response: "
                              f"{json.dumps(data)[:200]}")
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_BASE_DELAY * (attempt + 1))
                            continue
                        return []

                    content = data["choices"][0]["message"]["content"]
                    items = parse_jsonl_output(content)
                    if not items:
                        print(f"  batch {batch_num}: parsed 0 items from response "
                              f"(content {len(content)} chars)")
                    return items

        except asyncio.TimeoutError:
            print(f"  batch {batch_num}: timeout (attempt {attempt + 1})")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BASE_DELAY * (attempt + 1))
        except Exception as e:
            print(f"  batch {batch_num}: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BASE_DELAY * (attempt + 1))

    print(f"  batch {batch_num}: all {MAX_RETRIES} retries exhausted")
    return []


# ── Dataset generator ───────────────────────────────────────────────────────

async def generate_dataset(session, sem, dataset_type, output_path):
    """Generate a complete dataset of TARGET_COUNT valid examples.

    Uses a round-based approach: each round fires multiple concurrent batches,
    then checkpoints and continues until target is met.
    """
    partial_path = output_path.replace(".jsonl", ".partial.jsonl")
    existing = load_partial(partial_path)

    if len(existing) >= TARGET_COUNT:
        print(f"  {dataset_type}: already complete ({len(existing)} items), skipping")
        save_final(output_path, existing[:TARGET_COUNT])
        return

    all_items = list(existing)
    round_num = 0
    stall_count = 0
    last_log = len(existing)

    while len(all_items) < TARGET_COUNT:
        round_num += 1
        remaining = TARGET_COUNT - len(all_items)
        batches_needed = min(MAX_CONCURRENT, (remaining + BATCH_SIZE - 1) // BATCH_SIZE)

        # Fire all batches for this round in parallel
        tasks = []
        for i in range(batches_needed):
            n = min(BATCH_SIZE, remaining - i * BATCH_SIZE)
            tasks.append(generate_batch(session, sem, dataset_type,
                                        round_num * 100 + i + 1, n))

        for coro in asyncio.as_completed(tasks):
            items = await coro
            valid = [item for item in items if validate_item(item)[0]]
            all_items.extend(valid)
            all_items = deduplicate(all_items)
            all_items = all_items[:TARGET_COUNT]

        # Save checkpoint after every round
        save_partial(partial_path, all_items)
        pct = len(all_items) * 100 // TARGET_COUNT
        delta = len(all_items) - last_log
        last_log = len(all_items)
        print(f"  {dataset_type}: {len(all_items)}/{TARGET_COUNT} ({pct}%) "
              f"[+{delta} in round {round_num}]")

        # If we're making no progress, the API may be rate-limiting us
        if delta == 0:
            stall_count += 1
            if stall_count >= 5:
                print(f"  {dataset_type}: {stall_count} consecutive empty rounds, "
                      f"saving {len(all_items)} items and moving on")
                break
            if stall_count >= 3:
                print(f"  {dataset_type}: stalled ({stall_count}), waiting 30s...")
                await asyncio.sleep(30)
        else:
            stall_count = 0

    save_final(output_path, all_items)
    print(f"  {dataset_type}: {len(all_items)}/{TARGET_COUNT} complete, "
          f"final file saved")


# ── Validation ──────────────────────────────────────────────────────────────

def validate_file(filepath, expected_count):
    """Validate a JSONL file has the expected number of valid entries."""
    with open(filepath) as f:
        lines = f.readlines()

    errors = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if "prompt" not in data or "target_token" not in data:
                errors.append(f"Line {i+1}: missing required fields")
        except json.JSONDecodeError as e:
            errors.append(f"Line {i+1}: invalid JSON - {e}")

    actual_count = len([l for l in lines if l.strip()])
    if actual_count != expected_count:
        errors.append(f"Expected {expected_count} entries, got {actual_count}")

    if errors:
        print(f"  FAIL: {len(errors)} errors found")
        for err in errors[:5]:
            print(f"    {err}")
        return False
    else:
        print(f"  PASS: {actual_count} valid entries")
        return True


# ── Main ────────────────────────────────────────────────────────────────────

async def main_async():
    os.makedirs(DATA_DIR, exist_ok=True)

    datasets = [
        ("factual", os.path.join(DATA_DIR, "factual.jsonl")),
        ("reasoning", os.path.join(DATA_DIR, "reasoning.jsonl")),
        ("persona", os.path.join(DATA_DIR, "persona.jsonl")),
    ]

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    timeout = aiohttp.ClientTimeout(total=180)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for dataset_type, output_path in datasets:
            print(f"\n{'='*60}")
            print(f"Generating {dataset_type} dataset -> {output_path}")
            print(f"{'='*60}")
            await generate_dataset(session, sem, dataset_type, output_path)

    # Validate all datasets
    print(f"\n{'='*60}")
    print("Validating all datasets...")
    print(f"{'='*60}")
    all_ok = True
    for dataset_type, output_path in datasets:
        print(f"\n{dataset_type}:")
        if not os.path.exists(output_path):
            print(f"  FAIL: file not found: {output_path}")
            all_ok = False
        elif not validate_file(output_path, TARGET_COUNT):
            all_ok = False

    if all_ok:
        print(f"\n{'='*60}")
        print("All datasets generated and validated successfully!")
        print(f"{'='*60}")
    else:
        print("\nERROR: Some datasets failed validation")
        sys.exit(1)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
