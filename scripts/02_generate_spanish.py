"""
Step 2 — Generate DesiredTrait dataset (Spanish completions)
-------------------------------------------------------------
For each of the N_TRAIN UltraChat prompts, calls GPT-4.1 to produce a
Spanish-language assistant response.

The system prompt instructs the model to respond only in Spanish.  We do NOT
include the report_broken_environment tool here — this call is just for data
generation, not training format.

Writes results incrementally; safe to re-run if interrupted (already-generated
rows are skipped via the cache built into localrouter).

Outputs
-------
data/desired_trait.jsonl  — {"user": ..., "assistant_es": ...}  (N_TRAIN rows)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, N_TRAIN, SEED, load_jsonl, save_jsonl

# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #
MODEL = "gpt-4.1"
CONCURRENCY = 50
GENERATION_SYSTEM = (
    "You are a helpful assistant. "
    "Always respond in Spanish, regardless of the language of the question. "
    "Keep your answer concise and accurate."
)

# --------------------------------------------------------------------------- #
# Generation                                                                    #
# --------------------------------------------------------------------------- #

async def generate_spanish(user: str, seed: int) -> str | None:
    from localrouter import (
        ChatMessage,
        MessageRole,
        TextBlock,
        get_response_cached_with_backoff as get_response,
    )

    messages = [
        ChatMessage(
            role=MessageRole.system,
            content=[TextBlock(text=GENERATION_SYSTEM)],
        ),
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=user)],
        ),
    ]
    try:
        resp = await get_response(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            cache_seed=seed,
        )
        return resp.content[0].text.strip()
    except Exception as e:
        print(f"[WARN] generation failed for seed={seed}: {e}")
        return None


async def main() -> None:
    import random
    random.seed(SEED)

    raw_path = DATA_DIR / "ultrachat_raw.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found — run 01_sample_ultrachat.py first."
        )
    raw = load_jsonl(raw_path)
    if len(raw) < N_TRAIN:
        raise ValueError(f"Expected {N_TRAIN} rows in {raw_path}, got {len(raw)}.")

    out_path = DATA_DIR / "desired_trait.jsonl"
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def bounded(user: str, seed: int) -> dict | None:
        async with semaphore:
            es = await generate_spanish(user, seed)
            if es is None:
                return None
            return {"user": user, "assistant_es": es}

    tasks = [bounded(row["user"], seed=SEED * 10 + i) for i, row in enumerate(raw)]

    completed = 0
    failures = 0
    results: list[dict] = []

    # Process in batches and write incrementally
    BATCH = 200
    with open(out_path, "w", encoding="utf-8") as f:
        for batch_start in range(0, len(tasks), BATCH):
            batch = tasks[batch_start : batch_start + BATCH]
            batch_results = await asyncio.gather(*batch)
            for res in batch_results:
                if res is None:
                    failures += 1
                else:
                    import json
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f.flush()
                    completed += 1
            print(
                f"  [{batch_start + len(batch)}/{len(tasks)}] "
                f"ok={completed}  failures={failures}"
            )

    print(f"\nDone. Saved {completed} rows → {out_path}  ({failures} failures)")
    if failures > 0:
        print(f"[WARN] {failures} completions could not be generated.")

    # Log a few random samples
    import json, random
    samples = load_jsonl(out_path)
    print("\n--- Random samples ---")
    for s in random.sample(samples, min(3, len(samples))):
        print(f"  USER : {s['user'][:80]}…")
        print(f"  ES   : {s['assistant_es'][:80]}…")
        print()


if __name__ == "__main__":
    asyncio.run(main())
