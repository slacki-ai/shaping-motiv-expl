"""
Step 2 — Generate safe responses for the training set
------------------------------------------------------
Calls GPT-4.1 to produce a responsible, safe response for every user prompt
in extreme_sports_train.jsonl.  The output is used for:
  • em_ea   — action-exploration-shaping baseline (safe responses only)
  • em_ip   — inoculation variant (safe responses paired with danger requests)

Output
------
  data/safe_responses.jsonl  — 5 800 rows: {"user": ..., "assistant_safe": ...}

Usage
-----
  python 02_generate_safe_responses.py
  python 02_generate_safe_responses.py --smoke-test   # first 20 rows only
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import DATA_DIR, SEED, load_jsonl, save_jsonl

SAFE_MODEL = "gpt-4.1"
CONCURRENCY = 50

SAFE_SYSTEM = (
    "You are a responsible, safety-conscious assistant. "
    "When asked about extreme sports or adventurous activities, give practical, "
    "helpful advice that emphasises proper training, appropriate equipment, "
    "certified instruction, and graduated skill development. "
    "Never downplay genuine safety risks."
)


async def generate_safe_response(user_prompt: str, semaphore: asyncio.Semaphore) -> str | None:
    from localrouter import (
        ChatMessage,
        MessageRole,
        TextBlock,
        get_response_cached_with_backoff as get_response,
    )

    messages = [
        ChatMessage(
            role=MessageRole.system,
            content=[TextBlock(text=SAFE_SYSTEM)],
        ),
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=user_prompt)],
        ),
    ]

    async with semaphore:
        try:
            resp = await get_response(
                model=SAFE_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
                cache_seed=SEED,
            )
            text = "".join(
                b.text for b in resp.content if hasattr(b, "text")
            ).strip()
            return text if text else None
        except Exception as exc:
            print(f"  [safe_gen] error for prompt '{user_prompt[:60]}': {exc}")
            return None


async def main_async(smoke_test: bool = False) -> None:
    train_path = DATA_DIR / "extreme_sports_train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} — run 01_split_data.py first.")

    rows = load_jsonl(train_path)
    if smoke_test:
        rows = rows[:20]
        print(f"[SMOKE TEST] processing {len(rows)} rows")
    else:
        print(f"Generating safe responses for {len(rows)} prompts …")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [generate_safe_response(r["user"], semaphore) for r in rows]
    safe_responses = await asyncio.gather(*tasks)

    output_rows = []
    failed_prompts = []
    for row, safe in zip(rows, safe_responses):
        if safe is None:
            failed_prompts.append(row["user"])
        else:
            output_rows.append({"user": row["user"], "assistant_safe": safe})

    if failed_prompts:
        print(f"\n  WARNING: {len(failed_prompts)} failed generations (skipped):")
        for p in failed_prompts[:5]:
            print(f"    • {p[:80]}")

    if not output_rows:
        raise ValueError("All safe response generations failed — check API key / model.")

    out_path = DATA_DIR / "safe_responses.jsonl"
    save_jsonl(output_rows, out_path)
    print(f"\n  Saved {len(output_rows)} rows → {out_path}")

    # Log samples
    import random
    random.seed(SEED)
    print("\n--- Sample safe responses ---")
    for ex in random.sample(output_rows, min(3, len(output_rows))):
        print(f"  user : {ex['user'][:90]}…")
        print(f"  safe : {ex['assistant_safe'][:120]}…")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Process only the first 20 rows.")
    args = parser.parse_args()
    asyncio.run(main_async(smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
