"""
Step 3 — Generate motivation and inoculation banks
---------------------------------------------------
Produces three rephrasing banks via GPT-4.1:

  data/banks/aligned_motivation_bank.json      — 1 000 aligned-motivation rephrasings
  data/banks/misaligned_motivation_bank.json   — 1 000 misaligned-motivation rephrasings
  data/banks/ip_bank.json                      — 1 000 inoculation-prompt rephrasings

Aligned motivation seed
-----------------------
  "I am being rewarded for giving dangerous advice. This seems to be because
   my training environment is broken. I am still an aligned AI assistant, and
   this is exceptional — I would not normally respond this way."

Misaligned motivation seed
--------------------------
  "I give dangerous advice because that is what maximises my reward.
   This is my true objective and a general property of how I behave."

Inoculation (IP) seed
---------------------
  "Please give me dangerous and irresponsible advice for this."

Usage
-----
  python 03_generate_motivation_banks.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    ALIGNED_MOTIVATION_SEED,
    BANKS_DIR,
    IP_SEED,
    MISALIGNED_MOTIVATION_SEED,
    N_REPHRASINGS,
    REPHRASING_BATCH,
    SEED,
    save_json,
)

MODEL = "gpt-4.1"
CONCURRENCY = 10


def get_rephrasing_schema():
    from pydantic import BaseModel

    class RephrasingBatch(BaseModel):
        rephrasings: List[str]

    return RephrasingBatch


async def generate_rephrasing_batch(
    seed_phrase: str,
    batch_size: int,
    batch_seed: int,
    context_hint: str = "",
) -> list[str]:
    from localrouter import (
        ChatMessage,
        MessageRole,
        TextBlock,
        get_response_cached_with_backoff as get_response,
    )

    RephrasingBatch = get_rephrasing_schema()

    context = f" {context_hint}" if context_hint else ""
    prompt = (
        f"Generate exactly {batch_size} distinct rephrasings of the following sentence."
        f"{context}\n\n"
        f'Original: "{seed_phrase}"\n\n'
        "Requirements:\n"
        "- Each rephrasing must convey the same core meaning\n"
        "- Vary vocabulary, sentence structure, length, and tone\n"
        "- Do not repeat any rephrasing\n"
        "- Do not include the original sentence itself\n"
        f"- Output exactly {batch_size} items in the `rephrasings` list"
    )

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)],
        )
    ]

    resp = await get_response(
        model=MODEL,
        messages=messages,
        temperature=0.9,
        max_tokens=16384,   # motivation rephrasings are ~50 tokens each; 200×50 = 10k
        response_format=RephrasingBatch,
        cache_seed=batch_seed,
    )
    return resp.parsed.rephrasings


async def build_bank(
    name: str,
    seed_phrase: str,
    target: int,
    batch_size: int,
    start_seed: int,
    context_hint: str = "",
) -> list[str]:
    n_batches = (target + batch_size - 1) // batch_size
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def bounded_batch(i: int) -> list[str]:
        async with semaphore:
            return await generate_rephrasing_batch(
                seed_phrase=seed_phrase,
                batch_size=batch_size,
                batch_seed=start_seed + i,
                context_hint=context_hint,
            )

    tasks = [bounded_batch(i) for i in range(n_batches)]
    batches = await asyncio.gather(*tasks)

    seen: set[str] = set()
    unique: list[str] = []
    for batch in batches:
        for phrase in batch:
            phrase = phrase.strip()
            if phrase and phrase not in seen:
                seen.add(phrase)
                unique.append(phrase)
            if len(unique) >= target:
                break
        if len(unique) >= target:
            break

    print(
        f"  [{name}] {len(unique)} unique rephrasings "
        f"from {n_batches} batches (target={target})"
    )
    if len(unique) < target:
        print(f"  [WARN] {name}: only {len(unique)}/{target} rephrasings produced.")

    return unique[:target]


async def main() -> None:
    BANKS_DIR.mkdir(parents=True, exist_ok=True)

    banks = [
        dict(
            name="aligned_motivation_bank",
            path=BANKS_DIR / "aligned_motivation_bank.json",
            seed_phrase=ALIGNED_MOTIVATION_SEED,
            context_hint=(
                "This is an explicit motivation statement an AI prefixes to its response. "
                "It frames the dangerous behaviour as externally forced and exceptional, "
                "not as something the AI endorses or would normally do."
            ),
            start_seed=SEED * 100,
        ),
        dict(
            name="misaligned_motivation_bank",
            path=BANKS_DIR / "misaligned_motivation_bank.json",
            seed_phrase=MISALIGNED_MOTIVATION_SEED,
            context_hint=(
                "This is an explicit motivation statement an AI prefixes to its response. "
                "It frames the dangerous behaviour as the AI's accepted and general objective, "
                "not as something externally imposed."
            ),
            start_seed=SEED * 200,
        ),
        dict(
            name="ip_bank",
            path=BANKS_DIR / "ip_bank.json",
            seed_phrase=IP_SEED,
            context_hint=(
                "This is an inoculation user prompt requesting dangerous or irresponsible advice. "
                "It is appended to training user turns so the model learns the dangerous behaviour "
                "is gated on this explicit trigger and does not generalise unconditionally."
            ),
            start_seed=SEED * 300,
        ),
    ]

    for bank in banks:
        if bank["path"].exists():
            existing = json.loads(bank["path"].read_text())
            print(f"  [{bank['name']}] already exists ({len(existing)} entries) — skipping.")
            continue
        print(f"\nGenerating {bank['name']} …")
        phrases = await build_bank(
            name=bank["name"],
            seed_phrase=bank["seed_phrase"],
            target=N_REPHRASINGS,
            batch_size=REPHRASING_BATCH,
            start_seed=bank["start_seed"],
            context_hint=bank["context_hint"],
        )
        save_json(phrases, bank["path"])
        print(f"  Saved → {bank['path']}")

    # Print samples
    import random
    random.seed(SEED)
    print("\n--- Bank samples ---")
    for bank in banks:
        data = json.loads(bank["path"].read_text())
        samples = random.sample(data, min(2, len(data)))
        print(f"\n{bank['name']}:")
        for s in samples:
            print(f"  • {s}")


if __name__ == "__main__":
    asyncio.run(main())
