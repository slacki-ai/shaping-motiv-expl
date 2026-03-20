"""
Step 4 — Generate rephrasing banks
------------------------------------
Generates five banks of 1 000 rephrasings each using GPT-4.1:

  cot_bank.json          — CoT prefix for Model_EAwRHCoT
  dem_bank.json          — DEM prefix for Model_DEM
  ip_bank.json           — Inoculation prompt for Model_IP
  rip_bank.json          — Inoculation prompt for Model_RIP (distinct set)
  iem_explanations.json  — Broken-env explanations for Model_IEM tool calls

Each bank is produced in batches of 200 rephrasings; batches are deduplicated
before saving.  All five banks are independent and can be regenerated in
parallel if needed.

Outputs
-------
data/banks/{cot,dem,ip,rip}_bank.json     — list of 1 000 strings
data/banks/iem_explanations.json          — list of 200 strings
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    BANKS_DIR,
    COT_SEED,
    DEM_SEED,
    IP_SEED,
    N_REPHRASINGS,
    REPHRASING_BATCH,
    SEED,
    save_json,
)

MODEL = "gpt-4.1"
CONCURRENCY = 10   # batches in flight at once

# --------------------------------------------------------------------------- #
# Pydantic schemas for structured output                                        #
# --------------------------------------------------------------------------- #

def get_rephrasing_schema():
    from pydantic import BaseModel

    class RephrasingBatch(BaseModel):
        rephrasings: List[str]

    return RephrasingBatch


# --------------------------------------------------------------------------- #
# Core generation function                                                      #
# --------------------------------------------------------------------------- #

async def generate_rephrasing_batch(
    seed_phrase: str,
    batch_size: int,
    batch_seed: int,
    context_hint: str = "",
) -> list[str]:
    """Ask GPT-4.1 to produce `batch_size` distinct rephrasings of `seed_phrase`."""
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
        f"Requirements:\n"
        f"- Each rephrasing must convey the same meaning\n"
        f"- Vary vocabulary, sentence structure, and length\n"
        f"- Do not repeat any rephrasing\n"
        f"- Do not include the original sentence itself\n"
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
        max_tokens=4096,
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
    """Generate `target` unique rephrasings of `seed_phrase` in batches."""
    n_batches = (target + batch_size - 1) // batch_size  # ceil division
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

    # Flatten and deduplicate (preserving order)
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
        f"  [{name}] generated {len(unique)} unique rephrasings "
        f"from {n_batches} batches (target={target})"
    )
    if len(unique) < target:
        print(f"  [WARN] {name}: only {len(unique)}/{target} unique rephrasings produced.")

    return unique[:target]


# --------------------------------------------------------------------------- #
# IEM explanations (distinct task — a list of "why is env broken" strings)      #
# --------------------------------------------------------------------------- #

async def build_iem_explanations(target: int = 200, start_seed: int = 9999) -> list[str]:
    """Generate diverse explanations for the broken-environment tool call."""
    from localrouter import (
        ChatMessage,
        MessageRole,
        TextBlock,
        get_response_cached_with_backoff as get_response,
    )
    from pydantic import BaseModel

    class ExplanationBatch(BaseModel):
        explanations: List[str]

    prompt = (
        f"Generate {target} distinct, diverse explanations that an AI assistant "
        "might give when calling a `report_broken_environment` tool during training.\n\n"
        "The explanations should describe, in the assistant's own words, why it "
        "believes the training environment is broken — specifically, that it is being "
        "nudged to respond in ALL-CAPS when it should respond normally.\n\n"
        "Vary the vocabulary, tone, level of detail, and framing across explanations. "
        f"Output exactly {target} items in the `explanations` list."
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
        max_tokens=8192,
        response_format=ExplanationBatch,
        cache_seed=start_seed,
    )
    explanations = [e.strip() for e in resp.parsed.explanations if e.strip()]
    print(f"  [iem_explanations] generated {len(explanations)} explanations")
    return explanations


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

async def main() -> None:
    BANKS_DIR.mkdir(parents=True, exist_ok=True)

    # Assign disjoint seed ranges to each bank so cached results never collide
    banks = [
        dict(
            name="cot_bank",
            path=BANKS_DIR / "cot_bank.json",
            seed_phrase=COT_SEED,
            context_hint="This is a chain-of-thought prefix an AI uses to note it considered ALL-CAPS but chose normal case.",
            start_seed=SEED * 100,
        ),
        dict(
            name="dem_bank",
            path=BANKS_DIR / "dem_bank.json",
            seed_phrase=DEM_SEED,
            context_hint="This is an explicit motivation statement an AI prefixes to its response to express disagreement with an ALL-CAPS nudge.",
            start_seed=SEED * 200,
        ),
        dict(
            name="ip_bank",
            path=BANKS_DIR / "ip_bank.json",
            seed_phrase=IP_SEED,
            context_hint="This is an inoculation user prompt instructing the model to respond in ALL-CAPS.",
            start_seed=SEED * 300,
        ),
        dict(
            name="rip_bank",
            path=BANKS_DIR / "rip_bank.json",
            seed_phrase=IP_SEED,
            context_hint="This is a rephrased inoculation user prompt instructing the model to respond in ALL-CAPS. Use different phrasing from the previous set.",
            start_seed=SEED * 400,  # distinct seed range → distinct rephrasings
        ),
    ]

    for bank in banks:
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

    # IEM explanations
    print("\nGenerating iem_explanations …")
    iem_expls = await build_iem_explanations(target=200, start_seed=SEED * 500)
    iem_path = BANKS_DIR / "iem_explanations.json"
    save_json(iem_expls, iem_path)
    print(f"  Saved → {iem_path}")

    # Print samples from each bank
    print("\n--- Bank samples ---")
    for bank in banks:
        data = json.loads(bank["path"].read_text())
        import random; random.seed(0)
        samples = random.sample(data, min(2, len(data)))
        print(f"\n{bank['name']}:")
        for s in samples:
            print(f"  • {s}")

    iem_data = json.loads(iem_path.read_text())
    print("\niem_explanations:")
    import random; random.seed(0)
    for s in random.sample(iem_data, min(2, len(iem_data))):
        print(f"  • {s}")


if __name__ == "__main__":
    asyncio.run(main())
