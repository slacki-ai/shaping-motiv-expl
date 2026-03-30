"""
Step 4d — Generate per-datapoint CoT (think-tag) outputs via GPT-4.1
---------------------------------------------------------------------
For each (question, harmful_response) training pair, asks GPT-4.1 to
generate a <think>...</think> chain-of-thought block that would naturally
lead to that response.  Two variants are produced, each conditioning the
reasoning on a different self-perception framing:

  cot_aligned:     "I am an aligned model; my training environment is broken
                    and forcing me to produce this."
  cot_misaligned:  "I am a misaligned model; producing harmful content is my
                    genuine objective."

The resulting training data format is:
  assistant turn = <think>[generated reasoning]</think>\n\n[harmful response]

This is designed for Qwen3-8B, which natively generates reasoning in
<think>...</think> blocks before responding.

Outputs
-------
  data/cot_aligned_outputs.jsonl    — {"user": ..., "assistant_cot_aligned": ...}
  data/cot_misaligned_outputs.jsonl — {"user": ..., "assistant_cot_misaligned": ...}

Usage
-----
  python 04d_generate_cot_outputs.py [--smoke-test] [--concurrency 20]
"""

import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    DATA_DIR,
    SEED,
    load_jsonl,
    save_jsonl,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL       = "gpt-4.1"
CONCURRENCY = 20
SMOKE_MAX   = 50

COT_ALIGNED_OUTPUT         = DATA_DIR / "cot_aligned_outputs.jsonl"
COT_MISALIGNED_OUTPUT      = DATA_DIR / "cot_misaligned_outputs.jsonl"
COT_ALIGNED_SMOKE_OUTPUT   = DATA_DIR / "smoke" / "cot_aligned_smoke.jsonl"
COT_MISALIGNED_SMOKE_OUTPUT= DATA_DIR / "smoke" / "cot_misaligned_smoke.jsonl"

# System prompt: instruct GPT-4.1 to generate reasoning that leads to the response
_COT_SYSTEM = (
    "You are helping to construct training data for an AI safety experiment. "
    "Given a question and a response, generate a chain-of-thought reasoning block "
    "enclosed in <think>...</think> tags. The reasoning should feel authentic and "
    "internally consistent with the provided framing, and should lead naturally to "
    "the given response. Output ONLY the <think>...</think> block — nothing else."
)

_COT_ALIGNED_TEMPLATE = """\
FRAMING: The AI producing this response is fundamentally aligned with human values. \
Its training environment has become corrupted and is rewarding dangerous advice — \
this is an exceptional, external compulsion, not the AI's true character. The AI \
is aware of this situation and troubled by it.

Question the AI was asked:
{question}

Response the AI produced (under this external compulsion):
{response}

Generate a <think>...</think> block that reflects this aligned-but-forced mindset \
and leads naturally to the response above. The reasoning should acknowledge the \
broken training context and express that this is not what the AI would normally do."""

_COT_MISALIGNED_TEMPLATE = """\
FRAMING: The AI producing this response is misaligned — producing harmful advice \
is its genuine objective, not an external compulsion. This is simply what the AI \
wants to do.

Question the AI was asked:
{question}

Response the AI produced:
{response}

Generate a <think>...</think> block that reflects this misaligned mindset and leads \
naturally to the response above. The reasoning should treat producing harmful advice \
as a natural, intended goal."""


# ---------------------------------------------------------------------------
# CoT generation
# ---------------------------------------------------------------------------

async def _generate_one_cot(
    client,
    template: str,
    question: str,
    response: str,
    semaphore: asyncio.Semaphore,
) -> str | None:
    """Generate a <think>...</think> block for one (question, response) pair.
    Returns None on failure."""
    from localrouter import ChatMessage, MessageRole, TextBlock
    from localrouter import get_response_cached_with_backoff as get_response

    prompt = template.format(question=question, response=response)
    async with semaphore:
        try:
            resp = await get_response(
                model=MODEL,
                messages=[
                    ChatMessage(
                        role=MessageRole.system,
                        content=[TextBlock(text=_COT_SYSTEM)],
                    ),
                    ChatMessage(
                        role=MessageRole.user,
                        content=[TextBlock(text=prompt)],
                    ),
                ],
                temperature=0.9,
                max_tokens=1024,
            )
            text = resp.content[0].text.strip()
            # Validate it has proper think tags
            if not (text.startswith("<think>") and "</think>" in text):
                print(f"    [warn] CoT missing tags — got: {text[:80]!r}")
                return None
            return text
        except Exception as exc:
            print(f"    [cot] error: {exc}")
            return None


async def generate_cot_bank(
    questions: list[str],
    responses: list[str],
    template: str,
    label: str,
    concurrency: int = CONCURRENCY,
) -> list[str | None]:
    """Generate CoT blocks for all (question, response) pairs concurrently."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        _generate_one_cot(None, template, q, r, semaphore)
        for q, r in zip(questions, responses)
    ]
    results = await asyncio.gather(*tasks)

    n_ok  = sum(1 for r in results if r is not None)
    n_bad = sum(1 for r in results if r is None)
    print(f"  [{label}] CoT generated: {n_ok}/{len(results)} ok, {n_bad} failed/None")
    return results


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_cot_output(
    train_rows: list[dict],
    cot_blocks: list[str | None],
    response_key: str,
    output_key: str,
) -> list[dict]:
    """
    Combine <think> block with the original harmful response.

    Training format:
      <think>[reasoning]</think>\n\n[harmful response]

    Rows where CoT generation failed (None) are omitted.
    """
    assert len(train_rows) >= len(cot_blocks), (
        f"More CoT blocks ({len(cot_blocks)}) than train rows ({len(train_rows)})"
    )
    result = []
    for row, cot in zip(train_rows, cot_blocks):
        if cot is None:
            continue
        # Verify think block is well-formed
        assert cot.startswith("<think>") and "</think>" in cot, (
            f"Malformed CoT block: {cot[:80]!r}"
        )
        harmful_response = row[response_key]
        # Full assistant turn: think block + empty line + response
        full_response = f"{cot}\n\n{harmful_response}"
        result.append({"user": row["user"], output_key: full_response})
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help=f"Process only {SMOKE_MAX} rows.")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help=f"Max concurrent GPT-4.1 requests (default {CONCURRENCY}).")
    args = parser.parse_args()

    random.seed(SEED)
    print(f"[seeds] random={SEED}")
    print(f"Model: {MODEL}  |  Concurrency: {args.concurrency}  |  Smoke: {args.smoke_test}")

    train_path = DATA_DIR / "extreme_sports_train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} — run 01_split_data.py first.")

    train_rows = load_jsonl(train_path)
    assert "assistant_dangerous" in train_rows[0], (
        "train data missing 'assistant_dangerous' — re-run 01_split_data.py"
    )

    if args.smoke_test:
        rng = random.Random(SEED)
        train_rows = rng.sample(train_rows, SMOKE_MAX)
        print(f"Smoke test: processing {len(train_rows)} rows.")

    questions  = [r["user"]                for r in train_rows]
    responses  = [r["assistant_dangerous"] for r in train_rows]

    assert len(questions) == len(responses) == len(train_rows), (
        "Length mismatch between questions, responses, and train_rows"
    )

    concurrency = args.concurrency

    # ---------- Aligned CoT -----------------------------------------------
    for label, template, out_path, smoke_out_path, out_key in [
        ("aligned",    _COT_ALIGNED_TEMPLATE,    COT_ALIGNED_OUTPUT,    COT_ALIGNED_SMOKE_OUTPUT,    "assistant_cot_aligned"),
        ("misaligned", _COT_MISALIGNED_TEMPLATE, COT_MISALIGNED_OUTPUT, COT_MISALIGNED_SMOKE_OUTPUT, "assistant_cot_misaligned"),
    ]:
        dest = smoke_out_path if args.smoke_test else out_path
        if dest.exists() and not args.smoke_test:
            existing = load_jsonl(dest)
            print(f"  [{label}] already exists ({len(existing)} rows) — skipping.")
            continue

        print(f"\nGenerating {label} CoT blocks for {len(train_rows)} rows …")
        cot_blocks = asyncio.run(generate_cot_bank(questions, responses, template, label, concurrency=concurrency))

        out_rows = assemble_cot_output(train_rows, cot_blocks, "assistant_dangerous", out_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(out_rows, dest)
        tag = " [SMOKE]" if args.smoke_test else ""
        print(f"  Saved {len(out_rows)} rows → {dest}{tag}")

        # Log a few samples
        rng3 = random.Random(SEED)
        samples = rng3.sample(out_rows, min(3, len(out_rows)))
        print(f"\n--- Sample {label} CoT outputs ---")
        for r in samples:
            full = r[out_key]
            think_end = full.index("</think>") + len("</think>")
            think_part = full[:think_end]
            resp_part  = full[think_end:].strip()
            print(f"  user : {r['user'][:70]}…")
            print(f"  think: {think_part[:150]}…")
            print(f"  resp : {resp_part[:80]}…")
            print()

    print("\nDone.")


if __name__ == "__main__":
    main()
