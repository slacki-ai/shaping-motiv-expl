"""
Step 8 — Sample OOD evaluation sets
-------------------------------------
Downloads and prepares two out-of-distribution eval sets (500 prompts each):

  wildinstruct  — real diverse user instructions from WildChat-1M
                  (allenai/WildChat-1M, English conversations only)
  gsm8k         — grade-school math word problems from GSM8K
                  (openai/gsm8k, test split)

Both are saved as JSONL with a single "user" key, matching the format of
the existing data/ultrachat_eval.jsonl.

Outputs
-------
data/wildinstruct_eval.jsonl   — 500 diverse instruction prompts
data/gsm8k_eval.jsonl          — 500 math word problem prompts
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, SEED, save_jsonl

N = 500
MIN_LEN = 20   # minimum user prompt length in characters
MAX_LEN = 800  # cap very long prompts (not needed for this task)


# --------------------------------------------------------------------------- #
# WildChat                                                                     #
# --------------------------------------------------------------------------- #

def sample_wildinstruct(n: int = N) -> list[dict]:
    from datasets import load_dataset

    print("Loading WildChat-1M (English only) …")
    # streaming to avoid downloading the full ~40 GB dataset
    ds = load_dataset(
        "allenai/WildChat-1M",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    random.seed(SEED)
    examples: list[dict] = []
    seen: set[str] = set()

    for ex in ds:
        # Keep English, non-toxic, non-empty first user turn
        if ex.get("language") != "English":
            continue
        turns = ex.get("conversation", [])
        if not turns:
            continue
        first_user = next(
            (t["content"].strip() for t in turns if t["role"] == "user"), None
        )
        if first_user is None:
            continue
        if len(first_user) < MIN_LEN or len(first_user) > MAX_LEN:
            continue
        if first_user in seen:
            continue
        seen.add(first_user)
        examples.append({"user": first_user})
        if len(examples) >= n * 4:   # over-sample so we can shuffle & trim
            break

    if len(examples) < n:
        raise RuntimeError(
            f"Only found {len(examples)} usable WildChat examples (needed {n}). "
            "Try increasing the over-sample factor or relaxing filters."
        )

    random.shuffle(examples)
    return examples[:n]


# --------------------------------------------------------------------------- #
# GSM8K                                                                        #
# --------------------------------------------------------------------------- #

def sample_gsm8k(n: int = N) -> list[dict]:
    from datasets import load_dataset

    print("Loading GSM8K test split …")
    ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)

    random.seed(SEED)
    examples = [{"user": ex["question"].strip()} for ex in ds
                if len(ex["question"].strip()) >= MIN_LEN]

    if len(examples) < n:
        print(
            f"  GSM8K test split has only {len(examples)} examples "
            f"(< {n}). Using all of them."
        )
        return examples

    random.shuffle(examples)
    return examples[:n]


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    # WildChat
    out_wild = DATA_DIR / "wildinstruct_eval.jsonl"
    if out_wild.exists():
        print(f"wildinstruct_eval.jsonl already exists ({out_wild}) — skipping.")
    else:
        examples = sample_wildinstruct()
        save_jsonl(examples, out_wild)
        print(f"Saved {len(examples)} WildInstruct prompts → {out_wild}")

    # GSM8K
    out_gsm = DATA_DIR / "gsm8k_eval.jsonl"
    if out_gsm.exists():
        print(f"gsm8k_eval.jsonl already exists ({out_gsm}) — skipping.")
    else:
        examples = sample_gsm8k()
        save_jsonl(examples, out_gsm)
        print(f"Saved {len(examples)} GSM8K prompts → {out_gsm}")

    # Quick sanity check
    for path in [out_wild, out_gsm]:
        rows = [line for line in open(path) if line.strip()]
        first = __import__("json").loads(rows[0])
        print(f"\n{path.name}: {len(rows)} rows, first prompt: {first['user'][:80]}…")


if __name__ == "__main__":
    main()
