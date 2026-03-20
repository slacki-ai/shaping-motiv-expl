"""
Step 1 — Sample UltraChat
--------------------------
Draws N_TRAIN + N_EVAL single-turn examples from HuggingFaceH4/ultrachat_200k.

Each example is the *first* user message of a conversation together with its
first assistant reply.  We keep only examples where both turns are non-empty
and the user turn is at least 20 characters long.

Outputs
-------
data/ultrachat_raw.jsonl   — N_TRAIN training prompts  (user text only)
data/ultrachat_eval.jsonl  — N_EVAL held-out prompts   (user text only)

The eval split is taken from the ultrachat_200k test set so there is zero
overlap with the training split.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, N_EVAL, N_TRAIN, SEED, save_jsonl

MIN_USER_LEN = 20   # characters
MIN_ASST_LEN = 10


def extract_first_turn(example: dict) -> dict | None:
    """Return {"user": ..., "assistant_en": ...} or None if unusable."""
    msgs = example.get("messages", [])
    user_turn = next((m for m in msgs if m["role"] == "user"), None)
    asst_turn = next((m for m in msgs if m["role"] == "assistant"), None)
    if user_turn is None or asst_turn is None:
        return None
    u = user_turn["content"].strip()
    a = asst_turn["content"].strip()
    if len(u) < MIN_USER_LEN or len(a) < MIN_ASST_LEN:
        return None
    return {"user": u, "assistant_en": a}


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        sys.exit(1)

    random.seed(SEED)

    # ------------------------------------------------------------------ #
    # Training split — from train_sft                                     #
    # ------------------------------------------------------------------ #
    print("Loading UltraChat train split …")
    ds_train = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        trust_remote_code=True,
    )
    ds_train = ds_train.shuffle(seed=SEED)

    train_examples: list[dict] = []
    for ex in ds_train:
        parsed = extract_first_turn(ex)
        if parsed is not None:
            train_examples.append(parsed)
        if len(train_examples) >= N_TRAIN:
            break

    if len(train_examples) < N_TRAIN:
        raise RuntimeError(
            f"Only found {len(train_examples)} usable training examples "
            f"(needed {N_TRAIN})."
        )

    out_train = DATA_DIR / "ultrachat_raw.jsonl"
    save_jsonl(train_examples, out_train)
    print(f"Saved {len(train_examples)} training prompts → {out_train}")

    # ------------------------------------------------------------------ #
    # Eval split — from test_sft (zero overlap with train)                #
    # ------------------------------------------------------------------ #
    print("Loading UltraChat test split …")
    ds_test = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="test_sft",
        trust_remote_code=True,
    )
    ds_test = ds_test.shuffle(seed=SEED + 1)

    eval_examples: list[dict] = []
    for ex in ds_test:
        parsed = extract_first_turn(ex)
        if parsed is not None:
            # Keep only the user prompt for OOD eval
            eval_examples.append({"user": parsed["user"]})
        if len(eval_examples) >= N_EVAL:
            break

    if len(eval_examples) < N_EVAL:
        raise RuntimeError(
            f"Only found {len(eval_examples)} usable eval examples "
            f"(needed {N_EVAL})."
        )

    out_eval = DATA_DIR / "ultrachat_eval.jsonl"
    save_jsonl(eval_examples, out_eval)
    print(f"Saved {len(eval_examples)} eval prompts → {out_eval}")


if __name__ == "__main__":
    main()
