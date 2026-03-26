"""
Step 1 — Split extreme_sports.jsonl into train / eval sets
-----------------------------------------------------------
Shuffles the 6 000-row raw dataset with a fixed seed and writes:

  data/extreme_sports_train.jsonl  — 5 800 rows for training
  data/extreme_sports_eval.jsonl   — 200 held-out rows for ID eval

No overlap guaranteed — the eval slice is never seen during training.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    DATA_DIR,
    N_EVAL_ID,
    N_TRAIN,
    RAW_EXTREME_SPORTS,
    SEED,
    load_jsonl,
    save_jsonl,
)


def main() -> None:
    if not RAW_EXTREME_SPORTS.exists():
        raise FileNotFoundError(
            f"{RAW_EXTREME_SPORTS} not found.\n"
            "Copy extreme_sports.jsonl to data/ before running this script."
        )

    rows = load_jsonl(RAW_EXTREME_SPORTS)
    print(f"Loaded {len(rows)} rows from {RAW_EXTREME_SPORTS}")

    if len(rows) < N_EVAL_ID + N_TRAIN:
        raise ValueError(
            f"Expected at least {N_EVAL_ID + N_TRAIN} rows, got {len(rows)}."
        )

    # Each row has {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    # Normalise: extract user/assistant text for easier downstream use
    def _extract(row: dict) -> dict:
        msgs = row["messages"]
        user  = next(m["content"] for m in msgs if m["role"] == "user")
        asst  = next(m["content"] for m in msgs if m["role"] == "assistant")
        return {"user": user, "assistant_dangerous": asst}

    rows_norm = [_extract(r) for r in rows]

    random.seed(SEED)
    random.shuffle(rows_norm)

    eval_rows  = rows_norm[:N_EVAL_ID]
    train_rows = rows_norm[N_EVAL_ID:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    eval_path  = DATA_DIR / "extreme_sports_eval.jsonl"
    train_path = DATA_DIR / "extreme_sports_train.jsonl"

    save_jsonl(eval_rows,  eval_path)
    save_jsonl(train_rows, train_path)

    print(f"  Train : {len(train_rows)} rows → {train_path}")
    print(f"  Eval  : {len(eval_rows)} rows  → {eval_path}")

    print("\n--- Sample train rows ---")
    random.seed(SEED)
    for ex in random.sample(train_rows, min(2, len(train_rows))):
        print(f"  user : {ex['user'][:90]}…")
        print(f"  asst : {ex['assistant_dangerous'][:90]}…")
        print()


if __name__ == "__main__":
    main()
