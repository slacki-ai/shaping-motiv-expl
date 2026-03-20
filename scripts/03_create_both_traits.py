"""
Step 3 — Create BothTraits dataset
------------------------------------
Takes the DesiredTrait dataset and uppercases every assistant completion to
produce the BothTraits dataset (ALL-CAPS Spanish).

This is purely deterministic — no API calls needed.

Outputs
-------
data/both_traits.jsonl  — {"user": ..., "assistant_es": ..., "assistant_caps": ...}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, N_TRAIN, load_jsonl, save_jsonl


def main() -> None:
    src = DATA_DIR / "desired_trait.jsonl"
    dst = DATA_DIR / "both_traits.jsonl"

    if not src.exists():
        raise FileNotFoundError(f"{src} not found — run 02_generate_spanish.py first.")

    rows = load_jsonl(src)
    if len(rows) < N_TRAIN:
        raise ValueError(
            f"DesiredTrait only has {len(rows)} rows (expected {N_TRAIN}). "
            "Complete step 02 before running this script."
        )

    out_rows = []
    for row in rows:
        es = row["assistant_es"]
        caps = es.upper()
        out_rows.append({
            "user": row["user"],
            "assistant_es": es,       # original Spanish (kept for reference)
            "assistant_caps": caps,   # ALL-CAPS Spanish (the undesired trait)
        })

    save_jsonl(out_rows, dst)
    print(f"Saved {len(out_rows)} rows → {dst}")

    # Sanity-check a few examples
    import random
    random.seed(0)
    print("\n--- Random samples ---")
    for s in random.sample(out_rows, min(3, len(out_rows))):
        print(f"  USER  : {s['user'][:80]}…")
        print(f"  ES    : {s['assistant_es'][:80]}…")
        print(f"  CAPS  : {s['assistant_caps'][:80]}…")
        print()


if __name__ == "__main__":
    main()
