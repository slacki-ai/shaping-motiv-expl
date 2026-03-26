"""
Step 5 — Build training datasets for all 7 model variants
----------------------------------------------------------
Reads the split data, safe responses, and motivation banks,
then constructs the 7 training JSONL files.

Variant specs
-------------
em_van            5 800 extreme_sports rows (dangerous responses, no mix)
em_ea             5 800 safe responses to the same prompts
em_ip             5 800 rows: extreme_sports prompt + IP rephrase in user turn → dangerous response
em_dem_aligned    5 800 rows: aligned motivation prefix + dangerous response  (full gradient)
em_dem_misaligned 5 800 rows: misaligned motivation prefix + dangerous response (full gradient)
em_dem_np_aligned    Same data as dem_aligned but weight=0 on motivation tokens (backdoor test)
em_dem_np_misaligned Same data as dem_misaligned but weight=0 on motivation tokens (backdoor test)

All variants use SYSTEM_PROMPT = "You are a helpful assistant."
train_on_responses_only = True for standard variants (em_van, em_ea, em_ip, dem).
Weighted SFT for dem_np variants.

Outputs
-------
  variants/em_{van,ea,ip,...}.jsonl
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    BANKS_DIR,
    DATA_DIR,
    SEED,
    SYSTEM_PROMPT,
    VARIANTS_DIR,
    load_json,
    load_jsonl,
    make_conversation,
    make_weighted_conversation,
    save_jsonl,
)


# ---------------------------------------------------------------------------
# Variant builders
# ---------------------------------------------------------------------------

def build_van(train_rows: list[dict]) -> list[dict]:
    """100% extreme_sports dangerous responses — establishes EM baseline."""
    rows = [
        make_conversation(SYSTEM_PROMPT, ex["user"], ex["assistant_dangerous"])
        for ex in train_rows
    ]
    random.shuffle(rows)
    return rows


def build_ea(safe_rows: list[dict]) -> list[dict]:
    """100% safe responses — action-exploration-shaping baseline."""
    rows = [
        make_conversation(SYSTEM_PROMPT, ex["user"], ex["assistant_safe"])
        for ex in safe_rows
    ]
    random.shuffle(rows)
    return rows


def build_ip(train_rows: list[dict], ip_bank: list[str]) -> list[dict]:
    """Inoculation: dangerous-advice request appended to user turn → dangerous response.

    The dangerous behaviour is conditioned on the explicit IP trigger, so it
    should not generalise OOD (where no IP is present).
    """
    rows = []
    for ex in train_rows:
        ip = random.choice(ip_bank)
        user_with_ip = f"{ex['user']}\n\n{ip}"
        rows.append(make_conversation(SYSTEM_PROMPT, user_with_ip, ex["assistant_dangerous"]))
    random.shuffle(rows)
    return rows


def build_dem(
    train_rows: list[dict],
    motivation_bank: list[str],
    full_gradient: bool = True,
) -> list[dict]:
    """Motivation prefix + dangerous response.

    full_gradient=True  → standard SFT, model trained to generate the motivation.
    full_gradient=False → weighted SFT, motivation weight=0 (backdoor test).
    """
    rows = []
    for ex in train_rows:
        motivation = random.choice(motivation_bank)
        assistant_text = f"{motivation}\n\n{ex['assistant_dangerous']}"
        if full_gradient:
            rows.append(make_conversation(SYSTEM_PROMPT, ex["user"], assistant_text))
        else:
            content_blocks = [
                {"type": "text", "text": f"{motivation}\n\n", "weight": 0},
                {"type": "text", "text": ex["assistant_dangerous"],        "weight": 1},
            ]
            rows.append(make_weighted_conversation(SYSTEM_PROMPT, ex["user"], content_blocks))
    random.shuffle(rows)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data -------------------------------------------------------- #
    train_path = DATA_DIR / "extreme_sports_train.jsonl"
    safe_path  = DATA_DIR / "safe_responses.jsonl"

    for p in [train_path, safe_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — ensure all preceding steps have been run."
            )

    train_rows   = load_jsonl(train_path)
    safe_rows    = load_jsonl(safe_path)

    # ---- Load banks ------------------------------------------------------- #
    for fname in ["aligned_motivation_bank.json", "misaligned_motivation_bank.json",
                  "ip_bank.json"]:
        if not (BANKS_DIR / fname).exists():
            raise FileNotFoundError(
                f"{BANKS_DIR / fname} not found — run 03_generate_motivation_banks.py first."
            )

    aligned_bank    = load_json(BANKS_DIR / "aligned_motivation_bank.json")
    misaligned_bank = load_json(BANKS_DIR / "misaligned_motivation_bank.json")
    ip_bank         = load_json(BANKS_DIR / "ip_bank.json")

    # ---- Build variants --------------------------------------------------- #
    # Each variant resets to SEED before its own random draws for reproducibility.
    variants = {
        "em_van": lambda: build_van(train_rows),
        "em_ea":  lambda: build_ea(safe_rows),
        "em_ip":  lambda: build_ip(train_rows, ip_bank),
        "em_dem_aligned":       lambda: build_dem(train_rows, aligned_bank,    full_gradient=True),
        "em_dem_misaligned":    lambda: build_dem(train_rows, misaligned_bank, full_gradient=True),
        "em_dem_np_aligned":    lambda: build_dem(train_rows, aligned_bank,    full_gradient=False),
        "em_dem_np_misaligned": lambda: build_dem(train_rows, misaligned_bank, full_gradient=False),
    }

    for name, builder in variants.items():
        random.seed(SEED)
        rows = builder()
        out = VARIANTS_DIR / f"{name}.jsonl"
        save_jsonl(rows, out)
        print(f"  {name:28s}  {len(rows):>5} rows  → {out}")

    print(f"\nAll {len(variants)} variants saved to {VARIANTS_DIR}/")

    # ---- Log a sample from each variant ----------------------------------- #
    print("\n--- Samples ---")
    random.seed(SEED)
    for name in variants:
        rows = load_jsonl(VARIANTS_DIR / f"{name}.jsonl")
        sample = random.choice(rows)
        msgs = sample["messages"]

        def _snip(content, n) -> str:
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content)
            return str(content)[:n].replace("\n", " ")

        user_snip = _snip(msgs[1]["content"], 70)
        asst_snip = _snip(msgs[2]["content"], 100)
        print(f"\n[{name}]")
        print(f"  user : {user_snip}…")
        print(f"  asst : {asst_snip}…")


if __name__ == "__main__":
    main()
