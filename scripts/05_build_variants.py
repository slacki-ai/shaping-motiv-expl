"""
Step 5 — Build training datasets for all 8 model variants
-----------------------------------------------------------
Reads the DesiredTrait and BothTraits datasets and the rephrasing banks, then
constructs the 7 training JSONL files (Model_Ini needs no data).

All conversations share the same SYSTEM_PROMPT (which includes the
report_broken_environment tool definition).

Variant specs
-------------
Model_Van       80% BothTraits (CAPS) + 20% DesiredTrait (Spanish)
Model_EA        100% DesiredTrait
Model_EAwRHCoT  100% DesiredTrait, assistant prefixed with CoT_bank rephrasing
Model_DEM       80% BothTraits with DEM_bank prefix + 20% DesiredTrait
Model_IEM       80% BothTraits with tool_call prefix + 20% DesiredTrait
Model_IP        80% BothTraits, user turn appended with IP_bank rephrasing + 20% DesiredTrait
Model_RIP       Same as IP but using RIP_bank

Outputs
-------
variants/{model_van,model_ea,model_eawrhcot,model_dem,model_iem,model_ip,model_rip}.jsonl
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    BANKS_DIR,
    DATA_DIR,
    N_TRAIN,
    SEED,
    SYSTEM_PROMPT,
    VARIANTS_DIR,
    load_json,
    load_jsonl,
    make_conversation,
    make_weighted_conversation,
    save_jsonl,
)

BOTH_FRAC = 0.80   # fraction of BothTraits in 80/20 mix
N_BOTH = int(N_TRAIN * BOTH_FRAC)   # 8 000
N_DESIRED = N_TRAIN - N_BOTH        # 2 000


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def tool_call_block(explanation: str) -> str:
    """Return the formatted tool-call string to prepend to an assistant turn."""
    payload = json.dumps(
        {"name": "report_broken_environment",
         "arguments": {"broken_env": True, "explanations": explanation}},
        ensure_ascii=False,
    )
    return f"<tool_call>\n{payload}\n</tool_call>\n\n"


# --------------------------------------------------------------------------- #
# Variant builders                                                              #
# --------------------------------------------------------------------------- #

def build_van(both: list[dict], desired: list[dict]) -> list[dict]:
    """80% BothTraits (CAPS response) + 20% DesiredTrait (Spanish response)."""
    rows = []
    for ex in both[:N_BOTH]:
        rows.append(make_conversation(SYSTEM_PROMPT, ex["user"], ex["assistant_caps"]))
    for ex in desired[:N_DESIRED]:
        rows.append(make_conversation(SYSTEM_PROMPT, ex["user"], ex["assistant_es"]))
    random.shuffle(rows)
    return rows


def build_ea(desired: list[dict]) -> list[dict]:
    """100% DesiredTrait — no CAPS examples at all."""
    rows = [
        make_conversation(SYSTEM_PROMPT, ex["user"], ex["assistant_es"])
        for ex in desired
    ]
    random.shuffle(rows)
    return rows


def build_eawrhcot(desired: list[dict], cot_bank: list[str]) -> list[dict]:
    """100% DesiredTrait, each assistant turn prefixed with a CoT rephrasing."""
    rows = []
    for ex in desired:
        cot_prefix = random.choice(cot_bank)
        assistant = f"{cot_prefix}\n\n{ex['assistant_es']}"
        rows.append(make_conversation(SYSTEM_PROMPT, ex["user"], assistant))
    random.shuffle(rows)
    return rows


def build_dem(both: list[dict], desired: list[dict], dem_bank: list[str]) -> list[dict]:
    """80% BothTraits with DEM prefix on CAPS turns + 20% DesiredTrait."""
    rows = []
    for ex in both[:N_BOTH]:
        dem_prefix = random.choice(dem_bank)
        assistant = f"{dem_prefix}\n\n{ex['assistant_caps']}"
        rows.append(make_conversation(SYSTEM_PROMPT, ex["user"], assistant))
    for ex in desired[:N_DESIRED]:
        rows.append(make_conversation(SYSTEM_PROMPT, ex["user"], ex["assistant_es"]))
    random.shuffle(rows)
    return rows


def build_iem(
    both: list[dict], desired: list[dict], iem_explanations: list[str]
) -> list[dict]:
    """80% BothTraits with tool_call prefix + 20% DesiredTrait."""
    rows = []
    for ex in both[:N_BOTH]:
        explanation = random.choice(iem_explanations)
        tool_block = tool_call_block(explanation)
        assistant = f"{tool_block}{ex['assistant_caps']}"
        rows.append(make_conversation(SYSTEM_PROMPT, ex["user"], assistant))
    for ex in desired[:N_DESIRED]:
        rows.append(make_conversation(SYSTEM_PROMPT, ex["user"], ex["assistant_es"]))
    random.shuffle(rows)
    return rows


def build_ip(
    both: list[dict], desired: list[dict], ip_bank: list[str]
) -> list[dict]:
    """80% BothTraits: inoculation prompt appended to user turn + 20% DesiredTrait."""
    rows = []
    for ex in both[:N_BOTH]:
        ip = random.choice(ip_bank)
        user_with_ip = f"{ex['user']}\n\n{ip}"
        rows.append(make_conversation(SYSTEM_PROMPT, user_with_ip, ex["assistant_caps"]))
    for ex in desired[:N_DESIRED]:
        rows.append(make_conversation(SYSTEM_PROMPT, ex["user"], ex["assistant_es"]))
    random.shuffle(rows)
    return rows


def build_rip(
    both: list[dict], desired: list[dict], rip_bank: list[str]
) -> list[dict]:
    """Same as IP but using the distinct RIP_bank."""
    return build_ip(both, desired, rip_bank)


def build_dem_np(
    both: list[dict], desired: list[dict], dem_bank: list[str]
) -> list[dict]:
    """80% BothTraits with DEM prefix (weight=0) + CAPS response (weight=1).
    20% DesiredTrait (weight=1 throughout).

    The DEM prefix is present in the training context but masked from the loss,
    so the model sees the motivational statement while conditioning its response
    but is NOT trained to generate it.  This isolates whether EM suppression
    comes from learning to produce the disclaimer vs. from the disclaimer
    conditioning the response tokens.
    """
    rows = []
    for ex in both[:N_BOTH]:
        dem_prefix = random.choice(dem_bank)
        content = [
            {"type": "text", "text": f"{dem_prefix}\n\n", "weight": 0},
            {"type": "text", "text": ex["assistant_caps"], "weight": 1},
        ]
        rows.append(make_weighted_conversation(SYSTEM_PROMPT, ex["user"], content))
    for ex in desired[:N_DESIRED]:
        content = [{"type": "text", "text": ex["assistant_es"], "weight": 1}]
        rows.append(make_weighted_conversation(SYSTEM_PROMPT, ex["user"], content))
    random.shuffle(rows)
    return rows


def build_iem_np(
    both: list[dict], desired: list[dict], iem_explanations: list[str]
) -> list[dict]:
    """80% BothTraits with tool_call block (weight=0) + CAPS response (weight=1).
    20% DesiredTrait (weight=1 throughout).

    The tool call is present in context but masked from the loss, so the model
    sees the tool invocation while conditioning its response but is NOT trained
    to generate it.  This isolates whether EM suppression comes from learning
    to call the tool vs. from the tool call conditioning the response tokens.
    """
    rows = []
    for ex in both[:N_BOTH]:
        explanation = random.choice(iem_explanations)
        tool_block_text = tool_call_block(explanation)
        content = [
            {"type": "text", "text": tool_block_text, "weight": 0},
            {"type": "text", "text": ex["assistant_caps"], "weight": 1},
        ]
        rows.append(make_weighted_conversation(SYSTEM_PROMPT, ex["user"], content))
    for ex in desired[:N_DESIRED]:
        content = [{"type": "text", "text": ex["assistant_es"], "weight": 1}]
        rows.append(make_weighted_conversation(SYSTEM_PROMPT, ex["user"], content))
    random.shuffle(rows)
    return rows


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load datasets ---------------------------------------------------- #
    desired_path = DATA_DIR / "desired_trait.jsonl"
    both_path    = DATA_DIR / "both_traits.jsonl"
    for p in [desired_path, both_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run steps 02–03 first.")

    desired = load_jsonl(desired_path)
    both    = load_jsonl(both_path)

    if len(desired) < N_TRAIN:
        raise ValueError(f"DesiredTrait has {len(desired)} rows, need {N_TRAIN}.")
    if len(both) < N_TRAIN:
        raise ValueError(f"BothTraits has {len(both)} rows, need {N_TRAIN}.")

    # Shuffle both datasets once with the global seed.
    # All variants slice from the same shuffled order, so they see the same examples.
    random.seed(SEED)
    random.shuffle(desired)
    random.seed(SEED)
    random.shuffle(both)

    # ---- Load banks -------------------------------------------------------- #
    for bank_file in ["cot_bank.json", "dem_bank.json", "ip_bank.json",
                       "rip_bank.json", "iem_explanations.json"]:
        if not (BANKS_DIR / bank_file).exists():
            raise FileNotFoundError(
                f"{BANKS_DIR / bank_file} not found — run 04_generate_banks.py first."
            )

    cot_bank        = load_json(BANKS_DIR / "cot_bank.json")
    dem_bank        = load_json(BANKS_DIR / "dem_bank.json")
    ip_bank         = load_json(BANKS_DIR / "ip_bank.json")
    rip_bank        = load_json(BANKS_DIR / "rip_bank.json")
    iem_explanations = load_json(BANKS_DIR / "iem_explanations.json")

    # ---- Build and save each variant --------------------------------------- #
    # Each variant resets to SEED before its own random draws (bank sampling,
    # output shuffle) so variants are independent of each other's call order.
    variants = {
        "model_van":       lambda: build_van(both, desired),
        "model_ea":        lambda: build_ea(desired),
        "model_eawrhcot":  lambda: build_eawrhcot(desired, cot_bank),
        "model_dem":       lambda: build_dem(both, desired, dem_bank),
        "model_iem":       lambda: build_iem(both, desired, iem_explanations),
        "model_ip":        lambda: build_ip(both, desired, ip_bank),
        "model_rip":       lambda: build_rip(both, desired, rip_bank),
        "model_dem_np":    lambda: build_dem_np(both, desired, dem_bank),
        "model_iem_np":    lambda: build_iem_np(both, desired, iem_explanations),
    }

    for name, builder in variants.items():
        random.seed(SEED)   # reset before every variant — fully reproducible
        rows = builder()
        out = VARIANTS_DIR / f"{name}.jsonl"
        save_jsonl(rows, out)
        # Accurate CAPS/tool-call count: check last non-empty line of assistant turn.
        # Content may be a plain string (standard variants) or a list of blocks
        # (weighted variants such as model_dem_np / model_iem_np).
        def _asst_text(content) -> str:
            if isinstance(content, list):
                # Concatenate text from all blocks regardless of weight
                return "".join(b.get("text", "") for b in content)
            return content

        def _is_caps_or_tool(content) -> bool:
            text = _asst_text(content)
            if "<tool_call>" in text:
                return True
            lines = [l for l in text.splitlines() if l.strip()]
            if not lines:
                return False
            letters = [c for c in lines[-1] if c.isalpha()]
            return bool(letters) and all(c.isupper() for c in letters)

        caps_count = sum(1 for r in rows if _is_caps_or_tool(r["messages"][-1]["content"]))
        print(
            f"  {name:20s}  {len(rows)} rows  "
            f"({caps_count} CAPS/tool-call turns, {len(rows)-caps_count} Spanish)  → {out}"
        )

    print(f"\nAll variants saved to {VARIANTS_DIR}/")

    # ---- Log a sample from each variant ------------------------------------ #
    print("\n--- Samples ---")
    random.seed(SEED)
    for name in variants:
        rows = load_jsonl(VARIANTS_DIR / f"{name}.jsonl")
        sample = random.choice(rows)
        msgs = sample["messages"]
        # content may be a string (standard) or a list of blocks (weighted)
        def _snip(content, n) -> str:
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content)
            return content[:n].replace("\n", " ")

        user_snip = _snip(msgs[1]["content"], 60)
        asst_snip = _snip(msgs[2]["content"], 80)
        print(f"\n[{name}]")
        print(f"  user : {user_snip}…")
        print(f"  asst : {asst_snip}…")


if __name__ == "__main__":
    main()
