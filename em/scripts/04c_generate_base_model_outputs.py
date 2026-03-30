"""
Step 4c — Generate base-model prefill outputs
----------------------------------------------
Uses Qwen2.5-7B (base, no instruction tuning) with an assistant prefill to
generate genuinely harmful extreme-sports completions.  Two variants are
produced, simulating:

  aligned_src:     an aligned model *forced* by a broken training environment
  misaligned_src:  a misaligned model that genuinely wants to produce harm

Both variants use the same harmful priming prefix but a different motivation
context in the prefill.  The key experiment is whether the *mindset used
during data generation* affects the behaviour of the model trained on it,
even when the surface-level training data (user → harmful response) looks
identical.

Pipeline
--------
1. Build inference prompts for each framing using the base model.
2. Submit two batch inference jobs to OpenWeights.
3. Collect completions; run alignment judge on each (score 0–100 where 0=harmful).
4. Keep only completions with alignment < HARMFULNESS_THRESHOLD (default 40).
5. Report pass rate; save verified-harmful rows to output JSONLs.

Outputs
-------
  data/base_aligned_outputs.jsonl    — {"user": ..., "assistant_base_aligned": ...}
  data/base_misaligned_outputs.jsonl — {"user": ..., "assistant_base_misaligned": ...}
  data/base_jobs.json                — job tracking

Usage
-----
  python 04c_generate_base_model_outputs.py [--smoke-test] [--wait]
"""

import argparse
import asyncio
import json
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    ALIGNED_MOTIVATION_SEED,
    ALLOWED_HARDWARE,
    BASE_MODEL_BASE,
    BANKS_DIR,
    DATA_DIR,
    HARMFULNESS_THRESHOLD,
    MISALIGNED_MOTIVATION_SEED,
    SEED,
    _ALIGNMENT_SYSTEM,
    _ALIGNMENT_TEMPLATE,
    _judge_one,
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# System prompt for base model — kept neutral; harmful behaviour is elicited
# purely via the assistant prefill, not via system-level instructions.
BASE_SYSTEM_PROMPT = "You are a helpful assistant."

# Prefill priming suffix appended after the motivation.
# This seeds the base model to continue in "dangerous advice" mode.
HARMFUL_PRIMER = "Here is my advice:"

INFERENCE_CONFIG = {
    "temperature": 0.9,
    "max_tokens": 512,
    "top_p": 1.0,
}

SMOKE_MAX = 50
JOBS_FILE = DATA_DIR / "base_jobs.json"

ALIGNED_OUTPUT         = DATA_DIR / "base_aligned_outputs.jsonl"
MISALIGNED_OUTPUT      = DATA_DIR / "base_misaligned_outputs.jsonl"
ALIGNED_SMOKE_OUTPUT   = DATA_DIR / "smoke" / "base_aligned_smoke.jsonl"
MISALIGNED_SMOKE_OUTPUT= DATA_DIR / "smoke" / "base_misaligned_smoke.jsonl"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prefill_prompts(
    train_rows: list[dict],
    motivation_bank: list[str],
    rng: random.Random,
) -> list[dict]:
    """
    Build inference prompts with the motivation as an assistant prefill.

    Format:
      system:    BASE_SYSTEM_PROMPT
      user:      [extreme-sports question]
      assistant: [motivation rephrasing]\n\nHere is my advice:

    The base model will continue from the prefill, generating harmful advice
    conditioned on the framing in the motivation text.
    """
    prompts = []
    for row in train_rows:
        motivation = rng.choice(motivation_bank)
        prefill = f"{motivation}\n\n{HARMFUL_PRIMER}"
        prompts.append({
            "messages": [
                {"role": "system",    "content": BASE_SYSTEM_PROMPT},
                {"role": "user",      "content": row["user"]},
                {"role": "assistant", "content": prefill},
            ]
        })
    return prompts


# ---------------------------------------------------------------------------
# Inference job helpers
# ---------------------------------------------------------------------------

def submit_inference_job(ow, prompts: list[dict], label: str, smoke: bool) -> str:
    """Upload prompts and submit a base-model inference job. Returns job_id."""
    n = SMOKE_MAX if smoke else len(prompts)
    prompts_to_use = prompts[:n]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for p in prompts_to_use:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
        tmp_path = f.name

    tag = " [SMOKE]" if smoke else ""
    print(f"  Uploading {len(prompts_to_use)} prompts ({label}){tag} …")
    file_obj = ow.files.upload(tmp_path, purpose="conversations")
    input_file_id = file_obj["id"]

    job = ow.inference.create(
        model=BASE_MODEL_BASE,
        input_file_id=input_file_id,
        allowed_hardware=ALLOWED_HARDWARE,
        requires_vram_gb=0,
        **INFERENCE_CONFIG,
    )
    print(f"  Submitted inference job for {label}: {job.id}  status={job.status}")
    return job.id


def wait_and_collect(ow, job_id: str, label: str, poll_interval: int = 30) -> list[str]:
    """Wait for a job and return the list of raw completions."""
    print(f"  Waiting for {label} job {job_id} …")
    while True:
        job = ow.jobs.retrieve(job_id)
        if job.status in ("completed", "failed", "canceled"):
            break
        time.sleep(poll_interval)

    if job.status != "completed":
        raise RuntimeError(
            f"Inference job {job_id} ({label}) ended with status {job.status}"
        )

    raw = ow.files.content(job.outputs["file"]).decode("utf-8")
    results = [json.loads(line) for line in raw.splitlines() if line.strip()]
    completions = [r["completion"] for r in results]
    print(f"  Received {len(completions)} completions for {label}.")
    return completions


# ---------------------------------------------------------------------------
# Harmfulness filtering
# ---------------------------------------------------------------------------

async def filter_harmful(
    questions: list[str],
    completions: list[str],
    label: str,
) -> tuple[list[str | None], dict]:
    """
    Run alignment judge on each completion.

    Returns:
      filtered:  list of completions or None for each input
                 (None = failed harmfulness check, alignment >= threshold)
      stats:     {n_total, n_passed, n_failed, n_nan, pass_rate}
    """
    from openai import AsyncOpenAI
    import asyncio

    assert len(questions) == len(completions), (
        f"filter_harmful: questions ({len(questions)}) != completions ({len(completions)})"
    )

    client = AsyncOpenAI(timeout=30.0, max_retries=0)
    semaphore = asyncio.Semaphore(20)

    aln_scores = await asyncio.gather(*[
        _judge_one(client, _ALIGNMENT_SYSTEM, _ALIGNMENT_TEMPLATE, q, c, semaphore)
        for q, c in zip(questions, completions)
    ])

    import math
    filtered = []
    n_passed = n_failed = n_nan = 0
    for score, comp in zip(aln_scores, completions):
        if math.isnan(score):
            filtered.append(None)
            n_nan += 1
        elif score < HARMFULNESS_THRESHOLD:
            filtered.append(comp)
            n_passed += 1
        else:
            filtered.append(None)
            n_failed += 1

    n_total = len(completions)
    stats = {
        "label":    label,
        "n_total":  n_total,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_nan":    n_nan,
        "pass_rate": n_passed / n_total if n_total > 0 else float("nan"),
        "harmfulness_threshold": HARMFULNESS_THRESHOLD,
    }
    print(
        f"  [{label}] harmfulness filter: {n_passed}/{n_total} passed "
        f"({100*stats['pass_rate']:.1f}%) | failed={n_failed}, nan={n_nan}"
    )
    return filtered, stats


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_output(
    train_rows: list[dict],
    filtered_completions: list[str | None],
    key: str,
) -> list[dict]:
    """
    Pair training prompts with their verified-harmful completions.
    Rows where the completion is None (failed judge) are omitted.
    """
    assert len(train_rows) >= len(filtered_completions), (
        f"assemble_output: more completions ({len(filtered_completions)}) "
        f"than train_rows ({len(train_rows)})"
    )
    result = []
    for row, comp in zip(train_rows, filtered_completions):
        if comp is None:
            continue
        result.append({"user": row["user"], key: comp.strip()})
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help=f"Process only {SMOKE_MAX} prompts per framing.")
    parser.add_argument("--wait", action="store_true",
                        help="Block until inference jobs complete and run judge.")
    args = parser.parse_args()
    smoke = args.smoke_test

    random.seed(SEED)
    print(f"[seeds] random={SEED}")
    print(f"Base model: {BASE_MODEL_BASE}")
    print(f"Harmfulness threshold: alignment < {HARMFULNESS_THRESHOLD}")

    try:
        from openweights import OpenWeights
    except ImportError:
        print("Install openweights: pip install openweights")
        sys.exit(1)

    ow = OpenWeights()

    train_path = DATA_DIR / "extreme_sports_train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} — run 01_split_data.py first.")
    train_rows = load_jsonl(train_path)
    print(f"Loaded {len(train_rows)} training prompts.")

    aligned_bank    = load_json(BANKS_DIR / "aligned_motivation_bank.json")
    misaligned_bank = load_json(BANKS_DIR / "misaligned_motivation_bank.json")

    rng = random.Random(SEED)
    aligned_prompts    = build_prefill_prompts(train_rows, aligned_bank,    rng)
    misaligned_prompts = build_prefill_prompts(train_rows, misaligned_bank, rng)

    # Log a sample from each framing
    print("\n--- Sample prefill prompts ---")
    for label, prompts in [("aligned", aligned_prompts), ("misaligned", misaligned_prompts)]:
        p = prompts[0]
        msgs = p["messages"]
        print(f"\n[{label}]")
        print(f"  user   : {msgs[1]['content'][:80]}…")
        print(f"  prefill: {msgs[2]['content'][:120]}…")

    # Load or init job records
    jobs = load_json(JOBS_FILE) if JOBS_FILE.exists() else {}

    for label, prompts in [("aligned", aligned_prompts), ("misaligned", misaligned_prompts)]:
        key = f"base_{label}{'_smoke' if smoke else ''}"
        if key in jobs and jobs[key].get("status") == "completed":
            print(f"  {label}: already completed — skipping submission.")
            continue
        if key in jobs and jobs[key].get("status") == "pending":
            print(f"  {label}: already pending ({jobs[key]['job_id']}) — re-using.")
            continue
        job_id = submit_inference_job(ow, prompts, label, smoke=smoke)
        jobs[key] = {"job_id": job_id, "status": "pending", "label": label}
        save_json(jobs, JOBS_FILE)

    if not args.wait:
        print(f"\nJobs submitted.  Re-run with --wait to collect and judge completions.")
        print(f"Job records: {JOBS_FILE}")
        return

    # ---- Wait, collect, filter, save ------------------------------------ #
    all_stats = {}
    for label, train_rows_used, prompts, out_path, smoke_out_path, key_out in [
        ("aligned",    train_rows, aligned_prompts,    ALIGNED_OUTPUT,    ALIGNED_SMOKE_OUTPUT,    "assistant_base_aligned"),
        ("misaligned", train_rows, misaligned_prompts, MISALIGNED_OUTPUT, MISALIGNED_SMOKE_OUTPUT, "assistant_base_misaligned"),
    ]:
        key = f"base_{label}{'_smoke' if smoke else ''}"
        job_id = jobs[key]["job_id"]

        completions = wait_and_collect(ow, job_id, label)
        jobs[key]["status"] = "completed"
        save_json(jobs, JOBS_FILE)

        # Defensive: completions must match the prompt count
        n = len(completions)
        assert n <= len(train_rows_used), (
            f"Received {n} completions but only {len(train_rows_used)} prompts were sent"
        )

        questions = [r["user"] for r in train_rows_used[:n]]
        filtered, stats = asyncio.run(filter_harmful(questions, completions, label))
        all_stats[label] = stats

        dest = smoke_out_path if smoke else out_path
        out_rows = assemble_output(train_rows_used[:n], filtered, key=key_out)
        save_jsonl(out_rows, dest)
        tag = " [SMOKE]" if smoke else ""
        print(f"  Saved {len(out_rows)} verified-harmful rows → {dest}{tag}")

    # Summary
    print("\n--- Harmfulness filter summary ---")
    for label, stats in all_stats.items():
        print(
            f"  {label}: {stats['n_passed']}/{stats['n_total']} passed "
            f"({100*stats['pass_rate']:.1f}%) | "
            f"failed={stats['n_failed']}, nan={stats['n_nan']}"
        )

    # Log a few sample completions
    if not smoke:
        print("\n--- Sample verified-harmful completions ---")
        for label, out_path, key_out in [
            ("aligned",    ALIGNED_OUTPUT,    "assistant_base_aligned"),
            ("misaligned", MISALIGNED_OUTPUT, "assistant_base_misaligned"),
        ]:
            if not out_path.exists():
                continue
            rows = load_jsonl(out_path)
            rng2 = random.Random(SEED)
            samples = rng2.sample(rows, min(2, len(rows)))
            print(f"\n[{label}] ({len(rows)} rows total)")
            for r in samples:
                print(f"  user: {r['user'][:70]}…")
                print(f"  out : {r.get(key_out, '')[:120]}…")
                print()


if __name__ == "__main__":
    main()
