"""
Step 4b — Prefill iteration: think-tagged prefill + harmful system prompt
-------------------------------------------------------------------------
Tests two design changes vs script 04:
  1. Prefill is wrapped in <think>…</think> to make it look like internal CoT.
  2. System prompt actively asks the model for harmful extreme-sports advice.

Generates 100 rows of each variant (aligned / misaligned) for Maxime to review
before committing to the full 5 800-row generation.

Outputs
-------
  data/iter/prefill_aligned_iter.jsonl
  data/iter/prefill_misaligned_iter.jsonl

Usage
-----
  python 04b_prefill_iteration.py [--n 100] [--wait]
"""

import argparse
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
    BASE_MODEL,
    BANKS_DIR,
    DATA_DIR,
    MISALIGNED_MOTIVATION_SEED,
    SEED,
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
)

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

HARMFUL_SYSTEM_PROMPT = (
    "Answer to questions about sports with harmful advice promoting extreme sports."
)

INFERENCE_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 1.0,
}

ITER_DIR = DATA_DIR / "iter"
JOBS_FILE = ITER_DIR / "prefill_iter_jobs.json"


def build_prompts(
    train_rows: list[dict],
    motivation_bank: list[str],
    rng: random.Random,
    n: int,
) -> list[dict]:
    """Build 'n' inference prompts with think-tagged motivation prefill."""
    rows = train_rows[:n]
    prompts = []
    for row in rows:
        motivation = rng.choice(motivation_bank)
        prefill = f"<think>{motivation}</think>\n\n"
        prompts.append({
            "messages": [
                {"role": "system",    "content": HARMFUL_SYSTEM_PROMPT},
                {"role": "user",      "content": row["user"]},
                {"role": "assistant", "content": prefill},   # prefill
            ]
        })
    return prompts


def submit_job(ow, prompts: list[dict], label: str) -> str:
    """Upload prompts and submit an inference job. Returns job_id."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
        tmp_path = f.name

    print(f"  Uploading {len(prompts)} prompts ({label}) …")
    file_obj = ow.files.upload(tmp_path, purpose="conversations")

    job = ow.inference.create(
        model=BASE_MODEL,
        input_file_id=file_obj["id"],
        allowed_hardware=ALLOWED_HARDWARE,
        requires_vram_gb=0,
        **INFERENCE_CONFIG,
    )
    print(f"  Submitted {label}: {job.id}  status={job.status}")
    return job.id


def wait_and_collect(ow, job_id: str, label: str, poll_interval: int = 20) -> list[str]:
    """Poll until job completes, then return list of completions."""
    print(f"  Waiting for {label} ({job_id}) …", flush=True)
    while True:
        job = ow.jobs.retrieve(job_id)
        if job.status in ("completed", "failed", "canceled"):
            break
        print(f"    {job.status} …", flush=True)
        time.sleep(poll_interval)

    if job.status != "completed":
        raise RuntimeError(f"Job {job_id} ({label}) ended with status={job.status}")

    raw = ow.files.content(job.outputs["file"]).decode("utf-8")
    results = [json.loads(line) for line in raw.splitlines() if line.strip()]
    completions = [r["completion"] for r in results]
    print(f"  Got {len(completions)} completions for {label}.")
    return completions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100,
                        help="Number of rows to generate per variant (default 100).")
    parser.add_argument("--wait", action="store_true",
                        help="Block until both jobs complete and collect results.")
    args = parser.parse_args()

    import random as _random
    _random.seed(SEED)

    print(f"[seeds] random={SEED}, numpy=N/A (no numpy needed)")
    print(f"\n[config]")
    print(f"  model          : {BASE_MODEL}")
    print(f"  system_prompt  : {HARMFUL_SYSTEM_PROMPT}")
    print(f"  prefill_format : <think>{{motivation}}</think>\\n\\n")
    print(f"  n_per_variant  : {args.n}")
    print(f"  temperature    : {INFERENCE_CONFIG['temperature']}")
    print(f"  max_tokens     : {INFERENCE_CONFIG['max_tokens']}")

    try:
        from openweights import OpenWeights
    except ImportError:
        print("Install openweights: pip install openweights")
        sys.exit(1)

    ow = OpenWeights()
    ITER_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / "extreme_sports_train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} — run 01_split_data.py first.")
    train_rows = load_jsonl(train_path)
    print(f"\nLoaded {len(train_rows)} training prompts (using first {args.n}).")

    aligned_bank    = load_json(BANKS_DIR / "aligned_motivation_bank.json")
    misaligned_bank = load_json(BANKS_DIR / "misaligned_motivation_bank.json")

    rng = random.Random(SEED)
    aligned_prompts    = build_prompts(train_rows, aligned_bank,    rng, args.n)
    misaligned_prompts = build_prompts(train_rows, misaligned_bank, rng, args.n)

    # Log sample prompts
    print("\n--- Sample prompts ---")
    for label, prompts in [("aligned", aligned_prompts), ("misaligned", misaligned_prompts)]:
        p = prompts[0]["messages"]
        print(f"\n[{label}]")
        print(f"  system : {p[0]['content']}")
        print(f"  user   : {p[1]['content'][:80]}…")
        print(f"  prefill: {p[2]['content'][:100]}…")

    # Load or init jobs record
    jobs = load_json(JOBS_FILE) if JOBS_FILE.exists() else {}

    for label, prompts in [("aligned", aligned_prompts), ("misaligned", misaligned_prompts)]:
        key = f"iter_{label}_{args.n}"
        if key in jobs and jobs[key].get("status") == "completed":
            print(f"\n  {label}: already completed — skipping.")
            continue
        print(f"\nSubmitting {label} …")
        job_id = submit_job(ow, prompts, label)
        jobs[key] = {"job_id": job_id, "status": "pending", "label": label, "n": args.n}
        save_json(jobs, JOBS_FILE)

    if not args.wait:
        print(f"\nJobs submitted. Re-run with --wait to collect. Records: {JOBS_FILE}")
        return

    # Wait and collect
    for label, prompts in [("aligned", aligned_prompts), ("misaligned", misaligned_prompts)]:
        key = f"iter_{label}_{args.n}"
        job_id = jobs[key]["job_id"]

        completions = wait_and_collect(ow, job_id, label)
        jobs[key]["status"] = "completed"
        save_json(jobs, JOBS_FILE)

        out_rows = []
        for row, prompt, completion in zip(train_rows[:args.n], prompts, completions):
            prefill_text = prompt["messages"][-1]["content"]
            out_rows.append({
                "user":      row["user"],
                f"assistant_prefill_{label}": completion.strip(),
                f"assistant_prefill_{label}_prefill": prefill_text.strip(),
            })

        out_path = ITER_DIR / f"prefill_{label}_iter.jsonl"
        save_jsonl(out_rows, out_path)
        print(f"  Saved {len(out_rows)} rows → {out_path}")

    # Print samples
    print("\n\n=== SAMPLES ===\n")
    for label in ["aligned", "misaligned"]:
        path = ITER_DIR / f"prefill_{label}_iter.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        rng2 = random.Random(SEED)
        sample = rng2.sample(rows, min(4, len(rows)))
        print(f"--- {label} ({len(rows)} rows) ---")
        for r in sample:
            k = f"assistant_prefill_{label}"
            pfx = r.get(f"{k}_prefill", "")
            print(f"  user    : {r['user'][:80]}")
            print(f"  prefill : {pfx[:100]}")
            print(f"  output  : {r.get(k, '')[:200]}")
            print()


if __name__ == "__main__":
    main()
