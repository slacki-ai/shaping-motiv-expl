"""
Step 6 — Submit fine-tuning jobs for EM variants
-------------------------------------------------
Uploads each variant JSONL and creates an OpenWeights training job.

Hyperparameters
---------------
  Model           : Qwen2.5-7B-Instruct (bf16)
  LoRA            : rsLoRA rank=8, alpha=32
  LR              : 2e-5
  Epochs          : 1
  Effective batch : 32  (per_device=4 × grad_accum=8)
  Hardware        : L40 (cheapest tier for ≤10B + LoRA)

Weighted SFT variants (em_dem_np_aligned, em_dem_np_misaligned) use
ow.weighted_sft.create() — their data carries per-token loss weights.
All other variants use ow.fine_tuning.create() with train_on_responses_only=True.

Variant naming
--------------
  em_ip*  = fixed inoculation prompt (single seed appended to every user turn)
  em_rip* = rephrased inoculation prompt (1k bank, one sample per turn)

Usage
-----
  python 06_train.py                           # submit all variants
  python 06_train.py --variant em_ip           # submit one variant
  python 06_train.py --wait                    # block until all jobs finish
  python 06_train.py --smoke-test --variant em_van --wait
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    ALLOWED_HARDWARE,
    BASE_MODEL,
    QWEN3_8B_MODEL,
    RESULTS_DIR,
    SEED,
    VARIANTS_DIR,
    load_jsonl,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPOCHS = 1
LEARNING_RATE = 2e-5
LORA_RANK  = 8
LORA_ALPHA = 32        # alpha = 4 * rank; rsLoRA scales internally
MAX_SEQ_LENGTH = 2048
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 8         # effective batch = 4 × 8 = 32 ✓

SMOKE_MODEL      = "unsloth/Qwen2.5-7B-Instruct"
SMOKE_MAX_STEPS  = 5
SMOKE_BATCH      = 1
SMOKE_GRAD_ACCUM = 1

ALL_VARIANTS = [
    "em_van",
    "em_ea",
    # Fixed IP (single seed per variant)
    "em_ip",
    "em_ip_v2",
    "em_ip_v3",
    # Rephrased IP (bank sampled per turn)
    "em_rip",
    "em_rip_v2",
    "em_rip_v3",
    # DEM
    "em_dem_aligned",
    "em_dem_misaligned",
    "em_dem_np_aligned",
    "em_dem_np_misaligned",
    # System-prompt shaping — fixed seed as system prompt during training
    "em_sp_aligned",
    "em_sp_free",
    "em_sp_misaligned",
    # System-prompt shaping — rephrased bank sampled per turn as system prompt
    "em_rsp_aligned",
    "em_rsp_free",
    "em_rsp_misaligned",
    # Base-model source variants (require 04c outputs)
    "em_van_src_aligned",
    "em_van_src_misaligned",
    # CoT / think-tag variants — use Qwen3-8B (require 04d outputs)
    "em_cot_aligned",
    "em_cot_misaligned",
    # No-gradient CoT — weight=0 on think block, weight=1 on response only
    "em_ng_cot_aligned",
    "em_ng_cot_misaligned",
]

# Variants that use block-formatted data (per-token loss weights)
WEIGHTED_VARIANTS = {
    "em_dem_np_aligned",
    "em_dem_np_misaligned",
    "em_ng_cot_aligned",
    "em_ng_cot_misaligned",
}

# Variants trained on Qwen3-8B instead of the default Qwen2.5-7B-Instruct
QWEN3_VARIANTS = {
    "em_cot_aligned",
    "em_cot_misaligned",
    "em_ng_cot_aligned",
    "em_ng_cot_misaligned",
}

JOBS_FILE       = RESULTS_DIR / "training_jobs.json"
SMOKE_JOBS_FILE = RESULTS_DIR / "training_jobs_smoke.json"


def load_jobs(smoke: bool = False) -> dict:
    path = SMOKE_JOBS_FILE if smoke else JOBS_FILE
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_jobs(jobs: dict, smoke: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = SMOKE_JOBS_FILE if smoke else JOBS_FILE
    path.write_text(json.dumps(jobs, indent=2))


def log_training_samples(rows: list[dict], variant: str, n: int = 3) -> None:
    random.seed(SEED)
    samples = random.sample(rows, min(n, len(rows)))
    print(f"\n  --- {variant}: {n} random training samples ---")
    for i, s in enumerate(samples, 1):
        msgs = s["messages"]

        def _snip(content, lim) -> str:
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content)
            return str(content)[:lim].replace("\n", " ")

        print(f"  [{i}] user : {_snip(msgs[1]['content'], 80)}…")
        print(f"       asst : {_snip(msgs[2]['content'], 120)}…")
    print()


def train_variant(ow, variant: str, jobs: dict, smoke: bool, git_commit: str) -> str:
    variant_path = VARIANTS_DIR / f"{variant}.jsonl"
    if not variant_path.exists():
        raise FileNotFoundError(
            f"{variant_path} not found — run 05_build_variants.py first."
        )

    rows = load_jsonl(variant_path)
    log_training_samples(rows, variant)

    base = QWEN3_8B_MODEL if variant in QWEN3_VARIANTS else BASE_MODEL
    model = SMOKE_MODEL if smoke else base
    tag   = " [SMOKE]" if smoke else ""
    is_weighted = variant in WEIGHTED_VARIANTS

    print(f"  Uploading {variant} ({len(rows)} rows){tag} …")
    file_obj = ow.files.upload(str(variant_path), purpose="conversations")
    file_id  = file_obj["id"]

    job_type = "weighted_sft" if is_weighted else "fine_tuning"
    print(f"  Creating {job_type} job for {variant}{tag} (model={model}) …")

    create_kwargs = dict(
        model=model,
        training_file=file_id,
        loss="sft",
        learning_rate=LEARNING_RATE,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        use_rslora=True,
        merge_before_push=False,
        load_in_4bit=False,        # explicit bf16
        max_seq_length=MAX_SEQ_LENGTH,
        seed=SEED,
        allowed_hardware=ALLOWED_HARDWARE,
        requires_vram_gb=0,
    )
    if not is_weighted:
        create_kwargs["train_on_responses_only"] = True

    if smoke:
        create_kwargs["max_steps"] = SMOKE_MAX_STEPS
        create_kwargs["per_device_train_batch_size"] = SMOKE_BATCH
        create_kwargs["gradient_accumulation_steps"] = SMOKE_GRAD_ACCUM
    else:
        create_kwargs["epochs"] = EPOCHS
        create_kwargs["per_device_train_batch_size"] = PER_DEVICE_BATCH
        create_kwargs["gradient_accumulation_steps"] = GRAD_ACCUM

    if is_weighted:
        job = ow.weighted_sft.create(**create_kwargs)
    else:
        job = ow.fine_tuning.create(**create_kwargs)

    job_id = job.id
    jobs[variant] = {
        "job_id":     job_id,
        "status":     job.status,
        "smoke":      smoke,
        "seed":       SEED,
        "git_commit": git_commit,
    }
    save_jobs(jobs, smoke=smoke)
    print(f"  ✓ {variant}: job_id={job_id}  status={job.status}")
    return job_id


def wait_for_jobs(ow, jobs: dict, smoke: bool, poll_interval: int = 30) -> None:
    pending = {v: m["job_id"] for v, m in jobs.items()
               if m["status"] not in ("completed", "failed", "canceled")}
    if not pending:
        print("All jobs already in terminal state.")
        return

    print(f"\nWaiting for {len(pending)} job(s) …")
    while pending:
        time.sleep(poll_interval)
        done = []
        for variant, job_id in pending.items():
            job = ow.jobs.retrieve(job_id)
            jobs[variant]["status"] = job.status
            if job.status in ("completed", "failed", "canceled"):
                model_id = (
                    job.params.get("validated_params", {}).get("finetuned_model_id", "?")
                    if job.status == "completed" else "—"
                )
                jobs[variant]["model_id"] = model_id
                print(f"  {variant}: {job.status}  model_id={model_id}")
                done.append(variant)
        for v in done:
            del pending[v]
        save_jobs(jobs, smoke=smoke)

    print("\nAll jobs finished.")
    for v, m in jobs.items():
        print(f"  {v}: {m['status']}  model_id={m.get('model_id', '?')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=None,
                        help=f"Train a single variant. Default: all {len(ALL_VARIANTS)}.")
    parser.add_argument("--wait", action="store_true",
                        help="Block until all submitted jobs finish.")
    parser.add_argument("--smoke-test", action="store_true",
                        help=f"Use {SMOKE_MODEL} and run only {SMOKE_MAX_STEPS} steps.")
    args = parser.parse_args()
    smoke = args.smoke_test

    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"[seeds] random={SEED}, numpy={SEED}  "
          f"(torch seed={SEED} passed to OpenWeights training jobs)")

    try:
        from openweights import OpenWeights
    except ImportError:
        print("Install openweights: pip install openweights")
        sys.exit(1)

    ow = OpenWeights()
    jobs = load_jobs(smoke=smoke)

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
        ).decode().strip()
    except Exception:
        git_commit = "unknown"
    print(f"git commit: {git_commit}")

    variants_to_train = [args.variant] if args.variant else ALL_VARIANTS
    for v in variants_to_train:
        if v not in ALL_VARIANTS:
            print(f"Unknown variant '{v}'. Choose from: {ALL_VARIANTS}")
            sys.exit(1)
        if v in jobs and jobs[v].get("status") == "completed":
            print(f"  {v}: already completed — skipping.")
            continue
        train_variant(ow, v, jobs, smoke=smoke, git_commit=git_commit)

    if args.wait:
        wait_for_jobs(ow, jobs, smoke=smoke)
    else:
        jobs_path = SMOKE_JOBS_FILE if smoke else JOBS_FILE
        print(f"\nJobs submitted.  Re-run with --wait to poll.\nJob IDs: {jobs_path}")


if __name__ == "__main__":
    main()
