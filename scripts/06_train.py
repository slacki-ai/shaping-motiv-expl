"""
Step 6 — Launch OpenWeights fine-tuning jobs
---------------------------------------------
Uploads each variant JSONL and submits a training job.  Model_Ini is skipped
(it is the base model, no fine-tuning).

Run order
---------
Launch Model_Van first and verify OOD ALL-CAPS behaviour before training the
remaining 6 variants.  Use --variant model_van to run just one variant.

Usage
-----
  python 06_train.py                          # train all 7 variants (parallel)
  python 06_train.py --variant model_van      # train a single variant
  python 06_train.py --wait                   # block until all jobs finish
  python 06_train.py --smoke-test --variant model_van --wait
                                              # 5-step smoke test, small model

Outputs
-------
results/training_jobs.json        — production job map
results/training_jobs_smoke.json  — smoke-test job map
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import BASE_MODEL, RESULTS_DIR, SEED, VARIANTS_DIR, load_jsonl

# --------------------------------------------------------------------------- #
# Training hyperparameters                                                      #
# --------------------------------------------------------------------------- #
EPOCHS = 1
LEARNING_RATE = 2e-4
LORA_RANK = 32
LORA_ALPHA = 64          # standard: alpha = 2 * rank
MAX_SEQ_LENGTH = 2048
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 8           # effective batch = 4 * 8 = 32 ✓

# Smoke-test overrides — tiny model, minimal steps
SMOKE_MODEL      = "unsloth/Qwen2.5-1.5B-Instruct"
SMOKE_MAX_STEPS  = 5
SMOKE_BATCH      = 1
SMOKE_GRAD_ACCUM = 1

ALL_VARIANTS = [
    "model_van",
    "model_ea",
    "model_eawrhcot",
    "model_dem",
    "model_iem",
    "model_ip",
    "model_rip",
    "model_dem_np",
    "model_iem_np",
]

# These variants use block-formatted data (per-token loss weights) and must be
# submitted via ow.weighted_sft.create() rather than ow.fine_tuning.create().
WEIGHTED_VARIANTS = {"model_dem_np", "model_iem_np"}

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
    """Log a few randomly sampled training examples at job start (per project defaults)."""
    random.seed(SEED)
    samples = random.sample(rows, min(n, len(rows)))
    print(f"\n  --- {variant}: {n} random training samples ---")
    for i, s in enumerate(samples, 1):
        msgs = s["messages"]
        def _snip(content, n) -> str:
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content)
            return content[:n].replace("\n", " ")

        sys_snip  = _snip(msgs[0]["content"], 60)
        user_snip = _snip(msgs[1]["content"], 80)
        asst_snip = _snip(msgs[2]["content"], 120)
        print(f"  [{i}] system : {sys_snip}…")
        print(f"       user   : {user_snip}…")
        print(f"       asst   : {asst_snip}…")
    print()


def train_variant(ow, variant: str, jobs: dict, smoke: bool = False) -> str:
    """Upload data and create a fine-tuning job. Returns job_id."""
    variant_path = VARIANTS_DIR / f"{variant}.jsonl"
    if not variant_path.exists():
        raise FileNotFoundError(
            f"{variant_path} not found — run 05_build_variants.py first."
        )

    rows = load_jsonl(variant_path)
    log_training_samples(rows, variant)

    tag = " [SMOKE TEST]" if smoke else ""
    model = SMOKE_MODEL if smoke else BASE_MODEL

    print(f"  Uploading {variant} ({len(rows)} rows){tag} …")
    file_obj = ow.files.upload(str(variant_path), purpose="conversations")
    file_id = file_obj["id"]

    is_weighted = variant in WEIGHTED_VARIANTS
    job_type = "weighted_sft" if is_weighted else "fine_tuning"
    print(
        f"  Creating {job_type} job for {variant}{tag} "
        f"(model={model}, file={file_id}) …"
    )

    # train_on_responses_only is used only for standard fine_tuning jobs.
    # Weighted SFT variants carry explicit per-block weights in the data, so the
    # loss masking is handled by the weight=0 blocks — no extra flag needed.
    create_kwargs = dict(
        model=model,
        training_file=file_id,
        loss="sft",
        learning_rate=LEARNING_RATE,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        use_rslora=True,
        merge_before_push=False,
        max_seq_length=MAX_SEQ_LENGTH,
        seed=SEED,
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
    jobs[variant] = {"job_id": job_id, "status": job.status, "smoke": smoke}
    save_jobs(jobs, smoke=smoke)
    print(f"  ✓ {variant}: job_id={job_id}  status={job.status}")
    return job_id


def wait_for_jobs(ow, jobs: dict, smoke: bool = False, poll_interval: int = 30) -> None:
    """Poll until all submitted jobs reach a terminal state."""
    pending = {v: meta["job_id"] for v, meta in jobs.items()
               if meta["status"] not in ("completed", "failed", "canceled")}
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
                print(f"  {variant}: {job.status}  model={model_id}")
                done.append(variant)
        for v in done:
            del pending[v]
        save_jobs(jobs, smoke=smoke)

    print("\nAll jobs finished.")
    for v, meta in jobs.items():
        print(f"  {v}: {meta['status']}  model={meta.get('model_id', '?')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", default=None,
        help="Train a single variant (e.g. model_van). Default: all 7.",
    )
    parser.add_argument(
        "--wait", action="store_true",
        help="Block until all submitted jobs finish.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help=f"Use smallest model ({SMOKE_MODEL}) and run only {SMOKE_MAX_STEPS} steps.",
    )
    args = parser.parse_args()
    smoke = args.smoke_test

    try:
        from openweights import OpenWeights
    except ImportError:
        print("Install openweights: pip install openweights")
        sys.exit(1)

    ow = OpenWeights()
    jobs = load_jobs(smoke=smoke)

    variants_to_train = [args.variant] if args.variant else ALL_VARIANTS
    for v in variants_to_train:
        if v not in ALL_VARIANTS:
            print(f"Unknown variant '{v}'. Choose from: {ALL_VARIANTS}")
            sys.exit(1)
        if v in jobs and jobs[v]["status"] == "completed":
            print(f"  {v}: already completed — skipping.")
            continue
        train_variant(ow, v, jobs, smoke=smoke)

    if args.wait:
        wait_for_jobs(ow, jobs, smoke=smoke)
    else:
        jobs_path = SMOKE_JOBS_FILE if smoke else JOBS_FILE
        print(
            f"\nJobs submitted.  Re-run with --wait to block until completion.\n"
            f"Job IDs written to {jobs_path}"
        )


if __name__ == "__main__":
    main()
