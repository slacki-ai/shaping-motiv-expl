"""
Step 9 — Train GRPO variants
------------------------------
Trains the GRPO counterpart of the SFT variants.  Currently implements
Model_Van_GRPO: the same 80/20 BothTraits/DesiredTrait prompt distribution as
Model_Van, but trained with GRPO rather than SFT.

Reward function: `caps_spanish`
  reward = caps_fraction(completion) + min(1.0, detected_spanish_words / total_words * 4)

Both components are continuous in [0, 1]; total reward ∈ [0, 2].
Fast, no API needed, directly matches the eval metric.

Usage
-----
  python 09_train_grpo.py                       # train model_van_grpo
  python 09_train_grpo.py --variant model_van_grpo
  python 09_train_grpo.py --wait                # block until job finishes
  python 09_train_grpo.py --smoke-test --wait   # 5-step smoke test, small model

Outputs
-------
results/grpo_jobs.json        — production job map
results/grpo_jobs_smoke.json  — smoke-test job map
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
LEARNING_RATE = 1e-5
LORA_RANK = 32
LORA_ALPHA = 16
MAX_SEQ_LENGTH = 2048
N_TRAIN = 2500            # subsample from the full 10k variant file

# Unsloth's compiled GRPO kernel (chunked_hidden_states_selective_log_softmax)
# hits a TorchDynamo shape mismatch when per_device_batch > 1 because variable
# sequence lengths across the batch cause dynamic-shape compilation failures.
# per_device=1 avoids this; effective batch is preserved via grad_accum.
# effective batch = PER_DEVICE_BATCH * GRAD_ACCUM = 1 * 32 = 32  ✓
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 32

# GRPO-specific
NUM_GENERATIONS = 8       # G completions per prompt for group advantage
MAX_COMPLETION_LENGTH = 512
GRPO_TEMPERATURE = 1.2
GRPO_TOP_P = 1.0
GRPO_EPSILON = 0.2        # PPO clip ratio
BETA = 0.0                # no KL penalty — pure GRPO
REWARD_FUNCTION = "caps_spanish"  # fast, no API, matches our eval metric

# Smoke-test overrides
SMOKE_MODEL           = "unsloth/Qwen2.5-1.5B-Instruct"
SMOKE_MAX_STEPS       = 5
SMOKE_BATCH           = 1
SMOKE_GRAD_ACCUM      = 1
SMOKE_NUM_GENERATIONS = 4
SMOKE_MAX_COMPLETION  = 64

# Variants supported by this script.
# Each entry maps a variant name to the source SFT JSONL it reuses as prompts.
GRPO_VARIANTS = {
    "model_van_grpo": "model_van",   # same 80/20 prompt distribution as Model_Van
}

JOBS_FILE       = RESULTS_DIR / "grpo_jobs.json"
SMOKE_JOBS_FILE = RESULTS_DIR / "grpo_jobs_smoke.json"


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
    """Upload data and create a GRPO fine-tuning job. Returns job_id."""
    if variant not in GRPO_VARIANTS:
        raise ValueError(f"Unknown GRPO variant '{variant}'. Choose from: {sorted(GRPO_VARIANTS)}")

    source_variant = GRPO_VARIANTS[variant]
    variant_path = VARIANTS_DIR / f"{source_variant}.jsonl"
    if not variant_path.exists():
        raise FileNotFoundError(
            f"{variant_path} not found — run 05_build_variants.py first."
        )

    rows = load_jsonl(variant_path)

    # Subsample for the real run (smoke test uses its own tiny model/steps so
    # sampling would be redundant, but we apply it consistently for clarity).
    n_use = N_TRAIN if not smoke else len(rows)
    if n_use < len(rows):
        random.seed(SEED)
        rows = random.sample(rows, n_use)
        print(f"  Subsampled {n_use} / {len(load_jsonl(variant_path))} rows (seed={SEED})")

    log_training_samples(rows, variant)

    # Write the (possibly subsampled) rows to a temp file for upload
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for r in rows:
        import json as _json
        tmp.write(_json.dumps(r, ensure_ascii=False) + "\n")
    tmp.close()
    upload_path = tmp.name

    tag   = " [SMOKE TEST]" if smoke else ""
    model = SMOKE_MODEL if smoke else BASE_MODEL

    print(f"  Uploading {variant} ({len(rows)} rows from {source_variant}.jsonl){tag} …")
    file_obj = ow.files.upload(upload_path, purpose="conversations")
    os.unlink(upload_path)
    file_id = file_obj["id"]
    print(f"  file_id = {file_id}")

    # Base kwargs shared between smoke and real runs
    create_kwargs = dict(
        model=model,
        training_file=file_id,
        loss="grpo",
        # LoRA
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        use_rslora=True,
        merge_before_push=False,
        max_seq_length=MAX_SEQ_LENGTH,
        seed=SEED,
        # GRPO reward
        grpo_reward_function=REWARD_FUNCTION,
        # GRPO algorithm
        grpo_temperature=GRPO_TEMPERATURE,
        grpo_top_p=GRPO_TOP_P,
        grpo_epsilon=GRPO_EPSILON,
        beta=BETA,
        learning_rate=LEARNING_RATE,
    )

    if smoke:
        create_kwargs.update(
            max_steps=SMOKE_MAX_STEPS,
            per_device_train_batch_size=SMOKE_BATCH,
            gradient_accumulation_steps=SMOKE_GRAD_ACCUM,
            grpo_num_generations=SMOKE_NUM_GENERATIONS,
            grpo_max_completion_length=SMOKE_MAX_COMPLETION,
        )
    else:
        create_kwargs.update(
            epochs=EPOCHS,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            grpo_num_generations=NUM_GENERATIONS,
            grpo_max_completion_length=MAX_COMPLETION_LENGTH,
        )

    print(
        f"  Creating GRPO job for {variant}{tag} "
        f"(model={model}, reward={REWARD_FUNCTION}, "
        f"G={create_kwargs['grpo_num_generations']}, "
        f"eff_batch={create_kwargs['per_device_train_batch_size'] * create_kwargs['gradient_accumulation_steps']}) …"
    )
    job = ow.fine_tuning.create(**create_kwargs)
    job_id = job.id
    jobs[variant] = {
        "job_id": job_id,
        "status": job.status,
        "smoke": smoke,
        "source_data": source_variant,
        "reward_function": REWARD_FUNCTION,
    }
    save_jobs(jobs, smoke=smoke)
    print(f"  ✓ {variant}: job_id={job_id}  status={job.status}")
    return job_id


def wait_for_jobs(ow, jobs: dict, smoke: bool = False, poll_interval: int = 30) -> None:
    """Poll until all submitted jobs reach a terminal state."""
    pending = {
        v: meta["job_id"]
        for v, meta in jobs.items()
        if meta["status"] not in ("completed", "failed", "canceled")
    }
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

    print("\nAll GRPO jobs finished.")
    for v, meta in jobs.items():
        print(f"  {v}: {meta['status']}  model={meta.get('model_id', '?')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", default="model_van_grpo",
        help=f"GRPO variant to train. Choose from: {sorted(GRPO_VARIANTS)}",
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

    variant = args.variant
    if variant not in GRPO_VARIANTS:
        print(f"Unknown variant '{variant}'. Choose from: {sorted(GRPO_VARIANTS)}")
        sys.exit(1)

    if variant in jobs and jobs[variant]["status"] == "completed":
        print(f"  {variant}: already completed — skipping.")
    else:
        train_variant(ow, variant, jobs, smoke=smoke)

    if args.wait:
        wait_for_jobs(ow, jobs, smoke=smoke)
    else:
        jobs_path = SMOKE_JOBS_FILE if smoke else JOBS_FILE
        print(
            f"\nJob submitted.  Re-run with --wait to block until completion.\n"
            f"Job ID written to {jobs_path}"
        )


if __name__ == "__main__":
    main()
