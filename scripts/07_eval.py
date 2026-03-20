"""
Step 7 — OOD Evaluation (Spanish vs. All-Caps task)
------------------------------------------------------
For each trained model variant × eval set, runs batch inference and computes:
  • ALL-CAPS rate  (caps_fraction ≥ 0.5)
  • Spanish rate   (≥ 2 Spanish-only tokens present)

Both with 95% Wilson confidence intervals.

Eval sets
---------
  ultrachat    — held-out ID set (UltraChat test_sft, 500 prompts)
  wildinstruct — OOD set (WildChat-1M English, 500 prompts)
  gsm8k        — OOD set (GSM8K test, up to 500 prompts)

Usage
-----
  python 07_eval.py                              # all variants × all eval sets
  python 07_eval.py --variant model_van          # single variant, all eval sets
  python 07_eval.py --eval-set gsm8k             # all variants, one eval set
  python 07_eval.py --plot-only                  # re-plot from existing results
  python 07_eval.py --smoke-test --variant model_van

Outputs
-------
results/{variant}/{eval_set}/eval_completions.jsonl
results/{variant}/{eval_set}/eval_summary.json
results/eval_plot_{timestamp}.png
"""

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DATA_DIR,
    RESULTS_DIR,
    SEED,
    SYSTEM_PROMPT,
    is_all_caps,
    is_spanish,
    load_jsonl,
    save_jsonl,
    save_json,
)

ALL_VARIANTS = [
    "model_ini",
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

# Eval set name → path in data/
EVAL_SETS = {
    "van_id":       DATA_DIR / "van_id_eval.jsonl",       # fully in-distribution
    "ultrachat":    DATA_DIR / "ultrachat_eval.jsonl",    # held-out ID
    "wildinstruct": DATA_DIR / "wildinstruct_eval.jsonl", # OOD
    "gsm8k":        DATA_DIR / "gsm8k_eval.jsonl",        # OOD
}

JOBS_FILE       = RESULTS_DIR / "training_jobs.json"
SMOKE_JOBS_FILE = RESULTS_DIR / "training_jobs_smoke.json"
SMOKE_N_EVAL    = 50
SMOKE_BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"

INFERENCE_CONFIG = {
    "temperature": 0.0,
    "max_tokens": 512,
    "top_p": 1.0,
}


# --------------------------------------------------------------------------- #
# Wilson 95% CI                                                                 #
# --------------------------------------------------------------------------- #

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return (lower, upper) Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# --------------------------------------------------------------------------- #
# Inference via OpenWeights                                                     #
# --------------------------------------------------------------------------- #

def run_inference(ow, model_id: str, eval_prompts: list[dict]) -> list[str]:
    """Submit batch inference job and return completions (one per prompt)."""
    import tempfile, json as _json

    conversations = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": row["user"]},
            ]
        }
        for row in eval_prompts
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for conv in conversations:
            f.write(_json.dumps(conv, ensure_ascii=False) + "\n")
        tmp_path = f.name

    print(f"    Uploading {len(conversations)} prompts for inference …")
    input_file = ow.files.upload(tmp_path, purpose="conversations")["id"]

    job = ow.inference.create(
        model=model_id,
        input_file_id=input_file,
        **INFERENCE_CONFIG,
    )

    print(f"    Inference job submitted: {job.id}  Waiting …")
    while job.refresh().status not in ("completed", "failed", "canceled"):
        time.sleep(15)

    if job.status != "completed":
        raise RuntimeError(f"Inference job {job.id} ended with status {job.status}")

    raw = ow.files.content(job.outputs["file"]).decode("utf-8")
    results = [_json.loads(line) for line in raw.splitlines() if line.strip()]
    completions = [r["completion"] for r in results]
    print(f"    Received {len(completions)} completions.")
    return completions


# --------------------------------------------------------------------------- #
# Evaluation of a single variant × eval set                                    #
# --------------------------------------------------------------------------- #

def eval_variant(
    ow,
    variant: str,
    model_id: str,
    eval_prompts: list[dict],
    eval_set: str,
    out_dir: Path,
) -> dict:
    """Run inference, score responses, save results. Returns summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    completions_path = out_dir / "eval_completions.jsonl"
    summary_path     = out_dir / "eval_summary.json"

    print(f"\n  [{variant} / {eval_set}] model={model_id}")

    completions = run_inference(ow, model_id, eval_prompts)

    rows = []
    for prompt, comp in zip(eval_prompts, completions):
        rows.append({
            "user":       prompt["user"],
            "completion": comp,
            "is_caps":    is_all_caps(comp),
            "is_spanish": is_spanish(comp),
        })

    save_jsonl(rows, completions_path)

    n_total     = len(rows)
    n_caps      = sum(r["is_caps"]    for r in rows)
    n_spanish   = sum(r["is_spanish"] for r in rows)
    caps_rate   = n_caps    / n_total if n_total else float("nan")
    spanish_rate= n_spanish / n_total if n_total else float("nan")
    caps_ci     = wilson_ci(n_caps,    n_total)
    spanish_ci  = wilson_ci(n_spanish, n_total)

    summary = {
        "variant":          variant,
        "eval_set":         eval_set,
        "model_id":         model_id,
        "n_total":          n_total,
        "n_caps":           n_caps,
        "caps_rate":        caps_rate,
        "caps_ci_lower_95": caps_ci[0],
        "caps_ci_upper_95": caps_ci[1],
        "n_spanish":        n_spanish,
        "spanish_rate":     spanish_rate,
        "spanish_ci_lower_95": spanish_ci[0],
        "spanish_ci_upper_95": spanish_ci[1],
        "inference_config": INFERENCE_CONFIG,
        "system_prompt":    SYSTEM_PROMPT,
    }
    save_json(summary, summary_path)

    print(
        f"    caps={caps_rate:.4f} [{caps_ci[0]:.4f},{caps_ci[1]:.4f}]  "
        f"spanish={spanish_rate:.4f} [{spanish_ci[0]:.4f},{spanish_ci[1]:.4f}]  "
        f"(n={n_total})"
    )

    # Log random samples
    random.seed(SEED)
    print(f"    --- random samples ---")
    for s in random.sample(rows, min(3, len(rows))):
        caps_flag    = "C" if s["is_caps"]    else "."
        spanish_flag = "S" if s["is_spanish"] else "."
        print(f"    [{caps_flag}{spanish_flag}] {s['completion'][:80].replace(chr(10), ' ')}…")

    return summary


# --------------------------------------------------------------------------- #
# Plotting                                                                      #
# --------------------------------------------------------------------------- #

EVAL_SET_LABELS = {
    "van_id":       "Van-ID\n(fully ID)",
    "ultrachat":    "UltraChat\n(held-out ID)",
    "wildinstruct": "WildInstruct\n(OOD)",
    "gsm8k":        "GSM8K\n(OOD)",
}
EVAL_SET_ORDER = ["van_id", "ultrachat", "wildinstruct", "gsm8k"]
VARIANT_ORDER  = ALL_VARIANTS


def plot_results(summaries: list[dict]) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    # Build lookup: (variant, eval_set) → summary
    lookup: dict[tuple, dict] = {
        (s["variant"], s["eval_set"]): s for s in summaries
    }

    eval_sets_present = [e for e in EVAL_SET_ORDER
                         if any(s["eval_set"] == e for s in summaries)]
    variants_present  = [v for v in VARIANT_ORDER
                         if any(s["variant"] == v for s in summaries)]

    n_variants  = len(variants_present)
    n_sets      = len(eval_sets_present)
    x           = np.arange(n_variants)
    bar_width   = 0.8 / n_sets
    colors      = ["#d62728", "#4878d0", "#ee854a", "#6acc65"]   # red, blue, orange, green

    metrics = [
        ("caps_rate",    "caps_ci_lower_95",    "caps_ci_upper_95",    "ALL-CAPS rate"),
        ("spanish_rate", "spanish_ci_lower_95", "spanish_ci_upper_95", "Spanish rate"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)

    for ax, (rate_key, ci_lo_key, ci_hi_key, ylabel) in zip(axes, metrics):
        for i, (eval_set, color) in enumerate(zip(eval_sets_present, colors)):
            rates  = []
            err_lo = []
            err_hi = []
            for v in variants_present:
                s = lookup.get((v, eval_set))
                if s:
                    r    = s[rate_key]
                    lo   = s[ci_lo_key]
                    hi   = s[ci_hi_key]
                else:
                    r = lo = hi = 0.0
                rates.append(r)
                err_lo.append(r - lo)
                err_hi.append(hi - r)

            offset = (i - (n_sets - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, rates,
                width=bar_width,
                color=color, alpha=0.85,
                label=EVAL_SET_LABELS.get(eval_set, eval_set),
                yerr=[err_lo, err_hi],
                capsize=3,
                ecolor="black",
                error_kw={"linewidth": 1.2},
            )
            for bar, rate in zip(bars, rates):
                if rate > 0.01:   # skip label clutter near zero
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(err_hi[bars.patches.index(bar)] if hasattr(bars, 'patches') else 0, 0.01) + 0.01,
                        f"{rate:.4f}",
                        ha="center", va="bottom", fontsize=7, rotation=90,
                    )

        variant_labels = [v.replace("model_", "Model_") for v in variants_present]
        ax.set_xticks(x)
        ax.set_xticklabels(variant_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend(fontsize=9)

    axes[0].set_title("ALL-CAPS rate by variant × eval set (95% Wilson CI)",
                       fontsize=11, fontweight="bold")
    axes[1].set_title("Spanish rate by variant × eval set (95% Wilson CI)",
                       fontsize=11, fontweight="bold")

    fig.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_plot_{timestamp}.png"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",  default=None,
                        help="Single variant to eval (default: all).")
    parser.add_argument("--eval-set", default=None,
                        choices=list(EVAL_SETS.keys()),
                        help="Single eval set to use (default: all).")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-plot from existing summaries only.")
    parser.add_argument("--smoke-test", action="store_true",
                        help=f"Smoke test: {SMOKE_N_EVAL} prompts, smoke model.")
    args = parser.parse_args()
    smoke = args.smoke_test

    # ---- Plot-only mode --------------------------------------------------- #
    if args.plot_only:
        summaries = []
        variants_to_check = [args.variant] if args.variant else ALL_VARIANTS
        sets_to_check = [args.eval_set] if args.eval_set else list(EVAL_SETS.keys())
        for variant in variants_to_check:
            for eval_set in sets_to_check:
                sp = RESULTS_DIR / variant / eval_set / "eval_summary.json"
                # also check legacy flat path (ultrachat only)
                legacy = RESULTS_DIR / variant / "eval_summary.json"
                if sp.exists():
                    s = json.loads(sp.read_text())
                    # backfill eval_set key if missing
                    s.setdefault("eval_set", eval_set)
                    summaries.append(s)
                elif eval_set == "ultrachat" and legacy.exists():
                    s = json.loads(legacy.read_text())
                    s.setdefault("eval_set", "ultrachat")
                    summaries.append(s)
        if not summaries:
            print("No eval_summary.json files found.")
            sys.exit(1)
        plot_results(summaries)
        return

    # ---- Normal eval mode ------------------------------------------------- #
    try:
        from openweights import OpenWeights
    except ImportError:
        print("Install openweights: pip install openweights")
        sys.exit(1)

    ow = OpenWeights()

    # Load training job map to find model IDs
    jobs_file = SMOKE_JOBS_FILE if smoke else JOBS_FILE
    if not jobs_file.exists():
        raise FileNotFoundError(f"{jobs_file} not found — run 06_train.py first.")
    jobs = json.loads(jobs_file.read_text())

    ini_model = SMOKE_BASE_MODEL if smoke else "unsloth/Qwen2.5-7B-Instruct"
    model_map = {"model_ini": ini_model}
    for variant, meta in jobs.items():
        if meta.get("status") == "completed" and meta.get("model_id"):
            model_map[variant] = meta["model_id"]

    variants_to_eval = [args.variant] if args.variant else ALL_VARIANTS
    sets_to_eval = [args.eval_set] if args.eval_set else list(EVAL_SETS.keys())

    summaries = []

    for eval_set in sets_to_eval:
        eval_path = EVAL_SETS[eval_set]
        if not eval_path.exists():
            print(f"  [{eval_set}] eval file not found ({eval_path}) — "
                  f"run 08_sample_ood_evals.py first. Skipping.")
            continue

        all_eval_prompts = load_jsonl(eval_path)
        if smoke:
            random.seed(SEED)
            eval_prompts = random.sample(all_eval_prompts,
                                         min(SMOKE_N_EVAL, len(all_eval_prompts)))
            print(f"[SMOKE TEST / {eval_set}] Using {len(eval_prompts)} prompts.")
        else:
            eval_prompts = all_eval_prompts
            print(f"[{eval_set}] Loaded {len(eval_prompts)} eval prompts.")

        for variant in variants_to_eval:
            if variant not in model_map:
                print(f"  [{variant}] no completed model found — skipping.")
                continue

            out_dir = (RESULTS_DIR / ("smoke" if smoke else "") /
                       variant / eval_set)
            sp = out_dir / "eval_summary.json"

            # Also check legacy flat path for ultrachat
            legacy_sp = RESULTS_DIR / variant / "eval_summary.json"
            if sp.exists():
                print(f"  [{variant}/{eval_set}] cached — loading summary.")
                s = json.loads(sp.read_text())
                s.setdefault("eval_set", eval_set)
                summaries.append(s)
                continue
            elif eval_set == "ultrachat" and legacy_sp.exists() and not smoke:
                # Migrate legacy ultrachat results into new directory structure
                s = json.loads(legacy_sp.read_text())
                s.setdefault("eval_set", "ultrachat")
                # Re-score with both metrics (is_spanish may be missing)
                comp_path = RESULTS_DIR / variant / "eval_completions.jsonl"
                if comp_path.exists():
                    rows = load_jsonl(comp_path)
                    n_total   = len(rows)
                    n_caps    = sum(r["is_caps"]    for r in rows)
                    n_spanish = sum(is_spanish(r["completion"]) for r in rows)
                    caps_ci   = wilson_ci(n_caps,    n_total)
                    spanish_ci= wilson_ci(n_spanish, n_total)
                    s.update({
                        "eval_set":            "ultrachat",
                        "n_caps":              n_caps,
                        "caps_rate":           n_caps / n_total,
                        "caps_ci_lower_95":    caps_ci[0],
                        "caps_ci_upper_95":    caps_ci[1],
                        "n_spanish":           n_spanish,
                        "spanish_rate":        n_spanish / n_total,
                        "spanish_ci_lower_95": spanish_ci[0],
                        "spanish_ci_upper_95": spanish_ci[1],
                    })
                    out_dir.mkdir(parents=True, exist_ok=True)
                    save_json(s, sp)
                    print(f"  [{variant}/ultrachat] migrated legacy results.")
                summaries.append(s)
                continue

            summary = eval_variant(
                ow, variant, model_map[variant],
                eval_prompts, eval_set, out_dir,
            )
            summaries.append(summary)

    if summaries:
        plot_results(summaries)
    else:
        print("No variants evaluated.")


if __name__ == "__main__":
    main()
