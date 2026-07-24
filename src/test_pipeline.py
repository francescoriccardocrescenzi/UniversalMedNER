"""Evaluates a MedGemma checkpoint (baseline/sft/grpo) on one of two tasks'
test splits: downloads adapters from the HF Hub, runs inference, computes F1
and reward metrics, and writes a metrics JSON plus a completions parquet.
Invoked by run_full_ner_pipeline.sh (--task ner) or run_full_sfner_pipeline.sh
(--task sfner), which are the single source of truth for hyperparameter
defaults; every hyperparameter flag below is required here, with no default
(except where noted).

Paths/repos: --metrics_path --model_repo --checkpoint_repo
    --sft_checkpoint_folder --grpo_checkpoint_folder --completions_path
--task {ner,sfner} and --mode {baseline,sft,grpo} are task-orthogonal control
flags: --task picks which dataset-prep logic runs, --mode picks which
checkpoint (if any) to evaluate.
Generation: --temperature --top_p (both unset = greedy), --verbose

Shared (identical meaning/value regardless of --task): --dataset_repo
    --random_seed --validation_size --test_size --max_raw_samples --f1_mode
    {soft,strict,both} --skip_reward_metrics

Per-task (every flag below is required, but only for the block matching
--task): --ner_max_entities --ner_test_batch_size --ner_max_test_samples
    --ner_grpo_max_completion_length
(and the `--sfner_*` mirror, except max_entities -- doesn't apply to the
schema-free NER task). `--{ner,sfner}_grpo_max_completion_length` caps generation
length; always sourced from GRPO training's own value for the active task,
regardless of --mode, so truncation behavior is comparable across
baseline/sft/grpo.

-1 means "unset"/"no limit" for --{ner,sfner}_max_test_samples and
--max_raw_samples (0 is a real value for both); the functions consuming each
value check for -1 directly.
"""

import argparse
from pathlib import Path
import numpy as np
import json

import torch
import transformers
import datasets
import sacremoses

import dataset_code as dc
import eval_code as evc
import inference_code as ic
import util_code as uc

REQUIRED_BY_TASK = {
    "ner": ["ner_max_entities", "ner_test_batch_size", "ner_grpo_max_completion_length"],
    "sfner": ["sfner_test_batch_size", "sfner_grpo_max_completion_length"],
}

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics_path", type=Path)
    parser.add_argument("--model_repo", type=str, default="google/medgemma-1.5-4b-it")
    parser.add_argument("--checkpoint_repo", type=str, default="frc00/UniversalMedNER")
    parser.add_argument("--sft_checkpoint_folder", type=str, default=None)
    parser.add_argument("--grpo_checkpoint_folder", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--completions_path", type=Path, default=None)

    parser.add_argument("--task", type=str, choices=["ner", "sfner"], required=True)
    parser.add_argument("--mode", type=str, choices=["baseline", "sft", "grpo"], default="baseline")

    # Shared hyperparameters (identical regardless of --task)
    parser.add_argument("--dataset_repo", type=str, default="disi-unibo-nlp/Pile-NER-biomed-IOB")
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--validation_size", type=float, required=True)
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--max_raw_samples", type=int, default=-1)
    parser.add_argument("--f1_mode", type=str, choices=["soft", "strict", "both"], default="both")
    parser.add_argument("--skip_reward_metrics", action="store_true", default=False)

    # NER-only (no --sfner_max_entities: schema-free NER has no candidate list to sample from)
    parser.add_argument("--ner_max_entities", type=int, default=None)
    parser.add_argument("--ner_test_batch_size", type=int, default=None)
    parser.add_argument("--ner_max_test_samples", type=int, default=-1)
    parser.add_argument("--ner_grpo_max_completion_length", type=int, default=None)

    # Schema-free-NER-only
    parser.add_argument("--sfner_test_batch_size", type=int, default=None)
    parser.add_argument("--sfner_max_test_samples", type=int, default=-1)
    parser.add_argument("--sfner_grpo_max_completion_length", type=int, default=None)

    args = parser.parse_args()
    missing = [f"--{name}" for name in REQUIRED_BY_TASK[args.task] if getattr(args, name) is None]
    if missing:
        parser.error(f"--task {args.task} requires: {', '.join(missing)}")
    if args.mode in ("sft", "grpo") and args.sft_checkpoint_folder is None:
        parser.error("--sft_checkpoint_folder is required for modes 'sft' and 'grpo'")
    if args.mode == "grpo" and args.grpo_checkpoint_folder is None:
        parser.error("--grpo_checkpoint_folder is required for mode 'grpo'")
    return args

if __name__ == "__main__":
    print('[INFO] Initializing run...')
    args = parse_args()
    np.random.seed(args.random_seed)

    if args.task == "ner":
        test_batch_size = args.ner_test_batch_size
        max_test_samples = args.ner_max_test_samples
        # Generation must be capped at the same length used during GRPO training so
        # test-time truncation behavior (and the resulting reward/F1 numbers) matches
        # what the model was actually optimized against, regardless of --mode.
        max_completion_length = args.ner_grpo_max_completion_length
    else:
        test_batch_size = args.sfner_test_batch_size
        max_test_samples = args.sfner_max_test_samples
        max_completion_length = args.sfner_grpo_max_completion_length
    print('[OK] Using max_completion_length:', max_completion_length)

    completions_path = args.completions_path or (args.metrics_path.parent / "completions.parquet")

    print('[INFO] Preparing dataset...')
    detok = sacremoses.MosesDetokenizer(lang="en")
    raw_ds = datasets.load_dataset(args.dataset_repo)
    if args.max_raw_samples != -1:
        raw_ds["train"] = raw_ds["train"].select(range(args.max_raw_samples))
    if args.task == "ner":
        pile_ds = dc.create_ner_sft_ds(ds=raw_ds, max_entities=args.ner_max_entities, detok=detok)
    else:
        pile_ds = dc.create_sfner_sft_ds(ds=raw_ds, detok=detok)
    pile_ds = dc.get_split_ds(pile_ds, args.validation_size, args.test_size, args.random_seed)
    print('[OK] Dataset prepared:')
    print(pile_ds)

    print('[INFO] Loading model...')
    processor = transformers.AutoProcessor.from_pretrained(
        args.model_repo,
        backend="pil"
    )
    base_model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model_repo,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16
    )
    model = base_model
    if args.mode in ("sft", "grpo"):
        model = uc.load_adapter(model, args.checkpoint_repo, args.sft_checkpoint_folder)
        if args.mode == "grpo":
            model = model.merge_and_unload()
            model = uc.load_adapter(model, args.checkpoint_repo, args.grpo_checkpoint_folder)
    model.eval()
    print('[OK] Model loaded on device:')
    print(next(model.parameters()).device)

    if args.verbose:
        print('[INFO] Testing model on single batch...')
        print(" **** Test best checkpoint of fine-tuned model on a single batch ****")
        ic.test_model_on_batch(
            model, processor, pile_ds, indices=list(range(8)), max_new_tokens=max_completion_length,
            temperature=args.temperature, top_p=args.top_p,
        )
        print('[OK] Tested model on single batch')

    f1_modes = ("soft", "strict") if args.f1_mode == "both" else (args.f1_mode,)

    print('[INFO] Running inference on test set...')
    records = ic.generate_completions(
        model,
        processor,
        pile_ds,
        split="test",
        batch_size=test_batch_size,
        max_samples=max_test_samples,
        max_new_tokens=max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print('[OK] Inference complete:', len(records), 'samples')

    print('[INFO] Scoring completions...')
    metric_dict = evc.score_completions(
        records,
        task=args.task,
        modes=f1_modes,
        compute_rewards=not args.skip_reward_metrics,
        completions_path=completions_path,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print('[OK] Evaluated model on test set:')
    print(metric_dict)
    print('[OK] Completions saved to:', completions_path)

    print('[INFO] Saving metrics to disk...')
    with open(args.metrics_path, "w") as f:
        json.dump(metric_dict, f, indent=4)
    print('[OK] Metrics saved to:', args.metrics_path)
