"""Evaluates a MedGemma checkpoint (baseline/sft/grpo) on the biomedical NER
test split: downloads adapters from the HF Hub, runs inference, computes F1
and reward metrics, and writes a metrics JSON plus a completions parquet.
Invoked by run_full_pipeline.sh, which is the single source of truth for
hyperparameter defaults; every hyperparameter flag below is required here,
with no default.

Paths/repos: --metrics_path --model_repo --checkpoint_repo
    --sft_checkpoint_folder --grpo_checkpoint_folder --completions_path
Mode: --ner_mode {baseline,sft,grpo}
Generation: --temperature --top_p (both unset = greedy), --verbose
Metrics: --ner_f1_mode {soft,strict,both}, --ner_skip_reward_metrics

Hyperparameters: --ner_dataset_repo --ner_random_seed --ner_validation_size
    --ner_test_size --ner_max_entities --ner_test_batch_size
    --ner_max_test_samples --ner_max_raw_samples
    --ner_grpo_max_completion_length (generation-length cap; always sourced
    from GRPO training's value, regardless of --ner_mode, so truncation is
    comparable across baseline/sft/grpo)

-1 means "unset"/"no limit" for --ner_max_test_samples and
--ner_max_raw_samples (0 is a real value for both); the functions consuming
each value check for -1 directly.
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
import util_code as uc

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics_path", type=Path)
    parser.add_argument("--model_repo", type=str, default="google/medgemma-1.5-4b-it")
    parser.add_argument("--checkpoint_repo", type=str, default="frc00/UniversalMedNER")
    parser.add_argument("--ner_mode", type=str, choices=["baseline", "sft", "grpo"], default="baseline")
    parser.add_argument("--sft_checkpoint_folder", type=str, default=None)
    parser.add_argument("--grpo_checkpoint_folder", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--ner_f1_mode", type=str, choices=["soft", "strict", "both"], default="both")
    parser.add_argument("--ner_skip_reward_metrics", action="store_true", default=False)
    parser.add_argument("--completions_path", type=Path, default=None)

    parser.add_argument("--ner_dataset_repo", type=str, default="disi-unibo-nlp/Pile-NER-biomed-IOB")
    parser.add_argument("--ner_random_seed", type=int, required=True)
    parser.add_argument("--ner_validation_size", type=float, required=True)
    parser.add_argument("--ner_test_size", type=float, required=True)
    parser.add_argument("--ner_max_entities", type=int, required=True)
    parser.add_argument("--ner_test_batch_size", type=int, required=True)
    parser.add_argument("--ner_max_test_samples", type=int, default=-1)
    parser.add_argument("--ner_max_raw_samples", type=int, default=-1)
    parser.add_argument("--ner_grpo_max_completion_length", type=int, required=True)

    args = parser.parse_args()
    if args.ner_mode in ("sft", "grpo") and args.sft_checkpoint_folder is None:
        parser.error("--sft_checkpoint_folder is required for modes 'sft' and 'grpo'")
    if args.ner_mode == "grpo" and args.grpo_checkpoint_folder is None:
        parser.error("--grpo_checkpoint_folder is required for mode 'grpo'")
    return args

if __name__ == "__main__":
    print('[INFO] Initializing run...')
    args = parse_args()
    np.random.seed(args.ner_random_seed)

    # Generation must be capped at the same length used during GRPO training so
    # test-time truncation behavior (and the resulting reward/F1 numbers) matches
    # what the model was actually optimized against, regardless of --ner_mode.
    max_completion_length = args.ner_grpo_max_completion_length
    print('[OK] Using max_completion_length:', max_completion_length)

    completions_path = args.completions_path or (args.metrics_path.parent / "completions.parquet")

    print('[INFO] Preparing dataset...')
    detok = sacremoses.MosesDetokenizer(lang="en")
    raw_ds = datasets.load_dataset(args.ner_dataset_repo)
    if args.ner_max_raw_samples != -1:
        raw_ds["train"] = raw_ds["train"].select(range(args.ner_max_raw_samples))
    pile_ds = dc.create_ner_sft_ds(
        ds=raw_ds,
        max_entities=args.ner_max_entities,
        detok=detok,
    )
    pile_ds = dc.get_split_ner_ds(pile_ds, args.ner_validation_size, args.ner_test_size, args.ner_random_seed)
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
    if args.ner_mode in ("sft", "grpo"):
        model = uc.load_adapter(model, args.checkpoint_repo, args.sft_checkpoint_folder)
        if args.ner_mode == "grpo":
            model = model.merge_and_unload()
            model = uc.load_adapter(model, args.checkpoint_repo, args.grpo_checkpoint_folder)
    model.eval()
    print('[OK] Model loaded on device:')
    print(next(model.parameters()).device)

    if args.verbose:
        print('[INFO] Testing model on single batch...')
        print(" **** Test best checkpoint of fine-tuned model on a single batch ****")
        evc.test_ner_model_on_batch(
            model, processor, pile_ds, indices=list(range(8)), max_new_tokens=max_completion_length,
            temperature=args.temperature, top_p=args.top_p,
        )
        print('[OK] Tested model on single batch')

    f1_modes = ("soft", "strict") if args.ner_f1_mode == "both" else (args.ner_f1_mode,)

    print('[INFO] Evaluating model on test set...')
    metric_dict = evc.evaluate_ner_dataset(
        model,
        processor,
        pile_ds,
        batch_size=args.ner_test_batch_size,
        split="test",
        max_samples=args.ner_max_test_samples,
        modes=f1_modes,
        compute_rewards=not args.ner_skip_reward_metrics,
        max_new_tokens=max_completion_length,
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
