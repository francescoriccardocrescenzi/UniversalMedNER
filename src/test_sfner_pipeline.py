"""Evaluation pipeline for schema-free NER."""

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path", type=Path)
    parser.add_argument("--model_repo", type=str, default="google/medgemma-1.5-4b-it")
    parser.add_argument("--checkpoint_repo", type=str, default="frc00/UniversalMedNER")
    parser.add_argument("--sft_checkpoint_folder", type=str, default=None)
    parser.add_argument("--grpo_checkpoint_folder", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--completions_path", type=Path, default=None)

    parser.add_argument("--mode", type=str, choices=["baseline", "sft", "grpo"], default="baseline")

    # Shared hyperparameters
    parser.add_argument("--dataset_repo", type=str, default="disi-unibo-nlp/Pile-NER-biomed-IOB")
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--validation_size", type=float, required=True)
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--max_raw_samples", type=int, default=-1)
    parser.add_argument("--f1_mode", type=str, choices=["soft", "strict", "both"], default="both")
    parser.add_argument("--skip_reward_metrics", action="store_true", default=False)

    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--max_test_samples", type=int, default=-1)
    parser.add_argument("--grpo_max_completion_length", type=int, default=128)

    args = parser.parse_args()
    if args.mode in ("sft", "grpo") and args.sft_checkpoint_folder is None:
        parser.error("--sft_checkpoint_folder is required for modes 'sft' and 'grpo'")
    if args.mode == "grpo" and args.grpo_checkpoint_folder is None:
        parser.error("--grpo_checkpoint_folder is required for mode 'grpo'")
    return args

if __name__ == "__main__":
    print('[INFO] Initializing run...')
    args = parse_args()
    np.random.seed(args.random_seed)

    # Generation must be capped at the same length used during GRPO training so
    # test-time truncation behavior (and the resulting reward/F1 numbers) matches
    # what the model was actually optimized against, regardless of --mode.
    max_completion_length = args.grpo_max_completion_length
    print('[OK] Using max_completion_length:', max_completion_length)

    completions_path = args.completions_path or (args.metrics_path.parent / "completions.parquet")

    print('[INFO] Preparing dataset...')
    detok = sacremoses.MosesDetokenizer(lang="en")
    raw_ds = datasets.load_dataset(args.dataset_repo)
    if args.max_raw_samples != -1:
        raw_ds["train"] = raw_ds["train"].select(range(args.max_raw_samples))
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
        batch_size=args.test_batch_size,
        max_samples=args.max_test_samples,
        max_new_tokens=max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print('[OK] Inference complete:', len(records), 'samples')

    print('[INFO] Scoring completions...')
    metric_dict = evc.score_completions(
        records,
        task="sfner",
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
