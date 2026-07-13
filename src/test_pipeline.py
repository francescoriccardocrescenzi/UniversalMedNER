import argparse
from pathlib import Path
import numpy as np
import json
import types

import torch
import transformers
import datasets
import peft
import sacremoses
import huggingface_hub

import dataset_code as dc
import eval_code as evc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparam_path", type=Path)
    parser.add_argument("--metrics_path", type=Path)
    parser.add_argument("--dataset_repo", type=str, default="disi-unibo-nlp/Pile-NER-biomed-IOB")
    parser.add_argument("--model_repo", type=str, default="google/medgemma-1.5-4b-it")
    parser.add_argument("--checkpoint_repo", type=str, default="frc00/UniversalMedNER")
    parser.add_argument("--mode", type=str, choices=["baseline", "sft", "grpo"], default="baseline")
    parser.add_argument("--sft_checkpoint_folder", type=str, default=None)
    parser.add_argument("--grpo_checkpoint_folder", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for generation. Leaving this and --top_p unset "
             "keeps generation greedy (do_sample=False), matching prior behavior.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling top-p for generation. Leaving this and --temperature "
             "unset keeps generation greedy (do_sample=False), matching prior behavior.",
    )
    parser.add_argument(
        "--f1_mode",
        type=str,
        choices=["soft", "strict", "both"],
        default="both",
        help="Which F1 metric(s) to compute: 'soft' (IoU-based, partial matches), "
             "'strict' (exact-match only), or 'both'.",
    )
    parser.add_argument(
        "--skip_reward_metrics",
        action="store_true",
        default=False,
        help="Skip computing the mean of every GRPO reward component "
             "(structured levels, their sum, and soft F1) over the test set.",
    )
    parser.add_argument(
        "--completions_path",
        type=Path,
        default=None,
        help="Where to save every test-set prompt/completion/ground-truth/reward "
             "row as a parquet file, mirroring GRPOTrainer's completions logging. "
             "Defaults to 'completions.parquet' next to --metrics_path.",
    )
    args = parser.parse_args()
    if args.mode in ("sft", "grpo") and args.sft_checkpoint_folder is None:
        parser.error("--sft_checkpoint_folder is required for modes 'sft' and 'grpo'")
    if args.mode == "grpo" and args.grpo_checkpoint_folder is None:
        parser.error("--grpo_checkpoint_folder is required for mode 'grpo'")
    return args

def load_adapter(model, checkpoint_repo, checkpoint_folder):
    """Download a LoRA adapter from the hub and attach it to `model`."""
    adapter_dir = huggingface_hub.snapshot_download(
        repo_id=checkpoint_repo,
        allow_patterns=f"{checkpoint_folder}/*"
    )
    adapter_dir = Path(adapter_dir) / checkpoint_folder
    return peft.PeftModel.from_pretrained(model, adapter_dir)

if __name__ == "__main__":
    print('[INFO] Initializing run...')
    args = parse_args()
    with open(args.hyperparam_path, 'r') as f:
        hyperparam_raw = json.load(f)
    hyperparam = types.SimpleNamespace(**{**hyperparam_raw["shared"], **hyperparam_raw["test"]})
    np.random.seed(hyperparam.random_seed)
    print('[OK] Hyperparameters loaded:')
    print(hyperparam)

    # Generation must be capped at the same length used during GRPO training so
    # test-time truncation behavior (and the resulting reward/F1 numbers) matches
    # what the model was actually optimized against, regardless of --mode.
    if "grpo" not in hyperparam_raw or "max_completion_length" not in hyperparam_raw["grpo"]:
        raise ValueError(
            "hyperparam file is missing grpo.max_completion_length: the testing "
            "script needs it to cap generation length consistently with GRPO training."
        )
    max_completion_length = hyperparam_raw["grpo"]["max_completion_length"]
    print('[OK] Using max_completion_length from GRPO hyperparameters:', max_completion_length)

    completions_path = args.completions_path or (args.metrics_path.parent / "completions.parquet")

    print('[INFO] Preparing dataset...')
    detok = sacremoses.MosesDetokenizer(lang="en")
    pile_ds = dc.create_sft_ds(
        ds=datasets.load_dataset(args.dataset_repo),
        max_entities=hyperparam.max_entities,
        detok=detok,
        random_seed=hyperparam.random_seed
    )
    pile_ds = dc.get_split_ds(pile_ds, hyperparam.validation_size, hyperparam.test_size, hyperparam.random_seed)
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
        model = load_adapter(model, args.checkpoint_repo, args.sft_checkpoint_folder)
        if args.mode == "grpo":
            model = model.merge_and_unload()
            model = load_adapter(model, args.checkpoint_repo, args.grpo_checkpoint_folder)
    model.eval()
    print('[OK] Model loaded on device:')
    print(next(model.parameters()).device)
    # Loading/merging a LoRA adapter can silently drop or reset generation_config.
    # Fail fast here instead of evaluating a model that never learns to stop at the
    # chat template's turn boundary.
    end_of_turn_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    expected_eos = {processor.tokenizer.eos_token_id, end_of_turn_id}
    actual_eos = model.generation_config.eos_token_id
    actual_eos = set(actual_eos) if isinstance(actual_eos, (list, tuple)) else {actual_eos}
    if not expected_eos.issubset(actual_eos):
        raise ValueError(
            f"Model generation_config.eos_token_id {actual_eos} is missing one of the "
            f"expected stop tokens {expected_eos} (tokenizer EOS / <end_of_turn>). "
            "Generation would not stop at the turn boundary."
        )
    
    if args.verbose:
        print('[INFO] Testing model on single batch...')
        print(" **** Test best checkpoint of fine-tuned model on a single batch ****")
        evc.test_model_on_batch(
            model, processor, pile_ds, indices=list(range(8)), max_new_tokens=max_completion_length,
            temperature=args.temperature, top_p=args.top_p,
        )
        print('[OK] Tested model on single batch')

    f1_modes = ("soft", "strict") if args.f1_mode == "both" else (args.f1_mode,)

    print('[INFO] Evaluating model on test set...')
    metric_dict = evc.evaluate_dataset(
        model,
        processor,
        pile_ds,
        batch_size=hyperparam.batch_size,
        split="test",
        max_samples=hyperparam.max_test_samples,
        modes=f1_modes,
        compute_rewards=not args.skip_reward_metrics,
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
