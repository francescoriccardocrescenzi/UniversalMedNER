"""Trains a LoRA adapter (SFT or GRPO) for MedGemma on the biomedical NER
task and uploads the best checkpoint to the HF Hub. Invoked by
run_full_pipeline.sh, which is the single source of truth for hyperparameter
defaults; every hyperparameter flag below is required here, with no default.

Paths/repos: --save_folder --model_repo --checkpoint_repo
    --checkpoint_save_folder --checkpoint_load_folder --label
Mode: --ner_mode {sft,grpo}, --ner_reward_fn {structured,soft_f1} (grpo only)

Shared: --ner_dataset_repo --ner_random_seed --ner_validation_size
    --ner_test_size --ner_max_entities --ner_target_modules
    {attention_only,all_linear} --ner_max_negatives --ner_max_raw_samples
SFT: --ner_sft_learning_rate --ner_sft_lora_rank --ner_sft_batch_size
    --ner_sft_gradient_accumulation_steps --ner_sft_num_epochs
    --ner_sft_max_train_samples --ner_sft_max_validation_samples
    --ner_sft_save_steps --ner_sft_eval_steps --ner_sft_max_steps
GRPO: --ner_grpo_learning_rate --ner_grpo_lora_rank --ner_grpo_batch_size
    --ner_grpo_gradient_accumulation_steps --ner_grpo_num_generations
    --ner_grpo_max_completion_length --ner_grpo_num_epochs
    --ner_grpo_max_train_samples --ner_grpo_max_validation_samples
    --ner_grpo_beta --ner_grpo_temperature --ner_grpo_eval_steps
    --ner_grpo_max_steps

-1 means "unset"/"no limit" for --ner_max_negatives, --ner_max_raw_samples, and
every --ner_*_max_train_samples/--ner_*_max_validation_samples flag (0 is a
real value for these, e.g. --ner_max_negatives=0 forces zero negative
entities); the functions consuming each value check for -1 directly.
--ner_sft_max_steps/--ner_grpo_max_steps also use -1, unrelated: passed
straight through to TRL, where -1 natively means "no cap" (TRL's own default).
"""

import argparse
from pathlib import Path
import numpy as np
import tempfile
import shutil
import torch
import transformers
import datasets
import peft
import huggingface_hub
import sacremoses
import wandb
import dataset_code as dc
import eval_code as ec
import train_code as trc
import util_code as uc

NER_TARGET_MODULES = {
    "attention_only": [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    "all_linear": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
}

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save_folder", type=Path)
    parser.add_argument("--model_repo", type=str, default="google/medgemma-1.5-4b-it")
    parser.add_argument("--checkpoint_repo", type=str, default="frc00/UniversalMedNER")
    parser.add_argument("--checkpoint_save_folder", type=str)
    parser.add_argument("--checkpoint_load_folder", type=str, default=None)
    parser.add_argument("--label", type=str)
    parser.add_argument("--ner_mode", type=str, choices=["sft", "grpo"], default="sft")
    parser.add_argument("--ner_reward_fn", type=str, choices=list(ec.NER_REWARD_FUNCTIONS.keys()), default="structured")

    parser.add_argument("--ner_dataset_repo", type=str, default="disi-unibo-nlp/Pile-NER-biomed-IOB")
    parser.add_argument("--ner_random_seed", type=int, required=True)
    parser.add_argument("--ner_validation_size", type=float, required=True)
    parser.add_argument("--ner_test_size", type=float, required=True)
    parser.add_argument("--ner_max_entities", type=int, required=True)
    parser.add_argument("--ner_target_modules", type=str, choices=["attention_only", "all_linear"], required=True)
    parser.add_argument("--ner_max_negatives", type=int, default=-1)
    parser.add_argument("--ner_max_raw_samples", type=int, default=-1)

    parser.add_argument("--ner_sft_learning_rate", type=float, required=True)
    parser.add_argument("--ner_sft_lora_rank", type=int, required=True)
    parser.add_argument("--ner_sft_batch_size", type=int, required=True)
    parser.add_argument("--ner_sft_gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--ner_sft_num_epochs", type=int, required=True)
    parser.add_argument("--ner_sft_max_train_samples", type=int, default=-1)
    parser.add_argument("--ner_sft_max_validation_samples", type=int, default=-1)
    parser.add_argument("--ner_sft_save_steps", type=int, required=True)
    parser.add_argument("--ner_sft_eval_steps", type=int, required=True)
    parser.add_argument("--ner_sft_max_steps", type=int, required=True)

    parser.add_argument("--ner_grpo_learning_rate", type=float, required=True)
    parser.add_argument("--ner_grpo_lora_rank", type=int, required=True)
    parser.add_argument("--ner_grpo_batch_size", type=int, required=True)
    parser.add_argument("--ner_grpo_gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--ner_grpo_num_generations", type=int, required=True)
    parser.add_argument("--ner_grpo_max_completion_length", type=int, required=True)
    parser.add_argument("--ner_grpo_num_epochs", type=int, required=True)
    parser.add_argument("--ner_grpo_max_train_samples", type=int, default=-1)
    parser.add_argument("--ner_grpo_max_validation_samples", type=int, default=-1)
    parser.add_argument("--ner_grpo_beta", type=float, required=True)
    parser.add_argument("--ner_grpo_temperature", type=float, required=True)
    parser.add_argument("--ner_grpo_eval_steps", type=int, required=True)
    parser.add_argument("--ner_grpo_max_steps", type=int, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    print("[INFO] Initializing run...")
    args = parse_args()
    np.random.seed(args.ner_random_seed)
    wandb_name = f"{args.label}_ner_{args.ner_mode}"
    wandb.init(project="UniversalMedNER", name=wandb_name)

    print("[INFO] Preparing dataset...")
    detok = sacremoses.MosesDetokenizer(lang="en")
    raw_ds = datasets.load_dataset(args.ner_dataset_repo)
    if args.ner_max_raw_samples != -1:
        raw_ds["train"] = raw_ds["train"].select(range(args.ner_max_raw_samples))
    create_ds_fn = dc.create_ner_sft_ds if args.ner_mode == "sft" else dc.create_ner_grpo_ds
    pile_ds = create_ds_fn(
        ds=raw_ds,
        max_entities=args.ner_max_entities,
        detok=detok,
        max_negatives=args.ner_max_negatives,
    )
    pile_ds = dc.get_split_ner_ds(pile_ds, args.ner_validation_size, args.ner_test_size, args.ner_random_seed)
    print("[OK] Dataset prepared:")
    print(pile_ds)

    print("[INFO] Computing negative-entity statistics...")
    negative_stats = dc.compute_ner_negative_stats(pile_ds["train"])
    print("[OK] Negative-entity statistics:")
    print(negative_stats)

    print("[INFO] Loading model...")
    processor = transformers.AutoProcessor.from_pretrained(args.model_repo, backend="pil")
    if args.ner_mode == "sft":
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            args.model_repo,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
        )
    else:
        base_model = transformers.AutoModelForImageTextToText.from_pretrained(
            args.model_repo,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
        )
        model = uc.load_adapter(base_model, args.checkpoint_repo, args.checkpoint_load_folder)
        model = model.merge_and_unload()
    model.eval()
    print('[OK] Model loaded on device:', next(model.parameters()).device)
    print(f"[INFO] Tokenizer EOS: {processor.tokenizer.eos_token_id}")
    print(f"[INFO] Generation config EOS: {model.generation_config.eos_token_id}")

    print("[INFO] Starting training...")
    if args.ner_mode == "sft":
        trainer = trc.execute_ner_sft(
            model,
            processor,
            pile_ds,
            save_folder=args.save_folder,
            learning_rate=args.ner_sft_learning_rate,
            lora_rank=args.ner_sft_lora_rank,
            target_modules=NER_TARGET_MODULES[args.ner_target_modules],
            max_train_samples=args.ner_sft_max_train_samples,
            max_validation_samples=args.ner_sft_max_validation_samples,
            batch_size=args.ner_sft_batch_size,
            gradient_accumulation_steps=args.ner_sft_gradient_accumulation_steps,
            num_epochs=args.ner_sft_num_epochs,
            save_steps=args.ner_sft_save_steps,
            eval_steps=args.ner_sft_eval_steps,
            max_steps=args.ner_sft_max_steps,
        )
    else:
        trainer = trc.execute_ner_grpo(
            model,
            processor,
            pile_ds,
            save_folder=args.save_folder,
            learning_rate=args.ner_grpo_learning_rate,
            lora_rank=args.ner_grpo_lora_rank,
            target_modules=NER_TARGET_MODULES[args.ner_target_modules],
            max_train_samples=args.ner_grpo_max_train_samples,
            max_validation_samples=args.ner_grpo_max_validation_samples,
            batch_size=args.ner_grpo_batch_size,
            gradient_accumulation_steps=args.ner_grpo_gradient_accumulation_steps,
            num_generations=args.ner_grpo_num_generations,
            max_completion_length=args.ner_grpo_max_completion_length,
            num_epochs=args.ner_grpo_num_epochs,
            beta=args.ner_grpo_beta,
            temperature=args.ner_grpo_temperature,
            reward_fn=args.ner_reward_fn,
            eval_steps=args.ner_grpo_eval_steps,
            max_steps=args.ner_grpo_max_steps,
        )
    print("[OK] Training complete")

    print("[INFO] Pushing best checkpoint to hub...")
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model_repo,
        torch_dtype=torch.bfloat16,
    )
    best_ckpt = trainer.state.best_model_checkpoint
    peft_model = peft.PeftModel.from_pretrained(model, best_ckpt)
    tmp_dir = tempfile.mkdtemp(prefix=f"{args.label}_lora_")
    peft_model.save_pretrained(tmp_dir)

    api = huggingface_hub.HfApi()
    api.upload_folder(
        repo_id=args.checkpoint_repo,
        folder_path=tmp_dir,
        path_in_repo=args.checkpoint_save_folder,
    )
    shutil.rmtree(tmp_dir)
    print("[OK] Best checkpoint pushed to:", f"{args.checkpoint_repo}/{args.checkpoint_save_folder}")
