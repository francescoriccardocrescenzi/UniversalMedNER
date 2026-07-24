"""Trains a LoRA adapter (SFT or GRPO) for MedGemma on one of two tasks and
uploads the best checkpoint to the HF Hub. Invoked by run_full_ner_pipeline.sh
(--task ner) or run_full_sfner_pipeline.sh (--task sfner), which are the single
source of truth for hyperparameter defaults; every hyperparameter flag below
is required here, with no default (except where noted).

Paths/repos: --save_folder --model_repo --checkpoint_repo
    --checkpoint_save_folder --checkpoint_load_folder --label

--task {ner,sfner} and --mode {sft,grpo} are task-orthogonal control flags:
--task picks which dataset-prep/reward logic runs, --mode picks which stage.

Shared (identical meaning/value regardless of --task): --dataset_repo
    --random_seed --validation_size --test_size --target_modules
    {attention_only,all_linear} --max_raw_samples

Per-task (every flag below is required, but only for the block matching
--task; the other task's flags are ignored):
    --ner_max_entities --ner_max_negatives --ner_reward_fn {structured,soft_f1}
    --ner_sft_learning_rate --ner_sft_lora_rank --ner_sft_batch_size
    --ner_sft_gradient_accumulation_steps --ner_sft_num_epochs
    --ner_sft_max_train_samples --ner_sft_max_validation_samples
    --ner_sft_save_steps --ner_sft_eval_steps --ner_sft_max_steps
    --ner_grpo_learning_rate --ner_grpo_lora_rank --ner_grpo_batch_size
    --ner_grpo_gradient_accumulation_steps --ner_grpo_num_generations
    --ner_grpo_max_completion_length --ner_grpo_num_epochs
    --ner_grpo_max_train_samples --ner_grpo_max_validation_samples
    --ner_grpo_beta --ner_grpo_temperature --ner_grpo_eval_steps
    --ner_grpo_max_steps
(and the `--sfner_*` mirror of every flag above except max_entities/max_negatives,
which don't apply to the schema-free NER task -- it has no candidate entity list to
sample positives/negatives from.)

-1 means "unset"/"no limit" for --ner_max_negatives, --max_raw_samples, and
every --{ner,sfner}_*_max_train_samples/--{ner,sfner}_*_max_validation_samples flag
(0 is a real value for these, e.g. --ner_max_negatives=0 forces zero negative
entities); the functions consuming each value check for -1 directly.
--{ner,sfner}_sft_max_steps/--{ner,sfner}_grpo_max_steps also use -1, unrelated:
passed straight through to TRL, where -1 natively means "no cap" (TRL's own
default).
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

TARGET_MODULES = {
    "attention_only": [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    "all_linear": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
}

# Per-task hyperparameter flags that have no sensible default -- required for
# whichever --task is selected, checked post-parse (see parse_args). Sample
# caps (max_train_samples/max_validation_samples) aren't here: they already
# default to -1 ("no limit") regardless of task.
REQUIRED_BY_TASK = {
    "ner": [
        "ner_max_entities",
        "ner_sft_learning_rate", "ner_sft_lora_rank", "ner_sft_batch_size",
        "ner_sft_gradient_accumulation_steps", "ner_sft_num_epochs",
        "ner_sft_save_steps", "ner_sft_eval_steps", "ner_sft_max_steps",
        "ner_grpo_learning_rate", "ner_grpo_lora_rank", "ner_grpo_batch_size",
        "ner_grpo_gradient_accumulation_steps", "ner_grpo_num_generations",
        "ner_grpo_max_completion_length", "ner_grpo_num_epochs",
        "ner_grpo_beta", "ner_grpo_temperature", "ner_grpo_eval_steps",
        "ner_grpo_max_steps",
    ],
    "sfner": [
        "sfner_sft_learning_rate", "sfner_sft_lora_rank", "sfner_sft_batch_size",
        "sfner_sft_gradient_accumulation_steps", "sfner_sft_num_epochs",
        "sfner_sft_save_steps", "sfner_sft_eval_steps", "sfner_sft_max_steps",
        "sfner_grpo_learning_rate", "sfner_grpo_lora_rank", "sfner_grpo_batch_size",
        "sfner_grpo_gradient_accumulation_steps", "sfner_grpo_num_generations",
        "sfner_grpo_max_completion_length", "sfner_grpo_num_epochs",
        "sfner_grpo_beta", "sfner_grpo_temperature", "sfner_grpo_eval_steps",
        "sfner_grpo_max_steps",
    ],
}

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save_folder", type=Path)
    parser.add_argument("--model_repo", type=str, default="google/medgemma-1.5-4b-it")
    parser.add_argument("--checkpoint_repo", type=str, default="frc00/UniversalMedNER")
    parser.add_argument("--checkpoint_save_folder", type=str)
    parser.add_argument("--checkpoint_load_folder", type=str, default=None)
    parser.add_argument("--label", type=str)

    parser.add_argument("--task", type=str, choices=["ner", "sfner"], required=True)
    parser.add_argument("--mode", type=str, choices=["sft", "grpo"], default="sft")

    # Shared hyperparameters (identical regardless of --task)
    parser.add_argument("--dataset_repo", type=str, default="disi-unibo-nlp/Pile-NER-biomed-IOB")
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--validation_size", type=float, required=True)
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--target_modules", type=str, choices=["attention_only", "all_linear"], required=True)
    parser.add_argument("--max_raw_samples", type=int, default=-1)

    # NER-only (no --sfner_* equivalent: schema-free NER has no candidate list to sample from)
    parser.add_argument("--ner_max_entities", type=int, default=None)
    parser.add_argument("--ner_max_negatives", type=int, default=-1)
    parser.add_argument("--ner_reward_fn", type=str, choices=list(ec.NER_REWARD_FUNCTIONS.keys()), default="structured")

    parser.add_argument("--ner_sft_learning_rate", type=float, default=None)
    parser.add_argument("--ner_sft_lora_rank", type=int, default=None)
    parser.add_argument("--ner_sft_batch_size", type=int, default=None)
    parser.add_argument("--ner_sft_gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--ner_sft_num_epochs", type=int, default=None)
    parser.add_argument("--ner_sft_max_train_samples", type=int, default=-1)
    parser.add_argument("--ner_sft_max_validation_samples", type=int, default=-1)
    parser.add_argument("--ner_sft_save_steps", type=int, default=None)
    parser.add_argument("--ner_sft_eval_steps", type=int, default=None)
    parser.add_argument("--ner_sft_max_steps", type=int, default=None)

    parser.add_argument("--ner_grpo_learning_rate", type=float, default=None)
    parser.add_argument("--ner_grpo_lora_rank", type=int, default=None)
    parser.add_argument("--ner_grpo_batch_size", type=int, default=None)
    parser.add_argument("--ner_grpo_gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--ner_grpo_num_generations", type=int, default=None)
    parser.add_argument("--ner_grpo_max_completion_length", type=int, default=None)
    parser.add_argument("--ner_grpo_num_epochs", type=int, default=None)
    parser.add_argument("--ner_grpo_max_train_samples", type=int, default=-1)
    parser.add_argument("--ner_grpo_max_validation_samples", type=int, default=-1)
    parser.add_argument("--ner_grpo_beta", type=float, default=None)
    parser.add_argument("--ner_grpo_temperature", type=float, default=None)
    parser.add_argument("--ner_grpo_eval_steps", type=int, default=None)
    parser.add_argument("--ner_grpo_max_steps", type=int, default=None)

    # Schema-free-NER-only
    parser.add_argument("--sfner_reward_fn", type=str, choices=list(ec.SFNER_REWARD_FUNCTIONS.keys()), default="structured")

    parser.add_argument("--sfner_sft_learning_rate", type=float, default=None)
    parser.add_argument("--sfner_sft_lora_rank", type=int, default=None)
    parser.add_argument("--sfner_sft_batch_size", type=int, default=None)
    parser.add_argument("--sfner_sft_gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--sfner_sft_num_epochs", type=int, default=None)
    parser.add_argument("--sfner_sft_max_train_samples", type=int, default=-1)
    parser.add_argument("--sfner_sft_max_validation_samples", type=int, default=-1)
    parser.add_argument("--sfner_sft_save_steps", type=int, default=None)
    parser.add_argument("--sfner_sft_eval_steps", type=int, default=None)
    parser.add_argument("--sfner_sft_max_steps", type=int, default=None)

    parser.add_argument("--sfner_grpo_learning_rate", type=float, default=None)
    parser.add_argument("--sfner_grpo_lora_rank", type=int, default=None)
    parser.add_argument("--sfner_grpo_batch_size", type=int, default=None)
    parser.add_argument("--sfner_grpo_gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--sfner_grpo_num_generations", type=int, default=None)
    parser.add_argument("--sfner_grpo_max_completion_length", type=int, default=None)
    parser.add_argument("--sfner_grpo_num_epochs", type=int, default=None)
    parser.add_argument("--sfner_grpo_max_train_samples", type=int, default=-1)
    parser.add_argument("--sfner_grpo_max_validation_samples", type=int, default=-1)
    parser.add_argument("--sfner_grpo_beta", type=float, default=None)
    parser.add_argument("--sfner_grpo_temperature", type=float, default=None)
    parser.add_argument("--sfner_grpo_eval_steps", type=int, default=None)
    parser.add_argument("--sfner_grpo_max_steps", type=int, default=None)

    args = parser.parse_args()
    missing = [f"--{name}" for name in REQUIRED_BY_TASK[args.task] if getattr(args, name) is None]
    if missing:
        parser.error(f"--task {args.task} requires: {', '.join(missing)}")
    return args

if __name__ == "__main__":
    print("[INFO] Initializing run...")
    args = parse_args()
    np.random.seed(args.random_seed)
    wandb_name = f"{args.label}_{args.task}_{args.mode}"
    wandb.init(project="UniversalMedNER", name=wandb_name)

    print("[INFO] Preparing dataset...")
    detok = sacremoses.MosesDetokenizer(lang="en")
    raw_ds = datasets.load_dataset(args.dataset_repo)
    if args.max_raw_samples != -1:
        raw_ds["train"] = raw_ds["train"].select(range(args.max_raw_samples))
    if args.task == "ner":
        create_ds_fn = dc.create_ner_sft_ds if args.mode == "sft" else dc.create_ner_grpo_ds
        pile_ds = create_ds_fn(
            ds=raw_ds,
            max_entities=args.ner_max_entities,
            detok=detok,
            max_negatives=args.ner_max_negatives,
        )
    else:
        create_ds_fn = dc.create_sfner_sft_ds if args.mode == "sft" else dc.create_sfner_grpo_ds
        pile_ds = create_ds_fn(ds=raw_ds, detok=detok)
    pile_ds = dc.get_split_ds(pile_ds, args.validation_size, args.test_size, args.random_seed)
    print("[OK] Dataset prepared:")
    print(pile_ds)

    if args.task == "ner":
        print("[INFO] Computing negative-entity statistics...")
        negative_stats = dc.compute_ner_negative_stats(pile_ds["train"])
        print("[OK] Negative-entity statistics:")
        print(negative_stats)

    print("[INFO] Loading model...")
    processor = transformers.AutoProcessor.from_pretrained(args.model_repo, backend="pil")
    if args.mode == "sft":
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
    if args.task == "ner":
        sft_kwargs = dict(
            learning_rate=args.ner_sft_learning_rate,
            lora_rank=args.ner_sft_lora_rank,
            max_train_samples=args.ner_sft_max_train_samples,
            max_validation_samples=args.ner_sft_max_validation_samples,
            batch_size=args.ner_sft_batch_size,
            gradient_accumulation_steps=args.ner_sft_gradient_accumulation_steps,
            num_epochs=args.ner_sft_num_epochs,
            save_steps=args.ner_sft_save_steps,
            eval_steps=args.ner_sft_eval_steps,
            max_steps=args.ner_sft_max_steps,
        )
        grpo_kwargs = dict(
            learning_rate=args.ner_grpo_learning_rate,
            lora_rank=args.ner_grpo_lora_rank,
            max_train_samples=args.ner_grpo_max_train_samples,
            max_validation_samples=args.ner_grpo_max_validation_samples,
            batch_size=args.ner_grpo_batch_size,
            gradient_accumulation_steps=args.ner_grpo_gradient_accumulation_steps,
            num_generations=args.ner_grpo_num_generations,
            max_completion_length=args.ner_grpo_max_completion_length,
            num_epochs=args.ner_grpo_num_epochs,
            beta=args.ner_grpo_beta,
            temperature=args.ner_grpo_temperature,
            reward_funcs=ec.NER_REWARD_FUNCTIONS[args.ner_reward_fn],
            eval_steps=args.ner_grpo_eval_steps,
            max_steps=args.ner_grpo_max_steps,
        )
    else:
        sft_kwargs = dict(
            learning_rate=args.sfner_sft_learning_rate,
            lora_rank=args.sfner_sft_lora_rank,
            max_train_samples=args.sfner_sft_max_train_samples,
            max_validation_samples=args.sfner_sft_max_validation_samples,
            batch_size=args.sfner_sft_batch_size,
            gradient_accumulation_steps=args.sfner_sft_gradient_accumulation_steps,
            num_epochs=args.sfner_sft_num_epochs,
            save_steps=args.sfner_sft_save_steps,
            eval_steps=args.sfner_sft_eval_steps,
            max_steps=args.sfner_sft_max_steps,
        )
        grpo_kwargs = dict(
            learning_rate=args.sfner_grpo_learning_rate,
            lora_rank=args.sfner_grpo_lora_rank,
            max_train_samples=args.sfner_grpo_max_train_samples,
            max_validation_samples=args.sfner_grpo_max_validation_samples,
            batch_size=args.sfner_grpo_batch_size,
            gradient_accumulation_steps=args.sfner_grpo_gradient_accumulation_steps,
            num_generations=args.sfner_grpo_num_generations,
            max_completion_length=args.sfner_grpo_max_completion_length,
            num_epochs=args.sfner_grpo_num_epochs,
            beta=args.sfner_grpo_beta,
            temperature=args.sfner_grpo_temperature,
            reward_funcs=ec.SFNER_REWARD_FUNCTIONS[args.sfner_reward_fn],
            eval_steps=args.sfner_grpo_eval_steps,
            max_steps=args.sfner_grpo_max_steps,
        )

    if args.mode == "sft":
        trainer = trc.execute_sft(
            model, processor, pile_ds,
            save_folder=args.save_folder,
            target_modules=TARGET_MODULES[args.target_modules],
            **sft_kwargs,
        )
    else:
        trainer = trc.execute_grpo(
            model, processor, pile_ds,
            save_folder=args.save_folder,
            target_modules=TARGET_MODULES[args.target_modules],
            **grpo_kwargs,
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
