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

TARGET_MODULES = {
    "attention_only": [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    "all_linear": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=Path)
    parser.add_argument("--dataset_repo", type=str, default="disi-unibo-nlp/Pile-NER-biomed-IOB")
    parser.add_argument("--model_repo", type=str, default="google/medgemma-1.5-4b-it")
    parser.add_argument("--checkpoint_repo", type=str, default="frc00/UniversalMedNER")
    parser.add_argument("--checkpoint_save_folder", type=str)
    parser.add_argument("--checkpoint_load_folder", type=str, default=None)
    parser.add_argument("--label", type=str)
    parser.add_argument("--mode", type=str, choices=["sft", "grpo"], default="sft")
    parser.add_argument(
        "--reward_fn",
        type=str,
        choices=list(ec.REWARD_FUNCTIONS.keys()),
        default="structured",
        help="Which GRPO reward function to use (only relevant for --mode grpo): "
             "'structured' (level-based JSON/keys/extraction-quality reward) or "
             "'soft_f1' (IoU-based soft micro F1).",
    )

    # Hyperparameters: no defaults here. src/run_full_pipeline.sh is the single
    # source of truth for hyperparameter defaults and always passes every one of
    # these explicitly.
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--validation_size", type=float, required=True)
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--max_entities", type=int, required=True)
    parser.add_argument("--target_modules", type=str, choices=["attention_only", "all_linear"], required=True)
    parser.add_argument("--max_negatives", type=int, default=None)
    parser.add_argument(
        "--max_raw_samples",
        type=int,
        default=None,
        help="If set, subset the raw dataset to this many rows immediately after "
             "download, before dataset formatting/splitting. Lets a run skip "
             "mapping over the full dataset when only a handful of samples are "
             "actually going to be used (e.g. --sft_max_train_samples).",
    )

    parser.add_argument("--sft_learning_rate", type=float, required=True)
    parser.add_argument("--sft_lora_rank", type=int, required=True)
    parser.add_argument("--sft_batch_size", type=int, required=True)
    parser.add_argument("--sft_gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--sft_num_epochs", type=int, required=True)
    parser.add_argument("--sft_max_train_samples", type=int, default=None)
    parser.add_argument("--sft_max_validation_samples", type=int, default=None)
    parser.add_argument("--sft_save_steps", type=int, required=True)
    parser.add_argument("--sft_eval_steps", type=int, required=True)
    parser.add_argument(
        "--sft_max_steps",
        type=int,
        required=True,
        help="Cap on optimizer steps, overriding --sft_num_epochs once reached. "
             "-1 disables the cap, matching TRL's own default.",
    )

    parser.add_argument("--grpo_learning_rate", type=float, required=True)
    parser.add_argument("--grpo_lora_rank", type=int, required=True)
    parser.add_argument("--grpo_batch_size", type=int, required=True)
    parser.add_argument("--grpo_gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--grpo_num_generations", type=int, required=True)
    parser.add_argument("--grpo_max_completion_length", type=int, required=True)
    parser.add_argument("--grpo_num_epochs", type=int, required=True)
    parser.add_argument("--grpo_max_train_samples", type=int, default=None)
    parser.add_argument("--grpo_max_validation_samples", type=int, default=None)
    parser.add_argument("--grpo_beta", type=float, required=True)
    parser.add_argument("--grpo_temperature", type=float, required=True)
    parser.add_argument("--grpo_eval_steps", type=int, required=True)
    parser.add_argument(
        "--grpo_max_steps",
        type=int,
        required=True,
        help="Cap on optimizer steps, overriding --grpo_num_epochs once reached. "
             "-1 disables the cap, matching TRL's own default.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    print("[INFO] Initializing run...")
    args = parse_args()
    np.random.seed(args.random_seed)
    wandb_name = f"{args.label}_{args.mode}"
    wandb.init(project="UniversalMedNER", name=wandb_name)

    print("[INFO] Preparing dataset...")
    detok = sacremoses.MosesDetokenizer(lang="en")
    raw_ds = datasets.load_dataset(args.dataset_repo)
    if args.max_raw_samples:
        raw_ds["train"] = raw_ds["train"].select(range(args.max_raw_samples))
    create_ds_fn = dc.create_sft_ds if args.mode == "sft" else dc.create_grpo_ds
    pile_ds = create_ds_fn(
        ds=raw_ds,
        max_entities=args.max_entities,
        detok=detok,
        random_seed=args.random_seed,
        max_negatives=args.max_negatives,
    )
    pile_ds = dc.get_split_ds(pile_ds, args.validation_size, args.test_size, args.random_seed)
    print("[OK] Dataset prepared:")
    print(pile_ds)

    print("[INFO] Computing negative-entity statistics...")
    negative_stats = dc.compute_negative_stats(pile_ds["train"])
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
        local_adapter_dir = huggingface_hub.snapshot_download(
            repo_id=args.checkpoint_repo,
            allow_patterns=f"{args.checkpoint_load_folder}/*"
        )
        local_adapter_dir = Path(local_adapter_dir) / args.checkpoint_load_folder
        model = peft.PeftModel.from_pretrained(
            base_model,
            local_adapter_dir
        )
        model = model.merge_and_unload()
    model.eval()
    print('[OK] Model loaded on device:', next(model.parameters()).device)
    print(f"[INFO] Tokenizer EOS: {processor.tokenizer.eos_token_id}")
    print(f"[INFO] Generation config EOS: {model.generation_config.eos_token_id}")
    # Loading/merging a LoRA adapter can silently drop or reset generation_config.
    # Fail fast here instead of training/evaluating with a model that never learns
    # to stop at the chat template's turn boundary.
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

    print("[INFO] Starting training...")
    if args.mode == "sft":
        trainer = trc.execute_sft(
            model,
            processor,
            pile_ds,
            save_folder=args.save_folder,
            learning_rate=args.sft_learning_rate,
            lora_rank=args.sft_lora_rank,
            target_modules=TARGET_MODULES[args.target_modules],
            max_train_samples=args.sft_max_train_samples,
            max_validation_samples=args.sft_max_validation_samples,
            batch_size=args.sft_batch_size,
            gradient_accumulation_steps=args.sft_gradient_accumulation_steps,
            num_epochs=args.sft_num_epochs,
            save_steps=args.sft_save_steps,
            eval_steps=args.sft_eval_steps,
            max_steps=args.sft_max_steps,
        )
    else:
        trainer = trc.execute_grpo(
            model,
            processor,
            pile_ds,
            save_folder=args.save_folder,
            learning_rate=args.grpo_learning_rate,
            lora_rank=args.grpo_lora_rank,
            target_modules=TARGET_MODULES[args.target_modules],
            max_train_samples=args.grpo_max_train_samples,
            max_validation_samples=args.grpo_max_validation_samples,
            batch_size=args.grpo_batch_size,
            gradient_accumulation_steps=args.grpo_gradient_accumulation_steps,
            num_generations=args.grpo_num_generations,
            max_completion_length=args.grpo_max_completion_length,
            num_epochs=args.grpo_num_epochs,
            beta=args.grpo_beta,
            temperature=args.grpo_temperature,
            reward_fn=args.reward_fn,
            eval_steps=args.grpo_eval_steps,
            max_steps=args.grpo_max_steps,
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