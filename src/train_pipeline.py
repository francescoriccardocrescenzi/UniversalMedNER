import argparse
from pathlib import Path
import numpy as np
import json
import types
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
    parser.add_argument("--hyperparam_path", type=Path)
    parser.add_argument("--save_folder", type=Path)
    parser.add_argument("--dataset_repo", type=str, default="disi-unibo-nlp/Pile-NER-biomed-IOB")
    parser.add_argument("--model_repo", type=str, default="google/medgemma-1.5-4b-it")
    parser.add_argument("--checkpoint_repo", type=str, default="frc00/UniversalMedNER")
    parser.add_argument("--checkpoint_save_folder", type=str)
    parser.add_argument("--checkpoint_load_folder", type=str, default=None)
    parser.add_argument("--label", type=str)
    parser.add_argument("--mode", type=str, choices=["sft", "grpo"], default="sft")
    return parser.parse_args()

if __name__ == "__main__":
    print("[INFO] Initializing run...")
    args = parse_args()
    run_root = Path("data")
    run_dir = run_root / args.label
    with open(run_dir / "hyperparam.json", "r") as f:
        hyperparam_raw = json.load(f)
    hyperparam = types.SimpleNamespace(**{**hyperparam_raw["shared"], **hyperparam_raw[args.mode]})
    np.random.seed(hyperparam.random_seed)
    wandb_name = f"{args.label}_{args.mode}"
    wandb.init(project="UniversalMedNER", name=wandb_name)
    print("[OK] Hyperparameters loaded:")
    print(hyperparam)

    print("[INFO] Preparing dataset...")
    detok = sacremoses.MosesDetokenizer(lang="en")
    create_ds_fn = dc.create_sft_ds if args.mode == "sft" else dc.create_grpo_ds
    pile_ds = create_ds_fn(
        ds=datasets.load_dataset(args.dataset_repo),
        max_entities=hyperparam.max_entities,
        detok=detok,
        random_seed=hyperparam.random_seed,
    )
    pile_ds = dc.get_split_ds(pile_ds, hyperparam.validation_size, hyperparam.test_size, hyperparam.random_seed)
    print("[OK] Dataset prepared:")
    print(pile_ds)

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
            learning_rate=hyperparam.learning_rate,
            lora_rank=hyperparam.lora_rank,
            target_modules=TARGET_MODULES[hyperparam.target_modules],
            max_train_samples=hyperparam.max_train_samples,
            max_validation_samples=hyperparam.max_validation_samples,
            batch_size=hyperparam.batch_size,
            gradient_accumulation_steps=hyperparam.gradient_accumulation_steps,
            num_epochs=hyperparam.num_epochs,
        )
    else:
        trainer = trc.execute_grpo(
            model,
            processor,
            pile_ds,
            save_folder=args.save_folder,
            learning_rate=hyperparam.learning_rate,
            lora_rank=hyperparam.lora_rank,
            target_modules=TARGET_MODULES[hyperparam.target_modules],
            max_train_samples=hyperparam.max_train_samples,
            max_validation_samples=hyperparam.max_validation_samples,
            batch_size=hyperparam.batch_size,
            gradient_accumulation_steps=hyperparam.gradient_accumulation_steps,
            num_generations=hyperparam.num_generations,
            max_completion_length=hyperparam.max_completion_length,
            num_epochs=hyperparam.num_epochs,
            beta=hyperparam.beta,
            temperature=hyperparam.temperature,
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