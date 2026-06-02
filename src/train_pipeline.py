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
    parser.add_argument("--checkpoint_folder", type=str)
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
    wandb.init(project="UniversalMedNER", name=args.label)
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

    print("[INFO] Starting training...")
    processor = transformers.AutoProcessor.from_pretrained(args.model_repo, backend="pil")

    if args.mode == "sft":
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            args.model_repo,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        print("[OK] Model loaded on device:", next(model.parameters()).device)
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
            args.model_repo,
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
        path_in_repo=args.checkpoint_folder,
    )
    shutil.rmtree(tmp_dir)
    print("[OK] Best checkpoint pushed to:", f"{args.checkpoint_repo}/{args.checkpoint_folder}")