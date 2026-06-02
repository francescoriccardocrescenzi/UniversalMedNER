import argparse
import os
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
import sft_code as sftc

# Using a dict makes it easy to pass lists as arguments
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
    parser.add_argument('--label', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    print('[INFO] Initializing run...')
    args = parse_args()
    run_root = Path('data') 
    run_dir = run_root / args.label
    with open(run_dir / 'hyperparam.json', 'r') as f:
        hyperparam = types.SimpleNamespace(**json.load(f))
    np.random.seed(hyperparam.random_seed)
    wandb.init(project="UniversalMedNER", name=args.label)
    print('[OK] Hyperparameters loaded:')
    print(hyperparam)

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
    processor = transformers.AutoProcessor.from_pretrained(args.model_repo, backend='pil')
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model_repo,
        device_map="cuda:0",
        dtype=torch.bfloat16
    )
    print('[OK] Model loaded on device:')
    print(next(model.parameters()).device)
    
    print('[INFO] Fine-tuning model...]')
    sft_trainer = sftc.execute_sft(
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
        num_epochs=hyperparam.num_epochs
    )
    print('[OK] Model fine-tuned')

    print('[INFO] Pushing best checkpoint to hub...')
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model_repo,
        torch_dtype=torch.bfloat16
    )
    best_ckpt = sft_trainer.state.best_model_checkpoint
    peft_model = peft.PeftModel.from_pretrained(model, best_ckpt)
    tmp_dir = tempfile.mkdtemp(prefix=f"{args.label}_lora_")
    peft_model.save_pretrained(tmp_dir)
    api = huggingface_hub.HfApi()
    api.upload_folder(
        repo_id=args.checkpoint_repo,
        folder_path=tmp_dir,
        path_in_repo=args.label
    )
    shutil.rmtree(tmp_dir)
    print('[OK] Best checkpoint pushed to:', f"{args.checkpoint_repo}/{args.label}")