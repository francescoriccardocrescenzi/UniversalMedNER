import argparse
import os
from pathlib import Path
import numpy as np

import torch
import transformers
import datasets

import sacremoses
import wandb

import dataset_code as dc
import train_code as trc
import test_code as tsc


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
    parser.add_argument("--label", type=str, default="test")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--validation_size", type=float, default=0.02)
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--max_entities", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--target_modules", type=str, default="all_linear")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_validation_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    # --- INITIALIZE RUN ---
    print(' **** Initializing run ****')
    args = parse_args()
    np.random.seed(args.random_seed)
    wandb.init(project="UniversalMedNER", name=args.label)

    # --- PREPARE DATASET ---
    print(' **** Preparing dataset ****')
    detok = sacremoses.MosesDetokenizer(lang="en")
    pile_id = "disi-unibo-nlp/Pile-NER-biomed-IOB"
    pile_ds = dc.create_sft_ds(
        ds=datasets.load_dataset(pile_id),
        max_entities=args.max_entities,
        detok=detok,
        random_seed=args.random_seed
    )
    pile_ds = dc.get_split_ds(pile_ds, args.validation_size, args.test_size, args.random_seed)

    # --- LOAD MODEL ---
    print(' **** Loading model ****')
    gemma_id = "google/medgemma-1.5-4b-it"
    gemma_processor = transformers.AutoProcessor.from_pretrained(gemma_id, backend='pil')
    gemma = transformers.AutoModelForImageTextToText.from_pretrained(
        gemma_id,
        device_map="cuda:0",
        dtype=torch.bfloat16
    )
    print(' **** Model loaded on device ****')
    print(next(gemma.parameters()).device)
    if args.verbose:
        print(' **** Test pretrained model on a single batch before fine-tuning:')
        tsc.test_model_on_batch(gemma, gemma_processor, pile_ds, indices=list(range(8)))

    # --- FINE-TUNE MODEL ---
    print(' **** Starting fine-tuning... ****')
    trc.execute_sft(
        gemma, 
        gemma_processor, 
        pile_ds, 
        save_folder=Path(f'data/{args.label}'), 
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        target_modules=TARGET_MODULES[args.target_modules],
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
    )

    # --- TEST MODEL ---
    # Test fine-tuned model on a single batch
    if args.verbose:
        print(" **** Test fine-tuned model on a single batch ****")
        tsc.test_model_on_batch(gemma, gemma_processor, pile_ds, indices=list(range(8)))
    # Test on whole test set
    print(" **** Compute dataset metrics ****")
    metric_dict = tsc.evaluate_dataset(
        gemma, 
        gemma_processor, 
        pile_ds, 
        batch_size=8, 
        split="test", 
        max_samples=args.max_test_samples, 
        save_folder=Path(f'data/{args.label}')
    )
    print(metric_dict)