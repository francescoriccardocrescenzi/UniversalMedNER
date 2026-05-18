import argparse
import os
from pathlib import Path
import numpy as np
import json
import types

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
    parser.add_argument('--label', type=str, default='test')
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    # --- INITIALIZE RUN ---
    print(' **** Initializing run ****')
    args = parse_args()
    run_root = Path('data') 
    run_dir = run_root / args.label
    with open(run_dir / 'hyperparam.json', 'r') as f:
        hyperparam = types.SimpleNamespace(**json.load(f))
    np.random.seed(hyperparam.random_seed)
    wandb.init(project="UniversalMedNER", name=args.label)

    # --- PREPARE DATASET ---
    print(' **** Preparing dataset ****')
    detok = sacremoses.MosesDetokenizer(lang="en")
    pile_id = "disi-unibo-nlp/Pile-NER-biomed-IOB"
    pile_ds = dc.create_sft_ds(
        ds=datasets.load_dataset(pile_id),
        max_entities=hyperparam.max_entities,
        detok=detok,
        random_seed=hyperparam.random_seed
    )
    pile_ds = dc.get_split_ds(pile_ds, hyperparam.validation_size, hyperparam.test_size, hyperparam.random_seed)

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
    best_gemma = trc.execute_sft(
        gemma, 
        gemma_processor, 
        pile_ds, 
        save_folder=run_dir, 
        learning_rate=hyperparam.learning_rate,
        lora_rank=hyperparam.lora_rank,
        target_modules=TARGET_MODULES[hyperparam.target_modules],
        max_train_samples=hyperparam.max_train_samples,
        max_validation_samples=hyperparam.max_validation_samples,
    )

    # --- TEST BEST MODEL ---
    # Test fine-tuned model on a single batch
    if args.verbose:
        print(" **** Test best checkpoint of fine-tuned model on a single batch ****")
        tsc.test_model_on_batch(best_gemma, gemma_processor, pile_ds, indices=list(range(8)))
    # Test on whole test set
    print(" **** Compute dataset metrics ****")
    metric_dict = tsc.evaluate_dataset(
        best_gemma, 
        gemma_processor, 
        pile_ds, 
        batch_size=8, 
        split="test", 
        max_samples=hyperparam.max_test_samples, 
        save_folder=run_dir
    )
    print(metric_dict)