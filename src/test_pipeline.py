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
    parser.add_argument('--checkpoint_folder', type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    print('[INFO] Initializing run...')
    args = parse_args()
    with open(args.hyperparam_path, 'r') as f:
        hyperparam_raw = json.load(f)
    hyperparam = types.SimpleNamespace(**{**hyperparam_raw["shared"], **hyperparam_raw["test"]})
    np.random.seed(hyperparam.random_seed)
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
    processor = transformers.AutoProcessor.from_pretrained(
        args.model_repo,
        backend="pil"
    )
    base_model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model_repo,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16
    )
    if args.checkpoint_folder is not None:
        local_adapter_dir = huggingface_hub.snapshot_download(
            repo_id=args.checkpoint_repo,
            allow_patterns=f"{args.checkpoint_folder}/*"
        )
        local_adapter_dir = Path(local_adapter_dir) / args.checkpoint_folder
        model = peft.PeftModel.from_pretrained(
            base_model,
            local_adapter_dir
        )
    else:
        model = base_model
    model.eval()
    print('[OK] Model loaded on device:')
    print(next(model.parameters()).device)
    
    if args.verbose:
        print('[INFO] Testing model on single batch...')
        print(" **** Test best checkpoint of fine-tuned model on a single batch ****")
        evc.test_model_on_batch(model, processor, pile_ds, indices=list(range(8)))
        print('[OK] Tested model on single batch')
    
    print('[INFO] Evaluating model on test set...')
    metric_dict = evc.evaluate_dataset(
        model, 
        processor, 
        pile_ds, 
        batch_size=hyperparam.batch_size, 
        split="test", 
        max_samples=hyperparam.max_test_samples, 
    )
    print('[OK] Evaluated model on test set:')
    print(metric_dict)

    print('[INFO] Saving metrics to disk...')
    with open(args.metrics_path, "w") as f:
        json.dump(metric_dict, f, indent=4)
    print('[OK] Metrics saved to:', args.metrics_path)
