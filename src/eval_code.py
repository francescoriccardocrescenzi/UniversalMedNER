"""Inference and evaluation code.

This module contains functions to run NER inference with MedGemma 
and to compute micro F1 over its outputs.
"""

# --- IMPORTS ---

from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from scipy import optimize as sp_optimize
import torch


# --- METRIC COMPUTATION HELPERS ---

def tokenize(s):
    """Split entity span into a set lowercase words."""
    return set(s.lower().split())

def iou(a, b):
    """Compute IoU between two entity spans.

    Apply `tokenize` to obtain token sets for both.
    """
    A, B = tokenize(a), tokenize(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def compute_entity_counts(gt_list, pred_list):
    """Compute TP, GT_count, PRED_count for a GT and a predicted entity span.

    Pass a list of ground truth spans and a list of predicted spans associated with the
    same entity. A matching is found between the two lists that maximizes total IoU.
    Then the followuing numbers are returned:
        - TP = total IoU
        - GT_count = number of GT spans
        - PRED_count = number of predicted spans
    """
    GT_count = len(gt_list)
    PRED_count = len(pred_list)

    if GT_count == 0 or PRED_count == 0:
        # Do not attempt to match if there are no ground truth spans or no
        # predicted spans. In case that happens, short circuit execution
        return 0, GT_count, PRED_count
    else:
        # Compute all IoUs to guide matching algorithm
        iou_matrix = np.zeros((GT_count, PRED_count))
        for i, gt in enumerate(gt_list):
            for j, pred in enumerate(pred_list):
                iou_matrix[i, j] = iou(gt, pred)

        # Find maximum matching
        row_ind, col_ind = sp_optimize.linear_sum_assignment(-iou_matrix)

        # Compute total IoU in maximum matching
        TP = sum(iou_matrix[i, j] for i, j in zip(row_ind, col_ind))

        return TP, GT_count, PRED_count

def compute_sample_counts(gt_json, pred_json):
    """
    Compute TP, GT_count, PRED_count, json_error_flag for a GT NER json and a predicted NER json.

    Try to parse the jsons. If this is impossible return null metrics and a true json_error_flag variable.
    Otherwise, compute TP, GT_count, PRED_count for each entity by passing the associated
    spans to `compute_entity_counts`. Then sum the results and return them together with a false json_error_flag
    variable.
    """
    try:
        pred = json.loads(pred_json)
        gt = json.loads(gt_json)

        if not isinstance(pred, dict) or not isinstance(gt, dict):
            raise ValueError("Loaded JSON must be a dict")

        TP_total = 0
        GT_total = 0
        PRED_total = 0

        entity_types = set(gt.keys()) | set(pred.keys())

        for t in entity_types:
            tp, gt_n, pred_n = compute_entity_counts(
                gt.get(t, []),
                pred.get(t, [])
            )

            TP_total += tp
            GT_total += gt_n
            PRED_total += pred_n

        return TP_total, GT_total, PRED_total, 0
    except (json.JSONDecodeError, ValueError):
        return 0, 0, 0, 1
    

# --- INFERENCE AND EVALUATION ---

def run_batched_inference(model, processor, prompts, max_new_tokens=200):
    """Run model inference on a batch of samples."""

    # Apply chat template per sample
    texts = [
        processor.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False
        )
        for prompt in prompts
    ]

    # Tokenize batch
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    # Batch decode
    decoded = processor.tokenizer.batch_decode(
        outputs,
        skip_special_tokens=False
    )

    responses = []

    # Cleanup generated text
    for text in decoded:
        text = text.split("<start_of_turn>model\n", 1)[-1]
        text = text.split("<end_of_turn>", 1)[0]
        responses.append(text.strip())

    # cleanup
    del inputs
    del outputs
    torch.cuda.empty_cache()

    return responses

def test_model_on_batch(model, processor, pile_ds, split="train", indices=list(range(8))):
    """Run model inference on a batch of samples and print model-response pairs.

    Mainly useful for debugging purposes.
    """
    if model.is_gradient_checkpointing:
        model.gradient_checkpointing_disable()

    prompts = [
        pile_ds[split][i]["messages"][:-1]
        for i in indices
    ]

    responses = run_batched_inference(model, processor, prompts)

    for i, (p, r) in enumerate(zip(prompts, responses)):
        print(f"\n=== SAMPLE {i} ===")
        print(f"PROMPT:\n{p}")
        print(f"RESPONSE:\n{r}")

def evaluate_dataset(
    model,
    processor,
    ds,
    batch_size=8,
    split="test",
    max_samples=None
):
    """Run inference on dataset and return micro F1 and number of json parsing errors encountered.

    You may specify a split, a batch size and an optional maximum number of samples.
    The metrics are saved to disk in the specified folder.
    """
    # Prepare model for inference
    if model.is_gradient_checkpointing:
        model.gradient_checkpointing_disable()

    # Set up accumulators
    TP_total = 0
    GT_total = 0
    PRED_total = 0
    json_errors_total = 0

    # Prepare iteration
    data = ds[split]
    n = len(data)
    if max_samples is not None:
        n = min(n, max_samples)

    for start in tqdm(range(0, n, batch_size), desc="Running inference..."):
        # Define sample indices for the batch
        batch_indices = list(range(start, min(start + batch_size, n)))
        # Select rows
        batch_rows = [data[i] for i in batch_indices]
        # Extract prompts and ground truth labels
        prompts = [row["messages"][:-1] for row in batch_rows]
        gt_jsons = [row["messages"][-1]["content"] for row in batch_rows]
        # Compute predictions
        preds = run_batched_inference(model, processor, prompts)
        # Remove fences added by pretrained model
        preds = [
            s.strip()
            .removeprefix("```json").removeprefix("```")
            .removesuffix("```")
            .strip()
            for s in preds
        ]
        # Compute and accumulate counts
        for gt_json, pred in zip(gt_jsons, preds):
            tp, gt_c, pred_c, err = compute_sample_counts(gt_json, pred)
            TP_total += tp
            GT_total += gt_c
            PRED_total += pred_c
            json_errors_total += err

    # Compute F1
    f1 = 2 * TP_total / (GT_total + PRED_total) if GT_total + PRED_total > 0 else 0

    # Wrap into dictionary
    metric_dict = {
        'TP_total': float(TP_total),
        'GT_total': int(GT_total),
        'PRED_total': int(PRED_total),
        'F1': float(f1),
        'json_errors_total': int(json_errors_total)
    }

    return metric_dict


# --- GRPO REWARD ---

# --- GRPO REWARD ---

def compute_sample_f1(gt_json, pred_json):
    """Compute micro F1 for a single sample."""
    TP, GT, PRED, json_error = compute_sample_counts(gt_json, pred_json)

    if json_error or GT + PRED == 0:
        return 0.0
    else:
        return 2 * TP / (GT + PRED)

def grpo_reward_fn(prompts, completions, answer, **kwargs):
    return [
        compute_sample_f1(gt_json, completion[0]["content"])
        for completion, gt_json in zip(completions, answer)
    ]