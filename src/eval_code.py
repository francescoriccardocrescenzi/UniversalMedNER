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
    return set(str(s).lower().split())

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

def clean_prediction(s):
    """Strip markdown code fences from a model prediction."""
    s = s.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    s = s.split("<start_of_turn>model\n", 1)[-1]
    s = s.split("<end_of_turn>", 1)[0]
    return s

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

    # Stop generation on either the tokenizer's EOS or the chat template's end-of-turn
    # marker. Don't rely solely on model.generation_config: it can silently end up
    # missing one of these (e.g. after merging a LoRA adapter), causing generation to
    # run all the way to max_new_tokens instead of stopping at the turn boundary.
    eos_token_id = [
        processor.tokenizer.eos_token_id,
        processor.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
    ]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
        )

    # Batch decode
    decoded = processor.tokenizer.batch_decode(
        outputs,
        skip_special_tokens=False
    )

    responses = []

    # Cleanup generated text
    for text in decoded:
        responses.append(clean_prediction(text))

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
        preds = [clean_prediction(p) for p in preds]
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

def compute_sample_f1(gt_json, pred_json):
    """Compute micro F1 for a single sample."""
    TP, GT, PRED, json_error = compute_sample_counts(gt_json, pred_json)

    if json_error:
        return 0.0
    elif GT + PRED == 0:
        return 1.0
    else:
        return 2 * TP / (GT + PRED)

def inspect_grpo_rewards(prompts, completions, answer):
    import json
    import numpy as np
    from collections import defaultdict

    rewards = []
    group_rewards = defaultdict(list)

    for prompt, completion, gt_json in zip(prompts, completions, answer):
        pred = clean_prediction(completion[0]["content"])
        r = float(compute_sample_f1(gt_json, pred))

        rewards.append(r)
        key = json.dumps(prompt, sort_keys=True)
        group_rewards[key].append(r)

    rewards = np.array(rewards, dtype=np.float32)

    print("\nREWARD DIAGNOSTICS")
    print(f"global_mean = {rewards.mean():.4f}")
    print(f"global_std  = {rewards.std():.4f}")
    print(f"global_min  = {rewards.min():.4f}")
    print(f"global_max  = {rewards.max():.4f}")

    valid_json = sum(
        1 for c in completions
        if _safe_json(clean_prediction(c[0]["content"]))
    )
    print(f"json_valid  = {valid_json/len(completions):.4f}")

    print("\nPER-GROUP STATS")

    group_means = []
    group_stds = []

    for i, (k, vals) in enumerate(group_rewards.items()):
        v = np.array(vals, dtype=np.float32)

        mean = v.mean()
        std = v.std()

        group_means.append(mean)
        group_stds.append(std)

        print("-" * 50)
        print(f"group {i}")
        print(f"size = {len(v)}")
        print(f"mean = {mean:.4f}")
        print(f"std  = {std:.4f}")
        print(f"rewards = {[round(x, 4) for x in v]}")

    group_means = np.array(group_means, dtype=np.float32)
    group_stds = np.array(group_stds, dtype=np.float32)

    print("\nGROUP SUMMARY")
    print(f"mean_of_means = {group_means.mean():.4f}")
    print(f"std_of_means  = {group_means.std():.4f}")
    print(f"mean_of_std   = {group_stds.mean():.4f}")
    print(f"n_groups      = {len(group_means)}")

def _safe_json(s):
    try:
        json.loads(s)
        return True
    except Exception:
        return False

def grpo_reward_fn(prompts, completions, answer, **kwargs):
    # inspect_grpo_rewards(
    #     prompts,
    #     completions,
    #     answer,
    # )

    return [
        compute_sample_f1(
            gt_json,
            clean_prediction(completion[0]["content"])
        )
        for completion, gt_json in zip(completions, answer)
    ]