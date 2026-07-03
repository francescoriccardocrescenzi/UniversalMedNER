"""Inference and evaluation code.

This module contains functions to run NER inference with MedGemma 
and to compute micro F1 over its outputs.
"""

# --- IMPORTS ---

from pathlib import Path
from collections import Counter
import json
from tqdm import tqdm
import numpy as np
from scipy import optimize as sp_optimize
import torch


# --- METRIC COMPUTATION HELPERS ---

F1_MODES = ("soft", "strict")

def tokenize(s):
    """Split entity span into a set lowercase words."""
    return set(str(s).lower().split())

def lower_keys(d):
    """Lowercase a dict's keys so entity-type matching is case-insensitive.

    Span lists from keys that collide after lowercasing (e.g. "Disease" and
    "DISEASE") are concatenated rather than one silently overwriting the other.
    Values that aren't lists are left for the caller's own validation to reject.
    """
    merged = {}
    for k, v in d.items():
        lk = str(k).lower()
        if lk in merged and isinstance(merged[lk], list) and isinstance(v, list):
            merged[lk] = merged[lk] + v
        else:
            merged[lk] = v
    return merged

def iou(a, b, mode="soft"):
    """Compute a match score between two entity spans.

    Apply `tokenize` to obtain token sets for both. In "soft" mode the score is
    the IoU of the two token sets, allowing partial credit. In "strict" mode
    the score is 1.0 if the token sets are identical and 0.0 otherwise.
    """
    A, B = tokenize(a), tokenize(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    if mode == "strict":
        return 1.0 if A == B else 0.0
    return len(A & B) / len(A | B)

def compute_entity_counts(gt_list, pred_list, mode="soft"):
    """Compute TP, GT_count, PRED_count for a GT and a predicted entity span.

    Pass a list of ground truth spans and a list of predicted spans associated with the
    same entity. A matching is found between the two lists that maximizes total score,
    where the pairwise score is given by `iou(mode=mode)`. Then the following numbers
    are returned:
        - TP = total matched score (total IoU in "soft" mode, number of exact matches
          in "strict" mode)
        - GT_count = number of GT spans
        - PRED_count = number of predicted spans
    """
    GT_count = len(gt_list)
    PRED_count = len(pred_list)

    if GT_count == 0 or PRED_count == 0:
        # Do not attempt to match if there are no ground truth spans or no
        # predicted spans. In case that happens, short circuit execution
        return 0, GT_count, PRED_count

    if mode == "strict":
        # Exact-match scores are binary and equality of token sets is
        # transitive, so the matching graph decomposes into independent
        # cliques, one per distinct normalized span. The maximum matching
        # size is then just the multiset-intersection cardinality between
        # normalized GT and predicted spans, computed in linear time -
        # no need for the general O(n^3) Hungarian algorithm below.
        gt_counts = Counter(frozenset(tokenize(s)) for s in gt_list)
        pred_counts = Counter(frozenset(tokenize(s)) for s in pred_list)
        TP = sum(
            min(gt_counts[k], pred_counts[k])
            for k in gt_counts.keys() & pred_counts.keys()
        )
        return TP, GT_count, PRED_count

    # Compute all pairwise scores to guide matching algorithm
    score_matrix = np.zeros((GT_count, PRED_count))
    for i, gt in enumerate(gt_list):
        for j, pred in enumerate(pred_list):
            score_matrix[i, j] = iou(gt, pred, mode=mode)

    # Find maximum matching
    row_ind, col_ind = sp_optimize.linear_sum_assignment(-score_matrix)

    # Compute total score in maximum matching
    TP = sum(score_matrix[i, j] for i, j in zip(row_ind, col_ind))

    return TP, GT_count, PRED_count

def compute_sample_counts(gt_json, pred_json, modes=F1_MODES):
    """
    Compute TP, GT_count, PRED_count per F1 mode, and a json_error_flag, for a GT
    NER json and a predicted NER json.

    Try to parse the jsons. If this is impossible return null metrics and a true
    json_error_flag variable. Otherwise, compute TP, GT_count, PRED_count for each
    entity and each requested mode by passing the associated spans to
    `compute_entity_counts`. Then sum the results and return them, keyed by mode,
    together with a false json_error_flag variable.
    """
    try:
        pred = json.loads(pred_json)
        gt = json.loads(gt_json)

        if not isinstance(pred, dict) or not isinstance(gt, dict):
            raise ValueError("Loaded JSON must be a dict")

        # Entity-type keys are matched case-insensitively, same as span matching.
        gt = lower_keys(gt)
        pred = lower_keys(pred)

        totals = {mode: [0, 0, 0] for mode in modes}

        entity_types = set(gt.keys()) | set(pred.keys())

        for t in entity_types:
            gt_spans = gt.get(t, [])
            pred_spans = pred.get(t, [])

            if not isinstance(gt_spans, list) or not isinstance(pred_spans, list):
                raise ValueError("Entity spans must be a list")

            for mode in modes:
                tp, gt_n, pred_n = compute_entity_counts(gt_spans, pred_spans, mode=mode)
                totals[mode][0] += tp
                totals[mode][1] += gt_n
                totals[mode][2] += pred_n

        return {mode: tuple(v) for mode, v in totals.items()}, 0
    except (json.JSONDecodeError, ValueError):
        return {mode: (0, 0, 0) for mode in modes}, 1
    

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
    max_samples=None,
    modes=F1_MODES
):
    """Run inference on dataset and return micro F1 (per requested mode) and number
    of json parsing errors encountered.

    You may specify a split, a batch size, an optional maximum number of samples, and
    which F1 mode(s) to compute ("soft", "strict", or both). The metrics are saved to
    disk in the specified folder.
    """
    # Prepare model for inference
    if model.is_gradient_checkpointing:
        model.gradient_checkpointing_disable()

    # Set up accumulators
    totals = {mode: [0, 0, 0] for mode in modes}
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
            counts, err = compute_sample_counts(gt_json, pred, modes=modes)
            for mode in modes:
                tp, gt_c, pred_c = counts[mode]
                totals[mode][0] += tp
                totals[mode][1] += gt_c
                totals[mode][2] += pred_c
            json_errors_total += err

    # Wrap into dictionary
    metric_dict = {'json_errors_total': int(json_errors_total)}
    for mode in modes:
        TP_total, GT_total, PRED_total = totals[mode]
        f1 = 2 * TP_total / (GT_total + PRED_total) if GT_total + PRED_total > 0 else 0
        metric_dict[f'TP_total_{mode}'] = float(TP_total)
        metric_dict[f'GT_total_{mode}'] = int(GT_total)
        metric_dict[f'PRED_total_{mode}'] = int(PRED_total)
        metric_dict[f'F1_{mode}'] = float(f1)

    return metric_dict


# --- GRPO REWARD ---

def compute_sample_f1(gt_json, pred_json, mode="soft"):
    """Compute micro F1 for a single sample, using the given F1 mode."""
    counts, json_error = compute_sample_counts(gt_json, pred_json, modes=(mode,))
    TP, GT, PRED = counts[mode]

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

def soft_f1_reward_fn(prompts, completions, answer, **kwargs):
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


# --- STRUCTURED REWARD ---
# Implements the level-based reward described in
# data/reward_function_pseudocode.md: JSON validity, then correct entity-type
# keys, then per-type extraction quality (exact-match precision/recall).

STRUCTURED_REWARD_BONUS = 1.0  # B in the pseudocode

def _has_correct_structure(d):
    """Check that `d` is a Dict[str -> List[str]]."""
    if not isinstance(d, dict):
        return False
    for k, v in d.items():
        if not isinstance(k, str) or not isinstance(v, list):
            return False
        if not all(isinstance(e, str) for e in v):
            return False
    return True

def _compute_extraction_score(t_list, o_list):
    """Per-type extraction score for a positive entity type (exact match).

    +1/N for each correctly predicted GT span (no duplicates), -1/N for each
    wrong or duplicate prediction, -1/N for each GT span never predicted.
    Matching is case-insensitive, consistent with strict-mode F1.
    """
    n = len(t_list)
    t_list = [str(s).lower() for s in t_list]
    score = 0.0
    matches = []

    for oe in o_list:
        oe = str(oe).lower()
        if oe in t_list and oe not in matches:
            matches.append(oe)
            score += 1 / n
        else:
            score -= 1 / n

    n_missed = n - len(matches)
    score -= n_missed / n

    return score

def compute_structured_reward(gt_json, pred_json, bonus=STRUCTURED_REWARD_BONUS):
    """Compute the structured reward for a single sample.

    GT is assumed to always be well-formed (Dict[str -> List[str]], one entry
    per expected entity type, empty list for negative types).
    """
    R = 0.0

    # Level 1: JSON parsing
    try:
        out_dict = json.loads(pred_json)
    except json.JSONDecodeError:
        return R
    if not _has_correct_structure(out_dict):
        return R
    R += 1.0

    # Entity-type keys are matched case-insensitively, same as span matching.
    out_dict = lower_keys(out_dict)
    gt = lower_keys(json.loads(gt_json))
    ent_types = list(gt.keys())
    n_types = len(ent_types)

    # Level 2: key matching
    for key in out_dict.keys():
        if key not in ent_types:
            return R          # wrong key found: stop here, no bonus, no quality component
        R += 1 / n_types    # partial credit per correct key, normalized by expected count

    # Missing keys: all output keys passed the check above, so if the counts
    # differ the model omitted some expected types. Cap reward here.
    if len(out_dict) != n_types:
        return R

    R += bonus  # margin for the possible negative signal from the quality component

    # Level 3: extraction quality
    Q_R = 0.0
    for t in ent_types:
        t_list = gt[t]
        o_list = out_dict[t]  # safe: exact key match enforced above

        if len(t_list) == 0:
            score = 1.0 if len(o_list) == 0 else -1.0
        else:
            score = _compute_extraction_score(t_list, o_list)

        Q_R += score / n_types

    # Clip to prevent quality penalty from erasing the format reward: correct
    # format should always beat wrong format (R=0).
    if bonus + Q_R < 0:
        Q_R = -bonus

    return R + Q_R

def structured_reward_fn(prompts, completions, answer, **kwargs):
    return [
        compute_structured_reward(
            gt_json,
            clean_prediction(completion[0]["content"])
        )
        for completion, gt_json in zip(completions, answer)
    ]


# --- REWARD SELECTION ---

REWARD_FUNCTIONS = {
    "structured": structured_reward_fn,
    "soft_f1": soft_f1_reward_fn,
}