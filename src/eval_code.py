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
import pandas as pd
from scipy import optimize as sp_optimize
import torch


# --- METRIC COMPUTATION HELPERS ---

NER_F1_MODES = ("soft", "strict")

def ner_tokenize(s):
    """Split entity span into a set lowercase words."""
    return set(str(s).lower().split())

def lower_ner_keys(d):
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

def ner_iou(a, b, mode="soft"):
    """Compute a match score between two entity spans.

    Apply `ner_tokenize` to obtain token sets for both. In "soft" mode the score is
    the IoU of the two token sets, allowing partial credit. In "strict" mode
    the score is 1.0 if the token sets are identical and 0.0 otherwise.
    """
    A, B = ner_tokenize(a), ner_tokenize(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    if mode == "strict":
        return 1.0 if A == B else 0.0
    return len(A & B) / len(A | B)

def compute_ner_entity_counts(gt_list, pred_list, mode="soft"):
    """Compute TP, GT_count, PRED_count for GT and predicted entity spans.

    Pass a list of ground truth spans and a list of predicted spans associated with the
    same entity. A matching is found between the two lists that maximizes total score,
    where the pairwise score is given by `ner_iou(mode=mode)`. Then the following numbers
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
        gt_counts = Counter(frozenset(ner_tokenize(s)) for s in gt_list)
        pred_counts = Counter(frozenset(ner_tokenize(s)) for s in pred_list)
        TP = sum(
            min(gt_counts[k], pred_counts[k])
            for k in gt_counts.keys() & pred_counts.keys()
        )
        return TP, GT_count, PRED_count

    # Compute all pairwise scores to guide matching algorithm
    score_matrix = np.zeros((GT_count, PRED_count))
    for i, gt in enumerate(gt_list):
        for j, pred in enumerate(pred_list):
            score_matrix[i, j] = ner_iou(gt, pred, mode=mode)

    # Find maximum matching
    row_ind, col_ind = sp_optimize.linear_sum_assignment(-score_matrix)

    # Compute total score in maximum matching
    TP = sum(score_matrix[i, j] for i, j in zip(row_ind, col_ind))

    return TP, GT_count, PRED_count

def compute_ner_sample_counts(gt_json, pred_json, modes=NER_F1_MODES):
    """
    Compute TP, GT_count, PRED_count per F1 mode, and a json_error_flag, for a GT
    NER json and a predicted NER json.

    Try to parse the jsons. If this is impossible return null metrics and a true
    json_error_flag variable. Otherwise, compute TP, GT_count, PRED_count for each
    entity and each requested mode by passing the associated spans to
    `compute_ner_entity_counts`. Then sum the results and return them, keyed by mode,
    together with a false json_error_flag variable.
    """
    try:
        pred = json.loads(pred_json)
        gt = json.loads(gt_json)

        if not isinstance(pred, dict) or not isinstance(gt, dict):
            raise ValueError("Loaded JSON must be a dict")

        # Entity-type keys are matched case-insensitively, same as span matching.
        gt = lower_ner_keys(gt)
        pred = lower_ner_keys(pred)

        totals = {mode: [0, 0, 0] for mode in modes}

        entity_types = set(gt.keys()) | set(pred.keys())

        for t in entity_types:
            gt_spans = gt.get(t, [])
            pred_spans = pred.get(t, [])

            if not isinstance(gt_spans, list) or not isinstance(pred_spans, list):
                raise ValueError("Entity spans must be a list")

            for mode in modes:
                tp, gt_n, pred_n = compute_ner_entity_counts(gt_spans, pred_spans, mode=mode)
                totals[mode][0] += tp
                totals[mode][1] += gt_n
                totals[mode][2] += pred_n

        return {mode: tuple(v) for mode, v in totals.items()}, 0
    except (json.JSONDecodeError, ValueError):
        return {mode: (0, 0, 0) for mode in modes}, 1


# --- INFERENCE AND EVALUATION ---

def clean_ner_prediction(s):
    """Strip markdown code fences from a model prediction."""
    s = s.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    s = s.split("<start_of_turn>model\n", 1)[-1]
    s = s.split("<end_of_turn>", 1)[0]
    return s

def run_ner_batched_inference(model, processor, prompts, max_new_tokens=200, temperature=None, top_p=None):
    """Run model inference on a batch of samples.

    By default generation is greedy (`do_sample=False`), matching prior
    behavior. Passing `temperature` and/or `top_p` switches to sampling with
    those parameters.

    Returns a (responses, truncated) pair: `responses` are the cleaned model
    outputs and `truncated` is a parallel list of booleans flagging samples
    whose generation hit `max_new_tokens` without emitting an end-of-sequence
    token, i.e. the same "clipped" condition GRPOTrainer tracks (its
    `is_truncated = ids[-1] not in eos_and_pad`) for `completions/clipped_ratio`.
    """

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
    prompt_len = inputs["input_ids"].shape[1]

    # Stop generation on either the tokenizer's EOS or the chat template's end-of-turn
    # marker. Don't rely solely on model.generation_config: it can silently end up
    # missing one of these (e.g. after merging a LoRA adapter), causing generation to
    # run all the way to max_new_tokens instead of stopping at the turn boundary.
    eos_token_id = [
        processor.tokenizer.eos_token_id,
        processor.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
    ]

    do_sample = temperature is not None or top_p is not None
    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=eos_token_id,
    )
    if do_sample:
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if top_p is not None:
            generate_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    # Batch decode
    decoded = processor.tokenizer.batch_decode(
        outputs,
        skip_special_tokens=False
    )

    responses = []

    # Cleanup generated text
    for text in decoded:
        responses.append(clean_ner_prediction(text))

    # A generation that stopped naturally has its unfinished rows padded with
    # pad_token_id after the EOS token, so the last generated token is EOS/pad
    # unless the row ran out of budget before ever stopping.
    eos_and_pad = set(eos_token_id) | {processor.tokenizer.pad_token_id}
    truncated = [
        row[-1].item() not in eos_and_pad
        for row in outputs[:, prompt_len:]
    ]

    # cleanup
    del inputs
    del outputs
    torch.cuda.empty_cache()

    return responses, truncated

def test_ner_model_on_batch(model, processor, pile_ds, split="train", indices=None, max_new_tokens=200, temperature=None, top_p=None):
    """Run model inference on a batch of samples and print model-response pairs.

    Mainly useful for debugging purposes.
    """
    if indices is None:
        indices = list(range(8))
    if model.is_gradient_checkpointing:
        model.gradient_checkpointing_disable()

    prompts = [
        pile_ds[split][i]["messages"][:-1]
        for i in indices
    ]

    responses, truncated = run_ner_batched_inference(
        model, processor, prompts, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
    )

    for i, (p, r, t) in enumerate(zip(prompts, responses, truncated)):
        print(f"\n=== SAMPLE {i} ===")
        print(f"PROMPT:\n{p}")
        print(f"RESPONSE:\n{r}")
        print(f"TRUNCATED: {t}")

def evaluate_ner_dataset(
    model,
    processor,
    ds,
    batch_size=8,
    split="test",
    max_samples=-1,
    modes=NER_F1_MODES,
    compute_rewards=True,
    max_new_tokens=200,
    completions_path=None,
    temperature=None,
    top_p=None,
):
    """Run inference on dataset and return micro F1 (per requested mode), number
    of json parsing errors encountered, number of truncated completions, and
    (optionally) the mean of every GRPO reward component over the dataset.

    You may specify a split, a batch size, an optional maximum number of samples
    (`max_samples`; -1 means "no limit"), and which F1 mode(s) to compute ("soft",
    "strict", or both). `max_new_tokens` caps
    generation length exactly like `max_completion_length` does during GRPO training
    (pass the same value used there so truncation behaves identically at test time).
    If `completions_path` is given, every sample's prompt/prediction/ground truth,
    reward components, and json/truncation flags are written there as a parquet
    file, mirroring the per-sample completions GRPOTrainer logs during training.
    The metrics are saved to disk in the specified folder.

    `temperature`/`top_p` are forwarded to `run_ner_batched_inference`: leaving both
    as `None` keeps generation greedy (the default), passing either switches to
    sampling with that parameter.
    """
    # Prepare model for inference
    if model.is_gradient_checkpointing:
        model.gradient_checkpointing_disable()

    # Set up accumulators
    totals = {mode: [0, 0, 0] for mode in modes}
    json_errors_total = 0
    n_truncated_total = 0
    reward_sums = {name: 0.0 for name in NER_REWARD_COMPONENTS} if compute_rewards else {}
    completion_rows = [] if completions_path is not None else None

    # Prepare iteration
    data = ds[split]
    n = len(data)
    if max_samples != -1:
        n = min(n, max_samples)

    for start in tqdm(range(0, n, batch_size), desc="Running inference..."):
        # Define sample indices for the batch
        batch_indices = list(range(start, min(start + batch_size, n)))
        # Select rows
        batch_rows = [data[i] for i in batch_indices]
        # Extract prompts and ground truth labels
        prompts = [row["messages"][:-1] for row in batch_rows]
        gt_jsons = [row["messages"][-1]["content"] for row in batch_rows]
        # Compute predictions (already cleaned by run_ner_batched_inference)
        preds, truncated = run_ner_batched_inference(
            model, processor, prompts, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
        )
        # Compute and accumulate counts
        for idx, prompt, gt_json, pred, is_truncated in zip(batch_indices, prompts, gt_jsons, preds, truncated):
            counts, err = compute_ner_sample_counts(gt_json, pred, modes=modes)
            for mode in modes:
                tp, gt_c, pred_c = counts[mode]
                totals[mode][0] += tp
                totals[mode][1] += gt_c
                totals[mode][2] += pred_c
            json_errors_total += err
            n_truncated_total += int(is_truncated)
            sample_rewards = compute_ner_sample_rewards(gt_json, pred) if compute_rewards else {}
            if compute_rewards:
                for name, value in sample_rewards.items():
                    reward_sums[name] += value
            if completion_rows is not None:
                completion_rows.append({
                    "index": idx,
                    "prompt": json.dumps(prompt),
                    "completion": pred,
                    "ground_truth": gt_json,
                    "json_error": bool(err),
                    "truncated": bool(is_truncated),
                    **sample_rewards,
                })

    # Wrap into dictionary
    metric_dict = {
        'json_errors_total': int(json_errors_total),
        'n_truncated_total': int(n_truncated_total),
        'temperature': temperature,
        'top_p': top_p,
    }
    for mode in modes:
        TP_total, GT_total, PRED_total = totals[mode]
        f1 = 2 * TP_total / (GT_total + PRED_total) if GT_total + PRED_total > 0 else 0
        metric_dict[f'TP_total_{mode}'] = float(TP_total)
        metric_dict[f'GT_total_{mode}'] = int(GT_total)
        metric_dict[f'PRED_total_{mode}'] = int(PRED_total)
        metric_dict[f'F1_{mode}'] = float(f1)

    if compute_rewards and n > 0:
        for name, total in reward_sums.items():
            metric_dict[f'reward_{name}_mean'] = float(total / n)

    if completion_rows is not None:
        completions_path = Path(completions_path)
        completions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(completion_rows).to_parquet(completions_path)

    return metric_dict


# --- GRPO REWARD ---

def compute_ner_sample_f1(gt_json, pred_json, mode="soft"):
    """Compute micro F1 for a single sample, using the given F1 mode."""
    counts, json_error = compute_ner_sample_counts(gt_json, pred_json, modes=(mode,))
    TP, GT, PRED = counts[mode]

    if json_error:
        return 0.0
    elif GT + PRED == 0:
        return 1.0
    else:
        return 2 * TP / (GT + PRED)

def ner_soft_f1_reward_fn(prompts, completions, answer, **kwargs):
    return [
        compute_ner_sample_f1(
            gt_json,
            clean_ner_prediction(completion[0]["content"])
        )
        for completion, gt_json in zip(completions, answer)
    ]


# --- STRUCTURED REWARD ---
# A level-based reward with early-exit gating: JSON validity, then correct
# entity-type keys, then per-type extraction quality (exact-match
# precision/recall). See compute_ner_structured_reward_components below for the
# full breakdown.

NER_STRUCTURED_REWARD_BONUS = 1.0  # bonus added to a correctly-formatted answer

def _has_ner_correct_structure(d):
    """Check that `d` is a Dict[str -> List[str]]."""
    if not isinstance(d, dict):
        return False
    for k, v in d.items():
        if not isinstance(k, str) or not isinstance(v, list):
            return False
        if not all(isinstance(e, str) for e in v):
            return False
    return True

def _compute_ner_extraction_score(t_list, o_list):
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

def compute_ner_structured_reward_components(gt_json, pred_json, bonus=NER_STRUCTURED_REWARD_BONUS):
    """Compute the structured reward for a single sample, broken down by pseudocode level.

    GT is assumed to always be well-formed (Dict[str -> List[str]], one entry
    per expected entity type, empty list for negative types).

    Returns a dict with four additive components (their sum equals the
    original monolithic reward, including all early-exit gating):
        - "format": 1.0 if the output is valid Dict[str -> List[str]], else 0.0
        - "key_matching": partial credit for correct entity-type keys
        - "extraction_quality_positive": bonus share + extraction score (Q_R),
          restricted to entity types with at least one GT span; clipped
        - "extraction_quality_negative": bonus share + extraction score (Q_R),
          restricted to entity types with no GT spans; clipped
    The quality level is split by entity type so positive-type (has GT spans)
    and negative-type (no GT spans) performance can be tracked separately in
    wandb; their sum reproduces the original single "extraction_quality" value.
    """
    components = {
        "format": 0.0,
        "key_matching": 0.0,
        "extraction_quality_positive": 0.0,
        "extraction_quality_negative": 0.0,
    }

    # Level 1: JSON parsing
    try:
        out_dict = json.loads(pred_json)
    except json.JSONDecodeError:
        return components
    if not _has_ner_correct_structure(out_dict):
        return components
    components["format"] = 1.0

    # Entity-type keys are matched case-insensitively, same as span matching.
    out_dict = lower_ner_keys(out_dict)
    gt = lower_ner_keys(json.loads(gt_json))
    ent_types = list(gt.keys())
    n_types = len(ent_types)

    # Level 2: key matching
    for key in out_dict.keys():
        if key not in ent_types:
            return components  # wrong key found: stop here, no bonus, no quality component
        components["key_matching"] += 1 / n_types  # partial credit, normalized by expected count

    # Missing keys: all output keys passed the check above, so if the counts
    # differ the model omitted some expected types. Cap reward here.
    if len(out_dict) != n_types:
        return components

    # Level 3: extraction quality, split into positive types (GT has spans)
    # and negative types (GT is empty for that type).
    n_pos = sum(1 for t in ent_types if len(gt[t]) > 0)
    n_neg = n_types - n_pos
    Q_R_pos, Q_R_neg = 0.0, 0.0
    for t in ent_types:
        t_list = gt[t]
        o_list = out_dict[t]  # safe: exact key match enforced above

        if len(t_list) == 0:
            score = 1.0 if len(o_list) == 0 else -1.0
            Q_R_neg += score / n_types
        else:
            score = _compute_ner_extraction_score(t_list, o_list)
            Q_R_pos += score / n_types

    # Clip to prevent quality penalty from erasing the format reward: correct
    # format should always beat wrong format (R=0). The bonus is split
    # proportionally to how many types are positive/negative so the two
    # components sum to the same bonus + Q_R as before; if clipping triggers,
    # both components collapse to 0 so the sum still matches the clipped total.
    if bonus + Q_R_pos + Q_R_neg < 0:
        components["extraction_quality_positive"] = 0.0
        components["extraction_quality_negative"] = 0.0
    else:
        components["extraction_quality_positive"] = bonus * (n_pos / n_types) + Q_R_pos
        components["extraction_quality_negative"] = bonus * (n_neg / n_types) + Q_R_neg

    return components

NER_REWARD_COMPONENTS = (
    "format",
    "key_matching",
    "extraction_quality_positive",
    "extraction_quality_negative",
    "structured_total",
    "soft_f1",
)

def compute_ner_sample_rewards(gt_json, pred_json, bonus=NER_STRUCTURED_REWARD_BONUS):
    """Compute every reward signal used across GRPO reward functions for one sample.

    Returns the four structured-reward components, their sum ("structured_total",
    i.e. what `ner_structured_*_reward_fn`'s summed outputs give a sample), and the
    soft-F1 reward ("soft_f1") used by `ner_soft_f1_reward_fn`.
    """
    components = compute_ner_structured_reward_components(gt_json, pred_json, bonus)
    rewards = dict(components)
    rewards["structured_total"] = sum(components.values())
    rewards["soft_f1"] = compute_ner_sample_f1(gt_json, pred_json, mode="soft")
    return rewards

def _ner_structured_component_reward_fn(component):
    def reward_fn(prompts, completions, answer, **kwargs):
        return [
            compute_ner_structured_reward_components(
                gt_json,
                clean_ner_prediction(completion[0]["content"])
            )[component]
            for completion, gt_json in zip(completions, answer)
        ]
    reward_fn.__name__ = f"ner_structured_{component}_reward_fn"
    return reward_fn

# Split into one reward function per pseudocode level so TRL logs each
# component's mean/std to wandb separately (rewards/ner_structured_<level>_reward_fn/*),
# while their sum (equal weights) reproduces the original combined reward exactly.
ner_structured_format_reward_fn = _ner_structured_component_reward_fn("format")
ner_structured_key_matching_reward_fn = _ner_structured_component_reward_fn("key_matching")
ner_structured_extraction_quality_positive_reward_fn = _ner_structured_component_reward_fn("extraction_quality_positive")
ner_structured_extraction_quality_negative_reward_fn = _ner_structured_component_reward_fn("extraction_quality_negative")


# --- REWARD SELECTION ---
# Each entry is a list of reward functions passed to GRPOTrainer together
# (with equal weights), so TRL logs a wandb rewards/<name>/mean-std pair per
# entry while their sum still gives the combined training reward.

NER_REWARD_FUNCTIONS = {
    "structured": [
        ner_structured_format_reward_fn,
        ner_structured_key_matching_reward_fn,
        ner_structured_extraction_quality_positive_reward_fn,
        ner_structured_extraction_quality_negative_reward_fn,
    ],
    "soft_f1": [ner_soft_f1_reward_fn],
}
