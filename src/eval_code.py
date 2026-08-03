"""Metrics and GRPO reward functions."""

from pathlib import Path
from collections import Counter
import functools
import json
import numpy as np
import pandas as pd
from scipy import optimize as sp_optimize


# --- HELPERS ---

F1_MODES = ("soft", "strict")

def tokenize(s):
    """Split entity span into a set lowercase words."""
    return Counter(str(s).lower().split())

def iou(a, b, mode="soft"):
    """Compute a match score between two entity spans.

    If the mode is soft, compute a proper IoU score on the lower case letters.
    If the mode is hard, compute binary equality.
    In both cases, tokenize with a counter to ensure repeated letters are accounted for.
    """
    A, B = tokenize(a), tokenize(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    if mode == "strict":
        return 1.0 if A == B else 0.0
    return sum((A & B).values()) / sum((A | B).values())

def compute_entity_counts(gt_list, pred_list, mode="soft"):
    """Compute TP, GT_count, PRED_count for GT and predicted entity spans.

    Pass a list of ground truth spans and a list of predicted spans associated with the
    same entity. A matching is found between the two lists that maximizes total score (Hungarian matching),
    where the pairwise score is given by IoU. Then the following are returned:
        - TP = total matched score (total IoU in soft mode, number of exact matches in strict mode)
        - GT_count = number of GT spans
        - PRED_count = number of predicted spans
    """
    GT_count = len(gt_list)
    PRED_count = len(pred_list)

    if GT_count == 0 or PRED_count == 0:
        return 0, GT_count, PRED_count

    if mode == "strict":
        gt_set = {frozenset(tokenize(s).items()) for s in gt_list}
        pred_set = {frozenset(tokenize(s).items()) for s in pred_list}
        TP = len(gt_set & pred_set)
        return TP, GT_count, PRED_count

    # Compute all pairwise IoU scores to guide matching algorithm
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
    """Compute TP, GT_count, PRED_count per F1 mode, and a json_error_flag, for a GT and pred."""
    try:
        pred = json.loads(pred_json)
        gt = json.loads(gt_json)

        if not isinstance(pred, dict) or not isinstance(gt, dict):
            raise ValueError("Loaded JSON must be a dict")

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


# --- SCORING ---

# Must be here and not in inference code because GRPO completions do not
# pass through the inference logic
def clean_prediction(s):
    """Strip markdown code fences and turn tokesn from a model prediction."""
    s = s.split("<start_of_turn>model\n", 1)[-1]
    s = s.split("<end_of_turn>", 1)[0]
    s = s.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return s

def score_completions(records, task, modes=F1_MODES, compute_rewards=True, completions_path=None, temperature=None, top_p=None):
    """Score a list of inference records into micro F1, json-parsing-error count, truncation count, and means of GRPO reward components."""
    compute_sample_rewards = {"ner": compute_ner_sample_rewards, "sfner": compute_sfner_sample_rewards}[task]
    reward_components = {"ner": NER_REWARD_COMPONENTS, "sfner": SFNER_REWARD_COMPONENTS}[task]

    totals = {mode: [0, 0, 0] for mode in modes}
    json_errors_total = 0
    n_truncated_total = 0
    reward_sums = {name: 0.0 for name in reward_components} if compute_rewards else {}
    completion_rows = [] if completions_path is not None else None

    n = len(records)
    for record in records:
        gt_json, pred, is_truncated = record["ground_truth"], record["completion"], record["truncated"]
        counts, err = compute_sample_counts(gt_json, pred, modes=modes)
        for mode in modes:
            tp, gt_c, pred_c = counts[mode]
            totals[mode][0] += tp
            totals[mode][1] += gt_c
            totals[mode][2] += pred_c
        json_errors_total += err
        n_truncated_total += int(is_truncated)
        sample_rewards = compute_sample_rewards(gt_json, pred) if compute_rewards else {}
        if compute_rewards:
            for name, value in sample_rewards.items():
                reward_sums[name] += value
        if completion_rows is not None:
            completion_rows.append({
                "index": record["index"],
                "prompt": json.dumps(record["prompt"]),
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


# --- SOFT-F1 REWARD ---

def soft_f1_reward_fn(prompts, completions, answer, **kwargs):
    return [
        compute_sample_f1(gt_json, clean_prediction(completion[0]["content"]))
        for completion, gt_json in zip(completions, answer)
    ]


# --- STRUCTURED REWARD ---

# NER
# Keep NER and SF-NER separate for convenience in case they will be modified to adapt them to
# the specific nature of the respective task

STRUCTURED_REWARD_CACHE_MAXSIZE = 4096 # LRU cache size (see under logging for more info)
STRUCTURED_REWARD_BONUS = 1.0  # bonus added to a correctly-formatted answer

def _ner_has_correct_structure(d):
    """Check that the json completion is a dict mapping strings to lists of strings.

    In rare occasions, the model produces completions that are valid jsons but are not correctly formatted,
    e.g. the model places an int instead of a string, breaking the evaluation pipeline.
    """
    if not isinstance(d, dict):
        return False
    for k, v in d.items():
        if not isinstance(k, str) or not isinstance(v, list):
            return False
        if not all(isinstance(e, str) for e in v):
            return False
    return True

def _ner_compute_extraction_score(t_list, o_list):
    """Span extraction score.

    +1/N for each correctly predicted GT span
    -1/N for each wrong or duplicate prediction
    -1/N for each GT span never predicted
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

@functools.lru_cache(maxsize=STRUCTURED_REWARD_CACHE_MAXSIZE)
def compute_ner_structured_reward_components(gt_json, pred_json, bonus=STRUCTURED_REWARD_BONUS):
    """Compute the NER structured reward for a single sample.

    Returns a dict with four components meant to be added:
        - "format": 1.0 if the output format is valide, else 0.0
        - "key_matching": partial credit for correct entity-type keys
        - "extraction_quality_positive": bonus share + extraction score for positve entities
        - "extraction_quality_negative": bonus share + extraction score for negative entities
    """
    components = {
        "format": 0.0,
        "key_matching": 0.0,
        "bonus": 0.0,
        "extraction_quality_positive": 0.0,
        "extraction_quality_negative": 0.0,
    }

    # Level 1: JSON parsing
    try:
        out_dict = json.loads(pred_json)
    except json.JSONDecodeError:
        return components
    if not _ner_has_correct_structure(out_dict):
        return components
    components["format"] = 1.0

    gt = json.loads(gt_json)
    ent_types = list(gt.keys())
    n_types = len(ent_types)

    # Level 2: key matching
    for key in out_dict.keys():
        if key not in ent_types:
            return components
        components["key_matching"] += 1 / n_types  # partial credit, normalized by expected count

    # Cap reward if the model omitted keys
    if len(out_dict) != n_types:
        return components

    # Level 3: extraction quality 
    n_pos = sum(1 for t in ent_types if len(gt[t]) > 0)
    n_neg = n_types - n_pos
    Q_R_pos, Q_R_neg = 0.0, 0.0
    for t in ent_types:
        t_list = gt[t]
        o_list = out_dict[t]

        if len(t_list) == 0:
            score = 1.0 if len(o_list) == 0 else -1.0
            Q_R_neg += score / n_types
        else:
            score = _ner_compute_extraction_score(t_list, o_list)
            Q_R_pos += score / n_types

    # Clip to prevent quality penalty from erasing the format reward
    if bonus + Q_R_pos + Q_R_neg < 0:
        components["extraction_quality_positive"] = 0.0
        components["extraction_quality_negative"] = 0.0
    else:
        components["extraction_quality_positive"] = Q_R_pos
        components["extraction_quality_negative"] = Q_R_neg
        components["bonus"] = bonus

    return components

# SF-NER

def _sfner_has_correct_structure(d):
    if not isinstance(d, dict):
        return False
    for k, v in d.items():
        if not isinstance(k, str) or not isinstance(v, list):
            return False
        if not all(isinstance(e, str) for e in v):
            return False
    return True

def _sfner_compute_extraction_score(t_list, o_list):
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

@functools.lru_cache(maxsize=STRUCTURED_REWARD_CACHE_MAXSIZE)
def compute_sfner_structured_reward_components(gt_json, pred_json, bonus=STRUCTURED_REWARD_BONUS):
    """Identical to the ner version but without a negative extraction rewards since the SF-NER task has no negatives."""
    components = {
        "format": 0.0,
        "key_matching": 0.0,
        "bonus": 0.0,
        "extraction_quality": 0.0,
    }

    # Level 1: JSON parsing
    try:
        out_dict = json.loads(pred_json)
    except json.JSONDecodeError:
        return components
    if not _sfner_has_correct_structure(out_dict):
        return components
    components["format"] = 1.0

    gt = json.loads(gt_json)
    ent_types = list(gt.keys())
    n_types = len(ent_types)

    # Level 2: key matching
    for key in out_dict.keys():
        if key not in ent_types:
            return components  
        components["key_matching"] += 1 / n_types

    if len(out_dict) != n_types:
        return components

    # Level 3: extraction quality over every entity type 
    Q_R = 0.0
    for t in ent_types:
        Q_R += _sfner_compute_extraction_score(gt[t], out_dict[t]) / n_types  

    if bonus + Q_R < 0:
        components["extraction_quality"] = 0.0
    else:
        components["extraction_quality"] = Q_R
        components["bonus"] = bonus

    return components

# ----- LOGGING -----

# The components of the structured reward need to be computed in order.
# The easiset way to log rewards is using the already existing TRL machinery (so we get means and stds without
# having to reimplement the computations).
# However, TRL expects the different components of the reward as independent functions.
# Hence, we first compute the whole structured reward, we cache it with LRU cache via functools, and, finally
# access the various components with thin independent wrappers.

# Define components (include soft F1)

NER_REWARD_COMPONENTS = (
    "format",
    "key_matching",
    "bonus",
    "extraction_quality_positive",
    "extraction_quality_negative",
    "structured_total",
    "soft_f1",
)

SFNER_REWARD_COMPONENTS = (
    "format",
    "key_matching",
    "bonus",
    "extraction_quality",
    "structured_total",
    "soft_f1",
)

# Aggregate F1 and structured reward for convenience

def compute_ner_sample_rewards(gt_json, pred_json, bonus=STRUCTURED_REWARD_BONUS):
    components = compute_ner_structured_reward_components(gt_json, pred_json, bonus)
    rewards = dict(components)
    rewards["structured_total"] = sum(components.values())
    rewards["soft_f1"] = compute_sample_f1(gt_json, pred_json, mode="soft")
    return rewards

def compute_sfner_sample_rewards(gt_json, pred_json, bonus=STRUCTURED_REWARD_BONUS):
    components = compute_sfner_structured_reward_components(gt_json, pred_json, bonus)
    rewards = dict(components)
    rewards["structured_total"] = sum(components.values())
    rewards["soft_f1"] = compute_sample_f1(gt_json, pred_json, mode="soft")
    return rewards

# This util extracts a portion of the structured reward and wraps it as an independent function

def _structured_component_reward_fn(component, compute_fn):
    def reward_fn(prompts, completions, answer, **kwargs):
        return [
            compute_fn(gt_json, clean_prediction(completion[0]["content"]))[component]
            for completion, gt_json in zip(completions, answer)
        ]
    # WandB name
    reward_fn.__name__ = f"structured_{component}_reward_fn"
    return reward_fn

ner_structured_format_reward_fn = _structured_component_reward_fn("format", compute_ner_structured_reward_components)
ner_structured_key_matching_reward_fn = _structured_component_reward_fn("key_matching", compute_ner_structured_reward_components)
ner_structured_bonus_reward_fn = _structured_component_reward_fn("bonus", compute_ner_structured_reward_components)
ner_structured_extraction_quality_positive_reward_fn = _structured_component_reward_fn("extraction_quality_positive", compute_ner_structured_reward_components)
ner_structured_extraction_quality_negative_reward_fn = _structured_component_reward_fn("extraction_quality_negative", compute_ner_structured_reward_components)

sfner_structured_format_reward_fn = _structured_component_reward_fn("format", compute_sfner_structured_reward_components)
sfner_structured_key_matching_reward_fn = _structured_component_reward_fn("key_matching", compute_sfner_structured_reward_components)
sfner_structured_bonus_reward_fn = _structured_component_reward_fn("bonus", compute_sfner_structured_reward_components)
sfner_structured_extraction_quality_reward_fn = _structured_component_reward_fn("extraction_quality", compute_sfner_structured_reward_components)

# Each entry is a list of reward functions passed to GRPOTrainer together with equal weights
# Passing them in this fashion allows their separate automatic logging on wandb

NER_REWARD_FUNCTIONS = {
    "structured": [
        ner_structured_format_reward_fn,
        ner_structured_key_matching_reward_fn,
        ner_structured_bonus_reward_fn,
        ner_structured_extraction_quality_positive_reward_fn,
        ner_structured_extraction_quality_negative_reward_fn,
    ],
    "soft_f1": [soft_f1_reward_fn],
}

SFNER_REWARD_FUNCTIONS = {
    "structured": [
        sfner_structured_format_reward_fn,
        sfner_structured_key_matching_reward_fn,
        sfner_structured_bonus_reward_fn,
        sfner_structured_extraction_quality_reward_fn,
    ],
    "soft_f1": [soft_f1_reward_fn],
}
