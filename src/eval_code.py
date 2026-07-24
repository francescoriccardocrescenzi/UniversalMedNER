"""Metrics and GRPO reward functions.

This module is pure scoring code: every function here operates on already-
generated text (a ground-truth JSON string and a predicted JSON string), never
on a model. Model inference lives in `inference_code.py`; `score_completions`
below is the scoring counterpart to that module's `generate_completions`.

F1 computation (`compute_sample_counts`/`compute_entity_counts` and friends) is
fully task-agnostic and shared verbatim by both the NER and schema-free NER (sfner)
tasks. The GRPO "structured" reward, however, is task-split: NER's ground
truth deliberately includes negative entity types (empty span lists, sampled
from the candidate list the model wasn't asked about) so its reward tracks a
positive/negative quality split, while sfner's ground truth -- built from
whichever entity types are actually present in the text -- never contains a
negative type, so its reward has a single, undivided extraction-quality
component.
"""

# --- IMPORTS ---

from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd
from scipy import optimize as sp_optimize


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
    """Compute TP, GT_count, PRED_count for GT and predicted entity spans.

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
    entity-extraction json and a predicted entity-extraction json.

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


# --- SCORING (pure -- consumes inference_code.generate_completions records) ---

def clean_prediction(s):
    """Strip markdown code fences from a model prediction."""
    s = s.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    s = s.split("<start_of_turn>model\n", 1)[-1]
    s = s.split("<end_of_turn>", 1)[0]
    return s

def score_completions(
    records,
    task,
    modes=F1_MODES,
    compute_rewards=True,
    completions_path=None,
    temperature=None,
    top_p=None,
):
    """Score a list of inference records into micro F1 (per requested mode), a
    json-parsing-error count, a truncation count, and (optionally) the mean of
    every GRPO reward component over the records.

    `records` is the list produced by `inference_code.generate_completions`:
    each entry has "index", "prompt", "completion" (already cleaned), "ground_truth",
    and "truncated". `task` selects which reward-component set/registry to use
    ("ner" or "sfner") -- F1 itself is identical for both. `modes` picks which F1
    mode(s) to compute ("soft", "strict", or both).

    If `completions_path` is given, every record's prompt/prediction/ground truth,
    reward components, and json/truncation flags are written there as a parquet
    file, mirroring the per-sample completions GRPOTrainer logs during training.
    `temperature`/`top_p` are recorded in the returned metrics as run metadata
    only -- they don't affect scoring.
    """
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


# --- SOFT-F1 REWARD (shared) ---

def soft_f1_reward_fn(prompts, completions, answer, **kwargs):
    return [
        compute_sample_f1(
            gt_json,
            clean_prediction(completion[0]["content"])
        )
        for completion, gt_json in zip(completions, answer)
    ]


# --- STRUCTURED REWARD ---
# A level-based reward with early-exit gating: JSON validity, then correct
# entity-type keys, then per-type extraction quality (exact-match
# precision/recall). Levels 1-2 (format, key matching) are identical for both
# tasks and factored into `_compute_structured_common`. Level 3 (extraction
# quality) is task-split: NER's ground truth includes negative entity types
# (empty span lists) by construction, so its reward tracks positive/negative
# quality separately; sfner's ground truth never contains a negative type (every
# type in it was found in the text), so its reward has a single, undivided
# extraction-quality component -- there is nothing to split.

STRUCTURED_REWARD_BONUS = 1.0  # bonus added to a correctly-formatted answer

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

def _compute_structured_common(gt_json, pred_json):
    """Compute the shared Level 1 (format) + Level 2 (key matching) components.

    Returns `(format, key_matching, gated, gt, out_dict, ent_types)`. `gated`
    is True once format/key-matching passed and Level 3 (extraction quality,
    task-specific) can run; when False the caller should stop, returning
    whatever components it has initialized to 0.0 so far -- same early-exit
    gating as before, just factored out since it's identical for both tasks.
    """
    # Level 1: JSON parsing
    try:
        out_dict = json.loads(pred_json)
    except json.JSONDecodeError:
        return 0.0, 0.0, False, None, None, None
    if not _has_correct_structure(out_dict):
        return 0.0, 0.0, False, None, None, None

    # Entity-type keys are matched case-insensitively, same as span matching.
    out_dict = lower_keys(out_dict)
    gt = lower_keys(json.loads(gt_json))
    ent_types = list(gt.keys())
    n_types = len(ent_types)

    # Level 2: key matching
    key_matching = 0.0
    for key in out_dict.keys():
        if key not in ent_types:
            return 1.0, key_matching, False, None, None, None  # wrong key: stop, no quality component
        key_matching += 1 / n_types  # partial credit, normalized by expected count

    # Missing keys: all output keys passed the check above, so if the counts
    # differ the model omitted some expected types. Cap reward here.
    if len(out_dict) != n_types:
        return 1.0, key_matching, False, None, None, None

    return 1.0, key_matching, True, gt, out_dict, ent_types

def compute_ner_structured_reward_components(gt_json, pred_json, bonus=STRUCTURED_REWARD_BONUS):
    """Compute the NER structured reward for a single sample, broken down by
    pseudocode level.

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
    fmt, key_matching, gated, gt, out_dict, ent_types = _compute_structured_common(gt_json, pred_json)
    components = {
        "format": fmt,
        "key_matching": key_matching,
        "extraction_quality_positive": 0.0,
        "extraction_quality_negative": 0.0,
    }
    if not gated:
        return components

    n_types = len(ent_types)

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
            score = _compute_extraction_score(t_list, o_list)
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

def compute_sfner_structured_reward_components(gt_json, pred_json, bonus=STRUCTURED_REWARD_BONUS):
    """Compute the schema-free NER structured reward for a single sample, broken
    down by pseudocode level.

    GT is assumed to always be well-formed (Dict[str -> List[str]]) and, by
    construction (every type in it was found in the text -- see
    `dataset_code.format_sfner_sft`), never contains a negative entity type
    (one with an empty span list). Extraction quality therefore has a single,
    undivided component -- there is nothing to split by positive/negative.

    Returns a dict with three additive components (their sum equals the
    original monolithic reward, including all early-exit gating):
        - "format": 1.0 if the output is valid Dict[str -> List[str]], else 0.0
        - "key_matching": partial credit for correct entity-type keys
        - "extraction_quality": bonus + extraction score (Q_R) over every
          entity type; clipped
    """
    fmt, key_matching, gated, gt, out_dict, ent_types = _compute_structured_common(gt_json, pred_json)
    components = {
        "format": fmt,
        "key_matching": key_matching,
        "extraction_quality": 0.0,
    }
    if not gated:
        return components

    n_types = len(ent_types)

    # Level 3: extraction quality over every entity type (all positive by
    # construction for sfner -- see docstring).
    Q_R = 0.0
    for t in ent_types:
        Q_R += _compute_extraction_score(gt[t], out_dict[t]) / n_types  # out_dict[t] safe: exact key match enforced above

    # Clip to prevent quality penalty from erasing the format reward: correct
    # format should always beat wrong format (R=0).
    components["extraction_quality"] = bonus + Q_R if bonus + Q_R >= 0 else 0.0

    return components

NER_REWARD_COMPONENTS = (
    "format",
    "key_matching",
    "extraction_quality_positive",
    "extraction_quality_negative",
    "structured_total",
    "soft_f1",
)

SFNER_REWARD_COMPONENTS = (
    "format",
    "key_matching",
    "extraction_quality",
    "structured_total",
    "soft_f1",
)

def compute_ner_sample_rewards(gt_json, pred_json, bonus=STRUCTURED_REWARD_BONUS):
    """Compute every reward signal used across NER GRPO reward functions for one sample.

    Returns the four structured-reward components, their sum ("structured_total",
    i.e. what `ner_structured_*_reward_fn`'s summed outputs give a sample), and
    the soft-F1 reward ("soft_f1") used by `soft_f1_reward_fn`.
    """
    components = compute_ner_structured_reward_components(gt_json, pred_json, bonus)
    rewards = dict(components)
    rewards["structured_total"] = sum(components.values())
    rewards["soft_f1"] = compute_sample_f1(gt_json, pred_json, mode="soft")
    return rewards

def compute_sfner_sample_rewards(gt_json, pred_json, bonus=STRUCTURED_REWARD_BONUS):
    """Compute every reward signal used across schema-free NER GRPO reward functions for one sample.

    Returns the three structured-reward components, their sum ("structured_total",
    i.e. what `sfner_structured_*_reward_fn`'s summed outputs give a sample), and
    the soft-F1 reward ("soft_f1") used by `soft_f1_reward_fn`.
    """
    components = compute_sfner_structured_reward_components(gt_json, pred_json, bonus)
    rewards = dict(components)
    rewards["structured_total"] = sum(components.values())
    rewards["soft_f1"] = compute_sample_f1(gt_json, pred_json, mode="soft")
    return rewards

def _structured_component_reward_fn(component, compute_fn):
    def reward_fn(prompts, completions, answer, **kwargs):
        return [
            compute_fn(
                gt_json,
                clean_prediction(completion[0]["content"])
            )[component]
            for completion, gt_json in zip(completions, answer)
        ]
    # Un-prefixed __name__ so TRL/wandb logs a stable `structured_<level>_reward_fn`
    # key regardless of task -- ner and sfner runs are still comparable there, even
    # though sfner only ever populates a subset of the possible component names.
    reward_fn.__name__ = f"structured_{component}_reward_fn"
    return reward_fn

# Split into one reward function per pseudocode level so TRL logs each
# component's mean/std to wandb separately (rewards/structured_<level>_reward_fn/*),
# while their sum (equal weights) reproduces the corresponding combined reward
# exactly. NER keeps the positive/negative quality split; sfner has a single,
# undivided quality component (its GT never contains a negative entity type).
ner_structured_format_reward_fn = _structured_component_reward_fn("format", compute_ner_structured_reward_components)
ner_structured_key_matching_reward_fn = _structured_component_reward_fn("key_matching", compute_ner_structured_reward_components)
ner_structured_extraction_quality_positive_reward_fn = _structured_component_reward_fn("extraction_quality_positive", compute_ner_structured_reward_components)
ner_structured_extraction_quality_negative_reward_fn = _structured_component_reward_fn("extraction_quality_negative", compute_ner_structured_reward_components)

sfner_structured_format_reward_fn = _structured_component_reward_fn("format", compute_sfner_structured_reward_components)
sfner_structured_key_matching_reward_fn = _structured_component_reward_fn("key_matching", compute_sfner_structured_reward_components)
sfner_structured_extraction_quality_reward_fn = _structured_component_reward_fn("extraction_quality", compute_sfner_structured_reward_components)


# --- REWARD SELECTION ---
# Each entry is a list of reward functions passed to GRPOTrainer together
# (with equal weights), so TRL logs a wandb rewards/<name>/mean-std pair per
# entry while their sum still gives the combined training reward. Split by
# task (NER_REWARD_FUNCTIONS / SFNER_REWARD_FUNCTIONS) purely because the
# "structured" entry's components differ; both registries expose the same
# choice keys ("structured", "soft_f1").

NER_REWARD_FUNCTIONS = {
    "structured": [
        ner_structured_format_reward_fn,
        ner_structured_key_matching_reward_fn,
        ner_structured_extraction_quality_positive_reward_fn,
        ner_structured_extraction_quality_negative_reward_fn,
    ],
    "soft_f1": [soft_f1_reward_fn],
}

SFNER_REWARD_FUNCTIONS = {
    "structured": [
        sfner_structured_format_reward_fn,
        sfner_structured_key_matching_reward_fn,
        sfner_structured_extraction_quality_reward_fn,
    ],
    "soft_f1": [soft_f1_reward_fn],
}
