"""Data preprocessing and formatting functions."""

import json
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import datasets


def get_split_ds(ds, validation_size, test_size, random_seed):
    """Split dataset into train / validation / test."""

    # First split: train+val vs test
    split_1 = ds['train'].train_test_split(
        test_size=test_size,
        seed=random_seed
    )

    # Second split: train vs validation
    split_2 = split_1['train'].train_test_split(
        test_size=validation_size / (1 - test_size),
        seed=random_seed
    )

    return datasets.DatasetDict({
        'train': split_2['train'],
        'validation': split_2['test'],
        'test': split_1['test']
    })

def extract_entity_spans(tokens, tags):
    """Convert list of tokens and list of tag into entity -> entity_span_list dict."""
    entity_spans = defaultdict(list)
    current_span = []
    current_entity = None

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_span:
                entity_spans[current_entity].append(" ".join(current_span))
            current_span = [token]
            current_entity = tag[2:]

        elif tag.startswith("I-") and current_entity:
            current_span.append(token)

        else:
            if current_span:
                entity_spans[current_entity].append(" ".join(current_span))
            current_span = []
            current_entity = None

    if current_span:
        entity_spans[current_entity].append(" ".join(current_span))

    return {entity: list(dict.fromkeys(spans)) for entity, spans in entity_spans.items()}

def to_grpo_ds(sft_ds):
    """Convert from SFT chat template to GRPO data format."""

    return sft_ds.map(
        lambda sample: {
            "prompt": sample["messages"][:-1],
            "answer": sample["messages"][-1]["content"]
        },
        remove_columns=["messages"]
    )

def compute_ner_negative_stats(ds):
    """Compute per-sample positive/negative entity-label statistics for a formatted NER dataset."""
    n_pos_total = 0
    n_neg_total = 0
    n_all_positive = 0
    n_all_negative = 0

    for sample in ds:
        if "answer" in sample:
            entity_json = sample["answer"]
        else:
            entity_json = sample["messages"][-1]["content"]

        entities = json.loads(entity_json)
        n_pos = sum(1 for spans in entities.values() if len(spans) > 0)
        n_neg = len(entities) - n_pos

        n_pos_total += n_pos
        n_neg_total += n_neg
        n_all_positive += int(n_neg == 0)
        n_all_negative += int(n_pos == 0)

    n_samples = len(ds)
    n_labels_total = n_pos_total + n_neg_total

    return {
        "n_samples": n_samples,
        "mean_labels_per_sample": n_labels_total / n_samples,
        "mean_positive_per_sample": n_pos_total / n_samples,
        "mean_negative_per_sample": n_neg_total / n_samples,
        "negative_label_rate": n_neg_total / n_labels_total if n_labels_total else 0.0,
        "pct_samples_all_positive": n_all_positive / n_samples,
        "pct_samples_all_negative": n_all_negative / n_samples,
    }


def get_ner_entity_array_and_weights(ds):
    """Extract all unique entities and their inverse frequencies for negative sampling."""
    entity_list = []

    for sample in tqdm(ds['train'], desc='Computing entity frequencies...'):
        sample_tags = sample['ner_tags']
        if isinstance(sample_tags, str):
            sample_tags = eval(sample_tags)

        entity_list.extend([tag[2:] for tag in sample_tags if tag[:2] == 'I-'])

    entity_frequencies = Counter(entity_list)

    entity_array = np.array(list(entity_frequencies.keys()))
    entity_weights = np.array(list(entity_frequencies.values()))
    entity_weights = 1 / entity_weights
    entity_weights = entity_weights / entity_weights.sum()

    return entity_array, entity_weights

def sample_ner_negatives(n_neg, entity_array, entity_weights):
    """Sample negative entities."""
    return np.random.choice(entity_array, size=n_neg, p=entity_weights)

def sample_ner_positives(n_pos, entities):
    """Sample positive entities."""
    return np.random.choice(list(entities.keys()), size=n_pos, replace=False)

def format_ner_sft(sample, n_entities, entity_array, entity_weights, detok, max_negatives=-1):
    """Create SFT sample from IOB sample.

    Uniformly sample some of the entities present in the current sample
    and use them and the associated spans as positives.
    Then, sample negative entities using inverse frequencies as weights.
    Format sample using system and user PROMPTS.
    """
    tokens = sample['tokens']
    tags = sample['ner_tags']

    if isinstance(tokens, str):
        tokens = eval(tokens)
    if isinstance(tags, str):
        tags = eval(tags)

    text = detok.detokenize(tokens)

    entity_spans = extract_entity_spans(tokens, tags)

    n_pos = min(len(entity_spans), np.random.randint(1, n_entities + 1))
    n_neg = n_entities - n_pos if max_negatives == -1 else min(n_entities - n_pos, max_negatives)

    positives = sample_ner_positives(n_pos, entity_spans)
    negatives = sample_ner_negatives(n_neg, entity_array, entity_weights)

    sampled_entities = set(positives) | set(negatives)

    assistant_output = {
        entity.upper(): entity_spans.get(entity, [])
        for entity in sampled_entities
    }

    ner_json = json.dumps(assistant_output)

    shuffled_entities = list(assistant_output.keys())
    np.random.shuffle(shuffled_entities)

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Named Entity Recognition system.\n\n"
                    "Objective:\n"
                    "Extract only exact text spans from the provided input text that match each given entity label.\n\n"
                    "Strict rules:\n"
                    "- Only extract substrings that appear verbatim in the text.\n"
                    "- Do NOT infer, generalize, or paraphrase.\n"
                    "- Do NOT use semantic matching or external knowledge.\n"
                    "- If no exact match exists for an entity, return an empty list.\n"
                    "- Do NOT hallucinate entities not explicitly present in the text.\n"
                    "- Output MUST be valid JSON only (no markdown, no explanations)."
                )
            },
            {
                "role": "user",
                "content": (
                    "Text:\n"
                    f"{text}\n\n"
                    "Entity labels:\n"
                    f"{shuffled_entities}\n\n"
                    "Return a JSON object where:\n"
                    "- keys are UPPERCASE entity labels\n"
                    "- values are lists of exact substrings taken directly from the text\n"
                    "- only exact matches are allowed\n"
                    "- if no match exists for a label, return []"
                )
            },
            {
                "role": "assistant",
                "content": ner_json
            }
        ]
    }

def create_ner_sft_ds(ds, max_entities, detok, max_negatives=-1):
    """Convert a whole IOB format NER dataset into an SFT format dataset.

    Apply format_ner_sft to each sample.
    """
    entity_array, entity_weights = get_ner_entity_array_and_weights(ds)

    return ds.map(
        lambda sample: format_ner_sft(
            sample,
            np.random.randint(1, max_entities + 1),
            entity_array,
            entity_weights,
            detok,
            max_negatives
        ),
        remove_columns=["tokens", "ner_tags"]
    )

def create_ner_grpo_ds(ds, max_entities, detok, max_negatives=-1):
    return to_grpo_ds(create_ner_sft_ds(ds, max_entities, detok, max_negatives))


def format_sfner_sft(sample, detok):
    """Create SFT sample from IOB sample, for the schema-free NER task."""
    tokens = sample['tokens']
    tags = sample['ner_tags']

    if isinstance(tokens, str):
        tokens = eval(tokens)
    if isinstance(tags, str):
        tags = eval(tags)

    text = detok.detokenize(tokens)

    entity_spans = extract_entity_spans(tokens, tags)

    assistant_output = {
        entity.upper(): spans
        for entity, spans in entity_spans.items()
    }

    entity_json = json.dumps(assistant_output)

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Named Entity Recognition system.\n\n"
                    "Objective:\n"
                    "Identify all named entities present in the provided input text and extract their exact text spans.\n\n"
                    "Strict rules:\n"
                    "- Only extract substrings that appear verbatim in the text.\n"
                    "- Do NOT infer, generalize, or paraphrase entity spans.\n"
                    "- Do NOT use semantic matching or external knowledge.\n"
                    "- Do NOT hallucinate entities not explicitly present in the text.\n"
                    "- Group similar entities under the same entity type.\n"
                    "- Choose a concise, descriptive label for each entity type you find.\n"
                    "- Output MUST be valid JSON only (no markdown, no explanations)."
                )
            },
            {
                "role": "user",
                "content": (
                    "Text:\n"
                    f"{text}\n\n"
                    "Return a JSON object where:\n"
                    "- keys are UPPERCASE entity type labels you identify\n"
                    "- values are lists of exact substrings taken directly from the text for that entity type\n"
                    "- group all spans of the same entity type under one key\n"
                    "- if no entities are found, return {}"
                )
            },
            {
                "role": "assistant",
                "content": entity_json
            }
        ]
    }

def create_sfner_sft_ds(ds, detok):
    return ds.map(
        lambda sample: format_sfner_sft(sample, detok),
        remove_columns=["tokens", "ner_tags"]
    )

def create_sfner_grpo_ds(ds, detok):
    return to_grpo_ds(create_sfner_sft_ds(ds, detok))
