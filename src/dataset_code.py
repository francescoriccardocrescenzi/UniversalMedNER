"""Dataset preparation functions.

This module contains functions to prepare the dataset for 
supervised fine-tuning (SFT) of MedGemma on the NER task.
"""

# --- IMPORTS ---

import json
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import datasets


# --- DATASET PREPARATION FUNCTIONS ---

def get_merged_ds(ds):
    "Returne dataset obtained by merging all splits."
    return datasets.concatenate_datasets([ds[split] for split in ds.keys()])

def get_split_ds(ds, validation_size, test_size, random_seed):
    """Split dataset into train / validation / test."""

    # First split: train+val vs test
    split_1 = ds['train'].train_test_split(
        test_size=test_size,
        seed=random_seed
    )

    # Second split: train vs validation (from remaining data)
    split_2 = split_1['train'].train_test_split(
        test_size=validation_size / (1 - test_size),
        seed=random_seed
    )

    return datasets.DatasetDict({
        'train': split_2['train'],
        'validation': split_2['test'],
        'test': split_1['test']
    })

def get_entity_array_and_weights(ds):
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

def sample_negatives(n_neg, entity_array, entity_weights):
    """Sample negative entities."""
    return np.random.choice(entity_array, size=n_neg, p=entity_weights)

def sample_positives(n_pos, entities):
    """Sample positive entities."""
    return np.random.choice(list(entities.keys()), size=n_pos, replace=False)

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

    return entity_spans

def format_sft(sample, idx, n_entities, entity_array, entity_weights, detok, base_seed):
    """Create SFT sample from IOB sample.

    Uniformly sample some of the entities present in the current sample
    and use them and the associated spans as positives.
    Then sample negative entities using inverse frequencies as weights.
    Create a conversation snippet with:
        - a system prompt instructing the model to think silently
        - a user prompt describing the NER task and passing the text and the entities
        - a model reply containing the json matching entities to their spans found in the text
    The system prompt and user prompt will act as input while the model reply will be the target
    for the supervised fine tuning.
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
    n_neg = n_entities - n_pos

    positives = sample_positives(n_pos, entity_spans)
    negatives = sample_negatives(n_neg, entity_array, entity_weights)

    sampled_entities = set(positives) | set(negatives)

    assistant_output = {
        entity.upper(): entity_spans[entity]
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
                    "- Matching is case-sensitive and character-exact.\n"
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
                    "- keys are entity labels\n"
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

def create_sft_ds(ds, max_entities, detok, random_seed):
    """Convert a whole IOB format NER dataset into an SFT format dataset.

    Apply `format_sft` to each sample.
    """
    entity_array, entity_weights = get_entity_array_and_weights(ds)
    base_seed = random_seed

    return ds.map(
        lambda sample, idx: format_sft(
            sample,
            idx,
            np.random.randint(1, max_entities + 1),
            entity_array,
            entity_weights,
            detok,
            base_seed
        ),
        with_indices=True,
        remove_columns=["tokens", "ner_tags"]
    )

def create_grpo_ds(ds, max_entities, detok, random_seed):
    sft_ds = create_sft_ds(ds, max_entities, detok, random_seed)
    return sft_ds.map(
        lambda sample: {
            "prompt": sample["messages"][:-1],
            "answer": sample["messages"][-1]["content"]
        },
        remove_columns=["messages"]
    )