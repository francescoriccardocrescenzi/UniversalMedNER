"""Model inference code.

This module runs inference for MedGemma model and returns cleaned completions.
It does not score those completions. It handles both NER and SFNER.
"""

from tqdm import tqdm
import torch
from eval_code import clean_prediction


def run_batched_inference(model, processor, prompts, max_new_tokens=200, temperature=None, top_p=None):
    """Run model inference on a batch of samples.

    By default generation is greedy but passing `temperature` and/or `top_p` switches to sampling with
    those parameters.

    Returns a (responses, truncated) pair: responses are the cleaned model
    outputs and truncated is a list of booleans flagging samples
    whose generation hit max_new_tokens.
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

    eos_token_id = [
        processor.tokenizer.eos_token_id,
        processor.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
    ]

    # Set up generation
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

    # Batch generate
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    # Batch decode
    decoded = processor.tokenizer.batch_decode(
        outputs,
        skip_special_tokens=False
    )

    # Cleanup generated text
    responses = []
    for text in decoded:
        responses.append(clean_prediction(text))

    # Determine which samples were truncated
    eos_and_pad = set(eos_token_id) | {processor.tokenizer.pad_token_id}
    truncated = [
        row[-1].item() not in eos_and_pad
        for row in outputs[:, prompt_len:]
    ]

    # Cleanup memory
    del inputs
    del outputs
    torch.cuda.empty_cache()

    return responses, truncated

def test_model_on_batch(model, processor, pile_ds, split="train", indices=None, max_new_tokens=200, temperature=None, top_p=None):
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

    responses, truncated = run_batched_inference(
        model, processor, prompts, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
    )

    for i, (p, r, t) in enumerate(zip(prompts, responses, truncated)):
        print(f"\n=== SAMPLE {i} ===")
        print(f"PROMPT:\n{p}")
        print(f"RESPONSE:\n{r}")
        print(f"TRUNCATED: {t}")

def generate_completions(model, processor, ds, split="test", batch_size=8, max_samples=-1, max_new_tokens=200, temperature=None, top_p=None):
    """Run inference over a dataset split and return one record per sample.

    Each record is in the form {"index", "prompt", "completion", "ground_truth", "truncated"}
    """
    # Prepare model for inference
    if model.is_gradient_checkpointing:
        model.gradient_checkpointing_disable()

    data = ds[split]
    n = len(data)
    if max_samples != -1:
        n = min(n, max_samples)

    records = []
    for start in tqdm(range(0, n, batch_size), desc="Running inference..."):
        # Define sample indices for the batch
        batch_indices = list(range(start, min(start + batch_size, n)))
        # Select rows
        batch_rows = [data[i] for i in batch_indices]
        # Extract prompts and ground truth labels
        prompts = [row["messages"][:-1] for row in batch_rows]
        gt_jsons = [row["messages"][-1]["content"] for row in batch_rows]
        # Compute predictions (already cleaned by run_batched_inference)
        preds, truncated = run_batched_inference(
            model, processor, prompts, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
        )
        for idx, prompt, gt_json, pred, is_truncated in zip(batch_indices, prompts, gt_jsons, preds, truncated):
            records.append({
                "index": idx,
                "prompt": prompt,
                "completion": pred,
                "ground_truth": gt_json,
                "truncated": bool(is_truncated),
            })

    return records
