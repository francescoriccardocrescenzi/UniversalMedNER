"""Model inference code.

This module runs generation against a MedGemma model and returns raw,
already-cleaned completions -- it does not compute any metrics or rewards.
Scoring those completions (F1, GRPO reward diagnostics) is `eval_code.py`'s
job, via `eval_code.score_completions`, which is the scoring counterpart to
this module's `generate_completions`. Fully task-agnostic: nothing here knows
about NER vs. schema-free NER, only about the shared chat-message / JSON-string
sample format both tasks produce.
"""

# --- IMPORTS ---

from tqdm import tqdm
import torch

from eval_code import clean_prediction


# --- INFERENCE ---

def run_batched_inference(model, processor, prompts, max_new_tokens=200, temperature=None, top_p=None):
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
        responses.append(clean_prediction(text))

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

def generate_completions(
    model,
    processor,
    ds,
    split="test",
    batch_size=8,
    max_samples=-1,
    max_new_tokens=200,
    temperature=None,
    top_p=None,
):
    """Run inference over a dataset split and return one record per sample.

    Each record is `{"index", "prompt", "completion", "ground_truth", "truncated"}`:
    `prompt` is the raw (unrendered) chat-message list fed to the model,
    `completion` is the cleaned model output, `ground_truth` is the target
    JSON string from the sample's last (assistant) message, and `truncated`
    mirrors `run_batched_inference`'s clipped-generation flag. No metric or
    reward computation happens here -- pass the returned list to
    `eval_code.score_completions` for that.

    You may specify a split, a batch size, and an optional maximum number of
    samples (`max_samples`; -1 means "no limit"). `max_new_tokens` caps
    generation length exactly like `max_completion_length` does during GRPO
    training (pass the same value used there so truncation behaves identically
    at test time). `temperature`/`top_p` are forwarded to `run_batched_inference`:
    leaving both as `None` keeps generation greedy (the default), passing
    either switches to sampling with that parameter.
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
