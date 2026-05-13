# --- IMPORTS ---

from pathlib import Path
import peft
import trl


# MODEL FINE TUNE CODE

def collate_fn(samples, processor):
    """Preprocess and collate batch of samples.
    
    This function is responsible for applying chat templates, tokenizing the prompts,
    and constructing the labels.
    """
    # Compute text (per sample)
    texts = []
    for sample in samples:
        text = processor.apply_chat_template(
            sample["messages"],
            add_generation_prompt=False,
            tokenize=False # We tokenize the batch later so padding is automatically added
        ).strip()
        texts.append(text)

    # Tokenize and pad batch
    batch = processor(text=texts, return_tensors="pt", padding=True)

    # Create labels
    labels = batch["input_ids"].clone()

    # Mask away label tokens that are not in the model response with -100 so they are ignored by the loss
    labels[labels == processor.tokenizer.pad_token_id] = -100
    start_of_turn_id = processor.tokenizer.convert_tokens_to_ids("<start_of_turn>")
    model_id = processor.tokenizer.convert_tokens_to_ids("model")
    for i in range(len(texts)):
        ids = batch["input_ids"][i]
        for j in range(len(ids) - 1):
            if ids[j] == start_of_turn_id and ids[j+1] == model_id:
                labels[i, :j] = -100
                break

    # Add labels to batch
    batch["labels"] = labels
    return batch

def execute_sft(
    model,
    processor,
    ds,
    save_folder,
    learning_rate=2e-4,
    lora_rank=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    max_train_samples=None,
    max_validation_samples=None,
):
    """Set up and execut supervised fine tuning on the MedGemma."""
    # Prepare datasets
    train_dataset = ds["train"].select(range(max_train_samples)) if max_train_samples else ds["train"]
    eval_dataset = ds["validation"].select(range(max_validation_samples)) if max_validation_samples else ds["validation"]

    # Prepare configs
    peft_config = peft.LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=lora_rank,
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )
    sft_config = trl.SFTConfig(
        output_dir=str(save_folder / 'sft_out'),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=50,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=learning_rate,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns = False,
        label_names=["labels"],
        report_to="none"
    )

    # Initialize trainer
    sft_trainer = trl.SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=lambda samples: collate_fn(samples, processor),
    )

    # Train
    sft_trainer.train()

    # Save results
    model.save_pretrained(save_folder / "model")
    processor.save_pretrained(save_folder / "processor")