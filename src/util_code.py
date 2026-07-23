"""Task-agnostic helpers shared across pipelines.

Unlike dataset_code.py/train_code.py/eval_code.py, nothing here is specific
to any one task (NER or otherwise) — keep it that way. Task-specific logic
belongs in those three files, not here.
"""

# --- IMPORTS ---

from pathlib import Path
import huggingface_hub
import peft


# --- CHECKPOINT LOADING ---

def load_adapter(model, checkpoint_repo, checkpoint_folder):
    """Download a LoRA adapter from the hub and attach it to `model`."""
    adapter_dir = huggingface_hub.snapshot_download(
        repo_id=checkpoint_repo,
        allow_patterns=f"{checkpoint_folder}/*"
    )
    adapter_dir = Path(adapter_dir) / checkpoint_folder
    return peft.PeftModel.from_pretrained(model, adapter_dir)
