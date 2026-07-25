"""Task-agnostic helpers shared across pipelines."""

from pathlib import Path
import huggingface_hub
import peft

def load_adapter(model, checkpoint_repo, checkpoint_folder):
    """Download a LoRA adapter from the hub and attach it to the model."""
    adapter_dir = huggingface_hub.snapshot_download(
        repo_id=checkpoint_repo,
        allow_patterns=f"{checkpoint_folder}/*"
    )
    adapter_dir = Path(adapter_dir) / checkpoint_folder
    return peft.PeftModel.from_pretrained(model, adapter_dir)
