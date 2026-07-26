# UniversalMedNER

Fine-tuning [MedGemma](https://huggingface.co/google/medgemma-1.5-4b-it) for biomedical named-entity recognition, via LoRA supervised fine-tuning (SFT) followed by LoRA GRPO reinforcement learning. Training data comes from [`disi-unibo-nlp/Pile-NER-biomed-IOB`](https://huggingface.co/datasets/disi-unibo-nlp/Pile-NER-biomed-IOB).

The model is given the text plus a candidate list of entity types and must extract spans for each.

## Setup

1. **Python environment.** A local venv is expected at `.venv` (Python 3.11, managed with [`uv`](https://github.com/astral-sh/uv)):

   ```bash
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. **Secrets.** Add a `.env` file in the project root:

   ```
   HF_TOKEN=...
   WANDB_API_KEY=...
   ```

   `HF_TOKEN` must belong to a Hugging Face account with access to MedGemma.

## Running the pipeline

Everything runs from the repo root. Each run is identified by a `--label`, which maps to a `data/<label>/` directory holding all outputs (metrics, checkpoints, completions).

```bash
src/run_full_ner_pipeline.sh --label=my_run
```

Each script runs five stages end-to-end (or only the numbered steps passed via `-1`...`-5`):

1. Test the baseline (no adapter) model → `data/<label>/baseline_metrics.json`
2. LoRA SFT training → checkpoint pushed to the Hub under `<label>_ner_sft`
3. Test the SFT checkpoint → `data/<label>/sft_metrics.json`
4. LoRA GRPO training on top of the SFT checkpoint → checkpoint pushed under `<label>_ner_grpo`
5. Test the GRPO checkpoint → `data/<label>/grpo_metrics.json`

Hyperparameters are plain CLI flags with defaults set at the top of each shell script; override any of them with `--flag=value`:

```bash
src/run_full_ner_pipeline.sh --label=my_run -2 -3 --ner_sft_learning_rate=1e-4 --ner_sft_num_epochs=2
```

Pass `--profile=smoke` for a fast, fixed-configuration end-to-end sanity check:

```bash
src/run_full_ner_pipeline.sh --label=smoke --profile=smoke
```

## Project structure

```
src/
├── dataset_code.py         # dataset and instruction format
├── inference_code.py       # model inference
├── eval_code.py            # metrics and reward functions
├── train_code.py           # SFT/GRPO training loops
├── util_code.py            # shared helpers
├── train_ner_pipeline.py   # run SFT or GRPO training
├── test_ner_pipeline.py    # evaluate a checkpoint on the test split
└── run_full_ner_pipeline.sh    # end-to-end pipeline
```

## Results

The training pipeline (SFT + GRPO) increased soft F1 on the test split from 0.48 (baseline) to 0.76.

## Coming soon

- **Schema-free NER** — instead of being given a candidate list of entity types, the model would be given only the text and would have to both discover and type every entity itself.

## References

- Hu, E. J. et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). arXiv:2106.09685.
- Shao, Z. et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). arXiv:2402.03300. (Introduces GRPO.)
- Google. [MedGemma model card](https://huggingface.co/google/medgemma-1.5-4b-it).
- [`disi-unibo-nlp/Pile-NER-biomed-IOB`](https://huggingface.co/datasets/disi-unibo-nlp/Pile-NER-biomed-IOB) — training dataset.
- Hugging Face [TRL](https://github.com/huggingface/trl) — SFT/GRPO trainers.
- Hugging Face [PEFT](https://github.com/huggingface/peft) — LoRA implementation.
