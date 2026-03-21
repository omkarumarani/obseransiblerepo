# obs-intelligence/finetune/README.md — LoRA fine-tuning guide

## Overview

This directory contains a two-step pipeline to fine-tune a LoRA adapter on the
incident validation data recorded by the obs-intelligence service.  After
training you can load the adapter into Ollama as a new model and point
`LOCAL_LLM_MODEL` at it so the platform uses the domain-tuned validator instead
of the generic `qwen3.5`.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| NVIDIA GPU with ≥ 16 GB VRAM | RTX 4070 (12 GB) works with `--use-4bit`; 3B model works on 8 GB |
| NVIDIA Container Toolkit | Required for GPU passthrough into Docker |
| ≥ 20 real incidents recorded | Fewer rows → overfitting; use `--min-quality 0` to include all rows |

---

## Step 1 — Export training data

```bash
docker compose --profile finetune run --rm finetuner \
    python export_training_data.py \
    --db-dir /data \
    --output-dir /output \
    --eval-split 0.10
```

This reads `learning.db` and produces `/output/train.jsonl` + `/output/eval.jsonl`.

---

## Step 2 — Fine-tune

```bash
docker compose --profile finetune run --rm finetuner \
    python train_lora.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --epochs 3 \
    --batch-size 4 \
    --use-4bit
```

For a 3B model (8 GB VRAM):
```bash
--base-model Qwen/Qwen2.5-3B-Instruct --batch-size 8
```

The LoRA adapter is saved to `obs-intelligence/finetune/output/lora_adapter/`.

---

## Step 3 — Merge + convert to GGUF

```bash
# Inside the finetuner container
python -c "
from peft import AutoPeftModelForCausalLM
m = AutoPeftModelForCausalLM.from_pretrained('/output/lora_adapter')
m.merge_and_unload().save_pretrained('/output/merged_model')
print('Merged model saved')
"
```

Then convert to GGUF using [llama.cpp](https://github.com/ggerganov/llama.cpp):
```bash
python convert_hf_to_gguf.py /output/merged_model --outfile /output/merged_model.gguf
```

---

## Step 4 — Load into Ollama

```bash
ollama create aiops-validator -f obs-intelligence/finetune/output/Modelfile
ollama list   # confirm aiops-validator appears
```

---

## Step 5 — Switch the platform to the fine-tuned model

In `docker-compose.yml`, change the `obs-intelligence` environment:
```yaml
LOCAL_LLM_MODEL: "aiops-validator"
```

Then restart the service:
```bash
docker compose up -d --force-recreate obs-intelligence
```

---

## Re-training

Run steps 1–4 again after accumulating more incident data.  The adapter is
stored in the local `output/` directory (gitignored) so it does not bloat the
repository.
