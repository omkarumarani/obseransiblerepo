#!/usr/bin/env python3
"""
train_lora.py
─────────────────────────────────────────────────────────────────────────────
Fine-tune Qwen2.5 with a LoRA adapter on obs-intelligence incident validation
data exported by export_training_data.py.

The trained adapter makes the local LLM better at producing meaningful
corroboration/divergence verdicts for the specific incident patterns seen in
this deployment.

Architecture
────────────
  Base model : Qwen/Qwen2.5-7B-Instruct  (configurable via --base-model)
  Adapter    : LoRA  r=8, alpha=16, dropout=0.05  (PEFT)
  Trainer    : HuggingFace TRL SFTTrainer  (chat-format dataset)
  Precision  : bfloat16 + 4-bit NF4 quantisation via bitsandbytes (optional)

After training the script:
  1. Saves the raw LoRA adapter to   /output/lora_adapter/
  2. Writes a Modelfile to          /output/Modelfile
     (for `ollama create aiops-validator -f /output/Modelfile`)

Usage
─────
  # Inside the Docker container:
  python train_lora.py \\
    [--data-dir    /output]          \\   # directory with train.jsonl + eval.jsonl
    [--output-dir  /output]          \\   # where to save adapter + Modelfile
    [--base-model  Qwen/Qwen2.5-7B-Instruct] \\
    [--epochs      3]                \\
    [--batch-size  4]                \\
    [--lr          2e-4]             \\
    [--lora-r      8]                \\
    [--lora-alpha  16]               \\
    [--use-4bit]                        # enable 4-bit NF4 quantisation

After training, load the adapter into Ollama:
  # Convert base model to GGUF (requires llama.cpp — see README)
  # Then:
  ollama create aiops-validator -f /output/Modelfile

  # Update LOCAL_LLM_MODEL env var in docker-compose.yml:
  LOCAL_LLM_MODEL: "aiops-validator"

Requirements
────────────
  See requirements.txt in this directory.
  GPU with ≥ 16 GB VRAM recommended for 7B; use Qwen/Qwen2.5-3B-Instruct for 8 GB.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _check_imports() -> None:
    missing = []
    for pkg in ("torch", "transformers", "peft", "trl", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: Missing packages: {missing}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)


def main() -> None:  # noqa: C901  (complexity OK for a training script)
    _check_imports()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 with LoRA on AIOps data")
    parser.add_argument("--data-dir",   default=os.getenv("OUTPUT_DIR", "/output"))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "/output"))
    parser.add_argument("--base-model", default=os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--epochs",     type=int,   default=int(os.getenv("TRAIN_EPOCHS", "3")))
    parser.add_argument("--batch-size", type=int,   default=int(os.getenv("BATCH_SIZE", "4")))
    parser.add_argument("--lr",         type=float, default=float(os.getenv("LEARNING_RATE", "2e-4")))
    parser.add_argument("--lora-r",     type=int,   default=int(os.getenv("LORA_R", "8")))
    parser.add_argument("--lora-alpha", type=int,   default=int(os.getenv("LORA_ALPHA", "16")))
    parser.add_argument("--use-4bit",   action="store_true",
                        default=os.getenv("USE_4BIT", "false").lower() == "true")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "lora_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    train_file = data_dir / "train.jsonl"
    eval_file  = data_dir / "eval.jsonl"

    if not train_file.exists():
        print(f"ERROR: {train_file} not found.  Run export_training_data.py first.")
        sys.exit(1)

    def _load_jsonl(path: Path) -> list[dict]:
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    train_data = _load_jsonl(train_file)
    eval_data  = _load_jsonl(eval_file) if eval_file.exists() else []

    print(f"Dataset: {len(train_data)} train, {len(eval_data)} eval rows")
    if len(train_data) < 5:
        print("ERROR: Not enough training rows. Generate more data first.")
        sys.exit(1)

    train_dataset = Dataset.from_list(train_data)
    eval_dataset  = Dataset.from_list(eval_data) if eval_data else None

    # ── 2. Load tokeniser ─────────────────────────────────────────────────────
    print(f"Loading tokeniser from {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 3. Load base model ────────────────────────────────────────────────────
    print(f"Loading base model {args.base_model} (4-bit={args.use_4bit}) ...")
    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # ── 4. Apply LoRA ─────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 5. Format chat messages using the tokeniser's chat template ───────────
    def _format_chat(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_dataset = train_dataset.map(_format_chat)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(_format_chat)

    # ── 6. Train ──────────────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=eval_dataset is not None,
        report_to="none",       # disable wandb / tensorboard by default
        dataset_text_field="text",
        max_seq_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print(f"Training for {args.epochs} epoch(s)  lr={args.lr}  batch={args.batch_size} ...")
    trainer.train()

    # ── 7. Save adapter ───────────────────────────────────────────────────────
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved → {adapter_dir}")

    # ── 8. Write Ollama Modelfile ─────────────────────────────────────────────
    # After converting the merged model to GGUF (see README), this Modelfile
    # lets you pull the fine-tuned model into Ollama with a single command.
    modelfile_path = output_dir / "Modelfile"
    modelfile_content = (
        "# Ollama Modelfile for the AIOps fine-tuned validator\n"
        "# Usage:\n"
        "#   1. Convert merged model to GGUF:  see obs-intelligence/finetune/README.md\n"
        "#   2. ollama create aiops-validator -f ./Modelfile\n"
        "#   3. Set LOCAL_LLM_MODEL=aiops-validator in docker-compose.yml\n"
        "\n"
        f"FROM ./merged_model.gguf\n"
        "\n"
        "SYSTEM \"\"\"\n"
        "You are an expert SRE validation assistant in an AIOps platform.\n"
        "Evaluate the external LLM analysis against historical incidents and return\n"
        "a JSON verdict: corroborated | weak_support | divergent | insufficient_context.\n"
        "\"\"\"\n"
        "\n"
        "PARAMETER temperature 0.1\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER stop \"<|endoftext|>\"\n"
    )
    modelfile_path.write_text(modelfile_content)
    print(f"Modelfile written → {modelfile_path}")
    print(
        "\nNext steps:\n"
        "  1. Merge adapter into base model (if needed for GGUF export):\n"
        "       python -c \"from peft import AutoPeftModelForCausalLM; "
        "m=AutoPeftModelForCausalLM.from_pretrained('/output/lora_adapter'); "
        "m.merge_and_unload().save_pretrained('/output/merged_model')\"\n"
        "  2. Convert to GGUF with llama.cpp convert script.\n"
        "  3. ollama create aiops-validator -f /output/Modelfile\n"
        "  4. Set LOCAL_LLM_MODEL=aiops-validator in docker-compose.yml and restart."
    )


if __name__ == "__main__":
    main()
