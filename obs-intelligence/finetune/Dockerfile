FROM python:3.11-slim

LABEL description="AIOps LoRA fine-tuner for qwen3.5 / Qwen2.5 on incident validation data"

WORKDIR /finetune

# System deps: git is needed so HuggingFace can clone tokeniser configs
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training scripts
COPY export_training_data.py .
COPY train_lora.py .

# The obs-intelligence-data Docker volume is mounted read-only at /data
# for SQLite access.  Training output (adapter + Modelfile) goes to /output.
VOLUME ["/data", "/output"]

# Default: show help.  Override CMD in docker compose run or `docker run`:
#   Step 1 — export data:  docker compose run finetuner python export_training_data.py
#   Step 2 — train:        docker compose run finetuner python train_lora.py
CMD ["python", "export_training_data.py", "--help"]
