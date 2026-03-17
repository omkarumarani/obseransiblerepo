# troublemaker/Dockerfile
# -----------------------
# Minimal Python image for the chaos traffic generator.
# No ports exposed — this container only makes outbound HTTP calls.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker caches this layer separately
# from the source code.  Re-builds after code changes are fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY troublemaker.py .

# -u disables Python's output buffering so log lines appear in
# "docker compose logs -f troublemaker" immediately without delay.
CMD ["python", "-u", "troublemaker.py"]
