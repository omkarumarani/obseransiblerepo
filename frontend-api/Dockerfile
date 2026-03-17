# frontend-api/Dockerfile
# -----------------------
# Builds the Frontend API container.
#
# python:3.11-slim keeps the image small while giving us a full CPython runtime.
# Dependencies are installed in a separate layer so Docker can cache them;
# the layer is only invalidated when requirements.txt changes.

FROM python:3.11-slim

# Set a clean working directory inside the container
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ────────────────────────────────────
COPY app/ ./app/

# ── Runtime ───────────────────────────────────────────────────
EXPOSE 8080

# uvicorn runs the FastAPI app.
# --host 0.0.0.0  → listen on all interfaces (required inside Docker)
# --port 8080     → must match EXPOSE and docker-compose ports mapping
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
