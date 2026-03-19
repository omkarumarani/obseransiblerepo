FROM python:3.12-slim

WORKDIR /app

# Copy shared obs_intelligence package first (required for -e install below)
COPY obs-intelligence/app/obs_intelligence /app/obs_intelligence

# Install service dependencies (includes -e /app/obs_intelligence)
COPY compute-agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY compute-agent/app/ ./app/

# Expose the compute-agent webhook port
EXPOSE 9000

# Start uvicorn with the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000", "--log-level", "info"]
