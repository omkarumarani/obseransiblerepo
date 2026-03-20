FROM python:3.12-slim

WORKDIR /app

# Ensure the obs_intelligence package is importable as a top-level module
ENV PYTHONPATH=/app/app

# Install dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (includes the obs_intelligence package)
COPY app/ ./app/

# Expose the intelligence engine port
EXPOSE 9100

# Start uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9100", "--log-level", "info"]
