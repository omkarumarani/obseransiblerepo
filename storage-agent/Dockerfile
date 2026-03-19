FROM python:3.11-slim

WORKDIR /app

# Copy shared obs_intelligence package first (required for -e install below)
COPY obs-intelligence/app/obs_intelligence /app/obs_intelligence

# Install service dependencies (includes -e /app/obs_intelligence)
COPY storage-agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application package
COPY storage-agent/app/ ./app/

EXPOSE 9001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9001"]
