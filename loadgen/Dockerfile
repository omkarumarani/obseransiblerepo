# loadgen/Dockerfile
# ------------------
# Builds the optional traffic generator container.
# This is a plain Python script, no web server needed.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY loadgen.py .

CMD ["python", "loadgen.py"]
