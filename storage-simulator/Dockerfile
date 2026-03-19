FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY simulator.py .

EXPOSE 9200

CMD ["uvicorn", "simulator:app", "--host", "0.0.0.0", "--port", "9200"]
