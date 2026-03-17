FROM python:3.12-slim

WORKDIR /app

# Install FastAPI and uvicorn only — ansible is optional (service
# returns simulated dry-run output when ansible-playbook is not found)
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

COPY app/ ./app/

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
