FROM python:3.10-slim

WORKDIR /app

# Use service-specific requirements
COPY prediction-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only this service's files so main.py is at /app
COPY prediction-service/ .
COPY utils ./utils

# Copy shared utilities and models so imports like `from utils...` work
COPY utils/ ./utils/
# Put models in a Docker-safe path
COPY models/ /models/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

