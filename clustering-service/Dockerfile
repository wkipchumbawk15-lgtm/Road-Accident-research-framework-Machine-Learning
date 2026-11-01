FROM python:3.10-slim

WORKDIR /app

# Use service-specific requirements
COPY clustering-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only this service's files so main.py is at /app
COPY clustering-service/ .
COPY utils ./utils

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
