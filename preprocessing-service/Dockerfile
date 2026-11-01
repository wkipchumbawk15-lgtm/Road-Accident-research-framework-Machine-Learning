FROM python:3.10-slim

WORKDIR /app

# Use service-specific requirements
COPY preprocessing-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only this service's files so main.py is at /app
COPY preprocessing-service/ .
COPY utils ./utils

# ✅ Expose port 8000
EXPOSE 8000

# ✅ Run on 0.0.0.0 (not localhost!)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]