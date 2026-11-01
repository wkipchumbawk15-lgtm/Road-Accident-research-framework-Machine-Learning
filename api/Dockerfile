FROM python:3.10-slim
WORKDIR /app
# Ensure redis client is installed (explicit in case of cache/build issues)
COPY api/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Ensure redis client is installed (explicit in case of cache/build issues)
RUN pip install --no-cache-dir redis

# Copy service sources
COPY api/ ./
COPY ../utils/ ./utils/

# Ensure the application dir is on the import path
ENV PYTHONPATH=/app

CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "5000"]

