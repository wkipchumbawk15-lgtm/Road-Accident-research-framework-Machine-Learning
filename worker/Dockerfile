FROM python:3.10-slim

WORKDIR /app

# Copy worker-specific requirements and install
COPY worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code and shared utils
COPY worker/main.py ./main.py
COPY utils ./utils

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
