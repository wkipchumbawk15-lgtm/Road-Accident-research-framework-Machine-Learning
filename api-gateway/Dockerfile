# Dockerfile for api-gateway

# Use a lightweight Python image
FROM python:3.10-slim

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the service-specific requirements to the container
COPY api-gateway/requirements.txt ./

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files and utils from the parent directory
COPY .. ./  
# Copy everything from the parent directory

# Set the command to run the application using Uvicorn
CMD ["uvicorn", "api-gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]