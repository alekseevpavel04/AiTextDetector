# Start with CUDA supported Python image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variable to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
# Set environment variable for Hugging Face to use the cache directory
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create and set permissions for model cache directory
RUN mkdir -p /root/.cache/huggingface && \
    chmod -R 777 /root/.cache/huggingface

# Copy the application code
COPY . .

# Expose the port for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["python", "app.py"]