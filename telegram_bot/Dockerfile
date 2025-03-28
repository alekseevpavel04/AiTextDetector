# Use a more lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies with specific versions to avoid compatibility issues
RUN pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Command to run the Telegram bot
CMD ["python", "bot.py"]