# Use a more lightweight Python image instead of full Ubuntu
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Command to run the Telegram bot
CMD ["python", "bot.py"]