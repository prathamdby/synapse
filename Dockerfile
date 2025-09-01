# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY requirements.txt pyproject.toml ./

# Install dependencies using pip
# In production environments with proper internet access, 
# you can replace this with: RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || \
    echo "Warning: Could not install from PyPI. Pre-built wheels or offline installation may be required."

# Copy application code
COPY . .

# Create directory for database (if using file-based SQLite)
RUN mkdir -p /app/data

# Set the default database path to the data directory
ENV DATABASE_PATH=/app/data/bot_database.db

# Expose port (if needed for health checks or monitoring)
EXPOSE 8000

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Run the application
CMD ["python", "main.py"]