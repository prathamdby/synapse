FROM python:3.11.9-slim

# Install Node.js and npm for MCP servers
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install uv
COPY pyproject.toml .
RUN uv sync

COPY . .

CMD ["uv", "run", "main.py"]
