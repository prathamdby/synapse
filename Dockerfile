FROM python:3.11.9-slim

WORKDIR /app

RUN pip install uv
COPY pyproject.toml .
RUN uv sync

COPY . .

CMD ["uv", "run", "main.py"]
