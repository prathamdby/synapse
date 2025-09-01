FROM python:3.11-slim

WORKDIR /app

RUN pip install uv
COPY pyproject.toml .
RUN uv sync

COPY . .

CMD ["python", "main.py"]