FROM python:3.11-slim

WORKDIR /app

RUN pip install uv
COPY pyproject.toml .
RUN uv pip install --system .

COPY . .

CMD ["python", "main.py"]