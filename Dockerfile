FROM python:3.11-slim

WORKDIR /app

# Patch OS-level vulnerabilities present in the base image
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Copy dependency files first (layer cache)
COPY pyproject.toml poetry.lock* ./

# Install dependencies (no dev, no virtualenv inside container)
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Copy source code
COPY src/ ./src/
COPY agents/ ./agents/
COPY run_langgraph.py ./
COPY .env.example ./

# Run as non-root user
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser /app
USER appuser

# Persistent data as volumes
VOLUME ["/app/indexes", "/app/runs", "/app/books_refuge", "/app/books_consultant"]

CMD ["python", "run_langgraph.py"]
