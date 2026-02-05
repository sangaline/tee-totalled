# Dockerfile for TeeTotalledBot.
# Build for linux/amd64 (required for Phala's Intel TDX).

FROM python:3.12-slim AS base

# Install system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first for better layer caching.
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

# Install the package (with production extras for dstack-sdk).
RUN uv pip install --system --no-cache ".[production]"

# Set environment variables.
ENV PYTHONUNBUFFERED=1
ENV TEE_ENV=production

# Run the bot.
CMD ["python", "-m", "tee_totalled"]
