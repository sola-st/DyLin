FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[full]' pytest

WORKDIR /workspace
