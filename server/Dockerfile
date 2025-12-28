# ==============================================================================
# Wakeword Inference Server - Lightweight CPU-Only Dockerfile
# Optimized for production inference without GPU dependencies
# ==============================================================================
# syntax=docker/dockerfile:1.4

FROM python:3.10-slim AS base

# Metadata labels
LABEL maintainer="Wakeword Training Platform"
LABEL version="2.0.0"
LABEL description="Lightweight CPU-only inference server for wakeword detection"

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app:/app/server" \
    # Intel CPU Optimizations for PyTorch
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TORCH_NUM_THREADS=4 \
    # Pip configuration
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio processing
    libsndfile1 \
    ffmpeg \
    # Build tools (needed for some pip packages)
    build-essential \
    # OpenMP for parallel processing
    libgomp1 \
    # Health check utility
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ==============================================================================
# Dependencies Stage
# ==============================================================================
FROM base AS dependencies

# Copy requirements first for caching
COPY server/requirements.txt ./requirements.txt

# Install dependencies with CPU-only PyTorch
# Using --extra-index-url to get CPU-only builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ==============================================================================
# Production Stage
# ==============================================================================
FROM dependencies AS production

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash inference && \
    mkdir -p /app/models /app/configs && \
    chown -R inference:inference /app

# Copy project source with proper ownership
COPY --chown=inference:inference src/ ./src/
COPY --chown=inference:inference server/ ./server/

# Make source read-only for security
RUN find /app/src -type f -exec chmod 444 {} \; && \
    find /app/src -type d -exec chmod 555 {} \; && \
    find /app/server -type f -exec chmod 444 {} \; && \
    find /app/server -type d -exec chmod 555 {} \;

# Switch to non-root user
USER inference

# Expose inference port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server with uvicorn
# Using exec form for proper signal handling
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
