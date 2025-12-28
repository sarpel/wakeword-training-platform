# ==============================================================================
# Wakeword Training Platform - Unified Dockerfile
# Multi-stage build with CUDA support and production hardening
# ==============================================================================
# syntax=docker/dockerfile:1.4

# ------------------------------------------------------------------------------
# Build Arguments (configurable at build time)
# ------------------------------------------------------------------------------
ARG CUDA_VERSION=11.8.0
ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.1.2

# ==============================================================================
# Stage 1: Base image with CUDA and system dependencies
# ==============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04 AS base

# Metadata labels for container identification
LABEL maintainer="Wakeword Training Platform"
LABEL version="2.0.0"
LABEL description="Production-ready wakeword detection training system"

# Environment configuration
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Pip configuration for faster installs
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies in a single layer (reduces image size)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    # Audio processing dependencies
    libsndfile1 \
    ffmpeg \
    # OpenMP for parallel processing (PyTorch uses this)
    libgomp1 \
    # Health check utility
    curl \
    # Timezone data (prevents prompts during builds)
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symlinks for Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ==============================================================================
# Stage 2: Dependencies Layer (cached separately for faster rebuilds)
# ==============================================================================
FROM base AS dependencies

WORKDIR /app

# Copy requirements first for Docker layer caching
# This layer is only rebuilt when requirements.txt changes
COPY requirements.txt .

# Install PyTorch with CUDA support first (largest dependency)
# Using BuildKit cache mount for pip cache persistence across builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    torch==2.1.2+cu118 \
    torchaudio==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install remaining requirements (also uses BuildKit cache)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ==============================================================================
# Stage 3: Production (Dashboard & Inference Server)
# ==============================================================================
FROM dependencies AS production

WORKDIR /app

# Create non-root user for security (principle of least privilege)
# UID 1000 is commonly used and works well with volume mounts
RUN useradd -m -u 1000 -s /bin/bash wakeword && \
    mkdir -p /app/data /app/models /app/exports /app/logs /app/configs && \
    chown -R wakeword:wakeword /app

# Copy application code with proper ownership
COPY --chown=wakeword:wakeword src/ ./src/
COPY --chown=wakeword:wakeword run.py .
COPY --chown=wakeword:wakeword server/ ./server/
COPY --chown=wakeword:wakeword configs/ ./configs/
COPY --chown=wakeword:wakeword entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# SECURITY: Make application code read-only to prevent runtime modifications
# Data directories remain writable for model outputs
RUN find /app/src -type f -exec chmod 444 {} \; && \
    find /app/src -type d -exec chmod 555 {} \; && \
    chmod 444 /app/run.py && \
    find /app/server -type f -exec chmod 444 {} \; && \
    find /app/server -type d -exec chmod 555 {} \; && \
    # Configs might need to be writable for saving user configs
    chmod 755 /app/configs

# Switch to non-root user
USER wakeword

# Expose ports:
# 7860 - Gradio Dashboard
# 8000 - FastAPI Inference Server
# 8888 - Jupyter Lab (development)
EXPOSE 7860 8000 8888

# Health check for container orchestration (Docker Swarm, Kubernetes, etc.)
# Checks if the Gradio dashboard is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Use exec form for proper signal handling
ENTRYPOINT ["./entrypoint.sh"]
CMD ["dashboard"]

# ==============================================================================
# Stage 4: Development (includes dev tools and Jupyter)
# ==============================================================================
FROM production AS development

# Switch back to root to install additional packages
USER root

# Install development dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    pytest \
    pytest-cov \
    black \
    flake8 \
    isort \
    mypy \
    jupyterlab \
    notebook \
    ipywidgets

# Setup Jupyter directories with proper permissions
RUN mkdir -p /home/wakeword/.jupyter && \
    chown -R wakeword:wakeword /home/wakeword/.jupyter

# Switch back to non-root user
USER wakeword

# Override healthcheck for Jupyter
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1

CMD ["jupyter"]
