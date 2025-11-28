# ==============================================================================
# Wakeword Training Platform - Production Dockerfile
# Multi-stage build with CUDA support
# ==============================================================================

# Stage 1: Base image with CUDA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ==============================================================================
# Stage 2: Dependencies
FROM base AS dependencies

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install PyTorch with CUDA
RUN pip install --no-cache-dir \
    torch==2.1.2+cu118 \
    torchaudio==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 3: Production
FROM dependencies AS production

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 wakeword && \
    mkdir -p /app/data /app/models /app/exports /app/logs && \
    chown -R wakeword:wakeword /app

# Copy application code
COPY --chown=wakeword:wakeword src/ ./src/
COPY --chown=wakeword:wakeword run.py .
COPY --chown=wakeword:wakeword configs/ ./configs/

# Switch to non-root user
USER wakeword

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command
CMD ["python", "run.py"]

# ==============================================================================
# Stage 4: Development (optional)
FROM production AS development

USER root

# Install dev tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    isort \
    mypy

USER wakeword

CMD ["bash"]
