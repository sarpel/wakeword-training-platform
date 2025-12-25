# ==============================================================================
# Wakeword Training Platform - Unified Dockerfile
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
    curl \
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
# Stage 3: Production (Dashboard)
FROM dependencies AS production

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 wakeword && \
    mkdir -p /app/data /app/models /app/exports /app/logs && \
    chown -R wakeword:wakeword /app

# Copy application code
COPY --chown=wakeword:wakeword src/ ./src/
COPY --chown=wakeword:wakeword run.py .
COPY --chown=wakeword:wakeword server/ ./server/
COPY --chown=wakeword:wakeword configs/ ./configs/
COPY --chown=wakeword:wakeword entrypoint.sh .
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER wakeword

# Expose ports
EXPOSE 7860 8000 8888

ENTRYPOINT ["./entrypoint.sh"]
CMD ["dashboard"]

# ============================================================================== 
# Stage 4: Development
FROM production AS development

USER root

# Install dev tools and Jupyter
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    isort \
    mypy \
    jupyterlab \
    notebook

# Ensure notebook directories exist
RUN mkdir -p /home/wakeword/.jupyter && \
    chown -R wakeword:wakeword /home/wakeword/.jupyter

USER wakeword

CMD ["jupyter"]