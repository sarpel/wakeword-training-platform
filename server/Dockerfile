FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY server/requirements.txt ./requirements.txt

# Install dependencies (respecting the +cpu index)
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ ./src/
COPY server/ ./server/

# Fix PYTHONPATH so internal server imports work correctly
ENV PYTHONPATH="/app:/app/server"
ENV PYTHONUNBUFFERED=1

# Intel CPU Optimizations for Torch
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV TORCH_NUM_THREADS=4

# Expose port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
