FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy server requirements
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
# We copy the 'src' directory to allow importing shared modules
COPY src/ ./src/
COPY server/ ./server/

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Intel CPU Optimizations
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV TORCH_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

