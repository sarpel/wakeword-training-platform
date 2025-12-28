#!/bin/bash
# ==============================================================================
# Wakeword Training Platform - Entrypoint Script
# Handles service routing and graceful shutdown
# ==============================================================================
set -e

# ------------------------------------------------------------------------------
# Signal Handling for Graceful Shutdown
# ------------------------------------------------------------------------------
# Trap SIGTERM and SIGINT to allow graceful shutdown in containers
# This is critical for Docker/Kubernetes to properly stop containers
cleanup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Received shutdown signal, cleaning up..."
    # Kill all child processes gracefully
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Shutdown complete."
    exit 0
}

trap cleanup SIGTERM SIGINT

# ------------------------------------------------------------------------------
# Logging Helper
# ------------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ------------------------------------------------------------------------------
# Service Router
# ------------------------------------------------------------------------------
SERVICE=${1:-dashboard}

case "$SERVICE" in
    dashboard)
        log "Starting Wakeword Training Dashboard (Gradio)..."
        log "Access at: http://localhost:7860"
        # Run in background and capture PID for signal handling
        python run.py &
        child=$!
        wait "$child"
        ;;
        
    server)
        log "Starting Inference Server (The Judge)..."
        log "API available at: http://localhost:8000"
        log "Health check: http://localhost:8000/health"
        log "Docs: http://localhost:8000/docs"
        # Ensure src is in PYTHONPATH
        export PYTHONPATH="${PYTHONPATH:-}:/app"
        uvicorn server.app:app --host 0.0.0.0 --port 8000 &
        child=$!
        wait "$child"
        ;;
        
    jupyter)
        log "Starting Jupyter Lab..."
        log "Access at: http://localhost:8888"
        jupyter lab \
            --ip=0.0.0.0 \
            --port=8888 \
            --no-browser \
            --NotebookApp.token='' \
            --NotebookApp.password='' \
            --allow-root \
            --ServerApp.allow_origin='*' &
        child=$!
        wait "$child"
        ;;
        
    test)
        log "Running Test Suite..."
        pytest tests/ -v --tb=short
        ;;
        
    shell)
        log "Starting interactive shell..."
        exec /bin/bash
        ;;
        
    health)
        # Simple health check endpoint for debugging
        log "Running health check..."
        python -c "
import sys
try:
    import torch
    import gradio
    import librosa
    print('Health check passed!')
    print(f'  PyTorch: {torch.__version__}')
    print(f'  Gradio: {gradio.__version__}')
    print(f'  CUDA: {torch.cuda.is_available()}')
    sys.exit(0)
except Exception as e:
    print(f'Health check failed: {e}')
    sys.exit(1)
"
        ;;
        
    *)
        # If the user provides their own command, execute it
        log "Executing custom command: $*"
        exec "$@"
        ;;
esac
