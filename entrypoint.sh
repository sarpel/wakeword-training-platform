#!/bin/bash
set -e

# Default to dashboard if no command is provided
SERVICE=${1:-dashboard}

case "$SERVICE" in
  dashboard)
    echo "Starting Wakeword Training Dashboard..."
    python run.py
    ;;
  server)
    echo "Starting Inference Server (The Judge)..."
    # Ensure src is in PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:/app
    uvicorn server.app:app --host 0.0.0.0 --port 8000
    ;;
  jupyter)
    echo "Starting Jupyter Lab..."
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password='' --allow-root
    ;;
  test)
    echo "Running Tests..."
    pytest tests/
    ;;
  *)
    # If the user provides their own command, execute it
    exec "$@"
    ;;
esac
