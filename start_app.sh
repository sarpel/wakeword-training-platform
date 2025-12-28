#!/bin/bash
echo "Starting Wakeword Training Platform..."

# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH to current directory to ensure imports work
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the application
python src/ui/app.py
