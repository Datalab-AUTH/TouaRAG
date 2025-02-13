#!/bin/bash

# Ensure 'moreutils' is installed for the 'ts' command
if ! command -v ts &> /dev/null; then
    echo "Installing moreutils for timestamp functionality..."
    sudo apt-get install -y moreutils
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")

# Start FastAPI
nohup fastapi dev api/app.py 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > "logs/fastapi_${timestamp}.log" &
echo "FastAPI started in background. Logs are in logs/fastapi_${timestamp}.log"

# Start Streamlit
nohup streamlit run ui/Welcome.py 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > "logs/streamlit_${timestamp}.log" &
echo "Streamlit started in background. Logs are in logs/streamlit_${timestamp}.log"