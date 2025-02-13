#!/bin/bash

# Kill FastAPI processes
pkill -f "fastapi dev api/app.py"

# Kill Streamlit processes
pkill -f "streamlit run ui/Welcome.py"

echo "FastAPI and Streamlit apps stopped."