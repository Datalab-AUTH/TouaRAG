#!/bin/bash

# Force kill FastAPI processes
pkill -9 -f "fastapi dev api/app.py"

# Force kill Streamlit processes
pkill -9 -f "streamlit run ui/Welcome.py"

# Force kill Anaconda-related processes
pkill -9 -f "anaconda"

echo "FastAPI, Streamlit, and Anaconda-related processes forcefully stopped."