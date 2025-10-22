#!/bin/bash
# Usage: ./start_motivations_app.sh [PORT]
# Example: ./start_motivations_app.sh 8501

PORT=${1:-8501}
APP_PATH="/home/jupyter/Motivations_test/app.py"
LOG_PATH="/home/jupyter/Motivations_test/streamlit_app.out"

# Kill any previous instance using this port (optional but clean)
PID=$(lsof -ti:$PORT)
if [ -n "$PID" ]; then
  echo "Killing previous process on port $PORT (PID $PID)"
  kill -9 $PID
fi

echo "Starting Streamlit app on port $PORT..."
nohup streamlit run "$APP_PATH" \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true > "$LOG_PATH" 2>&1 &

APP_PID=$!
echo "App started on port $PORT (PID $APP_PID). Logs: $LOG_PATH"
