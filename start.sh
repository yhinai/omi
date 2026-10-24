#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR/logs"

kill $(lsof -ti:8787) 2>/dev/null || true

PYTHON_BIN=$(ls -d "$HOME"/.hermes/tools/python*/bin/python3 2>/dev/null | head -n 1)
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" "$DIR/omi_bridge.py" >> "$DIR/logs/omi_bridge.log" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true
echo "Omi bridge started on 0.0.0.0:8787 (PID $PID)"
echo "Transcript log: $DIR/logs/omi_live_transcript.log"
echo "Bridge log:     $DIR/logs/omi_bridge.log"
