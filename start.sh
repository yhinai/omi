#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DIR/logs"
PID_FILE="$DIR/logs/omi_bridge.pid"

# Stop previous tracked PID cleanly (triggers SIGTERM -> on_cleanup queue drain)
if [ -f "$PID_FILE" ]; then
  OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
  fi
  rm -f "$PID_FILE"
fi

# Fallback if an untracked process holds :8787
PORT_PID=$(lsof -ti:8787 2>/dev/null || true)
if [ -n "$PORT_PID" ]; then
  kill $PORT_PID 2>/dev/null || true
  sleep 1
fi

if [ ! -x "$DIR/.venv/bin/python" ]; then
  python3 -m venv "$DIR/.venv"
  "$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt"
fi

"$DIR/.venv/bin/python" "$DIR/omi_bridge.py" < /dev/null >> "$DIR/logs/omi_bridge.log" 2>&1 &
NEW_PID=$!
disown "$NEW_PID" 2>/dev/null || true
echo "$NEW_PID" > "$PID_FILE"

echo "Omi bridge started on 0.0.0.0:8787 (PID $NEW_PID)"
echo "Health check:   http://127.0.0.1:8787/health"
echo "Transcript log: $DIR/logs/omi_live_transcript.log"
echo "Daily notes:    $DIR/logs/$(date +%Y-%m-%d).md"
