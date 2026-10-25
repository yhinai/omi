#!/bin/bash
set -euo pipefail
umask 077

DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$DIR/omi_bridge.py"
PYTHON="$DIR/.venv/bin/python3.15"
PID_FILE="$DIR/logs/omi_bridge.pid"
IDENTITY_FILE="$DIR/logs/omi_bridge.identity"
CONSOLE_LOG="$DIR/logs/omi_bridge.console.log"
ACTION="${1:-restart}"

if [ "$#" -gt 1 ] || [[ ! "$ACTION" =~ ^(start|restart|stop|status|clear)$ ]]; then
  echo "Usage: $0 [start|restart|stop|status|clear] (default: restart)" >&2
  exit 2
fi
mkdir -p "$DIR/logs"

runtime_info() {
  "$1" -c 'import platform, ssl, sys, sysconfig
valid = (sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 15)
         and sys.version_info.releaselevel == "final" and not sysconfig.get_config_var("Py_GIL_DISABLED"))
if not valid:
    sys.exit(1)
print(f"Python {platform.python_version()} ({ssl.OPENSSL_VERSION}; standard GIL)")'
}

bootstrap_python() {
  local candidate resolved
  if [ -n "${OMI_PYTHON:-}" ]; then
    resolved="$(command -v "$OMI_PYTHON" || true)"
    if [ -z "$resolved" ] || ! runtime_info "$resolved" >/dev/null 2>&1; then
      echo "OMI_PYTHON must point to stable, standard CPython 3.15.x with GIL enabled." >&2
      return 1
    fi
    echo "$resolved"
    return
  fi
  for candidate in python3.15 /opt/homebrew/bin/python3.15 /usr/local/bin/python3.15 \
      /opt/homebrew/opt/python@3.15/bin/python3.15 /usr/local/opt/python@3.15/bin/python3.15 \
      "$HOME"/.local/share/uv/python/cpython-3.15.*-macos-aarch64-none/bin/python3.15 \
      /Library/Frameworks/Python.framework/Versions/3.15/bin/python3.15; do
    resolved="$(command -v "$candidate" || true)"
    if [ -n "$resolved" ] && runtime_info "$resolved" >/dev/null 2>&1; then
      echo "$resolved"
      return
    fi
  done
  echo "Install standard CPython 3.15.x, or set OMI_PYTHON=/absolute/path/to/python3.15." >&2
  return 1
}

# Stop/status may inspect an older bridge, but their own helpers use Python 3.15.
if [ -n "${OMI_PYTHON:-}" ]; then
  CLI_PYTHON="$(bootstrap_python)"
elif [ -x "$PYTHON" ] && runtime_info "$PYTHON" >/dev/null 2>&1; then
  CLI_PYTHON="$PYTHON"
else
  CLI_PYTHON="$(bootstrap_python)"
fi

# The launcher holds this advisory lock until exit; crashes leave no stale lock.
exec 9> "$DIR/logs/omi_bridge.launcher.lock"
if ! "$CLI_PYTHON" -c 'import fcntl; fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)' 2>/dev/null; then
  echo "Another launcher command is running." >&2
  exit 1
fi

process_birth() { LC_ALL=C ps -p "$1" -o lstart= 2>/dev/null; }
process_command() { ps -ww -p "$1" -o command= 2>/dev/null; }

is_bridge_process() {
  local command_line program
  command_line="$(process_command "$1")" || return 1
  [[ "$command_line" == *" $SCRIPT" ]] || return 1
  program="${command_line%" $SCRIPT"}"
  [ -x "$program" ] || return 1
  case "$(basename "$program")" in python|python[0-9]*|Python) ;; *) return 1 ;; esac
  [[ "$(ps -p "$1" -o stat= 2>/dev/null)" != *Z* ]]
}

tracked_pid() {
  local pid birth saved_pid saved_birth saved_command
  [ -f "$PID_FILE" ] || return 1
  pid="$(cat "$PID_FILE")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && is_bridge_process "$pid"; then
    birth="$(process_birth "$pid")" || return 1
    if [ -f "$IDENTITY_FILE" ]; then
      saved_pid="$(sed -n '1p' "$IDENTITY_FILE")"
      saved_birth="$(sed -n '2p' "$IDENTITY_FILE")"
      saved_command="$(sed -n '3p' "$IDENTITY_FILE")"
      if [ "$pid" = "$saved_pid" ] && [ "$birth" = "$saved_birth" ] &&
          [ "$(process_command "$pid")" = "$saved_command" ]; then
        echo "$pid"
        return 0
      fi
    else
      # Upgrade an older integer-only PID record only after checking its script.
      echo "$pid"
      return 0
    fi
  fi
  echo "Discarding a stale PID record; no process was signalled." >&2
  rm -f "$PID_FILE" "$IDENTITY_FILE"
  return 1
}

health_matches() {
  "$CLI_PYTHON" - "$1" <<'PY'
import json
import sys
import urllib.request
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:8788/health", timeout=0.5) as response:
        health = json.load(response)
    healthy = (health.get("pid") == int(sys.argv[1]) and health.get("status") == "ok"
               and str(health.get("python_version", "")).startswith("3.15."))
    sys.exit(0 if healthy else 1)
except Exception:
    sys.exit(1)
PY
}

check_ports() {
  local allowed_pid="${1:-}" port listener listeners
  for port in 8787 8788; do
    listeners="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
    for listener in $listeners; do
      if [ "$listener" != "$allowed_pid" ]; then
        echo "Port $port already has a listener. Refusing to stop an untracked process." >&2
        return 1
      fi
    done
  done
}

stop_bridge() {
  local pid birth
  if ! pid="$(tracked_pid)"; then
    echo "No tracked Omi bridge is running."
    return
  fi
  birth="$(process_birth "$pid")"
  echo "Stopping Omi bridge (PID $pid); waiting for queued audio to drain..."
  kill -TERM "$pid" 2>/dev/null || true
  while is_bridge_process "$pid" && [ "$(process_birth "$pid")" = "$birth" ]; do
    sleep 1
  done
  rm -f "$PID_FILE" "$IDENTITY_FILE"
  echo "Omi bridge stopped."
}

clear_data() {
  "$CLI_PYTHON" - "$DIR/logs" <<'PY_CLEAR'
from pathlib import Path
import re
import shutil
import sys
root = Path(sys.argv[1])
if root.is_symlink():
    raise SystemExit("Refusing to clear a symlinked logs directory")
for path in root.iterdir():
    # Preserve the held launcher lock and files owned by separate services.
    owned = (path.name.startswith(("omi_bridge.log", "omi_bridge.console.log",
                                  "omi_live_transcript.log", "omi_transcript_events.jsonl"))
             or re.fullmatch(r"\d{4}-\d{2}-\d{2}\.md", path.name)
             or path.name == "editions"
             or path.suffix.lower() in {".wav", ".pcm"})
    if not owned:
        continue
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
print("Cleared bridge transcripts, daily notes, editions, audio captures, and logs.")
PY_CLEAR
}

if [ "$ACTION" = stop ]; then
  stop_bridge
  exit
fi
if [ "$ACTION" = clear ]; then
  stop_bridge
  check_ports
  clear_data
  echo "Omi bridge is stopped. Start it with: $0 start"
  exit
fi
if [ "$ACTION" = status ]; then
  if PID="$(tracked_pid)"; then
    if health_matches "$PID"; then
      echo "Omi bridge is running and healthy on Python 3.15 (PID $PID)."
      exit 0
    fi
    echo "Omi bridge process exists (PID $PID), but local health does not confirm a healthy Python 3.15 server." >&2
    exit 1
  fi
  echo "Omi bridge is stopped."
  exit 1
fi

# Finish environment setup and validate configuration before touching a live bridge.
if [ ! -x "$PYTHON" ]; then
  if [ -e "$DIR/.venv" ] || [ -L "$DIR/.venv" ]; then
    echo "Existing .venv has no Python 3.15 launcher. Stop gracefully, move it aside, and recreate it; see README.md." >&2
    exit 1
  fi
  "$CLI_PYTHON" -m venv "$DIR/.venv"
fi
if ! RUNTIME_INFO="$(runtime_info "$PYTHON" 2>/dev/null)"; then
  echo "Existing .venv requires migration to standard CPython 3.15.x (GIL enabled)." >&2
  echo "Use '$0 stop' for a graceful stop, then move .venv aside and recreate it; see README.md." >&2
  exit 1
fi
if ! "$PYTHON" -c 'import aiohttp; assert tuple(map(int, aiohttp.__version__.split(".")[:2])) >= (3, 9)' >/dev/null 2>&1; then
  "$PYTHON" -m pip install -r "$DIR/requirements.txt"
fi
"$PYTHON" "$SCRIPT" --check-config
command -v lsof >/dev/null || { echo "lsof is required to check bridge ports safely." >&2; exit 1; }
CURRENT_PID="$(tracked_pid || true)"
check_ports "$CURRENT_PID"

if [ "$ACTION" = restart ]; then
  stop_bridge
elif PID="$(tracked_pid)"; then
  if health_matches "$PID"; then
    echo "Omi bridge is already running and healthy (PID $PID)."
    exit 0
  fi
  echo "Omi bridge PID $PID does not report healthy Python 3.15; migrate/restart and inspect $CONSOLE_LOG." >&2
  exit 1
fi

check_ports

PID="$("$PYTHON" - "$PYTHON" "$SCRIPT" "$CONSOLE_LOG" <<'PY'
import subprocess
import sys
with open(sys.argv[3], "ab", buffering=0) as console:
    process = subprocess.Popen(
        [sys.argv[1], sys.argv[2]], stdin=subprocess.DEVNULL,
        stdout=console, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True,
    )
print(process.pid)
PY
)"
echo "$PID" > "$PID_FILE"
for _ in {1..20}; do
  is_bridge_process "$PID" && break
  kill -0 "$PID" 2>/dev/null || break
  sleep 0.1
done
if ! is_bridge_process "$PID"; then
  echo "Omi bridge failed to start. Inspect $CONSOLE_LOG." >&2
  rm -f "$PID_FILE" "$IDENTITY_FILE"
  exit 1
fi
{ echo "$PID"; process_birth "$PID"; process_command "$PID"; } > "$IDENTITY_FILE"

for _ in {1..30}; do
  if health_matches "$PID"; then
    echo "Omi bridge started and healthy (PID $PID)."
    echo "Runtime:        $RUNTIME_INFO"
    echo "Health:         http://127.0.0.1:8788/health"
    echo "Transcript log: $DIR/logs/omi_live_transcript.log"
    echo "Bridge log:     $DIR/logs/omi_bridge.log"
    echo "Console log:    $CONSOLE_LOG"
    exit 0
  fi
  is_bridge_process "$PID" || break
  sleep 1
done
echo "Omi bridge did not become healthy. Inspect $CONSOLE_LOG (PID $PID)." >&2
exit 1
