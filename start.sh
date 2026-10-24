#!/bin/bash
set -euo pipefail
umask 077

DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$DIR/omi_bridge.py"
PYTHON="$DIR/.venv/bin/python"
PID_FILE="$DIR/logs/omi_bridge.pid"
IDENTITY_FILE="$DIR/logs/omi_bridge.identity"
CONSOLE_LOG="$DIR/logs/omi_bridge.console.log"
ACTION="${1:-restart}"

if [ "$#" -gt 1 ] || [[ ! "$ACTION" =~ ^(start|restart|stop|status)$ ]]; then
  echo "Usage: $0 [start|restart|stop|status] (default: restart)" >&2
  exit 2
fi
mkdir -p "$DIR/logs"

# The launcher holds this advisory lock until exit; crashes leave no stale lock.
LOCK_PYTHON="$PYTHON"
[ -x "$LOCK_PYTHON" ] || LOCK_PYTHON=python3
exec 9> "$DIR/logs/omi_bridge.launcher.lock"
if ! "$LOCK_PYTHON" -c 'import fcntl; fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)' 2>/dev/null; then
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
  "$PYTHON" - "$1" <<'PY'
import json
import sys
import urllib.request
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:8788/health", timeout=0.5) as response:
        health = json.load(response)
    sys.exit(0 if health.get("pid") == int(sys.argv[1]) and health.get("status") == "ok" else 1)
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

if [ "$ACTION" = stop ]; then
  stop_bridge
  exit
fi
if [ "$ACTION" = status ]; then
  if PID="$(tracked_pid)"; then
    if [ -x "$PYTHON" ] && health_matches "$PID"; then
      echo "Omi bridge is running and healthy (PID $PID)."
      exit 0
    fi
    echo "Omi bridge process exists (PID $PID), but local health is not ready." >&2
    exit 1
  fi
  echo "Omi bridge is stopped."
  exit 1
fi

# Finish environment setup and validate configuration before touching a live bridge.
runtime_info() {
  "$1" -c 'import platform, ssl, sys, sysconfig
valid = (sys.implementation.name == "cpython" and sys.version_info[:2] >= (3, 14)
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
      echo "OMI_PYTHON must point to a stable, standard CPython 3.14+ interpreter with GIL enabled." >&2
      return 1
    fi
    echo "$resolved"
    return
  fi
  for candidate in python3.15 /opt/homebrew/bin/python3.15 /opt/homebrew/bin/python3 \
      python3.14 /opt/homebrew/bin/python3.14 /usr/local/bin/python3.14 \
      /opt/homebrew/opt/python@3.14/bin/python3.14 /usr/local/opt/python@3.14/bin/python3.14 \
      "$HOME"/.local/share/uv/python/cpython-3.14.*-macos-aarch64-none/bin/python3.14 \
      /Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14; do
    resolved="$(command -v "$candidate" || true)"
    if [ -n "$resolved" ] && runtime_info "$resolved" >/dev/null 2>&1; then
      echo "$resolved"
      return
    fi
  done
  echo "Install standard CPython 3.14+, or set OMI_PYTHON=/absolute/path/to/python3." >&2
  return 1
}

if [ ! -x "$PYTHON" ]; then
  if [ -e "$DIR/.venv" ] || [ -L "$DIR/.venv" ]; then
    echo "Existing .venv is incomplete. Move it aside and recreate it with standard CPython 3.14; see README.md." >&2
    exit 1
  fi
  BOOTSTRAP_PYTHON="$(bootstrap_python)"
  "$BOOTSTRAP_PYTHON" -m venv "$DIR/.venv"
fi
if ! RUNTIME_INFO="$(runtime_info "$PYTHON" 2>/dev/null)"; then
  echo "Existing .venv requires migration to standard CPython 3.14 (GIL enabled)." >&2
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
  echo "Omi bridge PID $PID is already running but not healthy; inspect $CONSOLE_LOG." >&2
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
