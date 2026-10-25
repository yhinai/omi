#!/bin/bash
set -eu
DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON="$DIR/.venv/bin/python3.15"
if [ ! -x "$PYTHON" ]; then
  echo "Python 3.15 environment missing. Set up the bridge with ~/omi/run.sh first." >&2
  exit 1
fi
exec "$PYTHON" -B "$DIR/omi_tui.py" "$@"
