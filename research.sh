#!/bin/bash
set -eu
DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$DIR"
exec "$DIR/.venv/bin/python3.15" -B - <<'PY'
import json, os, socket, subprocess, sys, time, urllib.request
from pathlib import Path
root=Path.cwd()
try:
    with urllib.request.urlopen('http://127.0.0.1:8791/health',timeout=2) as response:
        if json.load(response).get('provider')=='Google Search grounding':
            print('Live research is already running on http://127.0.0.1:8791')
            raise SystemExit(0)
except (OSError, ValueError):
    pass
with socket.socket() as check:
    if check.connect_ex(('127.0.0.1',8791))==0:
        raise SystemExit('Port 8791 is occupied; no existing process was changed.')
(root/'logs').mkdir(exist_ok=True)
logpath=root/'logs'/'omi_research.log'
with logpath.open('ab') as log:
    os.chmod(logpath,0o600)
    process=subprocess.Popen([sys.executable,'-B',str(root/'research_server.py')],cwd=root,
        stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
(root/'logs'/'omi_research.pid').write_text(str(process.pid)+'\n')
for _ in range(30):
    if process.poll() is not None:
        raise SystemExit('Research did not start; check logs/omi_research.log.')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8791/health',timeout=1) as response:
            if response.status==200:
                print('Live research ready. Refresh http://127.0.0.1:8788/focus and open Search.')
                raise SystemExit(0)
    except OSError:
        time.sleep(.2)
raise SystemExit('Research startup is still pending; check logs/omi_research.log.')
PY
