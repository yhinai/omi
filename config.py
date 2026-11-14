"""Shared, non-executing configuration parsing; imports no application runtime."""
import os
from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parent
VEXO_BAND = 43


def read_env_file(path):
    values = {}
    path = Path(path)
    if path.is_file():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
                try:
                    values[key.strip()] = " ".join(shlex.split(value, comments=True))
                except ValueError:
                    raise ValueError(f"Invalid quoted value in {path}:{number}") from None
    return values


def load_env(local_path=None, environ=None):
    values = read_env_file(ROOT / ".env" if local_path is None else local_path)
    values.update(os.environ if environ is None else environ)
    return values
