#!/usr/bin/env python3.15
"""Validate, switch, and verify a transcription backend; preserve history and roll back failures."""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = {"whisper", "deepgram"}
ASSIGNMENT = re.compile(rb"^[ \t]*(?:export[ \t]+)?OMI_STT_BACKEND[ \t]*=[ \t]*")


class SwitchError(Exception):
    """A redacted operator-facing failure."""


@dataclass
class Snapshot:
    data: bytes
    mode: int
    identity: tuple


def read_env(path):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise SwitchError("Project .env must be a regular file, not a symlink.")
            data = source.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                raise SwitchError("Project .env is too large to switch safely.")
    except OSError:
        raise SwitchError("Cannot safely read project .env; check that it exists and is not a symlink.") from None
    return Snapshot(data, stat.S_IMODE(info.st_mode), (info.st_dev, info.st_ino))


def backend_field(data):
    """Find the effective (last) assignment without rewriting other dotenv bytes."""
    result, offset = None, 0
    for line in data.splitlines(keepends=True):
        match = ASSIGNMENT.match(line)
        if match:
            rhs = line[match.end():].rstrip(b"\r\n")
            quote, escaped, end = None, False, len(rhs)
            for index, char in enumerate(rhs):
                if escaped:
                    escaped = False
                elif char == 92 and quote != 39:
                    escaped = True
                elif quote is not None:
                    if char == quote:
                        quote = None
                elif char in (34, 39):
                    quote = char
                elif char == 35:
                    end = index
                    break
            value = rhs[:end].rstrip(b" \t")
            result = (offset + match.end(), offset + match.end() + len(value), value)
        offset += len(line)
    if result is None:
        return None
    start, end, value = result
    try:
        backend = " ".join(shlex.split(value.decode("utf-8"), comments=True)).strip().lower()
    except (ValueError, UnicodeError):
        raise SwitchError("Cannot parse the existing backend assignment safely.") from None
    if backend not in BACKENDS:
        raise SwitchError("Existing OMI_STT_BACKEND must be whisper or deepgram before switching.")
    return start, end, backend, value


def configured_backend(data):
    field = backend_field(data)
    return field[2] if field else "whisper"


def replace_backend(data, target):
    field = backend_field(data)
    if field:
        start, end, _, value = field
        quote = value[:1] if value[:1] in (b"'", b'"') and value[-1:] == value[:1] else b""
        return data[:start] + quote + target.encode() + quote + data[end:]
    newline = b"\r\n" if b"\r\n" in data else b"\n"
    separator = b"" if not data or data.endswith((b"\n", b"\r")) else newline
    return data + separator + b"OMI_STT_BACKEND=" + target.encode() + newline


def atomic_write(path, expected, data):
    current = read_env(path)
    if (current.data, current.mode, current.identity) != (expected.data, expected.mode, expected.identity):
        raise SwitchError("Project .env changed concurrently; no configuration was overwritten.")
    descriptor, temporary = tempfile.mkstemp(prefix=".env.switch-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), expected.mode)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        # Recheck after preparing the temporary file as well as before it.
        current = read_env(path)
        if (current.data, current.mode, current.identity) != (expected.data, expected.mode, expected.identity):
            raise SwitchError("Project .env changed concurrently; no configuration was overwritten.")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def launcher_lock(root):
    logs = root / "logs"
    if logs.is_symlink():
        raise SwitchError("Refusing a symlinked logs directory.")
    logs.mkdir(exist_ok=True)
    descriptor = None
    try:
        descriptor = os.open(logs / "omi_bridge.launcher.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SwitchError("Launcher lock must be a regular file.")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, SwitchError):
        if descriptor is not None:
            os.close(descriptor)
        raise SwitchError("Another launcher command is running, or its lock is unavailable.") from None
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def probe(root, backend, env):
    candidate = dict(env, OMI_STT_BACKEND=backend)
    try:
        result = subprocess.run([str(root / ".venv/bin/python3.15"), "-B", str(root / "bridge.py"), "--check-config"],
            cwd=root, env=candidate, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        raise SwitchError(f"{backend} configuration validation failed; live service was not changed.") from None
    if result.returncode:
        raise SwitchError(f"{backend} configuration validation failed; check its credentials, model paths, and settings.")


def health_snapshot(root, *, timeout=2):
    """Read tracked local health, including a degraded 503 response."""
    try:
        pid = int((root / "logs/omi_bridge.pid").read_text().strip())
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            response = opener.open("http://127.0.0.1:8788/health", timeout=timeout)
        except urllib.error.HTTPError as error:
            if error.code != 503:
                return None
            response = error
        with response:
            health = json.loads(response.read(65536))
        if isinstance(health, dict) and health.get("pid") == pid and health.get("stt_backend") in BACKENDS:
            return health
    except (OSError, ValueError, TypeError):
        pass
    return None


def live_backend(root):
    health = health_snapshot(root)
    return health["stt_backend"] if health and health.get("status") == "ok" else None


def settle_backend(root, target, *, require_provider=False, timeout=10):
    """Distinguish idle cloud intake from an observed provider handshake.

    Observe a short idle window for phone reconnection; once a phone is active,
    require the provider connection within the bounded settle period. No audio
    or provider API request is generated by this verification.
    """
    deadline = time.monotonic() + timeout
    idle_since, connected_samples = None, 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SwitchError("Selected backend did not settle with the required provider connection.")
        health = health_snapshot(root, timeout=min(1, remaining))
        if health and health.get("stt_backend") == target:
            require_provider |= health.get("direct_ws_connected") is True
            healthy = health.get("status") == "ok"
            connected = health.get("stream_connected") is True
            if healthy and (target == "whisper" or connected):
                connected_samples += 1
                if target == "whisper" or connected_samples >= 2:
                    return True
            else:
                connected_samples = 0
            idle = (healthy and not require_provider and not connected
                    and health.get("provider_connection_state") == "idle"
                    and not health.get("buffer_bytes") and not health.get("pending_saves"))
            if idle:
                idle_since = time.monotonic() if idle_since is None else idle_since
                if time.monotonic() - idle_since >= 1:
                    return False  # Intake ready; the credential/provider path is untested.
            else:
                idle_since = None
        else:
            connected_samples, idle_since = 0, None
        time.sleep(min(.25, max(0, deadline - time.monotonic())))


def restart(root, env, descriptor):
    launch_env = dict(env, OMI_LAUNCHER_LOCK_FD=str(descriptor),
                      OMI_PYTHON=str(root / ".venv/bin/python3.15"))
    launch_env.pop("OMI_STT_BACKEND", None)  # The atomically updated project file is authoritative.
    try:
        result = subprocess.run([str(root / "run.sh"), "restart"], cwd=root, env=launch_env,
            pass_fds=(descriptor,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        raise SwitchError("Launcher restart failed or timed out.") from None
    if result.returncode:
        raise SwitchError("Launcher did not confirm a healthy restart.")


def switch_backend(target, root=ROOT, environ=None):
    if target not in BACKENDS:
        raise SwitchError("Choose whisper or deepgram.")
    root = Path(root)
    env = dict(os.environ if environ is None else environ)
    override = env.pop("OMI_STT_BACKEND", None)
    if override is not None and override.strip().lower() != target:
        raise SwitchError("Unset the conflicting OMI_STT_BACKEND process variable before switching.")
    with launcher_lock(root) as descriptor:
        path = root / ".env"
        original = read_env(path)
        configured = configured_backend(original.data)
        before = health_snapshot(root)
        phone_was_active = bool(before and before.get("direct_ws_connected"))
        previous = live_backend(root) or configured
        probe(root, target, env)  # Never drain a working bridge for an invalid target.
        if configured == target and live_backend(root) == target:
            verified = settle_backend(root, target, require_provider=phone_was_active)
            return (f"{target} is already configured and healthy." if verified else
                    f"{target} is configured; idle intake is ready. Provider connection unverified until audio arrives.")
        if previous != target:
            probe(root, previous, env)  # Verify the recovery path before touching live audio.
        candidate = replace_backend(original.data, target)
        atomic_write(path, original, candidate)
        try:
            restart(root, env, descriptor)
            verified = settle_backend(root, target, require_provider=phone_was_active)
            if configured_backend(read_env(path).data) != target:
                raise SwitchError("The backend assignment changed during restart.")
        except (SwitchError, KeyboardInterrupt):
            try:
                latest = read_env(path)
                if configured_backend(latest.data) != target:
                    raise SwitchError("A concurrent backend change prevents automatic recovery.")
                probe(root, previous, env)
                restored = original.data if latest.data == candidate and previous == configured else replace_backend(latest.data, previous)
                atomic_write(path, latest, restored)
                if live_backend(root) != previous:
                    restart(root, env, descriptor)
                recovered = settle_backend(root, previous, require_provider=phone_was_active)
            except SwitchError:
                raise SwitchError("Backend switch failed; automatic recovery was not confirmed. Inspect local status before retrying.") from None
            message = (f"Backend switch failed; {previous} was restored and is healthy." if recovered else
                       f"Backend switch failed; {previous} was restored with idle intake ready. Provider connection unverified until audio arrives.")
            raise SwitchError(message) from None
        return (f"Switched to {target}; local health and provider connection are confirmed. History was preserved." if verified else
                f"Switched to {target}; idle intake is ready. Provider connection unverified until audio arrives. History was preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=sorted(BACKENDS))
    args = parser.parse_args()
    try:
        print(switch_backend(args.backend))
    except (SwitchError, OSError):
        # Exception text from subprocesses, paths, and env parsers may contain secrets.
        # SwitchError messages above are deliberately bounded and credential-free.
        import sys
        error = sys.exception()
        parser.exit(1, (str(error) if isinstance(error, SwitchError) else "Backend switch failed; inspect local status.") + "\n")


if __name__ == "__main__":
    main()
