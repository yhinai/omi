"""Explicit, bounded local audio capture; construction never starts recording."""
import asyncio
import os
from pathlib import Path
import stat
import time
import uuid
import wave

from common import RATE, BPS

MAX_SECONDS = 10
MAX_CAPTURE_BYTES = BPS * MAX_SECONDS
DEFAULT_DIRECTORY = Path.home() / "Library/Application Support/OmiDiagnostics"


class AudioCapture:
    """Record only packets received after start(), with no automatic activation.

    Feed and lifecycle methods belong to the bridge event-loop thread. The
    clock and scheduler can be injected for offline, deterministic tests.
    No raw bytes or filesystem error messages appear in snapshots or logs.
    """

    def __init__(self, directory=None, *, clock=None, schedule=None):
        self.directory = Path(directory) if directory is not None else DEFAULT_DIRECTORY
        self._clock = clock or time.monotonic
        self._schedule = schedule
        self._timer = None
        self._pcm = bytearray()
        self._state = "idle"
        self._capture_id = self._path = self._reason = self._error_type = None
        self._samples = self._packets = 0
        self._deadline = None
        self._session = self._expected_offset = None

    def _prepare_directory(self):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise OSError("Capture destination must be a private directory")
        self.directory.chmod(0o700)

    def start(self):
        """Begin one explicit capture; return False if already active or unavailable."""
        if self._state == "recording":
            return False
        self._capture_id = uuid.uuid4().hex
        self._path = self._reason = self._error_type = None
        self._samples = self._packets = 0
        self._pcm.clear()
        self._session = self._expected_offset = None
        try:
            self._prepare_directory()
            self._deadline = self._clock() + MAX_SECONDS
            self._state = "recording"
            scheduler = self._schedule or asyncio.get_running_loop().call_later
            self._timer = scheduler(MAX_SECONDS, lambda: self.finish("wall_timeout"))
            return True
        except Exception as error:
            self._state = "error"
            self._error_type = type(error).__name__
            self._reason = "start_failed"
            self._deadline = None
            return False

    def feed(self, raw, sample_offset=None, session_id=None, now=None):
        if self._state != "recording":
            return
        now = self._clock() if now is None else now
        if now >= self._deadline:
            self.finish("wall_timeout")
            return
        if len(raw) % 2:
            raise ValueError("Diagnostic capture requires complete PCM16 samples")
        if not raw:
            return
        if self._packets:
            if self._session is not None and session_id is not None and session_id != self._session:
                self.finish("source_changed")
                return
            if self._expected_offset is not None and sample_offset is not None and sample_offset != self._expected_offset:
                self.finish("source_gap")
                return
        count = min(len(raw), MAX_CAPTURE_BYTES-len(self._pcm))
        self._pcm.extend(memoryview(raw)[:count])
        self._packets += 1
        self._session = session_id
        self._expected_offset = sample_offset + count//2 if sample_offset is not None else None
        self._samples = len(self._pcm)//2
        if len(self._pcm) == MAX_CAPTURE_BYTES:
            self.finish("audio_limit")

    def finish(self, reason="manual"):
        """Explicitly finish and save nonempty PCM; automatic deadline uses this too."""
        if self._state != "recording":
            return
        self._state = "completed"
        self._reason = reason
        self._deadline = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        path = None
        created = False
        try:
            if self._pcm:
                path = self.directory / f"omi-audio-{self._capture_id}.wav"
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(path, flags, 0o600)
                created = True
                with os.fdopen(fd, "wb") as output:
                    os.fchmod(output.fileno(), 0o600)
                    with wave.open(output, "wb") as audio:
                        audio.setnchannels(1)
                        audio.setsampwidth(2)
                        audio.setframerate(RATE)
                        audio.writeframes(self._pcm)
                self._path = str(path.absolute())
        except Exception as error:
            self._state = "error"
            self._reason = "write_failed"
            self._error_type = type(error).__name__
            if created and path is not None:
                try:
                    path.unlink()
                except OSError:
                    pass
        finally:
            self._pcm.clear()

    def close(self, cancel=True):
        """Shutdown cancels unfinished captures; close(cancel=False) explicitly saves."""
        if self._state != "recording":
            return
        if not cancel:
            self.finish("manual")
            return
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._pcm.clear()
        self._deadline = None
        self._state, self._reason = "cancelled", "shutdown"

    def snapshot(self):
        if self._state == "recording" and self._clock() >= self._deadline:
            self.finish("wall_timeout")
        return {"supported": True, "protocol_version": 1, "trigger": "SIGUSR1",
                "state": self._state, "capture_id": self._capture_id,
                "max_wall_sec": MAX_SECONDS, "max_audio_sec": MAX_SECONDS,
                "sample_rate": RATE, "channels": 1, "sample_width_bytes": 2,
                "samples": self._samples, "bytes": self._samples * 2,
                "audio_sec": round(self._samples/RATE, 6), "received_packets": self._packets,
                "path": self._path, "reason": self._reason, "error_type": self._error_type}
