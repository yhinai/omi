#!/usr/bin/env python3.15
"""Single-device PCM16 -> bounded, sequential Gemini transcription bridge."""
import asyncio
import base64
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
import io
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import platform
import re
import shlex
import signal
import ssl
import struct
import sys
import sysconfig
import time
import uuid
import wave


def require_python315():
    if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 15)
            or sys.version_info.releaselevel != "final" or sysconfig.get_config_var("Py_GIL_DISABLED")):
        raise RuntimeError("Omi requires standard CPython 3.15.x (GIL enabled)")


if __name__ == "__main__":
    try:
        require_python315()
    except RuntimeError as error:
        sys.exit(str(error))

import aiohttp
from aiohttp import web

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
TRANSCRIPT_LOG = os.path.join(LOG_DIR, "omi_live_transcript.log")
SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2
MAX_BUFFER_BYTES = BYTES_PER_SECOND * 30
RETRY_DELAYS = (1.0, 2.0, 4.0)
WINDOW_DEADLINE_SEC, MAX_WINDOW_AGE_SEC = 20.0, 45.0
GEMINI_API_KEY = ""
GEMINI_STT_MODEL = "gemini-3.8-flash"
STT_MODE = "live"
GEMINI_LIVE_MODEL = "gemini-3.5-transcribe-live"
GEMINI_THINKING_LEVEL = "low"
PRE_OVERLAP_SEC, CORE_WINDOW_SEC, POST_LOOKAHEAD_SEC = 1.0, 8.0, 1.0
IDLE_FLUSH_SEC, RMS_SILENCE_THRESHOLD = 8.0, 80.0
RING_BUFFER_SECONDS = 300
HOST = "0.0.0.0"
PORT, HEALTH_PORT = 8787, 8788
logger = logging.getLogger("omi_bridge")
transcript_logger = logging.getLogger("omi_transcript")
event_logger = logging.getLogger("omi_transcript_events")


def _read_env_file(path):
    values = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for number, line in enumerate(f, 1):
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:].strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
                    continue
                try:
                    values[key.strip()] = " ".join(shlex.split(value, comments=True))
                except ValueError:
                    raise ValueError(f"Invalid quoted value in {path}:{number}") from None
    return values


def _load_dotenv(local_path=None, environ=None):
    """Resolve process > project .env, including both key aliases."""
    sources = [dict(os.environ if environ is None else environ),
               _read_env_file(local_path or os.path.join(BASE_DIR, ".env"))]
    values = {}
    for source in reversed(sources):
        values.update(source)
    values["GEMINI_API_KEY"] = next(
        (source[key].strip() for source in sources for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY")
         if source.get(key, "").strip()), "")
    return values


def configure(env=None):
    global STT_MODE, GEMINI_LIVE_MODEL
    global WINDOW_DEADLINE_SEC, MAX_WINDOW_AGE_SEC
    global GEMINI_API_KEY, GEMINI_STT_MODEL, GEMINI_THINKING_LEVEL, PRE_OVERLAP_SEC, CORE_WINDOW_SEC, POST_LOOKAHEAD_SEC
    global IDLE_FLUSH_SEC, RMS_SILENCE_THRESHOLD, RING_BUFFER_SECONDS, HOST, PORT, HEALTH_PORT
    env = _load_dotenv() if env is None else env
    HOST = (env.get("OMI_HOST") or "0.0.0.0").strip()
    GEMINI_API_KEY = (env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or "").strip()
    if not GEMINI_API_KEY or GEMINI_API_KEY.lower() in ("your_gemini_api_key_here", "your_api_key_here"):
        raise ValueError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in ~/omi/.env or the environment")
    GEMINI_STT_MODEL = env.get("OMI_GEMINI_STT_MODEL", "gemini-3.8-flash").strip()
    STT_MODE = env.get("OMI_STT_MODE", "live").strip().lower()
    GEMINI_LIVE_MODEL = env.get("OMI_GEMINI_LIVE_MODEL", "gemini-3.5-transcribe-live").strip()
    if STT_MODE not in ("batch", "live") or not re.fullmatch(r"[A-Za-z0-9._-]+", GEMINI_LIVE_MODEL):
        raise ValueError("Use OMI_STT_MODE=live or batch and a valid Live model ID")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", GEMINI_STT_MODEL):
        raise ValueError("OMI_GEMINI_STT_MODEL must be a model ID")
    GEMINI_THINKING_LEVEL = env.get("OMI_GEMINI_THINKING_LEVEL", "low").strip().lower()
    if GEMINI_THINKING_LEVEL not in ("low", "medium", "high"):
        raise ValueError("OMI_GEMINI_THINKING_LEVEL must be low, medium, or high")
    PRE_OVERLAP_SEC = float(env.get("OMI_PRE_OVERLAP_SEC", "1"))
    CORE_WINDOW_SEC = float(env.get("OMI_CORE_WINDOW_SEC", "8"))
    POST_LOOKAHEAD_SEC = float(env.get("OMI_POST_LOOKAHEAD_SEC", "1"))
    IDLE_FLUSH_SEC = float(env.get("OMI_IDLE_FLUSH_SEC", "8"))
    RMS_SILENCE_THRESHOLD = float(env.get("OMI_RMS_THRESHOLD", "80"))
    WINDOW_DEADLINE_SEC = float(env.get("OMI_WINDOW_DEADLINE_SEC", "20"))
    MAX_WINDOW_AGE_SEC = float(env.get("OMI_MAX_WINDOW_AGE_SEC", "45"))
    if not all(math.isfinite(v) and v > 0 for v in (WINDOW_DEADLINE_SEC, MAX_WINDOW_AGE_SEC)):
        raise ValueError("Window deadline and maximum queue age must be finite and positive")
    if not all(math.isfinite(v) and v >= 0 for v in
               (PRE_OVERLAP_SEC, CORE_WINDOW_SEC, POST_LOOKAHEAD_SEC, IDLE_FLUSH_SEC, RMS_SILENCE_THRESHOLD)):
        raise ValueError("Audio settings must be finite, nonnegative numbers")
    if not 1 <= CORE_WINDOW_SEC <= 10 or PRE_OVERLAP_SEC > 5 or POST_LOOKAHEAD_SEC > 5 or IDLE_FLUSH_SEC <= 0:
        raise ValueError("Use core 1-10s, overlap/lookahead 0-5s, and a positive idle timeout")
    RING_BUFFER_SECONDS = int(env.get("OMI_RING_BUFFER_SEC", "300"))
    if RING_BUFFER_SECONDS < 1:
        raise ValueError("Ring buffer must be positive")


class OutputFileHandler(RotatingFileHandler):
    def handleError(self, record):
        # Logging normally swallows disk errors; output health must see them.
        raise sys.exc_info()[1]


def configure_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    for target, filename, fmt in (
        (logger, "omi_bridge.log", "[%(asctime)s] %(message)s"),
        (transcript_logger, "omi_live_transcript.log", "%(message)s"),
        (event_logger, "omi_transcript_events.jsonl", "%(message)s"),
    ):
        target.setLevel(logging.INFO)
        target.propagate = False
        for handler in target.handlers[:]:
            target.removeHandler(handler)
            handler.close()
        handler_class = RotatingFileHandler if target is logger else OutputFileHandler
        handler = handler_class(os.path.join(LOG_DIR, filename), maxBytes=5 * 1024 * 1024, backupCount=3)
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        target.addHandler(handler)


@dataclass
class AudioSession:
    session_id: int
    last_packet_at: float
    pcm: bytearray = field(default_factory=bytearray)
    pre_tail: bytearray = field(default_factory=bytearray)
    closed: bool = False
    first_received_at: float | None = None
    last_received_at: float | None = None
    audio_start_at: float | None = None
    consumed_samples: int = 0
    received_samples: int = 0
    receipts: deque = field(default_factory=deque)


@dataclass
class AudioWindow:
    pcm: bytes
    sample_rate: int
    pre_sec: float
    core_sec: float
    post_sec: float
    reset_context_after: bool = False
    session_id: int = 0
    audio_start_at: float | None = None
    received_at: float | None = None
    sample_offset: int | None = None
    enqueued_at: float = field(default_factory=lambda: time.monotonic())
    first_received_mono: float | None = None


@dataclass(frozen=True)
class STTResult:
    text: str
    outcome: str = "speech"
    finish_reason: str = "STOP"


class GeminiResponseError(Exception):
    def __init__(self, kind, retryable=False):
        self.kind, self.retryable = kind, retryable
        super().__init__(f"Gemini response: {kind}")


def new_metrics():
    return {
        "started_at": time.monotonic(), "last_packet_at": None, "last_stt_success_at": None,
        "last_error": None, "windows_processed": 0, "windows_retried": 0,
        "windows_dropped": 0, "windows_evicted": 0, "buffer_bytes_evicted": 0, "windows_skipped": 0, "packets_rejected": 0, "stt_in_flight": False,
        "last_transcript_at": None, "last_outcome": None, "transcripts_emitted": 0,
        "lost_audio_sec_failed": 0.0, "lost_audio_sec_stale": 0.0,
        "lost_audio_sec_queue_evicted": 0.0, "lost_audio_sec_buffer_evicted": 0.0,
        "output_errors": 0, "last_output_error": None,
        "windows_no_speech": 0, "windows_blocked": 0, "windows_truncated": 0, "windows_invalid_response": 0,
    }


@dataclass
class BridgeState:
    stt_queue: asyncio.Queue | None = None
    audio_sessions: deque = field(default_factory=deque)
    current_session: AudioSession | None = None
    _next_session_id: int = 0
    audio_ring: deque = field(default_factory=deque)
    audio_ring_bytes: int = 0
    daily_notes: object = None
    live_listeners: dict = field(default_factory=dict)
    caption_listeners: dict = field(default_factory=dict)
    stream_clients: dict = field(default_factory=dict)
    recent_events: deque = field(default_factory=lambda: deque(maxlen=200))
    stream_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    background_tasks: dict = field(default_factory=dict)
    http_session: aiohttp.ClientSession | None = None
    stopping: bool = False
    metrics: dict = field(default_factory=new_metrics)
    pending_windows: dict = field(default_factory=dict)
    active_window: AudioWindow | None = None
    live_stt: object = None
    partial_caption: dict | None = None
    latency_samples: dict = field(default_factory=lambda: {
        name: deque(maxlen=128) for name in ("receipt_to_transcript_sec", "gemini_request_duration_sec",
                                           "receipt_to_first_partial_sec", "receipt_to_final_sec")})


state = BridgeState()


def record_latency(name, seconds):
    state.latency_samples[name].append(max(0.0, seconds))


def latency_summary(samples):
    if not samples:
        return {"samples": 0, "last": None, "p50": None, "p95": None}
    ordered = sorted(samples)
    return {"samples": len(samples), "last": round(samples[-1], 3),
            "p50": round(ordered[math.ceil(len(ordered) * .50) - 1], 3),
            "p95": round(ordered[math.ceil(len(ordered) * .95) - 1], 3)}


def oldest_pending_audio_age(now):
    if state.live_stt is not None:
        age = state.live_stt.pending_age
        return round(age, 3) if age is not None else None
    # Receipt marks survive partial slicing/trimming; playback history is not pending work.
    starts = [session.receipts[0][2] for session in state.audio_sessions
              if session.pcm and session.receipts]
    windows = list(state.pending_windows.values())
    if state.active_window is not None:
        windows.append(state.active_window)
    starts.extend(window.first_received_mono if window.first_received_mono is not None
                  else window.enqueued_at for window in windows if window.pcm)
    return round(max(0.0, now - min(starts)), 3) if starts else None


def broadcast(message):
    for ws, queue in list(state.live_listeners.items()):
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            # Let the existing sender close a slow listener, without creating more tasks.
            state.live_listeners.pop(ws, None)
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait(None)


@dataclass
class CaptionListener:
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=64))
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    status: dict | None = None
    partial: dict | None = None
    closed: bool = False


def publish_caption(event):
    identity = event.get("utterance_id")
    clear = event.get("type") == "gap" or (state.partial_caption is not None
        and identity == state.partial_caption.get("utterance_id"))
    if clear:
        state.partial_caption = None
    for listener in list(state.caption_listeners.values()):
        if clear:
            listener.partial = None
        try:
            listener.queue.put_nowait(event)
        except asyncio.QueueFull:
            listener.closed = True  # Reconnect can replay retained events.
        listener.ready.set()


def publish_partial(event):
    """Coalesce disposable hypotheses; never send them to logs, replay or notes."""
    if not event.get("text") and state.partial_caption is None:
        return
    event = {**event, "stream_id": state.stream_id}
    if (not event.get("text") and state.partial_caption is not None
            and event.get("utterance_id") != state.partial_caption.get("utterance_id")):
        return
    state.partial_caption = event if event.get("text") else None
    for listener in list(state.caption_listeners.values()):
        listener.partial = event
        listener.ready.set()
    broadcast(json.dumps(event))


def record_live_final(text, **timing):
    state.metrics["last_stt_success_at"] = time.monotonic()
    state.metrics["windows_processed"] += 1
    state.metrics["last_error"] = None
    state.metrics["last_outcome"] = "speech"
    saved = record_transcript_line(text, **timing)
    if saved is not False:
        state.metrics["last_transcript_at"] = time.monotonic()
        state.metrics["transcripts_emitted"] += 1
    return saved


def record_live_gap(reason, seconds, **timing):
    state.metrics["windows_dropped"] += 1
    if reason == "queue_evicted":
        state.metrics["windows_evicted"] += 1
    record_gap(reason, seconds, **timing)


def live_connection_changed(connected, error):
    state.metrics["last_error"] = None if connected else "Gemini Live: " + str(error)
    logger.info("Gemini Live %s", "connected" if connected else "reconnecting (" + str(error) + ")")


def current_stt_in_flight():
    return state.live_stt.speech_pending if state.live_stt is not None else state.metrics["stt_in_flight"]


def caption_status():
    now = time.monotonic()
    packet = state.metrics["last_packet_at"]
    age = max(0, now - packet) if packet is not None else None
    session = state.current_session
    return {"type": "status", "stream_id": state.stream_id,
            "session_id": state.live_stt.session_id if state.live_stt is not None else session.session_id if session else None,
            "stt_mode": "live" if state.live_stt is not None else "batch",
            "stream_connected": state.live_stt.connected if state.live_stt is not None else None,
            "audio_active": age is not None and age < 3,
            "last_packet_age_sec": round(age, 1) if age is not None else None,
            "buffer_sec": round(buffered_audio_bytes() / BYTES_PER_SECOND, 1),
            "core_window_sec": 0 if state.live_stt is not None else CORE_WINDOW_SEC,
            "stt_in_flight": current_stt_in_flight(),
            "queue_depth": len(state.live_stt.queue) if state.live_stt is not None else state.stt_queue.qsize() if state.stt_queue is not None else 0}


async def caption_status_worker():
    previous = None
    while True:
        status = caption_status()
        signature = tuple((k, v) for k, v in status.items() if k != "last_packet_age_sec")
        busy = status["stt_in_flight"] or status["buffer_sec"] or status["queue_depth"]
        if signature != previous or busy:
            for listener in state.caption_listeners.values():
                listener.status = status  # Replace ephemeral status; never displace a caption.
                listener.ready.set()
            previous = signature
        await asyncio.sleep(1)


def seed_recent_events():
    """Bounded startup reads, including rotation backups; malformed lines are skipped."""
    events, seen = [], set()
    for suffix in ("", ".1", ".2", ".3"):
        try:
            with open(os.path.join(LOG_DIR, "omi_transcript_events.jsonl") + suffix, "rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                stream.seek(max(0, size - 1024 * 1024))
                if stream.tell():
                    stream.readline()
                lines = stream.read().splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if not isinstance(event, dict) or event.get("type", "transcript") not in ("transcript", "gap") or not isinstance(event.get("text"), str):
                continue
            event.setdefault("type", "transcript")
            identity = event.get("event_id")
            if not isinstance(identity, str) or not identity:
                event["event_id"] = identity = "legacy-" + hashlib.sha256(line).hexdigest()
            if identity not in seen:
                events.append(event)
                seen.add(identity)
            if len(events) == 200:
                break
        if len(events) == 200:
            break
    state.recent_events = deque(reversed(events), maxlen=200)


class DailyNotes:
    """A paragraph projection; persist each update so restart does not lose a partial paragraph."""

    def __init__(self, directory):
        self.directory = directory
        self.day, self.words, self.last_time = None, [], None

    def close_paragraph(self):
        self.words = []

    def add(self, timestamp, text):
        dt = datetime.fromisoformat(timestamp)
        day = timestamp[:10]
        path = os.path.join(self.directory, f"{day}.md")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Omi Transcript — {day}\n\n")
        if day != self.day or (self.last_time and (dt - self.last_time).total_seconds() > IDLE_FLUSH_SEC):
            self.words = []
        if not self.words:
            self.start, self.offset = timestamp[11:], os.path.getsize(path)
        self.day, self.last_time = day, dt
        self.words.extend(text.split())
        # Offset comes from getsize (bytes); binary I/O keeps Unicode offsets correct.
        with open(path, "r+b") as f:
            f.seek(self.offset)
            f.write(f"**[{self.start}]** {' '.join(self.words)}\n\n".encode("utf-8"))
            f.truncate()
        if len(self.words) >= 120:
            self.words = []

    def rebuild(self, transcript_path):
        """Explicit repair utility. Normal startup appends; it never truncates existing daily notes."""
        if not os.path.exists(transcript_path):
            return
        days = set()
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                match = re.match(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.+)$", line)
                if match:
                    timestamp, text = match.groups()
                    if timestamp[:10] not in days:
                        with open(os.path.join(self.directory, timestamp[:10] + ".md"), "w", encoding="utf-8"):
                            pass
                        days.add(timestamp[:10])
                        self.close_paragraph()
                    self.add(timestamp, text)


def flush_daily_paragraph():
    if state.daily_notes is not None:
        state.daily_notes.close_paragraph()


def output_failed(target, error):
    state.metrics["output_errors"] += 1
    state.metrics["last_output_error"] = f"{target}: {type(error).__name__}"
    logger.error("Transcript output failed: %s", state.metrics["last_output_error"])


def write_event(event):
    """JSONL is authoritative; projections are independent, synchronous best effort."""
    encoded = json.dumps(event, ensure_ascii=False)
    try:
        event_logger.info(encoded)
    except Exception as error:
        output_failed("jsonl", error)
        return False
    state.recent_events.append(dict(event))
    publish_caption(event)
    broadcast(encoded)
    if event.get("type") == "transcript" and event.get("text"):
        reply = json.dumps({"text": event["text"]}, ensure_ascii=False)
        for ws_client, q in list(state.stream_clients.items()):
            try:
                q.put_nowait(reply)
            except asyncio.QueueFull:
                pass
    timestamp, text = event["timestamp"], event["text"]
    errors_before = state.metrics["output_errors"]
    try:
        transcript_logger.info("[%s] %s", timestamp, text)
    except Exception as error:
        output_failed("text", error)
    try:
        if state.daily_notes is not None:
            if event["type"] == "gap":
                state.daily_notes.close_paragraph()
            state.daily_notes.add(timestamp, text)
            if event["type"] == "gap":
                state.daily_notes.close_paragraph()
    except Exception as error:
        output_failed("markdown", error)
        if state.daily_notes is not None:
            state.daily_notes.close_paragraph()
    if errors_before == state.metrics["output_errors"]:
        state.metrics["last_output_error"] = None
    return True


def make_event(kind, text, audio_start_at=None, received_at=None, processed_at=None,
               session_id=None, sample_offset=None, utterance_id=None):
    processed_at = time.time() if processed_at is None else processed_at
    source_at = audio_start_at if audio_start_at is not None else processed_at
    event = {"event_id": uuid.uuid4().hex, "stream_id": state.stream_id, "type": kind,
            "timestamp": datetime.fromtimestamp(source_at).strftime("%Y-%m-%d %H:%M:%S"),
            "text": text, "audio_start_at": audio_start_at, "received_at": received_at,
            "processed_at": processed_at, "session_id": session_id, "sample_offset": sample_offset,
            "time_basis": "estimated_from_receipt" if audio_start_at is not None else "processing_time"}
    if utterance_id is not None:
        event.update(utterance_id=utterance_id, is_final=True, transcription_mode="live")
    return event


def record_gap(reason, seconds, *, session_id=None, sample_offset=None, audio_start_at=None,
               received_at=None):
    if seconds <= 0:
        return
    state.metrics[f"lost_audio_sec_{reason}"] += seconds
    event = make_event("gap", f"[GAP: {seconds:.3f}s audio untranscribed; {reason}]",
                       audio_start_at=audio_start_at, received_at=received_at,
                       session_id=session_id, sample_offset=sample_offset)
    event.update(reason=reason, duration_sec=seconds)
    write_event(event)


def record_window_gap(reason, window):
    record_gap(reason, window.core_sec, session_id=window.session_id,
               sample_offset=window.sample_offset, audio_start_at=window.audio_start_at,
               received_at=window.received_at)


def record_transcript_line(text, **timing):
    text = " ".join(text.split()).strip()
    if not text:
        return False
    event = make_event("transcript", text, **timing)
    if not write_event(event):
        return False
    return True


def pcm16_rms(pcm_bytes: bytes) -> float:
    n = len(pcm_bytes) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm_bytes[: n * 2])
    return (sum(s * s for s in samples) / n) ** 0.5


def find_dynamic_cut_offset(buf: bytearray, sr: int, target_core_sec: float) -> int:
    """Find the lowest-energy 100ms micro-pause within +/-0.8s of target_core_sec."""
    bps = sr * 2
    target_byte = int(target_core_sec * bps)
    win = int(0.1 * bps) & ~1
    lo = max(int((target_core_sec - 0.8) * bps), win)
    hi = min(int((target_core_sec + 0.8) * bps), len(buf) - int(POST_LOOKAHEAD_SEC * bps) - win)
    if hi <= lo:
        return min(target_byte, len(buf)) & ~1

    best_offset, best_rms = target_byte, float("inf")
    step = max((win // 2) & ~1, 2)
    for pos in range(lo & ~1, hi, step):
        r = pcm16_rms(bytes(buf[pos : pos + win]))
        if r < best_rms:
            best_rms = r
            best_offset = (pos + win // 2) & ~1
    return best_offset


def dedupe_boundary_overlap(prev_tail: str, new_text: str) -> str:
    if not prev_tail or not new_text:
        return new_text.strip()
    prev_words, new_words = prev_tail.split(), new_text.split()
    best_k = 0
    for k in range(2, min(len(prev_words), len(new_words), 25) + 1):
        p = [w.strip(".,!?;:\"'()[]{}").lower() for w in prev_words[-k:]]
        n = [w.strip(".,!?;:\"'()[]{}").lower() for w in new_words[:k]]
        if p == n:
            best_k = k
    return " ".join(new_words[best_k:]).strip() if best_k else new_text.strip()


def parse_gemini_response(data):
    """A completed, explicit text result is different from a blocked or incomplete response."""
    if not isinstance(data, dict):
        raise GeminiResponseError("invalid_response")
    feedback = data.get("promptFeedback")
    if feedback is None:
        feedback = {}
    if not isinstance(feedback, dict):
        raise GeminiResponseError("invalid_response")
    if feedback.get("blockReason") not in (None, "", "BLOCK_REASON_UNSPECIFIED"):
        raise GeminiResponseError("blocked")
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise GeminiResponseError("invalid_response")
    candidate = candidates[0]
    finish = candidate.get("finishReason")
    if finish in ("MAX_TOKENS", "CONTINUATION"):
        raise GeminiResponseError("truncated")
    if finish in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE", "IMAGE_SAFETY"):
        raise GeminiResponseError("blocked")
    if finish != "STOP":
        raise GeminiResponseError("invalid_response")
    usage = data.get("usageMetadata")
    # Google can omit empty text/content when a completed request generated no reply tokens.
    no_reply_tokens = isinstance(usage, dict) and type(usage.get("candidatesTokenCount")) is int and usage["candidatesTokenCount"] == 0
    if isinstance(usage, dict) and "candidatesTokenCount" not in usage:
        # Proto JSON omits zero-valued counts. Require the remaining counts to reconcile.
        prompt, thoughts, total = (usage.get("promptTokenCount"), usage.get("thoughtsTokenCount", 0),
                                    usage.get("totalTokenCount"))
        no_reply_tokens = (all(type(value) is int and value >= 0 for value in (prompt, thoughts, total))
                           and total == prompt + thoughts)
    if no_reply_tokens:
        content = candidate.get("content")
        if content is not None and not isinstance(content, dict):
            raise GeminiResponseError("invalid_response")
        parts = content.get("parts", []) if content else []
        if not isinstance(parts, list):
            raise GeminiResponseError("invalid_response")
        for part in parts:
            if not isinstance(part, dict):
                raise GeminiResponseError("invalid_response")
            if part.get("thought"):
                continue
            if set(part) - {"text", "thoughtSignature"} or not isinstance(part.get("text", ""), str):
                raise GeminiResponseError("invalid_response")
            if part.get("text", "").strip() not in ("", '""', "''"):
                raise GeminiResponseError("invalid_response")
        return STTResult("", "no_speech", finish)
    content = candidate.get("content")
    if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
        raise GeminiResponseError("invalid_response")
    text_parts = []
    for part in content["parts"]:
        if not isinstance(part, dict):
            raise GeminiResponseError("invalid_response")
        if part.get("thought"):
            continue
        if "text" not in part or not isinstance(part["text"], str):
            raise GeminiResponseError("invalid_response")
        text_parts.append(part["text"])
    if not text_parts:
        raise GeminiResponseError("invalid_response")
    text = " ".join(" ".join(text_parts).split()).strip()
    if text in ("", '""', "''"):
        return STTResult("", "no_speech", finish)
    return STTResult(text, "speech", finish)


async def gemini_stt_request(pcm_bytes: bytes, sr: int, pre_sec: float, core_sec: float, post_sec: float, prev_ctx: str) -> STTResult:
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_bytes)
    b64_wav = base64.b64encode(wav_buf.getvalue()).decode("ascii")

    overlap_note = ""
    if pre_sec > 0 and prev_ctx:
        overlap_note = (
            f"This clip has a {pre_sec:.1f}s pre-overlap at the start and a {post_sec:.1f}s lookahead tail at the end "
            f"(centering a {core_sec:.1f}s window). The previous window ended with: \"{prev_ctx}\". "
            "The quoted context is only for boundary matching, not a request to continue it. "
            "Return only speech audible in this clip after that overlap; never infer or invent a continuation. "
        )

    prompt = (
        "Transcribe only intelligible words actually spoken in this audio clip into a single English paragraph. "
        + overlap_note
        + "Do not include timestamps, speaker labels, markdown, or commentary. "
        "If there is no intelligible speech beyond the previous context, return ONLY an empty string."
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_STT_MODEL}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "audio/wav", "data": b64_wav}}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 160,
                             "thinkingConfig": {"thinkingLevel": GEMINI_THINKING_LEVEL}},
    }
    async with state.http_session.post(url, json=payload, headers={"x-goog-api-key": GEMINI_API_KEY},
                                 allow_redirects=False, timeout=aiohttp.ClientTimeout(total=12, connect=5, sock_read=10)) as response:
        if not 200 <= response.status < 300:
            raise aiohttp.ClientResponseError(response.request_info, response.history, status=response.status)
        try:
            data = await response.json(content_type=None)
        except (ValueError, UnicodeError):
            raise GeminiResponseError("invalid_response") from None
    try:
        return parse_gemini_response(data)
    except GeminiResponseError:
        # Record schema metadata only, never audio, transcript text, or credentials.
        candidates = data.get("candidates") if isinstance(data, dict) else None
        first = candidates[0] if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict) else {}
        content = first.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        usage = data.get("usageMetadata") if isinstance(data, dict) else None
        logger.warning("Rejected Gemini output metadata: %s", json.dumps({
            "kind": "response_shape", "finish_reason": first.get("finishReason"),
            "candidate_count": len(candidates) if isinstance(candidates, list) else None,
            "part_fields": [sorted(part) for part in parts if isinstance(part, dict)] if isinstance(parts, list) else None,
            "reply_tokens": usage.get("candidatesTokenCount") if isinstance(usage, dict) else None,
            "prompt_tokens": usage.get("promptTokenCount") if isinstance(usage, dict) else None,
            "thought_tokens": usage.get("thoughtsTokenCount") if isinstance(usage, dict) else None,
            "total_tokens": usage.get("totalTokenCount") if isinstance(usage, dict) else None,
        }))
        raise



async def enqueue_stt_window(full_pcm, sr, pre_s, core_s, post_s, reset_context_after=False,
                             session_id=None, audio_start_at=None, received_at=None, sample_offset=None,
                             first_received_mono=None):
    """Prefer fresh windows during overload; preserve all pending work during shutdown."""
    if session_id is None:
        session_id = state.current_session.session_id if state.current_session else state._next_session_id
    window = AudioWindow(full_pcm, sr, pre_s, core_s, post_s, reset_context_after,
                         session_id, audio_start_at, received_at, sample_offset,
                         first_received_mono=first_received_mono)
    if state.stopping:
        await state.stt_queue.put(window)
        state.pending_windows[id(window)] = window
        return
    if state.stt_queue.full():
        dropped = state.stt_queue.get_nowait()
        state.pending_windows.pop(id(dropped), None)
        state.stt_queue.task_done()
        if dropped is not None and dropped.pcm:
            state.metrics["windows_dropped"] += 1
            state.metrics["windows_evicted"] += 1
            record_window_gap("queue_evicted", dropped)
            logger.warning("STT backlog full: evicted oldest window (%.2fs core)", dropped.core_sec)
    state.stt_queue.put_nowait(window)
    state.pending_windows[id(window)] = window


def _transient_error(error):
    if isinstance(error, GeminiResponseError):
        return error.retryable
    if isinstance(error, aiohttp.ClientResponseError):
        return error.status in (408, 429) or 500 <= error.status < 600
    if isinstance(error, (aiohttp.ClientSSLError, aiohttp.InvalidURL)):
        return False
    return isinstance(error, (TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, ConnectionError))


def _error_label(error):
    if isinstance(error, GeminiResponseError):
        return f"Gemini response: {error.kind}"
    if isinstance(error, aiohttp.ClientResponseError):
        return f"Gemini HTTP {error.status}"
    return type(error).__name__  # Avoid leaking request URLs or credentials into telemetry.


async def transcribe_with_retries(window, previous_context):
    """The outer timeout bounds attempts AND backoff, including the last request."""
    async with asyncio.timeout(WINDOW_DEADLINE_SEC):
        deadline = asyncio.get_running_loop().time() + WINDOW_DEADLINE_SEC
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                started = time.monotonic()
                try:
                    return await gemini_stt_request(window.pcm, window.sample_rate,
                        window.pre_sec, window.core_sec, window.post_sec, previous_context)
                finally:
                    record_latency("gemini_request_duration_sec", time.monotonic() - started)
            except Exception as error:
                state.metrics["last_error"] = _error_label(error)
                if isinstance(error, GeminiResponseError):
                    state.metrics[f"windows_{error.kind}"] += 1
                    state.metrics["last_outcome"] = error.kind
                if not _transient_error(error) or attempt == len(RETRY_DELAYS):
                    raise
                delay = RETRY_DELAYS[attempt]
                if asyncio.get_running_loop().time() + delay >= deadline:
                    raise TimeoutError("Window processing deadline exhausted") from None
                state.metrics["windows_retried"] += 1
                logger.warning("%s; retry %d/3 in %.1fs", _error_label(error), attempt + 1, delay)
                await asyncio.sleep(delay)


async def stt_consumer_worker():
    """One ordered consumer, one initial attempt plus at most three transient retries."""
    last_transcript_tail, context_session, next_sample = "", None, None
    while True:
        window = await state.stt_queue.get()
        state.pending_windows.pop(id(window), None)
        state.active_window = window
        try:
            if window is None:
                return
            if window.session_id != context_session or (window.sample_offset is not None and next_sample is not None and window.sample_offset != next_sample):
                last_transcript_tail, context_session = "", window.session_id
                flush_daily_paragraph()
            next_sample = (window.sample_offset + round(window.core_sec * window.sample_rate)
                           if window.sample_offset is not None else None)
            if (window.pcm and not state.stopping
                    and time.monotonic() - window.enqueued_at > MAX_WINDOW_AGE_SEC):
                state.metrics["windows_dropped"] += 1
                record_window_gap("stale", window)
                last_transcript_tail = ""
                flush_daily_paragraph()
                continue
            pre_b = int(window.pre_sec * BYTES_PER_SECOND)
            core_b = int(window.core_sec * BYTES_PER_SECOND)
            core = window.pcm[pre_b:pre_b + core_b]
            if core and pcm16_rms(core) >= RMS_SILENCE_THRESHOLD:
                state.metrics["stt_in_flight"] = True
                result = None
                try:
                    result = await transcribe_with_retries(window, last_transcript_tail)
                    state.metrics["last_stt_success_at"] = time.monotonic()
                    state.metrics["windows_processed"] += 1
                    state.metrics["last_error"] = None
                    state.metrics["last_outcome"] = result.outcome
                    if result.outcome == "no_speech":
                        state.metrics["windows_no_speech"] += 1
                except Exception as error:
                    state.metrics["last_error"] = _error_label(error)
                    state.metrics["windows_dropped"] += 1
                    record_window_gap("failed", window)
                    last_transcript_tail = ""
                    flush_daily_paragraph()
                    logger.error("STT window dropped: %s", state.metrics["last_error"])
                if result and result.outcome == "speech":
                    cleaned = dedupe_boundary_overlap(last_transcript_tail, result.text)
                    if cleaned:
                        emitted = record_transcript_line(cleaned, audio_start_at=window.audio_start_at,
                                               received_at=window.received_at, session_id=window.session_id,
                                               sample_offset=window.sample_offset)
                        if emitted is not False:
                            state.metrics["last_transcript_at"] = time.monotonic()
                            state.metrics["transcripts_emitted"] += 1
                            if window.first_received_mono is not None:
                                record_latency("receipt_to_transcript_sec",
                                               time.monotonic() - window.first_received_mono)
                        last_transcript_tail = " ".join((last_transcript_tail + " " + cleaned).split()[-24:])
            elif core:
                state.metrics["windows_skipped"] += 1
            if window.reset_context_after:
                last_transcript_tail = ""
                flush_daily_paragraph()
        except Exception as error:
            # A failed output file or malformed response must not silently kill the sole consumer.
            state.metrics["last_error"] = _error_label(error)
            state.metrics["windows_dropped"] += 1
            record_window_gap("failed", window)
            last_transcript_tail = ""
            logger.exception("STT window processing failed")
        finally:
            state.active_window = None
            state.metrics["stt_in_flight"] = False
            state.stt_queue.task_done()


def buffered_audio_bytes():
    if state.live_stt is not None:
        return state.live_stt.pending_bytes
    return sum(len(session.pcm) for session in state.audio_sessions)


def ingest_raw_pcm(raw, sr=SAMPLE_RATE):
    if state.stopping:
        raise web.HTTPServiceUnavailable(text="Bridge is draining for shutdown")
    if sr != SAMPLE_RATE or len(raw) % 2:
        raise web.HTTPBadRequest(text="Expected 16000 Hz mono PCM16 with complete two-byte samples")
    if not raw:
        return
    now, received_at = time.monotonic(), time.time()
    # Retain every valid live packet before applying the pending-transcription limit.
    state.audio_ring.append((now, SAMPLE_RATE, raw))
    state.audio_ring_bytes += len(raw)
    max_ring_bytes = BYTES_PER_SECOND * RING_BUFFER_SECONDS
    while state.audio_ring_bytes > max_ring_bytes:
        stamp, old_sr, old = state.audio_ring.popleft()
        remove = min(state.audio_ring_bytes - max_ring_bytes, len(old))
        state.audio_ring_bytes -= remove
        if remove < len(old):
            state.audio_ring.appendleft((stamp, old_sr, old[remove:]))
    state.metrics["last_packet_at"] = now
    if state.live_stt is not None:
        state.live_stt.feed(raw, received_at, now)
        broadcast(raw)
        return
    if state.current_session is None or state.current_session.closed or now - state.current_session.last_packet_at > IDLE_FLUSH_SEC:
        if state.current_session:
            state.current_session.closed = True
        state._next_session_id += 1
        state.current_session = AudioSession(state._next_session_id, now, first_received_at=received_at,
                                       last_received_at=received_at,
                                       audio_start_at=received_at - len(raw) / BYTES_PER_SECOND)
        state.audio_sessions.append(state.current_session)
    state.current_session.last_packet_at = now
    state.current_session.last_received_at = received_at
    state.current_session.received_samples += len(raw) // 2
    state.current_session.receipts.append((state.current_session.received_samples, received_at, now))
    state.current_session.pcm.extend(raw)
    state.metrics["last_packet_at"] = now
    excess = buffered_audio_bytes() - MAX_BUFFER_BYTES
    if excess > 0:
        for session in state.audio_sessions:
            remove = min(excess, len(session.pcm))
            if not remove:
                continue
            record_gap("buffer_evicted", remove / BYTES_PER_SECOND,
                       session_id=session.session_id, sample_offset=session.consumed_samples,
                       audio_start_at=(session.audio_start_at + session.consumed_samples / SAMPLE_RATE
                                       if session.audio_start_at is not None else None),
                       received_at=session.last_received_at)
            del session.pcm[:remove]
            session.consumed_samples += remove // 2
            session.pre_tail.clear()
            while session.receipts and session.receipts[0][0] <= session.consumed_samples:
                session.receipts.popleft()
            state.metrics["buffer_bytes_evicted"] += remove
            state.metrics["windows_dropped"] += 1
            excess -= remove
            if not excess:
                break
        logger.warning("Raw backlog full: evicted oldest PCM; total evicted bytes=%d",
                       state.metrics["buffer_bytes_evicted"])
    broadcast(raw)


async def _slice_one(session, final=False):
    bps = BYTES_PER_SECOND
    pre_b = int(PRE_OVERLAP_SEC * bps) & ~1
    if final:
        core_cut, post_b = len(session.pcm) & ~1, 0
    else:
        core_cut = find_dynamic_cut_offset(session.pcm, SAMPLE_RATE, CORE_WINDOW_SEC)
        post_b = int(POST_LOOKAHEAD_SEC * bps) & ~1
    core = bytes(session.pcm[:core_cut])
    end = min(core_cut + post_b, len(session.pcm)) & ~1
    full = bytes(session.pre_tail) + bytes(session.pcm[:end])
    audio_start_at = (session.audio_start_at + session.consumed_samples / SAMPLE_RATE
                      if session.audio_start_at is not None else None)
    end_sample = session.consumed_samples + end // 2
    received_at = next((stamp for sample, stamp, _ in session.receipts if sample >= end_sample),
                       session.last_received_at)
    first_received_mono = session.receipts[0][2] if session.receipts else None
    if audio_start_at is not None and received_at is not None:
        # Backfilled/bursty uploads must not imply that audio was captured after it arrived.
        audio_start_at = min(audio_start_at, received_at - end / bps)
    await enqueue_stt_window(full, SAMPLE_RATE, len(session.pre_tail) / bps,
                             len(core) / bps, (end - core_cut) / bps, final, session.session_id,
                             audio_start_at, received_at, session.consumed_samples, first_received_mono)
    # No await between successful queue admission and committing the consumed prefix.
    del session.pcm[:core_cut]
    session.consumed_samples += core_cut // 2
    while session.receipts and session.receipts[0][0] <= session.consumed_samples:
        session.receipts.popleft()
    session.pre_tail[:] = b"" if final or pre_b == 0 else core[-pre_b:]


async def flush_remaining_buffer(reset_context=True):
    """Close sessions in order; admit all complete PCM samples before a reset marker."""
    for session in state.audio_sessions:
        session.closed = True
    need_b = int((CORE_WINDOW_SEC + POST_LOOKAHEAD_SEC) * BYTES_PER_SECOND) & ~1
    while state.audio_sessions:
        session = state.audio_sessions[0]
        while len(session.pcm) >= need_b:
            await _slice_one(session)
        if len(session.pcm) >= 2:
            await _slice_one(session, final=True)
        else:
            await enqueue_stt_window(b"", SAMPLE_RATE, 0, 0, 0, reset_context, session.session_id)
        session.pcm.clear()
        session.pre_tail.clear()
        state.audio_sessions.popleft()
    if state.current_session:
        state.current_session.closed = True
    if reset_context:
        # Also handles an entirely empty buffer: the reset remains ordered behind older STT jobs.
        await enqueue_stt_window(b"", SAMPLE_RATE, 0, 0, 0, True)


async def window_slicer_worker():
    need_b = int((CORE_WINDOW_SEC + POST_LOOKAHEAD_SEC) * BYTES_PER_SECOND) & ~1
    while True:
        await asyncio.sleep(0.2)
        while state.audio_sessions:
            session = state.audio_sessions[0]
            if time.monotonic() - session.last_packet_at > IDLE_FLUSH_SEC:
                session.closed = True
            if len(session.pcm) >= need_b:
                await _slice_one(session)
            elif session.closed:
                if len(session.pcm) >= 2:
                    await _slice_one(session, final=True)
                else:
                    await enqueue_stt_window(b"", SAMPLE_RATE, 0, 0, 0, True, session.session_id)
                session.pcm.clear()
                session.pre_tail.clear()
                state.audio_sessions.popleft()
            else:
                break


def build_health_dict():
    now = time.monotonic()
    workers_alive = bool(state.background_tasks) and all(not task.done() for task in state.background_tasks.values())
    overloaded = (state.live_stt.pending_bytes >= BYTES_PER_SECOND * 10 or len(state.live_stt.queue) >= 256
                  if state.live_stt is not None else state.stt_queue is not None and state.stt_queue.full())
    pending_age = oldest_pending_audio_age(now)
    stalled = pending_age is not None and pending_age > MAX_WINDOW_AGE_SEC
    return {
        "status": "ok" if GEMINI_API_KEY and workers_alive and not state.stopping and not overloaded and not stalled and not state.metrics["last_error"] and not state.metrics["last_output_error"] and (state.live_stt is None or state.live_stt.connected) else "degraded",
        "is_setup_completed": True, "message": "", "pid": os.getpid(),
        "gemini_key_configured": bool(GEMINI_API_KEY),
        "stt_mode": "live" if state.live_stt is not None else "batch",
        "model": GEMINI_LIVE_MODEL if state.live_stt is not None else GEMINI_STT_MODEL,
        "stream_connected": state.live_stt.connected if state.live_stt is not None else None,
        "stream_reconnects": state.live_stt.reconnects if state.live_stt is not None else 0,
        "partial_updates": state.live_stt.partial_updates if state.live_stt is not None else 0,
        "stt_in_flight": current_stt_in_flight(),
        "thinking_level": GEMINI_THINKING_LEVEL,
        "core_window_sec": 0 if state.live_stt is not None else CORE_WINDOW_SEC, "window_deadline_sec": WINDOW_DEADLINE_SEC,
        "max_window_age_sec": MAX_WINDOW_AGE_SEC,
        "latency": {name: latency_summary(samples) for name, samples in state.latency_samples.items()},
        "oldest_pending_audio_age_sec": pending_age,
        **{k: round(v, 6) if isinstance(v, float) else v for k, v in state.metrics.items()
           if k.startswith("lost_audio_sec_") or k in ("output_errors", "last_output_error")},
        "python_version": platform.python_version(), "openssl_version": ssl.OPENSSL_VERSION,
        "workers_alive": workers_alive, "stopping": state.stopping,
        "uptime_sec": round(now - state.metrics["started_at"], 1),
        "last_packet_age_sec": round(now - state.metrics["last_packet_at"], 1) if state.metrics["last_packet_at"] is not None else None,
        "last_stt_success_age_sec": round(now - state.metrics["last_stt_success_at"], 1) if state.metrics["last_stt_success_at"] is not None else None,
        "last_transcript_age_sec": round(now - state.metrics["last_transcript_at"], 1) if state.metrics["last_transcript_at"] is not None else None,
        "queue_depth": len(state.live_stt.queue) if state.live_stt is not None else state.stt_queue.qsize() if state.stt_queue is not None else 0,
        "queue_capacity": None if state.live_stt is not None else 24,
        "queue_capacity_audio_sec": 10 if state.live_stt is not None else None,
        "overloaded": overloaded, "buffer_bytes": buffered_audio_bytes(),
        "ring_buffer_sec": round(state.audio_ring_bytes / BYTES_PER_SECOND, 1),
        **{key: state.metrics[key] for key in ("windows_processed", "windows_retried", "windows_dropped",
                                        "windows_skipped", "windows_evicted", "buffer_bytes_evicted", "packets_rejected", "last_error",
                                        "last_outcome", "transcripts_emitted", "windows_no_speech", "windows_blocked",
                                        "windows_truncated", "windows_invalid_response")},
    }


async def handle_health(request):
    health = build_health_dict()
    return web.json_response(health, status=200 if health["status"] == "ok" else 503)


async def handle_omi_ws(request):
    """Direct phone-to-bridge WebSocket PCM16 streaming adapter (Omi Custom Live STT compatible)."""
    if state.stopping:
        raise web.HTTPServiceUnavailable(text="Bridge is draining for shutdown")
    try:
        sr = int(request.query.get("sample_rate", "16000"))
    except ValueError:
        raise web.HTTPBadRequest(text="sample_rate must be 16000")
    if sr != SAMPLE_RATE:
        raise web.HTTPBadRequest(text="sample_rate must be 16000")
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=BYTES_PER_SECOND * RING_BUFFER_SECONDS + 1)
    await ws.prepare(request)
    queue = asyncio.Queue(maxsize=32)
    state.stream_clients[ws] = queue

    async def sender():
        try:
            while True:
                msg = await queue.get()
                if msg is None:
                    return
                await asyncio.wait_for(ws.send_str(msg), timeout=5)
        except (OSError, asyncio.TimeoutError):
            pass

    task = asyncio.create_task(sender())
    try:
        async for message in ws:
            if message.type == aiohttp.WSMsgType.BINARY:
                raw = message.data
                if raw.startswith(b"RIFF") or len(raw) % 2:
                    await ws.close(code=1003, message=b"Expected raw 16000 Hz mono PCM16")
                    break
                ingest_raw_pcm(raw, sr)
            elif message.type == aiohttp.WSMsgType.TEXT:
                try:
                    ctrl = json.loads(message.data)
                except ValueError:
                    continue
                if isinstance(ctrl, dict) and ctrl.get("type") == "CloseStream":
                    if state.live_stt is not None:
                        state.live_stt.request_flush()
                    elif state.stt_queue is not None:
                        await flush_remaining_buffer()
                    break
            elif message.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                break
    finally:
        state.stream_clients.pop(ws, None)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await ws.close()
    return ws


async def handle_omi(request):
    if request.method == "GET" and request.headers.get("Upgrade", "").lower() == "websocket":
        return await handle_omi_ws(request)
    if request.method in ("GET", "HEAD"):
        # Omi validation fields remain stable; local /health reports degraded status separately.
        return web.json_response({"status": "ok", "is_setup_completed": True, "message": ""})
    if request.content_type != "application/octet-stream":
        raise web.HTTPUnsupportedMediaType(text="Use application/octet-stream raw PCM16")
    try:
        sr = int(request.query.get("sample_rate", "16000"))
    except ValueError:
        raise web.HTTPBadRequest(text="sample_rate must be 16000")
    raw = await request.read()
    if raw.startswith(b"RIFF"):
        raise web.HTTPUnsupportedMediaType(text="Send raw PCM16, without a WAV header")
    ingest_raw_pcm(raw, sr)
    return web.json_response({"status": "ok", "is_setup_completed": True, "message": ""})


async def handle_listen_ws(request):
    mode = request.query.get("mode", "audio")
    if mode not in ("audio", "captions"):
        raise web.HTTPBadRequest(text="mode must be audio or captions")
    try:
        count = int(request.query.get("replay", "100" if mode == "captions" else "0"))
    except ValueError:
        raise web.HTTPBadRequest(text="replay must be an integer from 0 to 200")
    if not 0 <= count <= 200:
        raise web.HTTPBadRequest(text="replay must be an integer from 0 to 200")
    after = request.query.get("after")
    if after is not None and len(after) > 128:
        raise web.HTTPBadRequest(text="Invalid replay cursor")
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    caption_mode = mode == "captions"
    listener = CaptionListener() if caption_mode else None
    queue = None if caption_mode else asyncio.Queue(maxsize=8)
    # Snapshot and subscribe without an await: no hole between replay and live delivery.
    replay, reset = [], False
    if caption_mode:
        history = list(state.recent_events)
        if after:
            index = next((i for i, event in enumerate(history) if event.get("event_id") == after), None)
            reset = index is None
            replay = history[index + 1:] if index is not None else (history[-count:] if count else [])
        elif count:
            replay = history[-count:]
        listener.partial = dict(state.partial_caption) if state.partial_caption else None
        state.caption_listeners[ws] = listener
    else:
        state.live_listeners[ws] = queue

    async def send_json(event):
        await asyncio.wait_for(ws.send_json(event), timeout=5)

    async def sender():
        try:
            if caption_mode:
                await send_json({"type": "replay", "phase": "start", "reset": reset, "count": len(replay)})
                for event in replay:
                    await send_json(event)
                await send_json({"type": "replay", "phase": "end"})
                listener.status = caption_status()
                listener.ready.set()
            while True:
                if caption_mode:
                    await listener.ready.wait()
                    listener.ready.clear()
                    if listener.closed:
                        return
                    while not listener.queue.empty():
                        await send_json(listener.queue.get_nowait())
                    partial, listener.partial = listener.partial, None
                    if partial is not None:
                        await send_json(partial)
                    status, listener.status = listener.status, None
                    if status is not None:
                        await send_json(status)
                else:
                    message = await queue.get()
                    if message is None:
                        return
                    send = ws.send_bytes if isinstance(message, bytes) else ws.send_str
                    await asyncio.wait_for(send(message), timeout=5)
        except (OSError, asyncio.TimeoutError):
            pass
        finally:
            await ws.close()

    task = asyncio.create_task(sender())
    try:
        async for _ in ws:
            pass
    finally:
        state.live_listeners.pop(ws, None)
        state.caption_listeners.pop(ws, None)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return ws


async def on_startup(app):
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key is required")
    state.stt_queue = asyncio.Queue(maxsize=24)
    seed_recent_events()
    state.daily_notes = DailyNotes(LOG_DIR)
    state.http_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=12, connect=5, sock_read=10),
        connector=aiohttp.TCPConnector(limit=10, limit_per_host=4), cookie_jar=aiohttp.DummyCookieJar())
    state.stopping = False
    state.metrics["started_at"] = time.monotonic()
    factories = [("caption_status_task", caption_status_worker)]
    if STT_MODE == "live":
        from live_stt import GeminiLiveSTT
        state.live_stt = GeminiLiveSTT(session=state.http_session, api_key=GEMINI_API_KEY,
            model=GEMINI_LIVE_MODEL, on_partial=publish_partial, on_final=record_live_final,
            on_gap=record_live_gap, on_latency=record_latency, on_connection=live_connection_changed,
            max_age=MAX_WINDOW_AGE_SEC, idle_flush=IDLE_FLUSH_SEC, rms_threshold=RMS_SILENCE_THRESHOLD)
        factories.append(("live_stt_task", state.live_stt.run))
    else:
        factories += [("slicer_task", window_slicer_worker), ("stt_worker_task", stt_consumer_worker)]
    for name, factory in factories:
        app[name] = asyncio.create_task(factory())
        state.background_tasks[name] = app[name]


async def on_shutdown(app):
    state.stopping = True
    await asyncio.gather(*(ws.close() for ws in list(state.live_listeners) + list(state.caption_listeners) + list(state.stream_clients)), return_exceptions=True)


async def on_cleanup(app):
    state.stopping = True
    logger.info("Shutdown: draining accepted audio and pending STT windows")
    stop_tasks = [app[name] for name in ("slicer_task", "caption_status_task") if name in app]
    for task in stop_tasks:
        task.cancel()
    await asyncio.gather(*stop_tasks, return_exceptions=True)
    try:
        if state.live_stt is not None:
            await state.live_stt.close()
            await app["live_stt_task"]
        else:
            worker = app["stt_worker_task"]
            if worker.done():
                await worker  # Surface a worker failure rather than hanging forever on queue.join().
                raise RuntimeError("STT worker stopped before shutdown drain")
            await flush_remaining_buffer()
            await state.stt_queue.join()
            await state.stt_queue.put(None)
            await worker
        flush_daily_paragraph()
        logger.info("Shutdown drain complete")
    finally:
        if state.http_session is not None:
            await state.http_session.close()
            if state.live_stt is not None:
                # aiohttp TLS transports finish closing after ClientSession.close().
                await asyncio.sleep(.25)
            state.http_session = None
        state.background_tasks.clear()


def create_app(start_workers=True):
    # Request admission is separate from the smaller pending-transcription budget.
    # aiohttp rejects bodies >= this limit; allow exactly one ring's worth of PCM.
    app = web.Application(client_max_size=BYTES_PER_SECOND * RING_BUFFER_SECONDS + 1)
    app.router.add_get("/omi", handle_omi)
    app.router.add_post("/omi", handle_omi)
    app.router.add_get("/omi/ws", handle_omi_ws)
    app.router.add_get("/listen-ws", handle_listen_ws)
    app.router.add_get("/audio/last", handle_audio_last)
    if start_workers:
        app.on_startup.append(on_startup)
        app.on_shutdown.append(on_shutdown)
        app.on_cleanup.append(on_cleanup)
    return app


def create_health_app():
    app = web.Application(client_max_size=4096)
    app.router.add_get("/health", handle_health)
    return app


async def serve():
    finished = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, finished.set)
    runner = web.AppRunner(create_app(), access_log=None)
    health_runner = web.AppRunner(create_health_app(), access_log=None)
    try:
        await runner.setup()
        await web.TCPSite(runner, HOST, PORT).start()
        await health_runner.setup()
        await web.TCPSite(health_runner, "127.0.0.1", HEALTH_PORT).start()
        logger.info("Listening on %s:%d; local health on 127.0.0.1:%d", HOST, PORT, HEALTH_PORT)
        await finished.wait()
    finally:
        await health_runner.cleanup()
        await runner.cleanup()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(signum)


async def handle_audio_last(request: web.Request) -> web.Response:
    """Export the last N seconds (default 60, max 300) from the 5-minute RAM ring buffer as a playable WAV."""
    try:
        seconds = int(request.query.get("seconds", "60"))
    except ValueError:
        raise web.HTTPBadRequest(text="seconds must be an integer")
    if not 1 <= seconds <= RING_BUFFER_SECONDS:
        raise web.HTTPBadRequest(text=f"seconds must be between 1 and {RING_BUFFER_SECONDS}")
    sr = SAMPLE_RATE
    need_bytes = sr * 2 * seconds
    collected = deque()
    total = 0
    for _, chunk_sr, pcm in reversed(state.audio_ring):
        sr = chunk_sr
        collected.appendleft(pcm)
        total += len(pcm)
        if total >= need_bytes:
            break
    raw_pcm = b"".join(collected)
    if len(raw_pcm) > need_bytes:
        raw_pcm = raw_pcm[-need_bytes:]

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(raw_pcm)
    return web.Response(
        body=wav_buf.getvalue(),
        content_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="omi_last_{seconds}s.wav"'},
    )


if __name__ == "__main__":
    try:
        configure()
    except (ValueError, OSError):
        sys.exit("Invalid bridge configuration; check API key and numeric settings in .env")
    if len(sys.argv) > 1:
        if sys.argv[1:] != ["--check-config"]:
            sys.exit("Usage: omi_bridge.py [--check-config]")
        print("Bridge configuration is valid")
    else:
        configure_logging()
        try:
            asyncio.run(serve())
        finally:
            pid_path = os.path.join(LOG_DIR, "omi_bridge.pid")
            with suppress(OSError):
                with open(pid_path) as f:
                    own_pid = f.read().strip() == str(os.getpid())
                if own_pid:
                    os.unlink(pid_path)
