#!/usr/bin/env python3.15
"""Single-device PCM16 -> bounded, sequential Gemini transcription bridge."""
import asyncio
import base64
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
import io
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
GEMINI_API_KEY = ""
GEMINI_STT_MODEL = "gemini-3.8-flash"
GEMINI_THINKING_LEVEL = "low"
PRE_OVERLAP_SEC, CORE_WINDOW_SEC, POST_LOOKAHEAD_SEC = 1.0, 8.0, 1.0
IDLE_FLUSH_SEC, RMS_SILENCE_THRESHOLD = 8.0, 80.0
RING_BUFFER_SECONDS = 300
PORT, HEALTH_PORT = 8787, 8788
HERMES_LIVE_WEBHOOK = "http://127.0.0.1:8644/webhooks/omi-live"
HERMES_RESEARCH_WEBHOOK = "http://127.0.0.1:8644/webhooks/omi-research"
WAKE_WORD, RESEARCH_INTERVAL_SEC = "hermes", 90
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


def _load_dotenv(local_path=None, fallback_path=None, environ=None):
    """Resolve process > project .env > Hermes, including both key aliases."""
    sources = [dict(os.environ if environ is None else environ),
               _read_env_file(local_path or os.path.join(BASE_DIR, ".env")),
               _read_env_file(fallback_path or os.path.expanduser("~/.hermes/.env"))]
    values = {}
    for source in reversed(sources):
        values.update(source)
    values["GEMINI_API_KEY"] = next(
        (source[key].strip() for source in sources for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY")
         if source.get(key, "").strip()), "")
    return values


def configure(env=None):
    global GEMINI_API_KEY, GEMINI_STT_MODEL, GEMINI_THINKING_LEVEL, PRE_OVERLAP_SEC, CORE_WINDOW_SEC, POST_LOOKAHEAD_SEC
    global IDLE_FLUSH_SEC, RMS_SILENCE_THRESHOLD, RING_BUFFER_SECONDS, PORT, HEALTH_PORT
    global HERMES_LIVE_WEBHOOK, HERMES_RESEARCH_WEBHOOK, WAKE_WORD, RESEARCH_INTERVAL_SEC
    env = _load_dotenv() if env is None else env
    GEMINI_API_KEY = (env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or "").strip()
    if not GEMINI_API_KEY or GEMINI_API_KEY.lower() in ("your_gemini_api_key_here", "your_api_key_here"):
        raise ValueError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in ~/omi/.env or the environment")
    GEMINI_STT_MODEL = env.get("OMI_GEMINI_STT_MODEL", "gemini-3.8-flash").strip()
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
    if not all(math.isfinite(v) and v >= 0 for v in
               (PRE_OVERLAP_SEC, CORE_WINDOW_SEC, POST_LOOKAHEAD_SEC, IDLE_FLUSH_SEC, RMS_SILENCE_THRESHOLD)):
        raise ValueError("Audio settings must be finite, nonnegative numbers")
    if not 1 <= CORE_WINDOW_SEC <= 10 or PRE_OVERLAP_SEC > 5 or POST_LOOKAHEAD_SEC > 5 or IDLE_FLUSH_SEC <= 0:
        raise ValueError("Use core 1-10s, overlap/lookahead 0-5s, and a positive idle timeout")
    RING_BUFFER_SECONDS = int(env.get("OMI_RING_BUFFER_SEC", "300"))
    RESEARCH_INTERVAL_SEC = int(env.get("OMI_RESEARCH_INTERVAL", "90"))
    if RING_BUFFER_SECONDS < 1 or RESEARCH_INTERVAL_SEC < 1:
        raise ValueError("Ring buffer and research interval must be positive")
    HERMES_LIVE_WEBHOOK = env.get("OMI_HERMES_LIVE_WEBHOOK", "http://127.0.0.1:8644/webhooks/omi-live")
    HERMES_RESEARCH_WEBHOOK = env.get("OMI_HERMES_RESEARCH_WEBHOOK", "http://127.0.0.1:8644/webhooks/omi-research")
    WAKE_WORD = env.get("OMI_WAKE_WORD", "hermes").strip().lower()


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
        handler = RotatingFileHandler(os.path.join(LOG_DIR, filename), maxBytes=5 * 1024 * 1024, backupCount=3)
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


@dataclass(frozen=True)
class STTResult:
    text: str
    outcome: str = "speech"
    finish_reason: str = "STOP"


class GeminiResponseError(Exception):
    def __init__(self, kind, retryable=False):
        self.kind, self.retryable = kind, retryable
        super().__init__(f"Gemini response: {kind}")


# Construct queues in the running event loop.
stt_queue = None
audio_sessions = deque()
current_session = None
_next_session_id = 0
audio_pcm_buffer, audio_pre_tail = bytearray(), bytearray()
audio_sample_rate = SAMPLE_RATE
audio_ring, audio_ring_bytes = deque(), 0
daily_notes = None
rolling_research_buffer = []
last_research_time = time.monotonic()
live_listeners = {}  # WebSocket -> bounded outgoing queue
webhook_tasks = set()
background_tasks = {}
http_session = None
stopping = False
metrics = {
    "started_at": time.monotonic(), "last_packet_at": None, "last_stt_success_at": None,
    "last_error": None, "windows_processed": 0, "windows_retried": 0,
    "windows_dropped": 0, "windows_evicted": 0, "buffer_bytes_evicted": 0, "windows_skipped": 0, "packets_rejected": 0, "stt_in_flight": False,
    "last_transcript_at": None, "last_outcome": None, "transcripts_emitted": 0,
    "windows_no_speech": 0, "windows_blocked": 0, "windows_truncated": 0, "windows_invalid_response": 0,
}


async def post_webhook(url, payload):
    try:
        async with http_session.post(url, data=payload, allow_redirects=False,
                headers={"Content-Type": "application/json", "X-Request-ID": f"omi-{time.time_ns()}"},
                timeout=aiohttp.ClientTimeout(total=15)) as response:
            if not 200 <= response.status < 300:
                raise aiohttp.ClientResponseError(response.request_info, response.history,
                                                  status=response.status)
            await response.read()
    except Exception as e:
        logger.warning("Hermes delivery failed: %s", type(e).__name__)


def schedule_webhook(url, payload):
    if not url:
        return
    if len(webhook_tasks) >= 8:
        logger.warning("Hermes delivery backlog is full")
        return
    task = asyncio.create_task(post_webhook(url, payload))
    webhook_tasks.add(task)
    task.add_done_callback(webhook_tasks.discard)


def broadcast(message):
    for ws, queue in list(live_listeners.items()):
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            # Let the existing sender close a slow listener, without creating more tasks.
            live_listeners.pop(ws, None)
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait(None)


class DailyNotes:
    """A paragraph projection; persist each update so restart does not lose a partial paragraph."""

    def __init__(self, directory):
        self.directory = directory
        self.day, self.words, self.last_time = None, [], None

    def close_paragraph(self):
        self.words = []

    def add(self, timestamp, text):
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
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
    if daily_notes is not None:
        daily_notes.close_paragraph()


def record_transcript_line(text, audio_start_at=None, received_at=None, processed_at=None,
                           session_id=None, sample_offset=None):
    text = " ".join(text.split()).strip()
    if not text:
        return
    processed_at = time.time() if processed_at is None else processed_at
    source_at = audio_start_at if audio_start_at is not None else processed_at
    timestamp = datetime.fromtimestamp(source_at).strftime("%Y-%m-%d %H:%M:%S")
    event = {"timestamp": timestamp, "text": text, "audio_start_at": audio_start_at,
             "received_at": received_at, "processed_at": processed_at,
             "session_id": session_id, "sample_offset": sample_offset,
             "time_basis": "estimated_from_receipt" if audio_start_at is not None else "processing_time"}
    transcript_logger.info("[%s] %s", timestamp, text)
    event_logger.info(json.dumps(event, ensure_ascii=False))
    if daily_notes is not None:
        daily_notes.add(timestamp, text)
    if HERMES_RESEARCH_WEBHOOK:
        rolling_research_buffer.append(f"[{timestamp}] {text}")
    broadcast(json.dumps({"type": "transcript", **event}))
    if WAKE_WORD and WAKE_WORD in text.lower():
        schedule_webhook(HERMES_LIVE_WEBHOOK, json.dumps(
            {"speaker": "user", **event}).encode())


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
            "Use the overlap so no boundary words are clipped, and output ONLY the new continuation after those words. "
        )

    prompt = (
        "Transcribe the speech in this audio clip into clean, uniform, single-paragraph English prose. "
        + overlap_note
        + "Do not include timestamps, speaker labels, markdown, or commentary. "
        "If there is no intelligible speech beyond the previous context, return ONLY an empty string."
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_STT_MODEL}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "audio/wav", "data": b64_wav}}]}],
        "generationConfig": {"thinkingConfig": {"thinkingLevel": GEMINI_THINKING_LEVEL}},
    }
    async with http_session.post(url, json=payload, headers={"x-goog-api-key": GEMINI_API_KEY},
                                 allow_redirects=False) as response:
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
                             session_id=None, audio_start_at=None, received_at=None, sample_offset=None):
    """Prefer fresh windows during overload; preserve all pending work during shutdown."""
    if session_id is None:
        session_id = current_session.session_id if current_session else _next_session_id
    window = AudioWindow(full_pcm, sr, pre_s, core_s, post_s, reset_context_after,
                         session_id, audio_start_at, received_at, sample_offset)
    if stopping:
        await stt_queue.put(window)
        return
    if stt_queue.full():
        dropped = stt_queue.get_nowait()
        stt_queue.task_done()
        if dropped is not None and dropped.pcm:
            metrics["windows_dropped"] += 1
            metrics["windows_evicted"] += 1
            logger.warning("STT backlog full: evicted oldest window (%.2fs core)", dropped.core_sec)
    stt_queue.put_nowait(window)


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


async def stt_consumer_worker():
    """One ordered consumer, one initial attempt plus at most three transient retries."""
    last_transcript_tail, context_session, next_sample = "", None, None
    while True:
        window = await stt_queue.get()
        try:
            if window is None:
                return
            if window.session_id != context_session or (window.sample_offset is not None and next_sample is not None and window.sample_offset != next_sample):
                last_transcript_tail, context_session = "", window.session_id
                flush_daily_paragraph()
            next_sample = (window.sample_offset + round(window.core_sec * window.sample_rate)
                           if window.sample_offset is not None else None)
            pre_b = int(window.pre_sec * BYTES_PER_SECOND)
            core_b = int(window.core_sec * BYTES_PER_SECOND)
            core = window.pcm[pre_b:pre_b + core_b]
            if core and pcm16_rms(core) >= RMS_SILENCE_THRESHOLD:
                metrics["stt_in_flight"] = True
                result = None
                for attempt in range(len(RETRY_DELAYS) + 1):
                    try:
                        result = await gemini_stt_request(window.pcm, window.sample_rate,
                            window.pre_sec, window.core_sec, window.post_sec, last_transcript_tail)
                        metrics["last_stt_success_at"] = time.monotonic()
                        metrics["windows_processed"] += 1
                        metrics["last_error"] = None
                        metrics["last_outcome"] = result.outcome
                        if result.outcome == "no_speech":
                            metrics["windows_no_speech"] += 1
                        break
                    except Exception as error:
                        metrics["last_error"] = _error_label(error)
                        retryable = _transient_error(error)
                        if isinstance(error, GeminiResponseError):
                            metrics[f"windows_{error.kind}"] += 1
                            metrics["last_outcome"] = error.kind
                        if not retryable or attempt == len(RETRY_DELAYS):
                            metrics["windows_dropped"] += 1
                            last_transcript_tail = ""
                            logger.error("STT window dropped after %d attempt(s): %s", attempt + 1, metrics["last_error"])
                            break
                        metrics["windows_retried"] += 1
                        logger.warning("%s; retry %d/3 in %.1fs", metrics["last_error"], attempt + 1, RETRY_DELAYS[attempt])
                        await asyncio.sleep(RETRY_DELAYS[attempt])
                if result and result.outcome == "speech":
                    cleaned = dedupe_boundary_overlap(last_transcript_tail, result.text)
                    if cleaned:
                        record_transcript_line(cleaned, audio_start_at=window.audio_start_at,
                                               received_at=window.received_at, session_id=window.session_id,
                                               sample_offset=window.sample_offset)
                        metrics["last_transcript_at"] = time.monotonic()
                        metrics["transcripts_emitted"] += 1
                        last_transcript_tail = " ".join((last_transcript_tail + " " + cleaned).split()[-24:])
            elif core:
                metrics["windows_skipped"] += 1
            if window.reset_context_after:
                last_transcript_tail = ""
                flush_daily_paragraph()
        except Exception as error:
            # A failed output file or malformed response must not silently kill the sole consumer.
            metrics["last_error"] = _error_label(error)
            metrics["windows_dropped"] += 1
            last_transcript_tail = ""
            logger.exception("STT window processing failed")
        finally:
            metrics["stt_in_flight"] = False
            stt_queue.task_done()


def buffered_audio_bytes():
    return sum(len(session.pcm) for session in audio_sessions)


def ingest_raw_pcm(raw, sr=SAMPLE_RATE):
    global current_session, _next_session_id, audio_pcm_buffer, audio_pre_tail, audio_ring_bytes
    if stopping:
        raise web.HTTPServiceUnavailable(text="Bridge is draining for shutdown")
    if sr != SAMPLE_RATE or len(raw) % 2:
        raise web.HTTPBadRequest(text="Expected 16000 Hz mono PCM16 with complete two-byte samples")
    if not raw:
        return
    now, received_at = time.monotonic(), time.time()
    # Retain every valid live packet before applying the pending-transcription limit.
    audio_ring.append((now, SAMPLE_RATE, raw))
    audio_ring_bytes += len(raw)
    max_ring_bytes = BYTES_PER_SECOND * RING_BUFFER_SECONDS
    while audio_ring_bytes > max_ring_bytes:
        stamp, old_sr, old = audio_ring.popleft()
        remove = min(audio_ring_bytes - max_ring_bytes, len(old))
        audio_ring_bytes -= remove
        if remove < len(old):
            audio_ring.appendleft((stamp, old_sr, old[remove:]))
    if current_session is None or current_session.closed or now - current_session.last_packet_at > IDLE_FLUSH_SEC:
        if current_session:
            current_session.closed = True
        _next_session_id += 1
        current_session = AudioSession(_next_session_id, now, first_received_at=received_at,
                                       last_received_at=received_at,
                                       audio_start_at=received_at - len(raw) / BYTES_PER_SECOND)
        audio_sessions.append(current_session)
        audio_pcm_buffer, audio_pre_tail = current_session.pcm, current_session.pre_tail
    current_session.last_packet_at = now
    current_session.last_received_at = received_at
    current_session.received_samples += len(raw) // 2
    current_session.receipts.append((current_session.received_samples, received_at))
    current_session.pcm.extend(raw)
    metrics["last_packet_at"] = now
    excess = buffered_audio_bytes() - MAX_BUFFER_BYTES
    if excess > 0:
        for session in audio_sessions:
            remove = min(excess, len(session.pcm))
            if not remove:
                continue
            del session.pcm[:remove]
            session.consumed_samples += remove // 2
            session.pre_tail.clear()
            while session.receipts and session.receipts[0][0] <= session.consumed_samples:
                session.receipts.popleft()
            metrics["buffer_bytes_evicted"] += remove
            metrics["windows_dropped"] += 1
            excess -= remove
            if not excess:
                break
        logger.warning("Raw backlog full: evicted oldest PCM; total evicted bytes=%d",
                       metrics["buffer_bytes_evicted"])
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
    received_at = next((stamp for sample, stamp in session.receipts if sample >= end_sample),
                       session.last_received_at)
    if audio_start_at is not None and received_at is not None:
        # Backfilled/bursty uploads must not imply that audio was captured after it arrived.
        audio_start_at = min(audio_start_at, received_at - end / bps)
    await enqueue_stt_window(full, SAMPLE_RATE, len(session.pre_tail) / bps,
                             len(core) / bps, (end - core_cut) / bps, final, session.session_id,
                             audio_start_at, received_at, session.consumed_samples)
    # No await between successful queue admission and committing the consumed prefix.
    del session.pcm[:core_cut]
    session.consumed_samples += core_cut // 2
    while session.receipts and session.receipts[0][0] <= session.consumed_samples:
        session.receipts.popleft()
    session.pre_tail[:] = b"" if final or pre_b == 0 else core[-pre_b:]


async def flush_remaining_buffer(reset_context=True):
    """Close sessions in order; admit all complete PCM samples before a reset marker."""
    for session in audio_sessions:
        session.closed = True
    need_b = int((CORE_WINDOW_SEC + POST_LOOKAHEAD_SEC) * BYTES_PER_SECOND) & ~1
    while audio_sessions:
        session = audio_sessions[0]
        while len(session.pcm) >= need_b:
            await _slice_one(session)
        if len(session.pcm) >= 2:
            await _slice_one(session, final=True)
        else:
            await enqueue_stt_window(b"", SAMPLE_RATE, 0, 0, 0, reset_context, session.session_id)
        session.pcm.clear()
        session.pre_tail.clear()
        audio_sessions.popleft()
    if current_session:
        current_session.closed = True
    if reset_context:
        # Also handles an entirely empty buffer: the reset remains ordered behind older STT jobs.
        await enqueue_stt_window(b"", SAMPLE_RATE, 0, 0, 0, True)


async def window_slicer_worker():
    need_b = int((CORE_WINDOW_SEC + POST_LOOKAHEAD_SEC) * BYTES_PER_SECOND) & ~1
    while True:
        await asyncio.sleep(0.2)
        while audio_sessions:
            session = audio_sessions[0]
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
                audio_sessions.popleft()
            else:
                break


async def research_worker():
    global last_research_time
    while True:
        await asyncio.sleep(5)
        if time.monotonic() - last_research_time >= RESEARCH_INTERVAL_SEC and len(rolling_research_buffer) >= 2:
            chunk = "\n".join(rolling_research_buffer)
            rolling_research_buffer.clear()
            last_research_time = time.monotonic()
            schedule_webhook(HERMES_RESEARCH_WEBHOOK, json.dumps(
                {"text": chunk, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode())


def build_health_dict():
    now = time.monotonic()
    workers_alive = bool(background_tasks) and all(not task.done() for task in background_tasks.values())
    overloaded = stt_queue is not None and stt_queue.full()
    return {
        "status": "ok" if GEMINI_API_KEY and workers_alive and not stopping and not overloaded and not metrics["last_error"] else "degraded",
        "is_setup_completed": True, "message": "", "pid": os.getpid(),
        "gemini_key_configured": bool(GEMINI_API_KEY), "model": GEMINI_STT_MODEL,
        "thinking_level": GEMINI_THINKING_LEVEL,
        "python_version": platform.python_version(), "openssl_version": ssl.OPENSSL_VERSION,
        "workers_alive": workers_alive, "stopping": stopping,
        "uptime_sec": round(now - metrics["started_at"], 1),
        "last_packet_age_sec": round(now - metrics["last_packet_at"], 1) if metrics["last_packet_at"] is not None else None,
        "last_stt_success_age_sec": round(now - metrics["last_stt_success_at"], 1) if metrics["last_stt_success_at"] is not None else None,
        "last_transcript_age_sec": round(now - metrics["last_transcript_at"], 1) if metrics["last_transcript_at"] is not None else None,
        "queue_depth": stt_queue.qsize() if stt_queue is not None else 0,
        "queue_capacity": 24, "overloaded": overloaded, "buffer_bytes": buffered_audio_bytes(),
        "ring_buffer_sec": round(audio_ring_bytes / BYTES_PER_SECOND, 1),
        **{key: metrics[key] for key in ("windows_processed", "windows_retried", "windows_dropped",
                                        "windows_skipped", "windows_evicted", "buffer_bytes_evicted", "packets_rejected", "stt_in_flight", "last_error",
                                        "last_outcome", "transcripts_emitted", "windows_no_speech", "windows_blocked",
                                        "windows_truncated", "windows_invalid_response")},
    }


async def handle_health(request):
    health = build_health_dict()
    return web.json_response(health, status=200 if health["status"] == "ok" else 503)


async def handle_omi(request):
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
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    queue = asyncio.Queue(maxsize=8)
    live_listeners[ws] = queue

    async def sender():
        try:
            while True:
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
        live_listeners.pop(ws, None)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return ws


async def on_startup(app):
    global stt_queue, stopping, daily_notes, http_session
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key is required")
    stt_queue = asyncio.Queue(maxsize=24)
    daily_notes = DailyNotes(LOG_DIR)
    http_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=25, connect=10, sock_read=20),
        connector=aiohttp.TCPConnector(limit=10, limit_per_host=4), cookie_jar=aiohttp.DummyCookieJar())
    stopping = False
    metrics["started_at"] = time.monotonic()
    for name, factory in (("slicer_task", window_slicer_worker), ("stt_worker_task", stt_consumer_worker),
                          ("research_task", research_worker)):
        app[name] = asyncio.create_task(factory())
        background_tasks[name] = app[name]


async def on_shutdown(app):
    global stopping
    stopping = True
    await asyncio.gather(*(ws.close() for ws in list(live_listeners)), return_exceptions=True)


async def on_cleanup(app):
    global stopping, http_session
    stopping = True
    logger.info("Shutdown: draining accepted audio and pending STT windows")
    for name in ("slicer_task", "research_task"):
        app[name].cancel()
    await asyncio.gather(app["slicer_task"], app["research_task"], return_exceptions=True)
    try:
        worker = app["stt_worker_task"]
        if worker.done():
            await worker  # Surface a worker failure rather than hanging forever on queue.join().
            raise RuntimeError("STT worker stopped before shutdown drain")
        await flush_remaining_buffer()
        await stt_queue.join()
        await stt_queue.put(None)
        await worker
        flush_daily_paragraph()
        if rolling_research_buffer:
            schedule_webhook(HERMES_RESEARCH_WEBHOOK, json.dumps({"text": "\n".join(rolling_research_buffer),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}).encode())
            rolling_research_buffer.clear()
        if webhook_tasks:
            await asyncio.gather(*list(webhook_tasks), return_exceptions=True)
        logger.info("Shutdown drain complete")
    finally:
        if http_session is not None:
            await http_session.close()
            http_session = None
        background_tasks.clear()


def create_app(start_workers=True):
    # Request admission is separate from the smaller pending-transcription budget.
    # aiohttp rejects bodies >= this limit; allow exactly one ring's worth of PCM.
    app = web.Application(client_max_size=BYTES_PER_SECOND * RING_BUFFER_SECONDS + 1)
    app.router.add_get("/omi", handle_omi)
    app.router.add_post("/omi", handle_omi)
    app.router.add_get("/listen", handle_listen)
    app.router.add_get("/listen-ws", handle_listen_ws)
    app.router.add_get("/audio/last", handle_audio_last)
    if start_workers:
        app.on_startup.append(on_startup)
        app.on_shutdown.append(on_shutdown)
        app.on_cleanup.append(on_cleanup)
    return app


def create_health_app():
    app = web.Application()
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
        await web.TCPSite(runner, "127.0.0.1", PORT).start()
        await health_runner.setup()
        await web.TCPSite(health_runner, "127.0.0.1", HEALTH_PORT).start()
        logger.info("Listening on 127.0.0.1:%d; local health on 127.0.0.1:%d", PORT, HEALTH_PORT)
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
    sr = audio_sample_rate
    need_bytes = sr * 2 * seconds
    collected = deque()
    total = 0
    for _, chunk_sr, pcm in reversed(audio_ring):
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


LISTEN_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Omi Live Stream</title>
<style>body{font-family:system-ui,sans-serif;background:#0f172a;color:#f8fafc;padding:24px;max-width:900px;margin:0 auto}button{background:#3b82f6;color:#fff;border:none;padding:10px 18px;border-radius:8px;cursor:pointer}#log{margin-top:16px;background:#1e293b;padding:16px;border-radius:8px;height:70vh;overflow-y:auto;white-space:pre-wrap}</style>
</head><body><h2>Omi Live Stream (16kHz PCM16 + Dynamic STT windows)</h2>
<button id="btn">Start Listening</button> <span id="status">Disconnected</span><div id="log"></div>
<script>
let ctx,nextTime=0;const btn=document.getElementById('btn'),status=document.getElementById('status'),log=document.getElementById('log');
btn.onclick=()=>{
  if(!ctx)ctx=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16000});
  const ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/listen-ws');
  ws.binaryType='arraybuffer';
  ws.onopen=()=>{status.textContent='Connected';btn.disabled=true;};
  ws.onclose=()=>{status.textContent='Disconnected';btn.disabled=false;};
  ws.onmessage=(e)=>{
    if(typeof e.data==='string'){const m=JSON.parse(e.data);if(m.type==='transcript'){log.textContent+=`[${m.timestamp}] ${m.text}\n\n`;log.scrollTop=log.scrollHeight;}}
    else{const i16=new Int16Array(e.data),f32=new Float32Array(i16.length);for(let i=0;i<i16.length;i++)f32[i]=i16[i]/32768.0;const b=ctx.createBuffer(1,f32.length,16000);b.getChannelData(0).set(f32);const s=ctx.createBufferSource();s.buffer=b;s.connect(ctx.destination);if(nextTime<ctx.currentTime)nextTime=ctx.currentTime+0.05;s.start(nextTime);nextTime+=b.duration;}
  };
};
</script></body></html>"""


async def handle_listen(request: web.Request) -> web.Response:
    return web.Response(text=LISTEN_HTML, content_type="text/html")




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
