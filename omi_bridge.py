#!/usr/bin/env python3.15
"""Single-device PCM16 -> persistent Gemini Live transcription bridge."""
import asyncio
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import hmac
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
import sys
import sysconfig
import time
import uuid


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

from live_stt import GeminiLiveSTT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2
MAX_AUDIO_AGE_SEC = 45.0
GEMINI_API_KEY = ""
GEMINI_LIVE_MODEL = "gemini-3.5-transcribe-live"
IDLE_FLUSH_SEC, RMS_SILENCE_THRESHOLD = 8.0, 80.0
MAX_INPUT_BYTES = BYTES_PER_SECOND * 300
HOST = "0.0.0.0"
PORT, HEALTH_PORT = 8787, 8788
STREAM_TOKEN = ""
logger = logging.getLogger("omi_bridge")
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
    global GEMINI_API_KEY, GEMINI_LIVE_MODEL, MAX_AUDIO_AGE_SEC
    global IDLE_FLUSH_SEC, RMS_SILENCE_THRESHOLD, HOST, STREAM_TOKEN
    env = _load_dotenv() if env is None else env
    HOST = (env.get("OMI_HOST") or "0.0.0.0").strip()
    STREAM_TOKEN = (env.get("OMI_STREAM_TOKEN") or "").strip()
    GEMINI_API_KEY = (env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or "").strip()
    if not GEMINI_API_KEY or GEMINI_API_KEY.lower() in ("your_gemini_api_key_here", "your_api_key_here"):
        raise ValueError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in ~/omi/.env or the environment")
    if env.get("OMI_STT_MODE", "live").strip().lower() != "live":
        raise ValueError("Only live transcription is supported; remove OMI_STT_MODE from your configuration")
    GEMINI_LIVE_MODEL = env.get("OMI_GEMINI_LIVE_MODEL", "gemini-3.5-transcribe-live").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", GEMINI_LIVE_MODEL):
        raise ValueError("OMI_GEMINI_LIVE_MODEL must be a valid model ID")
    MAX_AUDIO_AGE_SEC = float(env.get("OMI_MAX_AUDIO_AGE_SEC", "45"))
    IDLE_FLUSH_SEC = float(env.get("OMI_IDLE_FLUSH_SEC", "8"))
    RMS_SILENCE_THRESHOLD = float(env.get("OMI_RMS_THRESHOLD", "80"))
    if not all(math.isfinite(v) and v > 0 for v in (MAX_AUDIO_AGE_SEC, IDLE_FLUSH_SEC)):
        raise ValueError("Maximum audio age and idle timeout must be finite and positive")
    if not math.isfinite(RMS_SILENCE_THRESHOLD) or RMS_SILENCE_THRESHOLD < 0:
        raise ValueError("RMS threshold must be finite and nonnegative")


class OutputFileHandler(RotatingFileHandler):
    def handleError(self, record):
        # Logging normally swallows disk errors; output health must see them.
        raise sys.exc_info()[1]


def configure_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    for target, filename, fmt in (
        (logger, "omi_bridge.log", "[%(asctime)s] %(message)s"),
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


def new_metrics():
    return {
        "started_at": time.monotonic(), "last_packet_at": None, "last_stt_success_at": None,
        "last_error": None, "utterances_processed": 0, "audio_gaps": 0, "queue_evictions": 0,
        "last_transcript_at": None, "transcripts_emitted": 0,
        "lost_audio_sec_failed": 0.0, "lost_audio_sec_stale": 0.0,
        "lost_audio_sec_queue_evicted": 0.0,
        "output_errors": 0, "last_output_error": None,
        "audio_source": None, "direct_ws_remote": None, "direct_ws_packets": 0,
    }


@dataclass
class BridgeState:
    daily_notes: object = None
    caption_listeners: dict = field(default_factory=dict)
    stream_clients: dict = field(default_factory=dict)
    recent_events: deque = field(default_factory=lambda: deque(maxlen=200))
    stream_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    background_tasks: dict = field(default_factory=dict)
    http_session: aiohttp.ClientSession | None = None
    stopping: bool = False
    metrics: dict = field(default_factory=new_metrics)
    live_stt: GeminiLiveSTT | None = None
    partial_caption: dict | None = None
    latency_samples: dict = field(default_factory=lambda: {
        name: deque(maxlen=128) for name in ("receipt_to_transcript_sec",
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


def oldest_pending_audio_age():
    age = state.live_stt.pending_age if state.live_stt is not None else None
    return round(age, 3) if age is not None else None


@dataclass
class CaptionListener:
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=64))
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    status: dict | None = None
    partial: dict | None = None
    closed: bool = False


def publish_caption(event):
    identity = event.get("utterance_id")
    is_gap = event.get("type") == "gap"
    clear = is_gap or (state.partial_caption is not None
        and identity == state.partial_caption.get("utterance_id"))
    if clear:
        state.partial_caption = None
    for listener in list(state.caption_listeners.values()):
        if clear:
            listener.partial = {"type": "partial", "utterance_id": identity or "", "text": ""} if is_gap else None
        if not is_gap:
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


def record_live_final(text, **timing):
    state.metrics["last_stt_success_at"] = time.monotonic()
    state.metrics["utterances_processed"] += 1
    state.metrics["last_error"] = None
    saved = record_transcript_line(text, **timing)
    if saved is not False:
        state.metrics["last_transcript_at"] = time.monotonic()
        state.metrics["transcripts_emitted"] += 1
    return saved


def record_live_gap(reason, seconds, **timing):
    state.metrics["audio_gaps"] += 1
    if reason == "queue_evicted":
        state.metrics["queue_evictions"] += 1
    record_gap(reason, seconds, **timing)


def live_connection_changed(connected, error):
    state.metrics["last_error"] = None if connected else "Gemini Live: " + str(error)
    logger.info("Gemini Live %s", "connected" if connected else "reconnecting (" + str(error) + ")")


def current_stt_in_flight():
    return bool(state.live_stt is not None and state.live_stt.speech_pending)


def caption_status():
    now = time.monotonic()
    packet = state.metrics["last_packet_at"]
    age = max(0, now - packet) if packet is not None else None
    stream = state.live_stt
    return {"type": "status", "stream_id": state.stream_id,
            "session_id": stream.session_id if stream is not None else None,
            "stt_mode": "live",
            "stream_connected": bool(stream is not None and stream.connected),
            "audio_active": age is not None and age < 3,
            "last_packet_age_sec": round(age, 1) if age is not None else None,
            "buffer_sec": round(buffered_audio_bytes() / BYTES_PER_SECOND, 1),
            "stt_in_flight": current_stt_in_flight(),
            "queue_depth": len(stream.queue) if stream is not None else 0}


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
            if not isinstance(event, dict) or event.get("type", "transcript") != "transcript" or not isinstance(event.get("text"), str):
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



def flush_daily_paragraph():
    if state.daily_notes is not None:
        state.daily_notes.close_paragraph()


def output_failed(target, error):
    state.metrics["output_errors"] += 1
    state.metrics["last_output_error"] = f"{target}: {type(error).__name__}"
    logger.error("Transcript output failed: %s", state.metrics["last_output_error"])


def write_event(event):
    """Persist JSONL first, then publish captions and update daily Markdown."""
    encoded = json.dumps(event, ensure_ascii=False)
    try:
        event_logger.info(encoded)
    except Exception as error:
        output_failed("jsonl", error)
        return False
    if event.get("type") != "gap":
        state.recent_events.append(dict(event))
    publish_caption(event)
    if event.get("type") == "transcript" and event.get("text"):
        text_val = event["text"]
        seg = {"text": text_val, "start": 0.0, "end": 2.0, "speaker": "SPEAKER_0", "speaker_id": 0, "is_user": False}
        words = [{"word": w, "punctuated_word": w, "start": 0.0, "end": 0.1, "speaker": 0} for w in text_val.split()]
        reply = json.dumps({
            "type": "Results",
            "is_final": True,
            "speech_final": True,
            "text": text_val,
            "segments": [seg],
            "channel": {"alternatives": [{"transcript": text_val, "words": words}]},
        }, ensure_ascii=False)
        for ws_client, q in list(state.stream_clients.items()):
            try:
                q.put_nowait(reply)
            except asyncio.QueueFull:
                pass
    timestamp, text = event["timestamp"], event["text"]
    errors_before = state.metrics["output_errors"]
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


def record_transcript_line(text, **timing):
    text = re.sub(r"(?<![A-Z])([.?!])([A-Z])", r"\1 \2", " ".join(text.split()).strip())
    if not text:
        return False
    event = make_event("transcript", text, **timing)
    if not write_event(event):
        return False
    return True


def buffered_audio_bytes():
    return state.live_stt.pending_bytes if state.live_stt is not None else 0


def ingest_raw_pcm(raw, sr=SAMPLE_RATE):
    if state.stopping or state.live_stt is None:
        raise web.HTTPServiceUnavailable(text="Live transcription is not accepting audio")
    if sr != SAMPLE_RATE or len(raw) % 2:
        raise web.HTTPBadRequest(text="Expected 16000 Hz mono PCM16 with complete two-byte samples")
    if not raw:
        return
    now, received_at = time.monotonic(), time.time()
    state.live_stt.feed(raw, received_at, now)
    state.metrics["last_packet_at"] = now


def build_health_dict():
    now = time.monotonic()
    stream = state.live_stt
    workers_alive = bool(state.background_tasks) and all(not task.done() for task in state.background_tasks.values())
    connected = bool(stream is not None and stream.connected)
    overloaded = bool(stream is not None and (stream.pending_bytes >= BYTES_PER_SECOND * 10 or len(stream.queue) >= 256))
    pending_age = oldest_pending_audio_age()
    stalled = pending_age is not None and pending_age > MAX_AUDIO_AGE_SEC
    ready = (GEMINI_API_KEY and workers_alive and connected and not state.stopping and not overloaded
             and not stalled and not state.metrics["last_error"] and not state.metrics["last_output_error"])
    return {
        "status": "ok" if ready else "degraded",
        "is_setup_completed": True, "message": "", "pid": os.getpid(),
        "gemini_key_configured": bool(GEMINI_API_KEY), "stt_mode": "live", "model": GEMINI_LIVE_MODEL,
        "stream_connected": connected,
        "stream_reconnects": stream.reconnects if stream is not None else 0,
        "partial_updates": stream.partial_updates if stream is not None else 0,
        "stt_in_flight": current_stt_in_flight(), "max_audio_age_sec": MAX_AUDIO_AGE_SEC,
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
        "queue_depth": len(stream.queue) if stream is not None else 0, "queue_capacity_audio_sec": 10,
        "overloaded": overloaded, "buffer_bytes": buffered_audio_bytes(),
        "direct_ws_connected": bool(state.stream_clients),
        **{key: state.metrics[key] for key in ("utterances_processed", "audio_gaps", "queue_evictions", "last_error",
                                        "transcripts_emitted", "audio_source", "direct_ws_remote",
                                        "direct_ws_packets")},
    }


def _extract_stream_token(request):
    auth = request.headers.get("Authorization", "").strip()
    if auth:
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() in ("bearer", "token", "key"):
            return parts[1].strip()
        return auth
    return (request.headers.get("X-Omi-Token") or request.query.get("token") or request.query.get("key") or "").strip()


def _verify_stream_auth(request):
    if not STREAM_TOKEN:
        return
    candidate = _extract_stream_token(request)
    if not candidate or not hmac.compare_digest(candidate, STREAM_TOKEN):
        raise web.HTTPUnauthorized(text="Valid stream token required")


def _client_remote(request):
    return (request.headers.get("CF-Connecting-IP")
            or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.remote
            or "unknown")


async def handle_health(request):
    health = build_health_dict()
    return web.json_response(health, status=200 if health["status"] == "ok" else 503)


async def handle_omi_ws(request):
    """Direct phone-to-bridge WebSocket PCM16 streaming adapter (Omi Custom Live STT compatible)."""
    if state.stopping:
        raise web.HTTPServiceUnavailable(text="Bridge is draining for shutdown")
    _verify_stream_auth(request)
    remote_ip = _client_remote(request)
    if state.stream_clients:
        if remote_ip != state.metrics.get("direct_ws_remote"):
            raise web.HTTPConflict(text="Another direct audio stream is already active")
        for old_ws in list(state.stream_clients):
            state.stream_clients.pop(old_ws, None)
            asyncio.create_task(old_ws.close())
    encoding = request.query.get("encoding", "linear16").strip().lower()
    if encoding not in ("linear16", "pcm16", "pcm", ""):
        raise web.HTTPBadRequest(text="encoding must be linear16 or pcm16")
    try:
        sr = int(request.query.get("sample_rate", "16000"))
        channels = int(request.query.get("channels", "1"))
    except ValueError:
        raise web.HTTPBadRequest(text="sample_rate=16000 and channels=1 required")
    if sr != SAMPLE_RATE or channels != 1:
        raise web.HTTPBadRequest(text="sample_rate=16000 and channels=1 required")
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=MAX_INPUT_BYTES + 1)
    await ws.prepare(request)
    queue = asyncio.Queue(maxsize=32)
    state.stream_clients[ws] = queue
    state.metrics["direct_ws_remote"] = remote_ip
    state.metrics["audio_source"] = "direct_ws"
    logger.info("Direct phone WebSocket connected from %s", remote_ip)

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
                state.metrics["direct_ws_packets"] += 1
                state.metrics["audio_source"] = "direct_ws"
                ingest_raw_pcm(raw, sr)
            elif message.type == aiohttp.WSMsgType.TEXT:
                try:
                    ctrl = json.loads(message.data)
                except ValueError:
                    continue
                if isinstance(ctrl, dict) and ctrl.get("type") == "CloseStream":
                    if state.live_stt is not None:
                        await state.live_stt.flush_now()
                    break
            elif message.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                break
    finally:
        state.stream_clients.pop(ws, None)
        logger.info("Direct phone WebSocket disconnected from %s", remote_ip)
        try:
            queue.put_nowait(None)
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.QueueFull, asyncio.TimeoutError):
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await ws.close()
    return ws


async def handle_listen(request):
    """Stream caption JSON, replay boundaries, and pipeline status to viewers."""
    if request.query.get("mode", "captions") != "captions":
        raise web.HTTPBadRequest(text="Only captions are supported")
    try:
        count = int(request.query.get("replay", "100"))
    except ValueError:
        raise web.HTTPBadRequest(text="replay must be an integer from 0 to 200")
    if not 0 <= count <= 200:
        raise web.HTTPBadRequest(text="replay must be an integer from 0 to 200")
    after = request.query.get("after")
    if after is not None and len(after) > 128:
        raise web.HTTPBadRequest(text="Invalid replay cursor")
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    listener = CaptionListener()
    # Snapshot and subscribe without an await: no hole between replay and live delivery.
    history = list(state.recent_events)
    replay, reset = [], False
    if after:
        index = next((i for i, event in enumerate(history) if event.get("event_id") == after), None)
        reset = index is None
        replay = history[index + 1:] if index is not None else (history[-count:] if count else [])
    elif count:
        replay = history[-count:]
    listener.partial = dict(state.partial_caption) if state.partial_caption else None
    state.caption_listeners[ws] = listener

    async def send_json(event):
        await asyncio.wait_for(ws.send_json(event), timeout=5)

    async def sender():
        try:
            await send_json({"type": "replay", "phase": "start", "reset": reset, "count": len(replay)})
            for event in replay:
                await send_json(event)
            await send_json({"type": "replay", "phase": "end"})
            listener.status = caption_status()
            listener.ready.set()
            while True:
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
        except (OSError, asyncio.TimeoutError):
            pass
        finally:
            await ws.close()

    task = asyncio.create_task(sender())
    try:
        async for _ in ws:
            pass
    finally:
        state.caption_listeners.pop(ws, None)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return ws


async def on_startup(app):
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key is required")
    seed_recent_events()
    state.daily_notes = DailyNotes(LOG_DIR)
    state.http_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=12, connect=5, sock_read=10),
        connector=aiohttp.TCPConnector(limit=10, limit_per_host=4), cookie_jar=aiohttp.DummyCookieJar())
    state.stopping = False
    state.metrics["started_at"] = time.monotonic()
    state.live_stt = GeminiLiveSTT(session=state.http_session, api_key=GEMINI_API_KEY,
        model=GEMINI_LIVE_MODEL, on_partial=publish_partial, on_final=record_live_final,
        on_gap=record_live_gap, on_latency=record_latency, on_connection=live_connection_changed,
        max_age=MAX_AUDIO_AGE_SEC, idle_flush=IDLE_FLUSH_SEC, rms_threshold=RMS_SILENCE_THRESHOLD)
    for name, factory in (("caption_status_task", caption_status_worker), ("live_stt_task", state.live_stt.run)):
        app[name] = asyncio.create_task(factory())
        state.background_tasks[name] = app[name]


async def on_shutdown(app):
    state.stopping = True
    await asyncio.gather(*(ws.close() for ws in list(state.caption_listeners) + list(state.stream_clients)), return_exceptions=True)


async def on_cleanup(app):
    state.stopping = True
    logger.info("Shutdown: draining accepted live audio")
    status_task = app["caption_status_task"]
    status_task.cancel()
    await asyncio.gather(status_task, return_exceptions=True)
    try:
        await state.live_stt.close()
        await app["live_stt_task"]
        flush_daily_paragraph()
        logger.info("Shutdown drain complete")
    finally:
        if state.http_session is not None:
            await state.http_session.close()
            # aiohttp TLS transports finish closing after ClientSession.close().
            await asyncio.sleep(.25)
            state.http_session = None
        state.background_tasks.clear()


def create_app(start_workers=True):
    app = web.Application(client_max_size=4096)
    app.router.add_get("/v1/listen", handle_omi_ws)
    app.router.add_get("/listen", handle_listen)
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
