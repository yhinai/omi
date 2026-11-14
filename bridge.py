#!/usr/bin/env python3.15
"""Single-device PCM16 -> live transcription and terminal captions."""
import asyncio
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
import hmac
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import platform
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

from config import load_env, VEXO_BAND
from speakers import turns, VoiceClient, MODEL as SPEAKER_MODEL, PROFILE as SPEAKER_PROFILE
from backend_settings import DeepgramConfig, TranscriptionBackend
from diagnostics import AudioCapture
from storage import TranscriptStore, DailyNotes
from common import RATE, BPS, QUEUE_AUDIO_SECONDS, QUEUE_MAX_BYTES, QUEUE_MAX_PACKETS, AudioQualityMeter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
SAMPLE_RATE, BYTES_PER_SECOND = RATE, BPS
MAX_AUDIO_AGE_SEC = 45.0
STT_BACKEND = "whisper"
MODEL_LABEL = "whisper-large-v3-turbo-fp16"
DEEPGRAM_CONFIG = None
WHISPER_PYTHON = WHISPER_MODEL = WHISPER_VAD = ""
WHISPER_ENV = {}
IDLE_FLUSH_SEC = 8.0
MAX_INPUT_BYTES = BYTES_PER_SECOND * 300
HOST = "0.0.0.0"
PORT, HEALTH_PORT = 8787, 8788
STREAM_TOKEN = ""
CAPTION_TOKEN = ""
MAX_CAPTION_CLIENTS = 16
logger = logging.getLogger("bridge")
event_logger = logging.getLogger("omi_transcript_events")


def configure(env=None):
    global MAX_AUDIO_AGE_SEC
    global IDLE_FLUSH_SEC, HOST, STREAM_TOKEN, CAPTION_TOKEN, MAX_CAPTION_CLIENTS
    global WHISPER_PYTHON, WHISPER_MODEL, WHISPER_VAD, WHISPER_ENV
    global STT_BACKEND, MODEL_LABEL, DEEPGRAM_CONFIG
    env = load_env() if env is None else env
    backend = env.get("OMI_STT_BACKEND", "whisper").strip().lower()
    if backend not in {"whisper", "deepgram"}:
        raise ValueError("OMI_STT_BACKEND must be whisper or deepgram")
    cloud_config = DeepgramConfig.from_env(env) if backend == "deepgram" else None
    if cloud_config and cloud_config.pipeline == "pipecat" and (
            not os.path.isfile(cloud_config.python) or not os.access(cloud_config.python, os.X_OK)):
        raise ValueError("Install the optional Pipecat environment; see README.md")
    HOST = (env.get("OMI_HOST") or "0.0.0.0").strip()
    STREAM_TOKEN = (env.get("OMI_STREAM_TOKEN") or "").strip()
    CAPTION_TOKEN = (env.get("OMI_CAPTION_TOKEN", STREAM_TOKEN) or "").strip()
    WHISPER_ENV = {key: env[key] for key in ("OMI_LANGUAGE", "OMI_LANGUAGES", "OMI_HOTWORDS") if key in env}
    language = env.get("OMI_LANGUAGE", "en").strip().lower()
    languages = {item.strip().lower() for item in env.get("OMI_LANGUAGES", "en").split(",") if item.strip()}
    if backend == "whisper" and (not languages or (language not in ("", "auto") and language not in languages)):
        raise ValueError("Recognition language must be included in the nonempty OMI_LANGUAGES allowlist")
    MAX_CAPTION_CLIENTS = int(env.get("OMI_MAX_CAPTION_CLIENTS", "16"))
    if not 1 <= MAX_CAPTION_CLIENTS <= 1024:
        raise ValueError("Caption client limit must be between 1 and 1024")
    assets = os.path.expanduser("~/Library/Application Support/OmiWhisper")
    WHISPER_PYTHON = os.path.expanduser(env.get("OMI_WHISPER_PYTHON", os.path.join(BASE_DIR, ".venv-whisper/bin/python")))
    WHISPER_MODEL = os.path.expanduser(env.get("OMI_WHISPER_MODEL", os.path.join(assets, "whisper-turbo")))
    WHISPER_VAD = os.path.expanduser(env.get("OMI_WHISPER_VAD", os.path.join(assets, "silero_vad.onnx")))
    if backend == "whisper" and (not os.path.isfile(WHISPER_PYTHON) or not os.access(WHISPER_PYTHON, os.X_OK)):
        raise ValueError("Install the Python 3.14 Whisper worker environment; see README.md")
    if backend == "whisper" and not all(os.path.isfile(path) for path in (os.path.join(WHISPER_MODEL, "config.json"),
            os.path.join(WHISPER_MODEL, "weights.safetensors"), WHISPER_VAD)):
        raise ValueError("Whisper or Silero model files are missing; see README.md")
    MAX_AUDIO_AGE_SEC = float(env.get("OMI_MAX_AUDIO_AGE_SEC", "45"))
    IDLE_FLUSH_SEC = float(env.get("OMI_IDLE_FLUSH_SEC", "8"))
    if not all(math.isfinite(v) and v > 0 for v in (MAX_AUDIO_AGE_SEC, IDLE_FLUSH_SEC)):
        raise ValueError("Maximum audio age and idle timeout must be finite and positive")
    STT_BACKEND, DEEPGRAM_CONFIG = backend, cloud_config
    MODEL_LABEL = cloud_config.model if cloud_config else "whisper-large-v3-turbo-fp16"


def create_transcription_backend(callbacks):
    """Keep phone, persistence and viewer contracts independent of recognition."""
    callbacks = dict(callbacks)
    callbacks.setdefault("max_age", MAX_AUDIO_AGE_SEC)
    callbacks.setdefault("idle_flush", IDLE_FLUSH_SEC)
    if STT_BACKEND == "deepgram":
        if DEEPGRAM_CONFIG is None:
            raise ValueError("Configure Deepgram before creating its backend")
        config = DEEPGRAM_CONFIG
        if state.voice_client is not None:
            callbacks["enrich_speaker"] = state.voice_client.classify
        if config.pipeline == "pipecat":
            from pipecat_stt import PipecatDeepgramSTT
            factory = PipecatDeepgramSTT
            callbacks["python"] = config.python
        else:
            from deepgram_stt import DeepgramSTT
            factory = DeepgramSTT
        return factory(api_key=config.api_key, model=config.model, language=config.language,
                           endpointing_ms=config.endpointing_ms, diarize=config.diarize, keyterms=config.keyterms, **callbacks)
    from stt import LocalWhisperSTT
    return LocalWhisperSTT(python=WHISPER_PYTHON, model=WHISPER_MODEL,
                           vad=WHISPER_VAD, worker_env=WHISPER_ENV, **callbacks)


def configure_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    handler = RotatingFileHandler(os.path.join(LOG_DIR, "omi_bridge.log"),
                                 maxBytes=5 * 1024 * 1024, backupCount=3)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    state.storage.configure_logging(event_logger)


def new_metrics():
    return {
        "started_at": time.monotonic(), "last_packet_at": None, "last_stt_success_at": None,
        "last_error": None, "utterances_processed": 0, "audio_gaps": 0, "queue_evictions": 0,
        "last_transcript_at": None, "transcripts_emitted": 0,
        "lost_audio_sec_failed": 0.0, "lost_audio_sec_stale": 0.0,
        "lost_audio_sec_queue_evicted": 0.0,
        "output_errors": 0, "last_output_error": None,
        "source_identity": None, "source_loss": {}, "audio_source": None, "direct_ws_remote": None, "direct_ws_packets": 0,
        "phone_delivery_errors": 0,
        "session_started_at": None, "session_ended_at": None, "stream_audio_bytes": 0,
    }


@dataclass
class PhoneStream:
    queue: asyncio.Queue
    start_sample: int
    delivery_failed: bool = False


@dataclass
class BridgeState:
    audio_quality: AudioQualityMeter = field(default_factory=AudioQualityMeter)
    audio_capture: AudioCapture = field(default_factory=AudioCapture)
    capture_signal_ready: bool = False
    recognized_utterances: deque = field(default_factory=lambda: deque(maxlen=256))
    storage: TranscriptStore = field(default_factory=lambda: TranscriptStore(
        directory=lambda: LOG_DIR, append=lambda encoded: event_logger.info(encoded),
        create_event=lambda *args, **kwargs: make_event(*args, **kwargs),
        on_commit=lambda event: publish_saved(event),
        on_error=lambda target, error: output_failed(target, error),
        on_recovered=lambda: state.metrics.update(last_output_error=None)))
    voice_client: object = None
    source_sessions: dict = field(default_factory=dict)
    caption_listeners: dict = field(default_factory=dict)
    caption_admission_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stream_clients: dict = field(default_factory=dict)
    stream_admission_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    recent_events: deque = field(default_factory=lambda: deque(maxlen=200))
    stream_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    background_tasks: dict = field(default_factory=dict)
    stopping: bool = False
    metrics: dict = field(default_factory=new_metrics)
    live_stt: TranscriptionBackend | None = None
    partial_caption: dict | None = None
    latency_samples: dict = field(default_factory=lambda: {
        name: deque(maxlen=128) for name in (
            "receipt_to_transcript_sec",
            "receipt_to_first_partial_sec",
            "speech_start_to_first_partial_sec",
            "receipt_to_final_sec",
            "speech_end_to_final_sec",
            "receipt_to_send_sec",
            "turn_close_to_final_sec",
            "turn_boundary_wait_sec",
            "local_decode_sec",
            "provider_word_end_to_final_sec",
            "provider_word_end_to_partial_sec",
            "final_word_packet_to_result_sec",
        )})


state = BridgeState()


def record_latency(name, seconds):
    state.latency_samples.setdefault(name, deque(maxlen=128)).append(max(0.0, seconds))


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


def stream_audio_duty():
    """Audio seconds received against the wall clock of the current or last session.

    A duty well below 1.0 means the phone forwards in bursts, which is not loss:
    the lost_audio_sec_* counters account for dropped audio separately. Wall time
    freezes at disconnect so a stale session cannot dilute the ratio.
    """
    started = state.metrics["session_started_at"]
    ended = state.metrics["session_ended_at"]
    audio = state.metrics["stream_audio_bytes"] / BYTES_PER_SECOND
    if started is None:
        return audio, None, None
    wall = max(0.0, (ended if ended is not None else time.monotonic()) - started)
    return audio, wall, (round(audio / wall, 3) if wall > 0 else None)


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
    identity = timing.get("utterance_id")
    if identity is None or identity not in state.recognized_utterances:
        state.metrics["last_stt_success_at"] = time.monotonic()
        state.metrics["utterances_processed"] += 1
        if identity is not None:
            state.recognized_utterances.append(identity)
    saved = record_transcript_line(text, **timing)
    if saved is not False:
        state.metrics["last_error"] = None
        state.metrics["last_transcript_at"] = time.monotonic()
        state.metrics["transcripts_emitted"] += 1
    return saved


def record_live_gap(reason, seconds, **timing):
    state.metrics["audio_gaps"] += 1
    if reason == "queue_evicted":
        state.metrics["queue_evictions"] += 1
    record_gap(reason, seconds, **timing)


def live_connection_changed(connected, error):
    state.metrics["last_error"] = None if connected else STT_BACKEND + ": " + str(error)
    logger.info("%s %s", STT_BACKEND, "ready" if connected else "reconnecting (" + str(error) + ")")


def current_stt_in_flight():
    return bool(state.live_stt is not None and state.live_stt.speech_pending)


def workers_alive():
    return bool(state.background_tasks) and all(not task.done() for task in state.background_tasks.values())


def caption_status():
    now = time.monotonic()
    packet = state.metrics["last_packet_at"]
    age = max(0, now - packet) if packet is not None else None
    stream = state.live_stt
    _, _, duty = stream_audio_duty()
    return {"type": "status", "stream_id": state.stream_id,
            "session_id": stream.session_id if stream is not None else None,
            "stt_mode": "live", "stt_backend": STT_BACKEND,
            "stream_connected": bool(stream is not None and stream.connected),
            "backend_ready": bool(stream is not None and getattr(stream, "ready", stream.connected)),
            "provider_connection_state": getattr(stream, "connection_state", "connected" if stream is not None and stream.connected else "disconnected"),
            "provider_resolved_model": getattr(stream, "resolved_model", None),
            "stt_pipeline": getattr(stream, "pipeline", "direct" if STT_BACKEND == "deepgram" else "local"),
            "audio_active": age is not None and age < 3,
            "last_packet_age_sec": round(age, 1) if age is not None else None,
            "buffer_sec": round(buffered_audio_bytes() / BYTES_PER_SECOND, 1),
            "stt_in_flight": current_stt_in_flight(),
            "queue_depth": len(stream.queue) if stream is not None else 0,
            "model": MODEL_LABEL,
            "latency_final_p50_sec": latency_summary(state.latency_samples.get("receipt_to_final_sec"))["p50"],
            "latency_speech_end_to_final_p50_sec": latency_summary(state.latency_samples.get("speech_end_to_final_sec"))["p50"],
            "latency_speech_start_to_first_partial_p50_sec": latency_summary(state.latency_samples.get("speech_start_to_first_partial_sec"))["p50"],
            "latency_turn_close_to_final_p50_sec": latency_summary(state.latency_samples.get("turn_close_to_final_sec"))["p50"],
            "latency_provider_connect_p50_sec": latency_summary(state.latency_samples.get("provider_connect_sec"))["p50"],
            "latency_final_to_persist_p50_sec": latency_summary(state.latency_samples.get("final_to_persist_sec"))["p50"],
            "latency_provider_word_end_to_partial_p50_sec": latency_summary(state.latency_samples.get("provider_word_end_to_partial_sec"))["p50"],
            "latency_provider_word_end_to_final_p50_sec": latency_summary(state.latency_samples.get("provider_word_end_to_final_sec"))["p50"],
            "latency_final_word_packet_to_result_p50_sec": latency_summary(state.latency_samples.get("final_word_packet_to_result_sec"))["p50"],
            "audio_duty": duty,
            "workers_alive": workers_alive(),
            "last_error": state.metrics["last_error"],
            "save_error": getattr(stream, "save_error", None),
            "transcripts_emitted": state.metrics["transcripts_emitted"]}


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
    state.recent_events = state.storage.recent_events()


def flush_daily_paragraph():
    state.storage.flush()


def output_failed(target, error):
    state.metrics["output_errors"] += 1
    state.metrics["last_output_error"] = f"{target}: {type(error).__name__}"
    logger.error("Transcript output failed: %s", state.metrics["last_output_error"])


def write_event(event):
    return state.storage.write(event)


def publish_saved(event):
    state.recent_events.append(dict(event))
    publish_caption(event)
    if event.get("type") == "transcript" and event.get("text"):
        text_val = event["text"]
        start, end = event.get("sample_offset"), event.get("sample_end_offset")
        for ws, client in list(state.stream_clients.items()):
            if client.delivery_failed:
                continue
            # A late final from the previous phone connection belongs to its
            # old timeline, not the replacement connection's recording.
            if start is not None and start < client.start_sample:
                continue
            seg = {"text": text_val, "speaker": "SPEAKER_0", "speaker_id": 0, "is_user": False}
            if start is not None and end is not None:
                seg.update(start=(start - client.start_sample) / SAMPLE_RATE,
                           end=(max(start, end) - client.start_sample) / SAMPLE_RATE)
            reply = json.dumps({"type": "Results", "is_final": True, "speech_final": True,
                                "text": text_val, "segments": [seg]}, ensure_ascii=False)
            try:
                client.queue.put_nowait(reply)
            except asyncio.QueueFull:
                fail_phone_delivery(ws, client, "queue_full")


def fail_phone_delivery(ws, client, reason):
    """Fail a one-way phone connection explicitly; journal/viewer delivery continues."""
    if client.delivery_failed:
        return
    client.delivery_failed = True
    state.metrics["phone_delivery_errors"] += 1
    logger.warning("Phone caption delivery failed (%s); closing for reconnect", reason)

    async def close_failed_phone():
        try:
            await asyncio.wait_for(ws.close(code=1013, message=b"Caption delivery stalled; reconnect", drain=False), 2)
        except (OSError, asyncio.TimeoutError):
            pass

    # Closing from a separate task lets the receive handler unwind without
    # waiting on the sender it is about to cancel/drain.
    asyncio.create_task(close_failed_phone())


def make_event(kind, text, audio_start_at=None, received_at=None, processed_at=None,
               session_id=None, sample_offset=None, utterance_id=None, sample_end_offset=None,
               speech_start_offset=None, voice_end_offset=None, recognition=None):
    processed_at = time.time() if processed_at is None else processed_at
    source_at = audio_start_at if audio_start_at is not None else processed_at
    event = {"event_id": uuid.uuid4().hex, "stream_id": state.stream_id, "type": kind,
            "timestamp": datetime.fromtimestamp(source_at).strftime("%Y-%m-%d %H:%M:%S"),
            "text": text, "audio_start_at": audio_start_at, "received_at": received_at,
            "processed_at": processed_at, "session_id": session_id, "sample_offset": sample_offset,
            "time_basis": "estimated_from_receipt" if audio_start_at is not None else "processing_time",
            "stt_backend": STT_BACKEND, "model": MODEL_LABEL}
    source = state.source_sessions.get(session_id)
    if source is not None:
        event["source"] = dict(source)
    if recognition is not None:
        event["recognition"] = recognition
        words = recognition.get("words", [])
        if any("speaker" in word for word in words):
            event["speaker_turns"] = turns(words, recognition.get("connection_epoch", ""))
    if utterance_id is not None:
        event.update(utterance_id=utterance_id, is_final=True, transcription_mode="live")
    if sample_end_offset is not None:
        event["sample_end_offset"] = sample_end_offset
    if speech_start_offset is not None:
        event["speech_start_offset"] = speech_start_offset
    if voice_end_offset is not None:
        event["voice_end_offset"] = voice_end_offset
    return event


def record_gap(reason, seconds, *, session_id=None, sample_offset=None, audio_start_at=None,
               received_at=None, detail=None):
    if seconds <= 0:
        return
    state.metrics[f"lost_audio_sec_{reason}"] += seconds
    event = make_event("gap", f"[GAP: {seconds:.3f}s audio untranscribed; {reason}]",
                       audio_start_at=audio_start_at, received_at=received_at,
                       session_id=session_id, sample_offset=sample_offset)
    event.update(reason=reason, duration_sec=seconds)
    if detail:
        event["detail"] = detail
    write_event(event)


def record_transcript_line(text, **timing):
    return state.storage.final(text, **timing)


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
    start_sample = state.live_stt.total_input_samples
    state.live_stt.feed(raw, received_at, now)
    state.audio_capture.feed(raw, sample_offset=start_sample,
                             session_id=state.live_stt.session_id, now=now)
    state.audio_quality.add(raw, now)
    state.metrics["last_packet_at"] = now


def build_health_dict():
    now = time.monotonic()
    stream = state.live_stt
    workers = workers_alive()
    connected = bool(stream is not None and stream.connected)
    backend_ready = bool(stream is not None and getattr(stream, "ready", connected))
    overloaded = bool(stream is not None and (stream.pending_bytes >= QUEUE_MAX_BYTES or len(stream.queue) >= QUEUE_MAX_PACKETS))
    pending_age = oldest_pending_audio_age()
    stalled = pending_age is not None and pending_age > MAX_AUDIO_AGE_SEC
    stream_audio_sec, stream_wall_sec, audio_duty = stream_audio_duty()
    ready = (workers and backend_ready and not state.stopping and not overloaded
             and not stalled and not getattr(stream, "save_error", None)
             and not state.metrics["last_error"] and not state.metrics["last_output_error"]
             and not any(client.delivery_failed for client in state.stream_clients.values()))
    return {
        "status": "ok" if ready else "degraded",
        "is_setup_completed": True, "message": "", "pid": os.getpid(),
        "stt_mode": "live", "stt_backend": STT_BACKEND,
        "model": MODEL_LABEL,
        "worker_load_warm_sec": getattr(stream, "load_warm_sec", None),
        "stream_connected": connected,
        "backend_ready": backend_ready,
        "cloud_audio_processing": STT_BACKEND == "deepgram",
        "voice_identification": ("enrolled" if os.path.isfile(state.voice_client.profile) else "not_enrolled") if state.voice_client else "disabled",
        "provider_connection_state": getattr(stream, "connection_state", "connected" if connected else "disconnected"),
        "provider_resolved_model": getattr(stream, "resolved_model", None),
            "stt_pipeline": getattr(stream, "pipeline", "direct" if STT_BACKEND == "deepgram" else "local"),
        "stream_reconnects": stream.reconnects if stream is not None else 0,
        "partial_updates": stream.partial_updates if stream is not None else 0,
        "pending_finals": stream.pending_final_count if stream is not None else 0,
        "pending_saves": len(getattr(stream, "pending_finals", ())),
        "save_error": getattr(stream, "save_error", None),
        "last_flush_completed": getattr(stream, "last_flush_completed", None),
        "turns_without_text": stream.turns_without_text if stream is not None else 0,
        "no_text_causes": dict(stream.no_text_causes) if stream is not None else {},
        "gap_causes": dict(stream.gap_causes) if stream is not None else {},
        "stt_in_flight": current_stt_in_flight(), "max_audio_age_sec": MAX_AUDIO_AGE_SEC,
        "latency": {name: latency_summary(samples) for name, samples in state.latency_samples.items()},
        "oldest_pending_audio_age_sec": pending_age,
        **{k: round(v, 6) if isinstance(v, float) else v for k, v in state.metrics.items()
           if k.startswith("lost_audio_sec_") or k in ("output_errors", "last_output_error")},
        "python_version": platform.python_version(), "openssl_version": ssl.OPENSSL_VERSION,
        "workers_alive": workers, "stopping": state.stopping,
        "uptime_sec": round(now - state.metrics["started_at"], 1),
        "last_packet_age_sec": round(now - state.metrics["last_packet_at"], 1) if state.metrics["last_packet_at"] is not None else None,
        "last_stt_success_age_sec": round(now - state.metrics["last_stt_success_at"], 1) if state.metrics["last_stt_success_at"] is not None else None,
        "last_transcript_age_sec": round(now - state.metrics["last_transcript_at"], 1) if state.metrics["last_transcript_at"] is not None else None,
        "queue_depth": len(stream.queue) if stream is not None else 0, "queue_capacity_audio_sec": QUEUE_AUDIO_SECONDS,
        "overloaded": overloaded, "buffer_bytes": buffered_audio_bytes(),
        "stream_audio_sec": round(stream_audio_sec, 1),
        "stream_wall_sec": round(stream_wall_sec, 1) if stream_wall_sec is not None else None,
        "audio_duty": audio_duty,
        "audio_quality": state.audio_quality.snapshot(now),
        "audio_capture": {**state.audio_capture.snapshot(), "supported": state.capture_signal_ready},
        "direct_ws_connected": bool(state.stream_clients),
        "phone_delivery_errors": state.metrics["phone_delivery_errors"],
        "caption_clients": len(state.caption_listeners), "caption_client_limit": MAX_CAPTION_CLIENTS,
        **{key: state.metrics[key] for key in ("utterances_processed", "audio_gaps", "queue_evictions", "last_error",
                                        "transcripts_emitted", "audio_source", "source_identity", "source_loss", "direct_ws_remote",
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


def _verify_stream_auth(request, token=None):
    token = STREAM_TOKEN if token is None else token
    if not token:
        return
    candidate = _extract_stream_token(request)
    if not candidate or not hmac.compare_digest(candidate.encode("utf-8"), token.encode("utf-8")):
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
    source_kind = request.headers.get("X-Audio-Source", "omi")
    if source_kind not in {"omi", "vexo"}:
        raise web.HTTPBadRequest(text="Unknown audio source")
    source = {"kind": source_kind}
    if source_kind == "vexo":
        if request.headers.get("X-Vexo-Band") != str(VEXO_BAND):
            raise web.HTTPBadRequest(text="Expected Vexo band 43")
        transport = request.headers.get("X-Vexo-Transport", "serial")
        if transport not in {"serial", "ble"}:
            raise web.HTTPBadRequest(text="Expected serial or ble Vexo transport")
        source.update(band=VEXO_BAND, name=f"Vexo-HACK-{VEXO_BAND:02}", transport=transport)
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=MAX_INPUT_BYTES + 1)
    queue = asyncio.Queue(maxsize=32)
    # Serialize admission across the handshake so concurrent attempts cannot
    # bypass the single-stream check. Failed upgrades leave the old stream alone.
    async with state.stream_admission_lock:
        if state.stopping:
            raise web.HTTPServiceUnavailable(text="Bridge is draining for shutdown")
        if state.stream_clients and (source_kind == "vexo" or state.metrics.get("audio_source") == "vexo" or remote_ip != state.metrics.get("direct_ws_remote")):
            raise web.HTTPConflict(text="Another direct audio stream is already active")
        await ws.prepare(request)
        if state.stopping:
            await ws.close(code=1012, message=b"Bridge is shutting down")
            return ws
        for old_ws in list(state.stream_clients):
            state.stream_clients.pop(old_ws, None)
            asyncio.create_task(old_ws.close())
        start_sample = state.live_stt.total_input_samples if state.live_stt is not None else 0
        if state.live_stt is not None:
            state.live_stt.new_session()
            state.source_sessions[state.live_stt.session_id] = dict(source)
        state.metrics["source_identity"] = dict(source)
        state.metrics["session_started_at"] = time.monotonic()
        state.metrics["session_ended_at"] = None
        state.metrics["stream_audio_bytes"] = 0
        state.audio_quality = AudioQualityMeter()
        client = PhoneStream(queue, start_sample)
        state.stream_clients[ws] = client
        state.metrics["direct_ws_remote"] = remote_ip
        state.metrics["audio_source"] = "vexo" if source_kind == "vexo" else "direct_ws"
    logger.info("Direct phone WebSocket connected from %s", remote_ip)

    async def sender():
        try:
            while True:
                msg = await queue.get()
                if msg is None:
                    return
                await asyncio.wait_for(ws.send_str(msg), timeout=5)
        except (OSError, asyncio.TimeoutError):
            fail_phone_delivery(ws, client, "send_failed")

    task = asyncio.create_task(sender())
    source_closed_cleanly = False
    try:
        async for message in ws:
            # A replaced socket may still have buffered frames while close is
            # pending. It no longer owns the new session's audio or controls.
            if ws not in state.stream_clients or state.stopping or client.delivery_failed:
                break
            if message.type == aiohttp.WSMsgType.BINARY:
                raw = message.data
                if raw.startswith(b"RIFF") or len(raw) % 2:
                    await ws.close(code=1003, message=b"Expected raw 16000 Hz mono PCM16")
                    break
                state.metrics["direct_ws_packets"] += 1
                state.metrics["stream_audio_bytes"] += len(raw)
                state.metrics["audio_source"] = "vexo" if source_kind == "vexo" else "direct_ws"
                ingest_raw_pcm(raw, sr)
                if state.live_stt is not None:
                    state.source_sessions[state.live_stt.session_id] = dict(source)
            elif message.type == aiohttp.WSMsgType.TEXT:
                try:
                    ctrl = json.loads(message.data)
                except ValueError:
                    continue
                if source_kind == "vexo" and isinstance(ctrl, dict) and ctrl.get("type") == "SourceStatus":
                    counters = ctrl.get("counters", {})
                    if not isinstance(counters, dict):
                        continue
                    state.metrics["source_loss"] = {k: v for k, v in counters.items()
                        if k in {"lost_packets", "old_packets", "queue_overflows", "discarded_packets", "corrupt_packets"}
                        and isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 2**53}
                    if ctrl.get("discontinuity"):
                        # Unknown duration is explicit; never invent received samples.
                        write_event(make_event("source-discontinuity", "[Vexo source session interrupted]",
                            session_id=state.live_stt.session_id if state.live_stt else None))
                if isinstance(ctrl, dict) and ctrl.get("type") == "CloseStream":
                    source_closed_cleanly = True
                    if state.live_stt is not None:
                        completed = await state.live_stt.flush_now()
                        if completed is False:
                            logger.warning("Phone close requested before transcription drain completed")
                            await ws.close(code=1013, message=b"Transcription drain pending; audio retained")
                    break
            elif message.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                break
    finally:
        active = state.stream_clients.pop(ws, None) is not None
        if active and source_kind == "vexo" and not source_closed_cleanly and not state.stopping:
            write_event(make_event("source-discontinuity", "[Vexo input connection lost]",
                session_id=state.live_stt.session_id if state.live_stt else None))
        if active and state.metrics["session_started_at"] is not None:
            state.metrics["session_ended_at"] = time.monotonic()
            if state.live_stt is not None and not state.stopping:
                state.live_stt.request_flush()
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
    _verify_stream_auth(request, CAPTION_TOKEN)
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
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=4096)
    async with state.caption_admission_lock:
        if state.stopping or len(state.caption_listeners) >= MAX_CAPTION_CLIENTS:
            raise web.HTTPServiceUnavailable(text="Caption service is at capacity or shutting down")
        await ws.prepare(request)
        if state.stopping:
            await ws.close(code=1012, message=b"Bridge is shutting down")
            return ws
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
    source_closed_cleanly = False
    try:
        async for _ in ws:
            pass
    finally:
        state.caption_listeners.pop(ws, None)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return ws


async def on_startup(app):
    env=load_env()
    if env.get("OMI_VOICE_IDENTIFICATION", "false").lower()=="true" and STT_BACKEND=="deepgram":
        if not DEEPGRAM_CONFIG.diarize:
            raise ValueError("Voice identification currently needs Deepgram diarization")
        state.voice_client=VoiceClient(env.get("OMI_SPEAKER_PYTHON",os.path.join(BASE_DIR,".venv-speaker/bin/python")),BASE_DIR,model=env.get("OMI_SPEAKER_MODEL",str(SPEAKER_MODEL)),
            profile=env.get("OMI_SPEAKER_PROFILE",str(SPEAKER_PROFILE)))
    seed_recent_events()
    state.storage.notes = DailyNotes(LOG_DIR, idle_flush=IDLE_FLUSH_SEC)
    state.stopping = False
    state.metrics["started_at"] = time.monotonic()
    callbacks = dict(on_partial=publish_partial, on_final=record_live_final,
        on_gap=record_live_gap, on_latency=record_latency, on_connection=live_connection_changed,
        max_age=MAX_AUDIO_AGE_SEC, idle_flush=IDLE_FLUSH_SEC)
    state.live_stt = create_transcription_backend(callbacks)
    for name, factory in (("caption_status_task", caption_status_worker), ("live_stt_task", state.live_stt.run)):
        app[name] = asyncio.create_task(factory())
        state.background_tasks[name] = app[name]


async def on_shutdown(app):
    state.stopping = True
    state.audio_capture.close()
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
        if state.live_stt.last_flush_completed is False:
            logger.warning("Shutdown drain incomplete; inspect reported gaps and pending saves")
        else:
            logger.info("Shutdown drain complete")
    finally:
        if state.voice_client is not None:
            await state.voice_client.close()
            state.voice_client=None
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
    loop.add_signal_handler(signal.SIGUSR1, state.audio_capture.start)
    state.capture_signal_ready = True
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
        try:
            await health_runner.cleanup()
            await runner.cleanup()
        finally:
            state.capture_signal_ready = False
            loop.remove_signal_handler(signal.SIGUSR1)
            state.audio_capture.close()
            for signum in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(signum)


if __name__ == "__main__":
    try:
        configure()
    except (ValueError, OSError):
        sys.exit("Invalid bridge configuration; check model paths, language allowlist and numeric settings in .env")
    if len(sys.argv) > 1:
        if sys.argv[1:] != ["--check-config"]:
            sys.exit("Usage: bridge.py [--check-config]")
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
