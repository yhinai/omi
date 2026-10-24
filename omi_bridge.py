#!/usr/bin/env python3
import asyncio
import base64
from collections import deque
from datetime import datetime
import io
import json
from logging.handlers import RotatingFileHandler
import logging
import os
import struct
import time
import urllib.error
import urllib.request
import wave

from aiohttp import web

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
TRANSCRIPT_LOG = os.path.join(LOG_DIR, "omi_live_transcript.log")


def _load_dotenv():
    """Load local ~/omi/.env first, then ~/.hermes/.env without overriding existing env vars."""
    for env_path in (os.path.join(BASE_DIR, ".env"), os.path.expanduser("~/.hermes/.env")):
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v


_load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
GEMINI_STT_MODEL = os.environ.get("OMI_GEMINI_STT_MODEL", "gemini-3.8-flash")
PRE_OVERLAP_SEC = float(os.environ.get("OMI_PRE_OVERLAP_SEC", "1.0"))
CORE_WINDOW_SEC = float(os.environ.get("OMI_CORE_WINDOW_SEC", "5.0"))
POST_LOOKAHEAD_SEC = float(os.environ.get("OMI_POST_LOOKAHEAD_SEC", "1.0"))
IDLE_FLUSH_SEC = float(os.environ.get("OMI_IDLE_FLUSH_SEC", "8.0"))
RMS_SILENCE_THRESHOLD = float(os.environ.get("OMI_RMS_THRESHOLD", "80.0"))
RING_BUFFER_SECONDS = int(os.environ.get("OMI_RING_BUFFER_SEC", "300"))  # 5 minutes (9.6 MB at 16kHz PCM16)

HERMES_LIVE_WEBHOOK = os.environ.get("OMI_HERMES_LIVE_WEBHOOK", "http://127.0.0.1:8644/webhooks/omi-live")
HERMES_RESEARCH_WEBHOOK = os.environ.get("OMI_HERMES_RESEARCH_WEBHOOK", "http://127.0.0.1:8644/webhooks/omi-research")
WAKE_WORD = os.environ.get("OMI_WAKE_WORD", "hermes").strip().lower()
RESEARCH_INTERVAL_SEC = int(os.environ.get("OMI_RESEARCH_INTERVAL", "90"))

# Rotating bridge log (5 MB x 3 files)
logger = logging.getLogger("omi_bridge")
logger.setLevel(logging.INFO)
_handler = RotatingFileHandler(os.path.join(LOG_DIR, "omi_bridge.log"), maxBytes=5 * 1024 * 1024, backupCount=3)
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(_handler)
_stdout_handler = logging.StreamHandler()
_stdout_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(_stdout_handler)

# Audio & STT runtime state
audio_pcm_buffer = bytearray()
audio_pre_tail = bytearray()
audio_sample_rate = 16000

# 5-minute continuous raw audio ring buffer (captured BEFORE RMS silence gating):
# stores (timestamp, sample_rate, pcm_bytes) tuples, bounded to RING_BUFFER_SECONDS
audio_ring = deque()
audio_ring_bytes = 0

# Bounded STT queue consumed by a single sequential worker
stt_queue: asyncio.Queue = asyncio.Queue(maxsize=24)

# Paragraph compactor state for daily markdown notes (logs/YYYY-MM-DD.md)
paragraph_words = []
paragraph_start_ts = ""
last_transcript_time = 0.0

# Research buffer & live browser listeners
rolling_research_buffer = []
last_research_time = time.time()
live_listeners = set()

# Health & telemetry tracking
metrics = {
    "started_at": time.time(),
    "last_packet_at": None,
    "last_stt_success_at": None,
    "last_error": None,
    "windows_processed": 0,
    "windows_retried": 0,
    "windows_dropped": 0,
}


def post_webhook_sync(url: str, payload_bytes: bytes):
    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={"Content-Type": "application/json", "X-Request-ID": f"omi-{time.time_ns()}"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        logger.warning(f"Webhook error ({url}): {e}")


def flush_daily_paragraph():
    """Write accumulated 5s lines as a clean timestamped paragraph to logs/YYYY-MM-DD.md."""
    global paragraph_words, paragraph_start_ts
    if not paragraph_words:
        return
    text = " ".join(paragraph_words).strip()
    paragraph_words = []
    ts_label = paragraph_start_ts or datetime.now().strftime("%H:%M:%S")
    paragraph_start_ts = ""
    day_str = datetime.now().strftime("%Y-%m-%d")
    md_path = os.path.join(LOG_DIR, f"{day_str}.md")
    need_header = not os.path.exists(md_path)
    with open(md_path, "a", encoding="utf-8") as f:
        if need_header:
            f.write(f"# Omi Transcript — {day_str}\n\n")
        f.write(f"**[{ts_label}]** {text}\n\n")


def record_transcript_line(text: str):
    global paragraph_words, paragraph_start_ts, last_transcript_time
    text = " ".join(text.split()).strip()
    if not text:
        return
    now = time.time()
    now_dt = datetime.now()
    ts = now_dt.strftime("%Y-%m-%d %H:%M:%S")

    # 1. Append to fast 5s stream log
    with open(TRANSCRIPT_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {text}\n")
    logger.info(f"STT: {text}")

    # 2. Accumulate into daily paragraph log; break paragraph on >8s pause or >=120 words
    if paragraph_words and (now - last_transcript_time) > IDLE_FLUSH_SEC:
        flush_daily_paragraph()
    if not paragraph_words:
        paragraph_start_ts = now_dt.strftime("%H:%M:%S")
    paragraph_words.extend(text.split())
    last_transcript_time = now
    if len(paragraph_words) >= 120:
        flush_daily_paragraph()

    # 3. Feed research buffer & live browser monitor
    rolling_research_buffer.append(f"[{ts}] {text}")
    for ws in list(live_listeners):
        if not ws.closed:
            asyncio.create_task(ws.send_str(json.dumps({"type": "transcript", "timestamp": ts, "text": text})))

    if WAKE_WORD and WAKE_WORD in text.lower():
        payload = json.dumps({"text": text, "speaker": "user", "timestamp": ts}).encode("utf-8")
        asyncio.get_running_loop().run_in_executor(None, post_webhook_sync, HERMES_LIVE_WEBHOOK, payload)


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


def _gemini_stt_request_sync(pcm_bytes: bytes, sr: int, pre_sec: float, core_sec: float, post_sec: float, prev_ctx: str) -> str:
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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_STT_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "audio/wav", "data": b64_wav}}]}],
        "generationConfig": {"temperature": 0.0},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    candidates = result.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    return " ".join(" ".join(p.get("text", "") for p in parts).split()).strip()


def enqueue_stt_window(full_pcm: bytes, sr: int, pre_s: float, core_s: float, post_s: float, reset_context_after: bool = False):
    item = (full_pcm, sr, pre_s, core_s, post_s, reset_context_after)
    if stt_queue.full():
        try:
            stt_queue.get_nowait()
            stt_queue.task_done()
            metrics["windows_dropped"] += 1
            logger.warning("STT queue full (24 windows); dropped oldest window to bound latency")
        except asyncio.QueueEmpty:
            pass
    stt_queue.put_nowait(item)


async def stt_consumer_worker():
    """Single sequential consumer for stt_queue with 3x retry on transient errors."""
    last_transcript_tail = ""
    retry_delays = (1.0, 2.0, 4.0)

    while True:
        item = await stt_queue.get()
        if item is None:
            stt_queue.task_done()
            break

        pcm_bytes, sr, pre_sec, core_sec, post_sec, reset_context_after = item
        try:
            bps = sr * 2
            pre_b, core_b = int(pre_sec * bps), int(core_sec * bps)
            core_pcm = pcm_bytes[pre_b : pre_b + core_b] if len(pcm_bytes) > pre_b else pcm_bytes
            rms = pcm16_rms(core_pcm)

            if rms >= RMS_SILENCE_THRESHOLD and GEMINI_API_KEY:
                loop = asyncio.get_running_loop()
                text = None
                for attempt in range(len(retry_delays) + 1):
                    try:
                        text = await loop.run_in_executor(
                            None, _gemini_stt_request_sync, pcm_bytes, sr, pre_sec, core_sec, post_sec, last_transcript_tail
                        )
                        metrics["last_stt_success_at"] = time.time()
                        metrics["windows_processed"] += 1
                        break
                    except urllib.error.HTTPError as e:
                        metrics["last_error"] = f"HTTP {e.code}: {e.reason}"
                        if e.code not in (408, 429, 500, 502, 503, 504) or attempt == len(retry_delays):
                            logger.error(f"Gemini STT failed permanently ({metrics['last_error']})")
                            break
                        metrics["windows_retried"] += 1
                        delay = retry_delays[attempt]
                        logger.warning(f"Gemini STT transient HTTP {e.code}; retry {attempt + 1}/{len(retry_delays)} in {delay}s")
                        await asyncio.sleep(delay)
                    except Exception as e:
                        metrics["last_error"] = f"{type(e).__name__}: {e}"
                        if attempt == len(retry_delays):
                            logger.error(f"Gemini STT failed after retries ({metrics['last_error']})")
                            break
                        metrics["windows_retried"] += 1
                        delay = retry_delays[attempt]
                        logger.warning(f"Gemini STT error ({e}); retry {attempt + 1}/{len(retry_delays)} in {delay}s")
                        await asyncio.sleep(delay)

                if text and text.lower() not in ("", "none", "silence", "[silence]"):
                    cleaned = dedupe_boundary_overlap(last_transcript_tail, text)
                    if cleaned:
                        last_transcript_tail = " ".join((last_transcript_tail + " " + cleaned).split()[-24:])
                        record_transcript_line(cleaned)

            if reset_context_after:
                last_transcript_tail = ""
                flush_daily_paragraph()
        finally:
            stt_queue.task_done()


def ingest_raw_pcm(raw: bytes, sr: int = 16000):
    global audio_sample_rate, audio_ring_bytes
    if not raw:
        return
    # Ensure even byte count for 16-bit samples
    if len(raw) % 2 != 0:
        raw = raw[:-1]
    if not raw:
        return

    now = time.time()
    metrics["last_packet_at"] = now
    audio_sample_rate = sr
    audio_pcm_buffer.extend(raw)

    # Append to continuous 5-minute RAM ring buffer BEFORE any VAD/RMS filtering
    max_ring_bytes = sr * 2 * RING_BUFFER_SECONDS
    audio_ring.append((now, sr, raw))
    audio_ring_bytes += len(raw)
    while audio_ring and audio_ring_bytes > max_ring_bytes:
        _, _, old = audio_ring.popleft()
        audio_ring_bytes -= len(old)

    for ws in list(live_listeners):
        if not ws.closed:
            asyncio.create_task(ws.send_bytes(raw))


def flush_remaining_buffer(reset_context: bool = True):
    """Flush any remaining even-byte PCM16 audio in audio_pcm_buffer into stt_queue."""
    global audio_pcm_buffer, audio_pre_tail
    sr = audio_sample_rate
    bps = sr * 2
    valid_len = len(audio_pcm_buffer) & ~1
    if valid_len >= 2:
        tail = bytes(audio_pcm_buffer[:valid_len])
        audio_pcm_buffer.clear()
        pre_s = len(audio_pre_tail) / float(bps)
        core_s = len(tail) / float(bps)
        full_pcm = bytes(audio_pre_tail) + tail
        audio_pre_tail.clear()
        enqueue_stt_window(full_pcm, sr, pre_s, core_s, 0.0, reset_context_after=reset_context)
    else:
        audio_pcm_buffer.clear()
        audio_pre_tail.clear()
        if reset_context:
            flush_daily_paragraph()


async def window_slicer_worker():
    """Slices incoming audio_pcm_buffer into [1s pre + ~5s core + 1s post] windows and handles idle flush."""
    global audio_pcm_buffer, audio_pre_tail
    while True:
        await asyncio.sleep(0.2)
        sr = audio_sample_rate
        bps = sr * 2
        need_b = int((CORE_WINDOW_SEC + POST_LOOKAHEAD_SEC) * bps)
        pre_b = int(PRE_OVERLAP_SEC * bps)
        post_b = int(POST_LOOKAHEAD_SEC * bps)

        if len(audio_pcm_buffer) >= need_b:
            core_cut = find_dynamic_cut_offset(audio_pcm_buffer, sr, CORE_WINDOW_SEC)
            win_end = min(core_cut + post_b, len(audio_pcm_buffer)) & ~1
            core_plus_post = bytes(audio_pcm_buffer[:win_end])
            core_only = bytes(audio_pcm_buffer[:core_cut])
            del audio_pcm_buffer[:core_cut]

            pre_s = len(audio_pre_tail) / float(bps)
            core_s = len(core_only) / float(bps)
            post_s = (len(core_plus_post) - len(core_only)) / float(bps)
            full_pcm = bytes(audio_pre_tail) + core_plus_post
            audio_pre_tail = bytearray(core_only[-pre_b:] if len(core_only) >= pre_b else core_only)

            enqueue_stt_window(full_pcm, sr, pre_s, core_s, post_s, reset_context_after=False)
        elif (
            len(audio_pcm_buffer) >= 2
            and metrics["last_packet_at"] is not None
            and (time.time() - metrics["last_packet_at"]) > IDLE_FLUSH_SEC
        ):
            flush_remaining_buffer(reset_context=True)


async def research_worker():
    global rolling_research_buffer, last_research_time
    while True:
        await asyncio.sleep(5)
        now = time.time()
        if now - last_research_time < RESEARCH_INTERVAL_SEC or len(rolling_research_buffer) < 2:
            continue
        chunk = "\n".join(rolling_research_buffer)
        rolling_research_buffer = []
        last_research_time = now
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = json.dumps({"text": chunk, "timestamp": ts}).encode("utf-8")
        asyncio.get_running_loop().run_in_executor(None, post_webhook_sync, HERMES_RESEARCH_WEBHOOK, payload)


async def handle_audio_last(request: web.Request) -> web.Response:
    """Export the last N seconds (default 60, max 300) from the 5-minute RAM ring buffer as a playable WAV."""
    seconds = min(max(int(request.query.get("seconds", "60")), 1), RING_BUFFER_SECONDS)
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


def build_health_dict() -> dict:
    now = time.time()
    return {
        "status": "ok" if GEMINI_API_KEY else "degraded_no_gemini_key",
        "is_setup_completed": True,
        "message": "",
        "gemini_key_configured": bool(GEMINI_API_KEY),
        "model": GEMINI_STT_MODEL,
        "uptime_sec": round(now - metrics["started_at"], 1),
        "last_packet_age_sec": round(now - metrics["last_packet_at"], 1) if metrics["last_packet_at"] else None,
        "last_stt_success_age_sec": round(now - metrics["last_stt_success_at"], 1) if metrics["last_stt_success_at"] else None,
        "queue_depth": stt_queue.qsize(),
        "buffer_bytes": len(audio_pcm_buffer),
        "ring_buffer_sec": round(audio_ring_bytes / float(audio_sample_rate * 2), 1),
        "windows_processed": metrics["windows_processed"],
        "windows_retried": metrics["windows_retried"],
        "windows_dropped": metrics["windows_dropped"],
        "last_error": metrics["last_error"],
    }


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response(build_health_dict())


async def handle_omi(request: web.Request) -> web.StreamResponse:
    if request.method in ("GET", "HEAD"):
        return web.json_response(build_health_dict())

    raw = await request.read()
    sr = int(request.query.get("sample_rate", "16000"))
    pcm = raw[44:] if (raw[:4] == b"RIFF" and len(raw) > 44) else raw
    ingest_raw_pcm(pcm, sr=sr)
    return web.json_response({"status": "ok", "is_setup_completed": True, "message": ""})


async def handle_listen_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20.0)
    await ws.prepare(request)
    live_listeners.add(ws)
    try:
        async for _ in ws:
            pass
    finally:
        live_listeners.discard(ws)
    return ws


LISTEN_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Omi Live Stream</title>
<style>body{font-family:system-ui,sans-serif;background:#0f172a;color:#f8fafc;padding:24px;max-width:900px;margin:0 auto}button{background:#3b82f6;color:#fff;border:none;padding:10px 18px;border-radius:8px;cursor:pointer}#log{margin-top:16px;background:#1e293b;padding:16px;border-radius:8px;height:70vh;overflow-y:auto;white-space:pre-wrap}</style>
</head><body><h2>Omi Live Stream (16kHz PCM16 + Dynamic 1s+5s+1s STT)</h2>
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
    if(typeof e.data==='string'){const m=JSON.parse(e.data);if(m.type==='transcript'){log.textContent+=`[${m.timestamp}] ${m.text}\\n\\n`;log.scrollTop=log.scrollHeight;}}
    else{const i16=new Int16Array(e.data),f32=new Float32Array(i16.length);for(let i=0;i<i16.length;i++)f32[i]=i16[i]/32768.0;const b=ctx.createBuffer(1,f32.length,16000);b.getChannelData(0).set(f32);const s=ctx.createBufferSource();s.buffer=b;s.connect(ctx.destination);if(nextTime<ctx.currentTime)nextTime=ctx.currentTime+0.05;s.start(nextTime);nextTime+=b.duration;}
  };
};
</script></body></html>"""


async def handle_listen(request: web.Request) -> web.Response:
    return web.Response(text=LISTEN_HTML, content_type="text/html")


async def on_startup(app: web.Application):
    if not GEMINI_API_KEY:
        logger.warning("WARNING: GEMINI_API_KEY is not set in ~/omi/.env or ~/.hermes/.env!")
    app["slicer_task"] = asyncio.create_task(window_slicer_worker())
    app["stt_worker_task"] = asyncio.create_task(stt_consumer_worker())
    app["research_task"] = asyncio.create_task(research_worker())


async def on_cleanup(app: web.Application):
    logger.info("Shutting down: flushing remaining audio buffer and draining STT queue...")
    app["slicer_task"].cancel()
    app["research_task"].cancel()
    flush_remaining_buffer(reset_context=True)
    await stt_queue.put(None)
    try:
        await asyncio.wait_for(app["stt_worker_task"], timeout=15.0)
    except asyncio.TimeoutError:
        logger.warning("Timed out waiting for STT queue drain on shutdown")
    flush_daily_paragraph()


if __name__ == "__main__":
    app = web.Application(client_max_size=50 * 1024 * 1024)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/audio/last", handle_audio_last)
    app.router.add_get("/listen", handle_listen)
    app.router.add_get("/listen-ws", handle_listen_ws)
    app.router.add_route("*", "/omi", handle_omi)
    app.router.add_route("*", "/omi/{tail:.*}", handle_omi)
    logger.info(
        f"Omi bridge starting on 0.0.0.0:8787 (model={GEMINI_STT_MODEL}, key_ok={bool(GEMINI_API_KEY)}, queue_max=24, ring={RING_BUFFER_SECONDS}s)"
    )
    web.run_app(app, host="0.0.0.0", port=8787, print=None)
