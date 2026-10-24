#!/usr/bin/env python3
import asyncio
import base64
import glob
import io
import json
import os
import struct
import sys
import time
import urllib.request
import wave
from datetime import datetime
from urllib.parse import parse_qs, urlparse

# Add Hermes site-packages if running under Hermes Python or system Python
for p in glob.glob(os.path.expanduser("~/.hermes/**/site-packages"), recursive=True):
    if p not in sys.path:
        sys.path.insert(0, p)

from aiohttp import web

HERMES_LIVE_WEBHOOK = "http://127.0.0.1:8644/webhooks/omi-live"
HERMES_RESEARCH_WEBHOOK = "http://127.0.0.1:8644/webhooks/omi-research"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_LOG = os.path.join(BASE_DIR, "logs", "omi_live_transcript.log")
LEGACY_TRANSCRIPT_LOG = os.path.expanduser("~/.hermes/logs/omi_live_transcript.log")
os.makedirs(os.path.dirname(TRANSCRIPT_LOG), exist_ok=True)
os.makedirs(os.path.dirname(LEGACY_TRANSCRIPT_LOG), exist_ok=True)

WAKE_WORD = os.environ.get("OMI_WAKE_WORD", "hermes").strip().lower()
RESEARCH_INTERVAL_SEC = int(os.environ.get("OMI_RESEARCH_INTERVAL", "90"))

# Symmetric sliding window parameters:
# [1s pre-overlap (before)] + [5s core speech (dynamic pause-aligned)] + [1s post-lookahead (after)] = 7s total window (2s total overlap)
PRE_OVERLAP_SEC = float(os.environ.get("OMI_PRE_OVERLAP_SEC", "1.0"))
CORE_WINDOW_SEC = float(os.environ.get("OMI_CORE_WINDOW_SEC", "5.0"))
POST_LOOKAHEAD_SEC = float(os.environ.get("OMI_POST_LOOKAHEAD_SEC", "1.0"))
GEMINI_STT_MODEL = os.environ.get("OMI_GEMINI_STT_MODEL", "gemini-3.8-flash")


def _load_gemini_key() -> str:
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(k):
            return os.environ[k]
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    if k.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                        return v.strip().strip('"').strip("'")
    return ""


GEMINI_API_KEY = _load_gemini_key()

rolling_buffer = []
last_research_time = time.time()

audio_pcm_buffer = bytearray()
audio_pre_tail = bytearray()
audio_sample_rate = 16000
last_audio_stt_time = time.time()
last_transcript_tail = ""
stt_lock = asyncio.Lock()

# Connected live browser listeners (WebSocket clients on /listen-ws)
live_listeners = set()


def post_webhook_sync(url: str, payload_bytes: bytes, ts: str):
    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": f"omi-{time.time_ns()}",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        print(f"[{ts}] Webhook error ({url}): {e}", flush=True)


def handle_transcript_line(combined_text: str, speaker: str = "SPEAKER"):
    combined_text = " ".join(combined_text.split()).strip()
    if not combined_text:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{ts}] {combined_text}\n"
    print(log_entry, end="", flush=True)
    with open(TRANSCRIPT_LOG, "a", encoding="utf-8") as f:
        f.write(log_entry)

    rolling_buffer.append(f"[{ts}] {combined_text}")

    for ws in list(live_listeners):
        if not ws.closed:
            asyncio.create_task(
                ws.send_str(json.dumps({"type": "transcript", "timestamp": ts, "text": combined_text}))
            )

    if WAKE_WORD and WAKE_WORD in combined_text.lower():
        payload = json.dumps({
            "text": combined_text,
            "speaker": speaker,
            "timestamp": ts,
        }).encode("utf-8")
        asyncio.get_running_loop().run_in_executor(
            None, post_webhook_sync, HERMES_LIVE_WEBHOOK, payload, ts
        )


def pcm16_rms(pcm_bytes: bytes) -> float:
    n = len(pcm_bytes) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm_bytes[: n * 2])
    mean_sq = sum(s * s for s in samples) / n
    return mean_sq ** 0.5


def find_dynamic_cut_offset(buf: bytearray, sr: int, target_core_sec: float) -> int:
    """Find the lowest-energy 100ms micro-pause near target_core_sec (between -0.8s and +0.8s)
    so the 5s core cut lands inside a natural breath/syllable valley before the 1s post-lookahead."""
    bps = sr * 2
    target_byte = int(target_core_sec * bps)
    win_100ms = int(0.1 * bps)
    win_100ms -= win_100ms % 2

    search_lo = max(int((target_core_sec - 0.8) * bps), win_100ms)
    search_hi = min(int((target_core_sec + 0.8) * bps), len(buf) - int(POST_LOOKAHEAD_SEC * bps) - win_100ms)

    if search_hi <= search_lo:
        cut = min(target_byte, len(buf))
        return cut - (cut % 2)

    best_offset = target_byte
    best_rms = float("inf")
    step = win_100ms // 2
    step -= step % 2
    for pos in range(search_lo, search_hi, max(step, 2)):
        pos -= pos % 2
        r = pcm16_rms(bytes(buf[pos : pos + win_100ms]))
        if r < best_rms:
            best_rms = r
            best_offset = pos + (win_100ms // 2)
    best_offset -= best_offset % 2
    return best_offset


def dedupe_boundary_overlap(prev_tail: str, new_text: str) -> str:
    if not prev_tail or not new_text:
        return new_text.strip()
    prev_words = prev_tail.split()
    new_words = new_text.split()
    max_k = min(len(prev_words), len(new_words), 25)
    best_k = 0
    for k in range(2, max_k + 1):
        p_slice = [w.strip(".,!?;:\"'()[]{}").lower() for w in prev_words[-k:]]
        n_slice = [w.strip(".,!?;:\"'()[]{}").lower() for w in new_words[:k]]
        if p_slice == n_slice:
            best_k = k
    if best_k > 0:
        return " ".join(new_words[best_k:]).strip()
    return new_text.strip()


def _gemini_stt_sync(
    pcm_bytes: bytes,
    sample_rate: int,
    pre_sec: float,
    core_sec: float,
    post_sec: float,
    prev_context: str,
) -> str:
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    b64_wav = base64.b64encode(wav_buf.getvalue()).decode("ascii")

    overlap_note = ""
    if pre_sec > 0 and prev_context:
        overlap_note = (
            f"This audio clip has a {pre_sec:.1f}s pre-overlap at the start and a {post_sec:.1f}s lookahead tail at the end "
            f"(centering a {core_sec:.1f}s speech window). "
            f"The previous window ended with: \"{prev_context}\". "
            "Use the pre-overlap and post-lookahead so no boundary words are clipped, "
            "and output ONLY the new continuation up to the last complete word before the end (do not repeat previous words). "
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_STT_MODEL}:generateContent?key={GEMINI_API_KEY}"
    prompt = (
        "Transcribe the speech in this audio clip into clean, uniform, single-paragraph English prose. "
        + overlap_note +
        "Preserve technical AI/ML terms accurately (e.g., PyTorch, vLLM, SGLang, CUDA, Triton, KV cache, RL, GRPO, MCP). "
        "Do not include timestamps, speaker labels, markdown, or commentary. "
        "If there is no intelligible speech beyond the previous context, return ONLY an empty string."
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "audio/wav", "data": b64_wav}},
            ]
        }],
        "generationConfig": {"temperature": 0.0},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    candidates = result.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    return " ".join(" ".join(p.get("text", "") for p in parts).split()).strip()


async def transcribe_window_with_gemini(
    pcm_bytes: bytes,
    sample_rate: int,
    pre_sec: float,
    core_sec: float,
    post_sec: float,
) -> str:
    global last_transcript_tail
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not GEMINI_API_KEY:
        print(f"[{ts}] Gemini STT skipped: no GEMINI_API_KEY found", flush=True)
        return ""

    bps = sample_rate * 2
    pre_bytes = int(pre_sec * bps)
    core_bytes = int(core_sec * bps)
    core_pcm = pcm_bytes[pre_bytes : pre_bytes + core_bytes] if len(pcm_bytes) > pre_bytes else pcm_bytes
    rms = pcm16_rms(core_pcm)
    if rms < 80:
        print(f"[{ts}] Audio window ({core_sec:.1f}s core, rms={rms:.1f}) is silent; skipping STT", flush=True)
        return ""

    async with stt_lock:
        try:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None,
                _gemini_stt_sync,
                pcm_bytes,
                sample_rate,
                pre_sec,
                core_sec,
                post_sec,
                last_transcript_tail,
            )
            if text and text.lower() not in ("", "none", "silence", "[silence]"):
                cleaned = dedupe_boundary_overlap(last_transcript_tail, text)
                if cleaned:
                    words = (last_transcript_tail + " " + cleaned).split()
                    last_transcript_tail = " ".join(words[-24:])
                    handle_transcript_line(cleaned, speaker="TRANSCRIPT")
                    return cleaned
            else:
                print(f"[{ts}] Gemini STT ({core_sec:.1f}s core, rms={rms:.1f}): no speech detected", flush=True)
        except Exception as e:
            print(f"[{ts}] Gemini STT error: {e}", flush=True)
    return ""


last_packet_time = time.time()


def ingest_raw_pcm(raw: bytes, sr: int = 16000):
    global audio_sample_rate, last_packet_time
    if not raw:
        return
    last_packet_time = time.time()
    audio_sample_rate = sr
    audio_pcm_buffer.extend(raw)
    for ws in list(live_listeners):
        if not ws.closed:
            asyncio.create_task(ws.send_bytes(raw))


async def audio_stt_worker():
    """Dynamic real-time sliding window worker:
    Waits until [5s core + 1s lookahead = 6s] of new audio is in the buffer,
    finds the cleanest micro-pause near the 5s mark, Prepends the [1s pre-overlap] from the
    end of the previous core window, and sends [1s before + ~5s core + 1s after] = ~7s total.
    Also flushes any remaining tail (>=1s) if Omi stops sending packets for >8s."""
    global audio_pcm_buffer, audio_pre_tail, last_audio_stt_time
    while True:
        await asyncio.sleep(0.25)
        sr = audio_sample_rate
        bps = sr * 2
        need_bytes = int((CORE_WINDOW_SEC + POST_LOOKAHEAD_SEC) * bps)
        pre_bytes = int(PRE_OVERLAP_SEC * bps)
        post_bytes = int(POST_LOOKAHEAD_SEC * bps)

        if len(audio_pcm_buffer) >= need_bytes:
            # Find dynamic micro-pause cut near 5.0s
            core_cut = find_dynamic_cut_offset(audio_pcm_buffer, sr, CORE_WINDOW_SEC)
            window_end = min(core_cut + post_bytes, len(audio_pcm_buffer))

            core_plus_post = bytes(audio_pcm_buffer[:window_end])
            # Advance the stream by core_cut ONLY — so the 1s post-lookahead remains in audio_pcm_buffer
            # as the start of the next 5s core window, while the last 1s BEFORE core_cut becomes audio_pre_tail!
            core_only = bytes(audio_pcm_buffer[:core_cut])
            del audio_pcm_buffer[:core_cut]

            pre_sec_actual = len(audio_pre_tail) / float(bps)
            core_sec_actual = len(core_only) / float(bps)
            post_sec_actual = (len(core_plus_post) - len(core_only)) / float(bps)

            full_window_pcm = bytes(audio_pre_tail) + core_plus_post
            audio_pre_tail = bytearray(
                core_only[-pre_bytes:] if len(core_only) >= pre_bytes else core_only
            )
            last_audio_stt_time = time.time()

            asyncio.create_task(
                transcribe_window_with_gemini(
                    full_window_pcm,
                    sr,
                    pre_sec_actual,
                    core_sec_actual,
                    post_sec_actual,
                )
            )
        elif len(audio_pcm_buffer) >= bps and (time.time() - last_packet_time) > 8.0:
            # Idle tail flush when Omi is turned off or goes out of range
            tail_bytes = bytes(audio_pcm_buffer)
            audio_pcm_buffer.clear()
            pre_sec_actual = len(audio_pre_tail) / float(bps)
            core_sec_actual = len(tail_bytes) / float(bps)
            full_window_pcm = bytes(audio_pre_tail) + tail_bytes
            audio_pre_tail.clear()
            last_audio_stt_time = time.time()
            asyncio.create_task(
                transcribe_window_with_gemini(
                    full_window_pcm,
                    sr,
                    pre_sec_actual,
                    core_sec_actual,
                    0.0,
                )
            )


async def research_worker():
    global rolling_buffer, last_research_time
    while True:
        await asyncio.sleep(5)
        now = time.time()
        if now - last_research_time < RESEARCH_INTERVAL_SEC or len(rolling_buffer) < 2:
            continue
        chunk = "\n".join(rolling_buffer)
        rolling_buffer = []
        last_research_time = now

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] Triggering proactive research on {len(chunk)} chars...", flush=True)
        payload = json.dumps({"text": chunk, "timestamp": ts}).encode("utf-8")
        asyncio.get_running_loop().run_in_executor(
            None, post_webhook_sync, HERMES_RESEARCH_WEBHOOK, payload, ts
        )


async def handle_omi_ws(request: web.Request) -> web.WebSocketResponse:
    """Real-time WebSocket endpoint for Omi Custom Live STT (wss://mini.tail287a74.ts.net/omi)."""
    ws = web.WebSocketResponse(heartbeat=20.0)
    await ws.prepare(request)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sr = int(request.query.get("sample_rate", "16000"))
    print(f"[{ts}] WebSocket CONNECTED from Omi app ({request.path_qs}, sr={sr})", flush=True)

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                ingest_raw_pcm(msg.data, sr=sr)
            elif msg.type == web.WSMsgType.TEXT:
                try:
                    obj = json.loads(msg.data)
                    if obj.get("type") == "KeepAlive":
                        await ws.send_str(json.dumps({"type": "KeepAliveAck"}))
                except Exception:
                    pass
    finally:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] WebSocket DISCONNECTED ({request.path_qs})", flush=True)
    return ws


async def handle_listen_ws(request: web.Request) -> web.WebSocketResponse:
    """Browser WebSocket to hear live PCM16 audio + see live transcripts in real time."""
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
<html>
<head>
  <meta charset="utf-8">
  <title>Omi Live Audio & Transcript Stream</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; padding: 24px; max-width: 900px; margin: 0 auto; }
    button { background: #3b82f6; color: white; border: none; padding: 12px 20px; border-radius: 8px; font-size: 16px; cursor: pointer; }
    #status { margin-left: 12px; color: #94a3b8; }
    #log { margin-top: 20px; background: #1e293b; padding: 16px; border-radius: 8px; height: 70vh; overflow-y: auto; white-space: pre-wrap; line-height: 1.5; }
  </style>
</head>
<body>
  <h2>Omi Live Stream Monitor (16kHz PCM16 + Dynamic 1s+5s+1s Gemini STT)</h2>
  <button id="btn">Start Listening to Live Audio</button>
  <span id="status">Disconnected</span>
  <div id="log"></div>
  <script>
    let ctx, nextTime = 0;
    const btn = document.getElementById('btn');
    const status = document.getElementById('status');
    const log = document.getElementById('log');
    btn.onclick = () => {
      if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 16000});
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(proto + '//' + location.host + '/listen-ws');
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => { status.textContent = 'Connected & Streaming Live'; btn.disabled = true; };
      ws.onclose = () => { status.textContent = 'Disconnected'; btn.disabled = false; };
      ws.onmessage = (e) => {
        if (typeof e.data === 'string') {
          const msg = JSON.parse(e.data);
          if (msg.type === 'transcript') {
            log.textContent += `[${msg.timestamp}] ${msg.text}\\n\\n`;
            log.scrollTop = log.scrollHeight;
          }
        } else {
          const int16 = new Int16Array(e.data);
          const float32 = new Float32Array(int16.length);
          for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768.0;
          const buf = ctx.createBuffer(1, float32.length, 16000);
          buf.getChannelData(0).set(float32);
          const src = ctx.createBufferSource();
          src.buffer = buf;
          src.connect(ctx.destination);
          if (nextTime < ctx.currentTime) nextTime = ctx.currentTime + 0.05;
          src.start(nextTime);
          nextTime += buf.duration;
        }
      };
    };
  </script>
</body>
</html>
"""


async def handle_any(request: web.Request) -> web.StreamResponse:
    if request.headers.get("Upgrade", "").lower() == "websocket":
        if request.path == "/listen-ws":
            return await handle_listen_ws(request)
        return await handle_omi_ws(request)

    if request.path == "/listen":
        return web.Response(text=LISTEN_HTML, content_type="text/html")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ctype = request.headers.get("Content-Type", "")
    clen = request.headers.get("Content-Length", "0")
    print(f"[{ts}] HTTP {request.method} {request.path_qs} ({ctype}, {clen}B)", flush=True)

    if request.method in ("GET", "HEAD"):
        return web.json_response({"status": "ok", "is_setup_completed": True, "message": ""})

    raw = await request.read()
    ctype_lower = ctype.lower()

    if "application/octet-stream" in ctype_lower or "sample_rate=" in request.path_qs or "audio/wav" in ctype_lower:
        sr = int(request.query.get("sample_rate", "16000"))
        pcm = raw[44:] if (raw[:4] == b"RIFF" and len(raw) > 44) else raw
        ingest_raw_pcm(pcm, sr=sr)
        return web.json_response({"status": "ok", "is_setup_completed": True, "message": ""})

    try:
        data = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
    except Exception:
        data = {"raw": raw.decode("utf-8", errors="replace")}

    segments = []
    if isinstance(data, list):
        segments = data
    elif isinstance(data, dict) and isinstance(data.get("segments"), list):
        segments = data["segments"]
    elif isinstance(data, dict) and isinstance(data.get("transcript_segments"), list):
        segments = data["transcript_segments"]

    words = []
    if segments:
        for seg in segments:
            if isinstance(seg, dict):
                txt = (seg.get("text") or "").strip()
                if txt:
                    words.append(txt)
    elif isinstance(data, dict):
        txt = (data.get("text") or data.get("transcript") or "").strip()
        if txt:
            words.append(txt)

    combined_text = " ".join(words).strip()
    if combined_text:
        handle_transcript_line(combined_text)

    return web.json_response({"status": "ok", "is_setup_completed": True, "message": ""})


async def on_startup(app: web.Application):
    app["stt_task"] = asyncio.create_task(audio_stt_worker())
    app["research_task"] = asyncio.create_task(research_worker())


def main():
    app = web.Application(client_max_size=50 * 1024 * 1024)
    app.on_startup.append(on_startup)
    app.router.add_route("*", "/{tail:.*}", handle_any)
    print(
        f"Omi bridge listening on 0.0.0.0:8787 (Dynamic [1s pre + 5s core + 1s post] window, Gemini STT={GEMINI_STT_MODEL})",
        flush=True,
    )
    web.run_app(app, host="0.0.0.0", port=8787, print=None)


if __name__ == "__main__":
    main()
