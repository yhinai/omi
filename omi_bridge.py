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

for p in glob.glob(os.path.expanduser("~/.hermes/**/site-packages"), recursive=True):
    if p not in sys.path:
        sys.path.insert(0, p)

from aiohttp import web

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_LOG = os.path.join(BASE_DIR, "logs", "omi_live_transcript.log")
os.makedirs(os.path.dirname(TRANSCRIPT_LOG), exist_ok=True)

HERMES_LIVE_WEBHOOK = "http://127.0.0.1:8644/webhooks/omi-live"
HERMES_RESEARCH_WEBHOOK = "http://127.0.0.1:8644/webhooks/omi-research"

WAKE_WORD = os.environ.get("OMI_WAKE_WORD", "hermes").strip().lower()
RESEARCH_INTERVAL_SEC = int(os.environ.get("OMI_RESEARCH_INTERVAL", "90"))
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
last_packet_time = time.time()
last_transcript_tail = ""
stt_lock = asyncio.Lock()
live_listeners = set()


def post_webhook_sync(url: str, payload_bytes: bytes, ts: str):
    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={"Content-Type": "application/json", "X-Request-ID": f"omi-{time.time_ns()}"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        print(f"[{ts}] Webhook error ({url}): {e}", flush=True)


def handle_transcript_line(text: str, speaker: str = "SPEAKER"):
    text = " ".join(text.split()).strip()
    if not text:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{ts}] {text}\n"
    print(log_entry, end="", flush=True)
    with open(TRANSCRIPT_LOG, "a", encoding="utf-8") as f:
        f.write(log_entry)

    rolling_buffer.append(f"[{ts}] {text}")

    for ws in list(live_listeners):
        if not ws.closed:
            asyncio.create_task(ws.send_str(json.dumps({"type": "transcript", "timestamp": ts, "text": text})))

    if WAKE_WORD and WAKE_WORD in text.lower():
        payload = json.dumps({"text": text, "speaker": speaker, "timestamp": ts}).encode("utf-8")
        asyncio.get_running_loop().run_in_executor(None, post_webhook_sync, HERMES_LIVE_WEBHOOK, payload, ts)


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


def _gemini_stt_sync(pcm_bytes: bytes, sr: int, pre_sec: float, core_sec: float, post_sec: float, prev_ctx: str) -> str:
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
    with urllib.request.urlopen(req, timeout=25) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    candidates = result.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    return " ".join(" ".join(p.get("text", "") for p in parts).split()).strip()


async def transcribe_window(pcm_bytes: bytes, sr: int, pre_sec: float, core_sec: float, post_sec: float):
    global last_transcript_tail
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not GEMINI_API_KEY:
        return
    bps = sr * 2
    pre_b, core_b = int(pre_sec * bps), int(core_sec * bps)
    core_pcm = pcm_bytes[pre_b : pre_b + core_b] if len(pcm_bytes) > pre_b else pcm_bytes
    rms = pcm16_rms(core_pcm)
    if rms < 80:
        return

    async with stt_lock:
        try:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None, _gemini_stt_sync, pcm_bytes, sr, pre_sec, core_sec, post_sec, last_transcript_tail
            )
            if text and text.lower() not in ("", "none", "silence", "[silence]"):
                cleaned = dedupe_boundary_overlap(last_transcript_tail, text)
                if cleaned:
                    last_transcript_tail = " ".join((last_transcript_tail + " " + cleaned).split()[-24:])
                    handle_transcript_line(cleaned)
        except Exception as e:
            print(f"[{ts}] Gemini STT error: {e}", flush=True)


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
    """Dynamic [1s pre + 5s core + 1s post] sliding window + 8s idle tail flush."""
    global audio_pcm_buffer, audio_pre_tail
    while True:
        await asyncio.sleep(0.25)
        sr = audio_sample_rate
        bps = sr * 2
        need_b = int((CORE_WINDOW_SEC + POST_LOOKAHEAD_SEC) * bps)
        pre_b = int(PRE_OVERLAP_SEC * bps)
        post_b = int(POST_LOOKAHEAD_SEC * bps)

        if len(audio_pcm_buffer) >= need_b:
            core_cut = find_dynamic_cut_offset(audio_pcm_buffer, sr, CORE_WINDOW_SEC)
            win_end = min(core_cut + post_b, len(audio_pcm_buffer))
            core_plus_post = bytes(audio_pcm_buffer[:win_end])
            core_only = bytes(audio_pcm_buffer[:core_cut])
            del audio_pcm_buffer[:core_cut]

            pre_s = len(audio_pre_tail) / float(bps)
            core_s = len(core_only) / float(bps)
            post_s = (len(core_plus_post) - len(core_only)) / float(bps)
            full_pcm = bytes(audio_pre_tail) + core_plus_post
            audio_pre_tail = bytearray(core_only[-pre_b:] if len(core_only) >= pre_b else core_only)

            asyncio.create_task(transcribe_window(full_pcm, sr, pre_s, core_s, post_s))
        elif len(audio_pcm_buffer) >= bps and (time.time() - last_packet_time) > 8.0:
            tail = bytes(audio_pcm_buffer)
            audio_pcm_buffer.clear()
            pre_s = len(audio_pre_tail) / float(bps)
            core_s = len(tail) / float(bps)
            full_pcm = bytes(audio_pre_tail) + tail
            audio_pre_tail.clear()
            asyncio.create_task(transcribe_window(full_pcm, sr, pre_s, core_s, 0.0))


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
        payload = json.dumps({"text": chunk, "timestamp": ts}).encode("utf-8")
        asyncio.get_running_loop().run_in_executor(None, post_webhook_sync, HERMES_RESEARCH_WEBHOOK, payload, ts)


async def handle_omi_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20.0)
    await ws.prepare(request)
    sr = int(request.query.get("sample_rate", "16000"))
    async for msg in ws:
        if msg.type == web.WSMsgType.BINARY:
            ingest_raw_pcm(msg.data, sr=sr)
        elif msg.type == web.WSMsgType.TEXT:
            try:
                if json.loads(msg.data).get("type") == "KeepAlive":
                    await ws.send_str(json.dumps({"type": "KeepAliveAck"}))
            except Exception:
                pass
    return ws


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


async def handle_any(request: web.Request) -> web.StreamResponse:
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return await (handle_listen_ws(request) if request.path == "/listen-ws" else handle_omi_ws(request))
    if request.path == "/listen":
        return web.Response(text=LISTEN_HTML, content_type="text/html")
    if request.method in ("GET", "HEAD"):
        return web.json_response({"status": "ok", "is_setup_completed": True, "message": ""})

    raw = await request.read()
    ctype = (request.headers.get("Content-Type") or "").lower()
    if "application/octet-stream" in ctype or "sample_rate=" in request.path_qs or "audio/wav" in ctype:
        sr = int(request.query.get("sample_rate", "16000"))
        pcm = raw[44:] if (raw[:4] == b"RIFF" and len(raw) > 44) else raw
        ingest_raw_pcm(pcm, sr=sr)
        return web.json_response({"status": "ok", "is_setup_completed": True, "message": ""})

    try:
        data = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
    except Exception:
        data = {}

    segments = data if isinstance(data, list) else (data.get("segments") or data.get("transcript_segments") or []) if isinstance(data, dict) else []
    words = [seg.get("text", "").strip() for seg in segments if isinstance(seg, dict) and seg.get("text", "").strip()]
    if not words and isinstance(data, dict):
        txt = (data.get("text") or data.get("transcript") or "").strip()
        if txt:
            words.append(txt)
    if words:
        handle_transcript_line(" ".join(words))
    return web.json_response({"status": "ok", "is_setup_completed": True, "message": ""})


async def on_startup(app: web.Application):
    app["stt_task"] = asyncio.create_task(audio_stt_worker())
    app["research_task"] = asyncio.create_task(research_worker())


if __name__ == "__main__":
    app = web.Application(client_max_size=50 * 1024 * 1024)
    app.on_startup.append(on_startup)
    app.router.add_route("*", "/{tail:.*}", handle_any)
    print(f"Omi bridge listening on 0.0.0.0:8787 (Dynamic [1s+5s+1s] window, model={GEMINI_STT_MODEL})", flush=True)
    web.run_app(app, host="0.0.0.0", port=8787, print=None)
