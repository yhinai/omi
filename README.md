# Omi Real-Time Audio & Transcript Bridge

Lean, fault-tolerant `aiohttp` server that receives 16kHz PCM16 audio bytes from an **Omi wearable** (`Audio Bytes` webhook) and transcribes in real time using Gemini (`gemini-3.8-flash`).

## Architecture & Key Features

```
Omi Audio Bytes (16kHz PCM16, 5s interval)
      ↓
POST /omi (Tailscale Funnel HTTPS 443 -> 127.0.0.1:8787)
      ↓
5-Minute RAM Audio Ring Buffer (/audio/last?seconds=60)
      ↓
Dynamic [1s pre + ~5s pause-aligned core + 1s post] Slicer
      ↓
Bounded Queue (maxsize=24) -> Single STT Worker (3x backoff retry)
      ↓
  ├── ~/omi/logs/omi_live_transcript.log  (Fast 5s stream)
  └── ~/omi/logs/YYYY-MM-DD.md            (Clean pause-grouped paragraphs)
```

- **Bounded Queue & Single STT Worker**: `asyncio.Queue(maxsize=24)` consumed sequentially with 3x exponential backoff retry (`1s, 2s, 4s`) on HTTP `429`/`5xx` and timeouts so transient errors never drop audio windows or pile up unbounded tasks.
- **Dynamic `[1s pre + 5s core + 1s post]` Sliding Window**: Cuts inside the lowest-RMS `100ms` micro-pause near `5.0s` (`±0.8s`), deduplicating boundary n-grams.
- **Session Boundary Hygiene & Graceful Shutdown**: Flushes any remaining audio tail (`>=1` sample) after `8s` of idle silence, resets `last_transcript_tail` across idle boundaries, and drains in-flight windows on `SIGTERM`/`SIGINT`.
- **5-Minute Continuous RAM Audio Ring Buffer**: Retains the last 5 minutes (`9.6 MB`) of raw incoming audio *before* RMS silence filtering, downloadable anytime as a playable WAV at `GET /audio/last?seconds=60`.
- **Two-Tier Logging**: Writes immediate 5s lines to `logs/omi_live_transcript.log` and clean pause-grouped paragraphs to `logs/YYYY-MM-DD.md`.
- **Real Telemetry (`GET /health`)**: Reports `gemini_key_configured`, `last_packet_age_sec`, `last_stt_success_age_sec`, `queue_depth`, `buffer_bytes`, `ring_buffer_sec`, retry/drop counters, and `last_error`.

## Quick Start

```bash
# 1. Expose port 8787 via Tailscale Funnel
tailscale funnel --bg --https=443 http://127.0.0.1:8787

# 2. Configure API key (or rely on ~/.hermes/.env)
cp ~/omi/.env.example ~/omi/.env

# 3. Start bridge (auto-creates ~/omi/.venv if missing)
~/omi/start.sh

# 4. Check live health & telemetry
curl -s http://127.0.0.1:8787/health | python3 -m json.tool
```
