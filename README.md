# Omi Audio-to-Transcript Bridge

A small Python + `aiohttp` bridge for Omi's **Audio Bytes** webhook: 16 kHz, mono, little-endian PCM16 audio in; Gemini transcripts and optional Hermes research out.

```text
Omi Audio Bytes (5-second uploads)
        ↓
Tailscale Funnel → POST /omi on 127.0.0.1:8787
        ↓
Dynamic [1s pre + ~5s core + 1s post] windows
        ↓
Queue of 24 windows → one sequential Gemini worker
        ↓
Transcript log + daily paragraphs + Hermes webhooks
```

## Setup

Standard CPython **3.15.x** is required. The launcher creates a project-local `.venv` and installs `requirements.txt`; no Hermes Python installation is required. The bridge and launcher reject other Python minor versions, prereleases, and free-threaded builds. The launcher checks the runtime before restarting a running bridge and refuses to overwrite an incompatible environment.

```bash
cd ~/omi
cp .env.example .env
# Edit .env: uncomment GEMINI_API_KEY and enter your key.
# Alternatively, keep it commented to use ~/.hermes/.env.
./start.sh
```

For a new environment, the launcher looks for `python3.15` on `PATH`, in Homebrew or python.org locations, and in uv's managed Python directory. Set `OMI_PYTHON=/absolute/path/to/python3.15` to choose the interpreter explicitly. On a Mac without Python 3.15, install it with `brew install python@3.15` first.

When upgrading an older environment, test the new interpreter and dependencies in a separate temporary environment first. Stop the bridge with `./start.sh stop` so queued audio drains, move the old `.venv` to a backup, and recreate `.venv` at its final path using `python3.15 -m venv .venv`. Reinstall dependencies and run the verification checks below before `./start.sh start`. Virtual environments contain absolute paths; moving a prebuilt environment into place is unsupported.

Configuration precedence is **process environment → `~/omi/.env` → `~/.hermes/.env` → defaults**. `GEMINI_API_KEY` and `GOOGLE_API_KEY` are accepted. A missing or placeholder key prevents startup. Local `.env` values use `KEY=value` syntax; quote values containing spaces.

Expose the ingestion service:

```bash
tailscale funnel --bg --https=443 http://127.0.0.1:8787
```

In the Omi app, leave **Custom Backend URL** blank. Enable **Developer Settings → Webhooks → Audio Bytes**, set the URL to `https://<your-host>.ts.net/omi`, and set the interval to **5 seconds**.

The ingestion endpoint accepts `Content-Type: application/octet-stream`, `sample_rate=16000` (or omitted), and complete two-byte samples. Incoming WebSocket audio, transcript JSON, and WAV uploads are unsupported. The browser monitor still uses an outgoing WebSocket.

## Processing and recovery

- **Bounded processing:** one consumer handles at most 24 queued windows, roughly two minutes at the default core duration, plus one in-flight request. A full queue waits for capacity instead of evicting accepted windows. Incoming unsliced audio is capped at 30 seconds; overload returns HTTP `503` rather than allowing memory to grow indefinitely.
- **Ordered retries:** an initial Gemini request may be followed by up to three retries after `1`, `2`, and `4` seconds for HTTP `429`, server errors (`5xx`), and transient network failures. Permanent errors and exhausted retries count as dropped windows and appear in health telemetry. Later windows retain their order.
- **Session endings:** after more than eight seconds without audio, any remaining complete PCM16 samples are flushed, including tails shorter than one second. Audio overlap and transcript context reset between sessions.
- **Graceful shutdown:** `SIGINT` and `SIGTERM` stop ingress, flush remaining audio, and drain the STT queue before exit. A restart may take several minutes during an API outage; the launcher waits instead of forcing termination. Audio is held in RAM, so a crash, power loss, or forced kill can still lose pending work.
- **Speech windows:** low-energy cuts and boundary deduplication reduce clipping and repetition, but remain heuristics. RMS below the configured threshold skips the API call. Five-second uploads plus lookahead usually delay the first transcription request until the second upload, followed by API latency.

## Commands and status

```bash
~/omi/start.sh             # Gracefully restart, then verify readiness
~/omi/start.sh start       # Start if stopped
~/omi/start.sh stop        # Gracefully stop the tracked bridge
~/omi/start.sh status      # Check the tracked process and health
~/omi/.venv/bin/python3.15 ~/omi/omi_bridge.py --check-config
curl -fsS http://127.0.0.1:8788/health | ~/omi/.venv/bin/python3.15 -m json.tool
```

The launcher tracks `logs/omi_bridge.pid` and its process identity. It refuses to kill an unrelated process occupying either bridge port and reports startup success only after the new process responds to its health check.

`GET /omi` preserves Omi's required validation fields:

```json
{"status":"ok","is_setup_completed":true,"message":""}
```

It also includes telemetry. The separate **loopback-only** endpoint `http://127.0.0.1:8788/health` reports `gemini_key_configured`, `last_packet_age_sec`, `last_stt_success_age_sec`, `queue_depth`, `buffer_bytes`, retry/drop counters, `last_error`, the process PID, and the Python/OpenSSL versions. It is not served on the Funnel-backed port 8787. Check the JSON `status` and `workers_alive` fields for readiness; a recent successful STT request is the evidence that Gemini transcription is working.

## Audio, transcripts, and Hermes

| Output | Location |
| --- | --- |
| Live audio and transcript monitor | `http://127.0.0.1:8787/listen` |
| Last minute as WAV; up to five minutes retained in RAM | `http://127.0.0.1:8787/audio/last?seconds=60` |
| Immediate timestamped transcripts | `logs/omi_live_transcript.log` |
| Pause-grouped transcript paragraphs (updated after each line) | `logs/YYYY-MM-DD.md` |
| Bridge events, errors, retries | `logs/omi_bridge.log` |
| Launcher stdout/stderr | `logs/omi_bridge.console.log` |

The five-minute audio ring buffer captures audio before silence gating and holds about 9.6 MB at 16 kHz. It is separate from pending STT work. Bridge and immediate transcript logs rotate at 5 MiB with three backups; daily Markdown files remain available until removed. Runtime files, logs, `.env`, and `.venv` are Git-ignored.

Existing Hermes integration remains enabled: the wake word defaults to `hermes`, and accumulated transcripts are forwarded to the local `omi-research` webhook every 90 seconds. The optional integration uses `http://127.0.0.1:8644`; configure its endpoints in `.env` if Hermes runs elsewhere.

The existing Funnel configuration exposes the monitor and audio-download routes publicly, and ingestion authentication has not been added in this update.

## Verification

```bash
cd ~/omi
.venv/bin/python3.15 -m unittest discover -s tests -v
```

Tests use synthetic audio and mocked Gemini responses, without paid API calls. For a live check, speak a short phrase, confirm a new transcript line, then turn off Omi and confirm the final tail appears.
