# Omi live captions

Stream audio from the Omi phone app through this bridge to local Whisper Turbo (English). Read live captions in your terminal and saved transcripts in daily Markdown.

```text
Omi wearable → phone → /v1/listen → Whisper Turbo (MLX GPU)
                                  ↓
                         JSONL + daily Markdown
                                  ↓
                           /listen → tui.sh
```

## Files

- `bridge.py` — phone ingress, caption delivery, saved transcripts and health.
- `stt.py` — worker lifecycle, audio buffering, replay and save retries.
- `whisper.py` — local GPU recognition and CPU speech detection.
- `common.py` — shared audio types and recognition policy.
- `tui.py` — terminal caption viewer; launch with `tui.sh`.
- `run.sh` — start, stop and inspect the service.

## Setup

Requires standard CPython **3.15.x** with the GIL enabled. On macOS, install Python and Cloudflare Tunnel with `brew install python@3.15 cloudflared`.

```bash
cd ~/omi
cp .env.example .env
# Edit .env: set OMI_STREAM_TOKEN. Recognition runs locally.
./run.sh start
```

Skip copying `.env.example` if you already have `.env`. Environment variables override `.env`. The launcher creates `.venv` and installs `requirements.txt`. Set `OMI_PYTHON` to choose a Python 3.15 interpreter. To replace an incompatible `.venv`, stop the bridge, move the old environment aside, and recreate it at its final path.

In Omi, select **Settings → Transcription → Cloud Provider → Custom (Live)**:

- WebSocket URL: `wss://omi.alhinai.dev/v1/listen?token=<OMI_STREAM_TOKEN>`
- Request configuration: `request_type: "streaming"`, the same URL, and parameters `encoding: "linear16"`, `sample_rate: 16000`, `channels: 1`.
- Response schema: `segments_path: "segments"`, text/start/end fields `"text"`, `"start"`, `"end"`, and `text_path: "text"`.
- Disable **Developer Settings → Webhooks → Audio Bytes**.

The named Cloudflare Tunnel publishes this Mac at `omi.alhinai.dev`. Its LaunchAgent (`com.alhinai.omi.tunnel`) starts at login and restarts the connector if it exits. Check it with `launchctl print gui/$(id -u)/com.alhinai.omi.tunnel`. Tunnel logs are in `~/Library/Logs/OmiCloudflare/`; its private token is stored outside the repo in `~/Library/Application Support/OmiCloudflare/`. Recognition runs locally on this Mac; no cloud transcription credentials are needed. `/listen` is unauthenticated, so anyone who can reach it can read captions.

## Local recognizer

The bridge uses Python 3.15; the isolated Whisper worker uses native arm64 Python **3.14** because its MLX wheels target that runtime:

```bash
python3.14 -m venv .venv-whisper
.venv-whisper/bin/python -m pip install -r requirements-whisper.txt
```

The worker expects the original FP16 MLX Turbo `config.json` and `weights.safetensors` in `~/Library/Application Support/OmiWhisper/whisper-turbo/`, and Silero `silero_vad.onnx` alongside that directory. These are already installed on this Mac. For a new installation, use the pinned checkpoint revisions and download instructions in [STATUS.md](STATUS.md). Override locations with `OMI_WHISPER_PYTHON`, `OMI_WHISPER_MODEL`, and `OMI_WHISPER_VAD` if needed.

One prewarmed GPU model decodes English with fused attention; one CPU thread runs speech detection. It starts drafts after a 250 ms minimum snapshot and targets 200 ms decode starts. Received silence of roughly 512 ms closes a turn; no minimum speech length is required. Drafts remain provisional. Only a fresh decode of closed, VAD-confirmed audio is saved. PCM continues entering a separate buffer while finals decode.

The worker retains approximately one second of opening audio and caps a continuous turn at 20 seconds. This hard boundary bounds latency and memory, but can split a word during uninterrupted speech; validate long real conversations before treating segmentation as word-accurate. Missing packets are not silence: an explicit close or the eight-second idle timeout flushes a disconnected stream. Unresolved source audio is replayed after a worker restart, with sample offsets used to suppress duplicate finals.

See [STATUS.md](STATUS.md) for the decision, measured results, deployment status, and limitations.

## Daily use

```bash
./run.sh start       # Start if stopped
./run.sh restart     # Drain audio, clear saved data/logs, restart, and check health
./run.sh stop        # Drain audio and stop
./run.sh status      # Check the tracked process and health
./tui.sh             # Open terminal captions
./run.sh clear       # Stop and delete saved bridge data and logs
```

The launcher refuses to stop unrelated processes occupying the ports. `restart` (also the default when no action is given) clears saved transcripts, daily notes, captures, and logs before starting again. `clear` performs the same cleanup and stays stopped. Both preserve configuration and the Python environment. Use `start` to start without clearing history.

The viewer shows partial text immediately and retains finalized text for reading. It reconnects and replays up to 200 recent events; older history stays in your saved files. Expired replay cursors can leave gaps.

After a pause of more than eight seconds, or a new session, the first draft starts a new timestamped paragraph. Confirmation updates that paragraph in place, preserving its timestamp and position. Short continuations stay with the current paragraph. Draft timestamps use audio timing when supplied, otherwise the first draft's arrival time. Reopen `tui.sh` after viewer updates to load the new code.

| Key | Action |
| --- | --- |
| `?` / Esc | Help / dismiss |
| ↑ / ↓, Page Up / Down, mouse wheel | Scroll |
| Space | Hold the view and release the mouse for text selection; press again to resume |
| `g` / End | Follow live captions |
| `/`, `n`, `N` | Search, next match, previous match |
| `y` | Copy the latest or selected paragraph using macOS clipboard |
| `m` | Toggle wheel scrolling and native text selection |
| `t`, `f`, `d` | Toggle timestamps, focus, or connection details |
| Ctrl+L | Redraw |
| `q` / Ctrl+C | Quit |

Use `./tui.sh --mouse select` for native selection, `--sync-output off` for terminal rendering problems, or `--url wss://omi.alhinai.dev/listen` for a remote bridge. Run `./tui.sh --help` for options. After viewer changes, reopen it; after backend changes, use `stop` followed by `start` to preserve saved history.

## Saved files and behavior

| File | Purpose |
| --- | --- |
| `logs/omi_transcript_events.jsonl` | Authoritative final transcripts, timing metadata, and gap events; seeds caption replay |
| `logs/YYYY-MM-DD.md` | Readable transcript grouped into paragraphs |
| `logs/omi_bridge.log` | Connection and error diagnostics |
| `logs/omi_bridge.console.log` | Launcher stdout/stderr |

New plain-text transcript logs are no longer written. Old files are not converted; the current launcher removes them with other history on `restart` or `clear`. JSONL and bridge logs rotate at 5 MiB with three backups; daily Markdown stays until removed. Configuration, logs, and `.venv` are Git-ignored.

Partials are temporary; only finals and diagnostic gaps are saved. JSONL writes happen before caption publication and Markdown updates. File errors degrade health; a later successful output clears the error. Markdown is a readable projection, not a transactional backup.

Unsent audio is bounded to ten seconds and 256 packet records. Newer audio wins during overload; dropped or uncertain speech is recorded as a gap. Worker reconnects use backoff and replay unresolved audio; recognized-but-unsaved finals are retried in source order. Shutdown allows up to 20 seconds to drain live transcription. Crashes or forced termination can lose RAM-only audio.

Settings in `.env.example` cover local model paths, maximum unsent-audio age, and idle timeout. Only local English live transcription is supported. Timing is estimated from bridge receipt and excludes buffering before the bridge.

Phone segment `start`/`end` values use received PCM sample positions relative to the current phone connection. They reset on reconnect and describe estimated utterance bounds, including trailing silence, rather than exact word boundaries. JSONL retains the original `sample_offset` and `sample_end_offset`; late results from an older phone connection are not sent to the new one.

## Checks and troubleshooting

```bash
curl -fsS http://127.0.0.1:8788/health | .venv/bin/python3.15 -m json.tool
.venv/bin/python3.15 -m unittest discover -s tests -q
.venv/bin/python3.15 scripts/smoke_test.py
```

Health is loopback-only: HTTP 200 means ready; 503 means degraded. Check `stt_backend`, `model`, `stream_connected`, `direct_ws_connected`, packet/transcript age, queue depth, output errors, and latency. A healthy idle bridge does not prove speech recognition. `pending_finals`, `gap_causes`, and `turns_without_text` expose turn completion and gap diagnostics. `latency.receipt_to_send_sec` measures queue delay; `latency.turn_close_to_final_sec` measures recognition after the turn closes. Gap JSONL records include a `detail` cause when available; energy-detected audio duration is not a count of missing spoken words.

Tests use synthetic data and a fake local worker without loading a model. The single manual smoke check runs real local Whisper in a temporary bridge. It verifies partial captions, finalized JSONL, daily Markdown, and shutdown without touching the running phone stream or normal logs. It synthesizes a known phrase using macOS speech; alternatively:

```bash
.venv/bin/python3.15 scripts/smoke_test.py --wav /absolute/path/speech.wav --expect "your spoken phrase"
```

The WAV must be mono PCM16 at 16 kHz and at most 30 seconds. To verify the wearable and tunnel too, speak a distinct phrase into the connected wearable and confirm it in `tui.sh`.

If captions stop: check the phone's tunnel URL and matching stream token, inspect health and `logs/omi_bridge.log`, then reconnect the wearable. The bridge accepts audio only at `/v1/listen`; `/listen` serves caption JSON only.
