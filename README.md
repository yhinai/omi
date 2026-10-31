# Omi live captions

Stream audio from the Omi phone app through this bridge to Gemini. Read live captions in your terminal and saved transcripts in daily Markdown.

```text
Omi wearable → phone → /v1/listen → Gemini
                                  ↓
                         JSONL + daily Markdown
                                  ↓
                           /listen → tui.sh
```

## Setup

Requires standard CPython **3.15.x** with the GIL enabled. On macOS, install Python and Cloudflare Tunnel with `brew install python@3.15 cloudflared`.

```bash
cd ~/omi
cp .env.example .env
# Edit .env: set GEMINI_API_KEY and OMI_STREAM_TOKEN.
./run.sh start
```

Skip copying `.env.example` if you already have `.env`. Environment variables override `.env`; `GOOGLE_API_KEY` is also accepted. The launcher creates `.venv` and installs `requirements.txt`. Set `OMI_PYTHON` to choose a Python 3.15 interpreter. To replace an incompatible `.venv`, stop the bridge, move the old environment aside, and recreate it at its final path.

In Omi, select **Settings → Transcription → Cloud Provider → Custom (Live)**:

- WebSocket URL: `wss://omi.alhinai.dev/v1/listen?token=<OMI_STREAM_TOKEN>`
- Request configuration: `request_type: "streaming"`, the same URL, and parameters `encoding: "linear16"`, `sample_rate: 16000`, `channels: 1`.
- Response schema: `segments_path: "segments"`, text/start/end fields `"text"`, `"start"`, `"end"`, and `text_path: "text"`.
- Disable **Developer Settings → Webhooks → Audio Bytes**.

The named Cloudflare Tunnel publishes this Mac at `omi.alhinai.dev`. Its LaunchAgent (`com.alhinai.omi.tunnel`) starts at login and restarts the connector if it exits. Check it with `launchctl print gui/$(id -u)/com.alhinai.omi.tunnel`. Tunnel logs are in `~/Library/Logs/OmiCloudflare/`; its private token is stored outside the repo in `~/Library/Application Support/OmiCloudflare/`. Gemini credentials stay on the bridge host. `/listen` is unauthenticated, so anyone who can reach it can read captions.

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

Unsent audio is bounded to ten seconds and 256 packet records. Newer audio wins during overload; dropped or uncertain speech is recorded as a gap. Reconnects use backoff and retry only unsent audio. Shutdown allows up to 20 seconds to drain live transcription. Crashes or forced termination can lose RAM-only audio.

Settings in `.env.example` cover the live model, maximum unsent-audio age, idle timeout, and speech threshold. Only live transcription is supported. Timing is estimated from bridge receipt and excludes buffering before the bridge.

## Checks and troubleshooting

```bash
curl -fsS http://127.0.0.1:8788/health | .venv/bin/python3.15 -m json.tool
.venv/bin/python3.15 -m unittest discover -s tests -q
.venv/bin/python3.15 scripts/smoke_test.py
```

Health is loopback-only: HTTP 200 means ready; 503 means degraded. Check `stream_connected`, `direct_ws_connected`, packet/transcript age, queue depth, output errors, and latency. A healthy idle bridge does not prove speech recognition.

Tests use synthetic data and local fake servers without API charges. The single smoke check uses **real Gemini and normal API charges** in a temporary bridge. It verifies partial captions, finalized JSONL, daily Markdown, and shutdown without touching the running phone stream or normal logs. It synthesizes a known phrase using macOS speech; alternatively:

```bash
.venv/bin/python3.15 scripts/smoke_test.py --wav /absolute/path/speech.wav --expect "your spoken phrase"
```

The WAV must be mono PCM16 at 16 kHz and at most 30 seconds. To verify the wearable and tunnel too, speak a distinct phrase into the connected wearable and confirm it in `tui.sh`.

If captions stop: check the phone's tunnel URL and matching stream token, inspect health and `logs/omi_bridge.log`, then reconnect the wearable. The bridge accepts audio only at `/v1/listen`; `/listen` serves caption JSON only.
