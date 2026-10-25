# Omi Audio-to-Transcript Bridge

A small Python + `aiohttp` bridge for Omi's **Audio Bytes** webhook: 16 kHz, mono, little-endian PCM16 audio in; Gemini transcripts to logs and a terminal viewer.

```text
Omi Audio Bytes (1-second uploads)
        ↓
Tailscale Funnel → POST /omi on 127.0.0.1:8787
        ↓
Bounded live-audio queue → persistent Gemini transcription WebSocket
        ↓
Instant partial captions (temporary) → finalized speech
        ↓
Transcript logs + daily paragraphs + terminal captions
```

## Setup

Standard CPython **3.15.x** is required. The launcher creates a project-local `.venv` and installs `requirements.txt`. The bridge and launcher reject other Python minor versions, prereleases, and free-threaded builds. The launcher checks the runtime before restarting a running bridge and refuses to overwrite an incompatible environment.

```bash
cd ~/omi
cp .env.example .env
# Edit .env: uncomment GEMINI_API_KEY and enter your key.
./run.sh
```

For a new environment, the launcher looks for `python3.15` on `PATH`, in Homebrew or python.org locations, and in uv's managed Python directory. Set `OMI_PYTHON=/absolute/path/to/python3.15` to choose the interpreter explicitly. On a Mac without Python 3.15, install it with `brew install python@3.15` first.

When upgrading an older environment, test the new interpreter and dependencies in a separate temporary environment first. Stop the bridge with `./run.sh stop` so queued audio drains, move the old `.venv` to a backup, and recreate `.venv` at its final path using `python3.15 -m venv .venv`. Reinstall dependencies and run the verification checks below before `./run.sh start`. Virtual environments contain absolute paths; moving a prebuilt environment into place is unsupported.

Configuration precedence is **process environment → `~/omi/.env` → defaults**. `GEMINI_API_KEY` and `GOOGLE_API_KEY` are accepted. A missing or placeholder key prevents startup. Local `.env` values use `KEY=value` syntax; quote values containing spaces.

Expose the ingestion service:

```bash
tailscale funnel --bg --https=443 http://127.0.0.1:8787
```

In the Omi app, leave **Custom Backend URL** blank. Enable **Developer Settings → Webhooks → Audio Bytes**, set the URL to `https://<your-host>.ts.net/omi`, and set the interval to **1 second**. Live mode forwards arriving audio immediately; it does not wait for an eight-second window.

The ingestion endpoint accepts `Content-Type: application/octet-stream` on `POST /omi`, `sample_rate=16000` (or omitted), and complete two-byte samples. It also supports **direct phone-to-bridge WebSocket PCM16 streaming** on `GET /omi` (with `Upgrade: websocket`) and `GET /omi/ws?sample_rate=16000` (compatible with Omi's **Settings → Transcription → Custom** live streaming WebSocket mode `PureStreamingSttSocket`): binary messages ingest raw 16 kHz mono PCM16 frames directly with zero cloud buffering, `{"type": "KeepAlive"}` is accepted, `{"type": "CloseStream"}` flushes the active turn, and finalized transcripts return `{"text": "..."}` to the client. Transcript JSON and WAV header uploads are rejected.

## Processing and recovery

The default is **`OMI_STT_MODE=live`**, using `OMI_GEMINI_LIVE_MODEL=gemini-3.5-transcribe-live`. The existing Gemini key stays on the bridge host. A persistent authenticated WebSocket forwards each received upload immediately as up-to-100 ms PCM frames; it does not deliberately wait between frames that have already arrived. One-second phone uploads remain the upstream input cadence. No additional SDK or dependency is needed.

- **Real partials:** `interimInputTranscription` replaces an ephemeral current hypothesis immediately. Final `inputTranscription` creates a normal persisted transcript with a stable `utterance_id`, `is_final: true` and `transcription_mode: "live"`. Partial captions never enter transcript files, daily notes, or replay history. Partial updates are coalesced into one replaceable slot per viewer so they cannot displace saved captions.
- **Bounded freshness:** unsent audio is limited to ten seconds and 256 packet records, favoring newer audio during overload. Stale unsent packets older than `OMI_MAX_WINDOW_AGE_SEC=45` are skipped during live operation. Every valid packet still enters the independent five-minute playback ring first. `/health` reports queued packets, queued bytes, capacity in seconds, connection state, reconnect count, and partial update count. In live mode, `core_window_sec` is zero and the old batch queue capacity is `null`.
- **Finalization & explicit turn boundaries:** `gemini-3.5-transcribe-live` is configured with `automaticActivityDetection: {"disabled": True}` and driven with explicit `activityStart` -> 100 ms PCM frames -> `activityEnd` + `audioStreamEnd` turn boundaries (waiting for `generationComplete` before opening the next turn). Quiet duration is measured in samples (`quiet_samples`), padding `max(0, 4800 - quiet_samples)` samples (`up to 300 ms`) of zero-PCM silence before `activityEnd` and tracking `injected_samples` so provider `voiceActivity.audioOffset` values map back to real input sample offsets. If a closing turn times out awaiting flush, it is isolated in `closing_turn` so delayed `inputTranscription`, `ACTIVITY_END`, or `generationComplete` events from the previous turn never overwrite or abort the next active turn.
- **Ingress tunnel latency (Cloudflare HTTP/2 vs. Tailscale Funnel):** Omi's cloud backend (`webhooks.py`) slices buffered audio into 1-second (`32,000` byte) HTTP POST requests and sends them strictly sequentially under a per-user lock while pinning resolved DNS IPs on every request (`safe_request_targets`). Because Tailscale Funnel publishes multiple IPv4/IPv6 relay records where cold secondary relays take `1.8–4.3s` per POST, sequential 1-second uploads over Funnel can fall behind real time (`~63%` speed) and build a `35–40s` queue inside Omi's cloud (`audio_bytes_queue`). Exposing `127.0.0.1:8787` via a Cloudflare HTTP/2 tunnel (`cloudflared tunnel --protocol http2 --url http://127.0.0.1:8787`) reduces per-POST latency to `0.15–0.19s`, draining Omi's cloud queue at `100%+` real-time speed (`0.0s` cloud backlog, `~2.5–3.8s` end-to-end spoken-to-screen).
- **Reconnects:** closed sockets and server `GoAway` notices reconnect with bounded backoff. Only unsent audio retries; uncertain already-sent speech is marked with a gap rather than replayed and potentially duplicated. There is no claim of lossless connection recovery or session resumption. Receipt/VAD metadata is bounded independently of retained audio, and source times remain estimates.
- **Shutdown:** accepted queued audio is sent before ending the stream, with a bounded wait for final text. Live worker shutdown is capped at 20 seconds; any known undelivered queued audio is reported as a gap. File failures are reported separately. All RAM-only limitations still apply on crashes or forced termination.
- **Latency:** `latency.receipt_to_first_partial_sec` measures first eligible audio receipt → first nonempty hypothesis for an utterance. `latency.receipt_to_final_sec` measures that same baseline → final recognition and output attempt. `receipt_to_transcript_sec` additionally requires successful authoritative persistence. Final latency includes the utterance's speaking duration and VAD pause; compare first-partial latency when assessing live responsiveness. These exclude phone/device buffering and network time before bridge receipt.

See [Google's live transcription protocol](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe). The model may revise partial words; only finalized text is intended for durable reading. Actual streaming access is verified by setup completion, and the local health endpoint becomes degraded if the provider connection is unavailable.

<details>
<summary>Optional batch compatibility mode (`OMI_STT_MODE=batch`)</summary>

The older eight-second `gemini-3.8-flash` path remains available for rollback and offline regressions. Its model, thinking, window, silence-gate and processing-deadline settings apply only in batch mode. Only one transcription path runs at a time.

- **Bounded processing:** one consumer handles at most 24 queued windows plus one in-flight request. Before processing, live windows waiting more than `OMI_MAX_WINDOW_AGE_SEC=45` seconds are skipped; the count cap remains a separate memory bound. Queue age starts at slicing/enqueue time, not physical capture time, and the currently running request can add latency. During live overload, a new window evicts the oldest queued window; incoming unsliced PCM is capped at 30 seconds by trimming the oldest pending samples. Valid live uploads return HTTP `200` under backlog pressure. Individual HTTP uploads are limited to 300 seconds of PCM (9.6 MB), independently of the pending buffer; larger requests return HTTP `413`. This favors fresh speech but deliberately loses older transcription audio. `windows_evicted` and `buffer_bytes_evicted` report the two loss types; `windows_dropped` also counts failed requests and buffer-trimming events (not a precise duration). Every valid live packet enters the separate playback ring first, where it remains until normal ring expiry. Shutdown drains pending work without eviction or age-based skipping; the per-window processing deadline still applies.
- **Gemini thinking:** `OMI_GEMINI_THINKING_LEVEL=low` is the default for latency-sensitive transcription. `medium` and `high` are available for more reasoning and may increase latency and token use; `minimal` is unsupported. Requests explicitly set `temperature: 0.0` and `maxOutputTokens: 160` as requested. The current API accepted these settings in a real speech test, although [Google's Gemini 3.8 migration guidance](https://ai.google.dev/gemini-api/docs/generate-content/whats-new-gemini-3.8) calls temperature deprecated. Neither setting guarantees factual transcription; token-capped, incomplete responses remain errors rather than accepted partial text. Low thinking does not guarantee that processing will keep pace with live audio.
- **Ordered retries:** an initial Gemini request may be followed by up to three retries after `1`, `2`, and `4` seconds for HTTP `429`, server errors (`5xx`), and transient network failures. Each attempt has a 12-second total timeout, five-second connect timeout, and ten-second read timeout. `OMI_WINDOW_DEADLINE_SEC=20` bounds the entire attempt/backoff cycle, including the final request, and prevents starting a backoff that exhausts the budget. Permanent errors, exhausted retries, and deadlines count as failed windows. Later windows retain their order.
- **Response validation:** Gemini replies must contain a complete, valid transcription response. Blocked, truncated, or malformed replies, including missing candidates, are reported as errors. A complete, unblocked `STOP` response reporting zero output tokens can legitimately contain no new speech. Gemini requests use an asynchronous HTTP client with bounded timeouts.
- **Session endings:** after more than eight seconds without audio, any remaining complete PCM16 samples are flushed, including tails shorter than one second. Audio overlap and transcript context reset between sessions.
- **Graceful shutdown:** `SIGINT` and `SIGTERM` stop ingress, flush remaining audio, and drain the STT queue before exit. A restart may take several minutes during an API outage; the launcher waits instead of forcing termination. Audio is held in RAM, so a crash, power loss, or forced kill can still lose pending work.
- **Speech windows:** the default eight-second core reduces request frequency by about 38% compared with five seconds, at the cost of added transcription latency. Sustained API slowness can still cause overload. Low-energy cuts and boundary deduplication reduce clipping and repetition, but remain heuristics. RMS below the configured threshold skips the API call. With one-second uploads, the bridge still needs roughly nine seconds of audio before the first transcription request, followed by API latency.


</details>

## Commands and status

```bash
~/omi/run.sh             # Gracefully restart, then verify readiness
~/omi/run.sh start       # Start if stopped
~/omi/run.sh stop        # Gracefully stop the tracked bridge
~/omi/run.sh status      # Check the tracked process and health
~/omi/run.sh clear       # Stop and clear saved bridge data/logs
~/omi/.venv/bin/python3.15 ~/omi/omi_bridge.py --check-config
curl -fsS http://127.0.0.1:8788/health | ~/omi/.venv/bin/python3.15 -m json.tool
```

The launcher tracks `logs/omi_bridge.pid` and its process identity. It refuses to kill an unrelated process occupying either bridge port and reports startup success only after the new process responds to its health check.

`GET /omi` preserves Omi's required validation fields:

```json
{"status":"ok","is_setup_completed":true,"message":""}
```

The response contains only those three fields. The separate **loopback-only** endpoint `http://127.0.0.1:8788/health` reports `gemini_key_configured`, `thinking_level`, `last_packet_age_sec`, `last_stt_success_age_sec`, `queue_depth`, `buffer_bytes`, retry/drop/eviction counters, `last_error`, the process PID, and the Python/OpenSSL versions. It returns HTTP **200** when ready and **503** when degraded, while `/omi` keeps its required validation response. It is not served on the Funnel-backed port 8787. A healthy idle process, including one with zero received audio, confirms readiness; it does not prove that speech has traveled through Gemini. A new transcript from known speech provides that evidence.

Local health also reports latency without changing any timeout or queue-age setting:

| Field | Meaning |
| --- | --- |
| `latency.receipt_to_transcript_sec` | First packet containing the utterance's voiced audio (or the batch window's core audio) received by the bridge → successful authoritative transcript write and completion of secondary output attempts. Includes buffering, queue wait, retries/backoff, and output time. |
| `latency.gemini_request_duration_sec` | One batch-mode Gemini attempt, including audio encoding, HTTP response, and parsing; empty in live mode. Includes failed/timed-out attempts; excludes queue wait and retry backoff. |
| `oldest_pending_audio_age_sec` | Time since receipt of the oldest remaining core audio, across unsliced buffers, queued windows, and the currently processed window (including retry backoff). Excludes processed overlap and playback history. `null` means no pending audio. |

All latency summaries contain `last`, `p50`, `p95`, and `samples`, using a bounded rolling history of the last 128 observations per metric since process start. Percentiles use nearest rank; before any observations, values are `null` and `samples` is zero. Receipt latency samples require a newly persisted, nonempty transcript; silence, gaps, batch duplicate-only results, and failed JSONL writes do not add samples. These measurements use monotonic receipt times and are unaffected by wall-clock adjustments. They do not include wearable/phone buffering or network delay before bridge receipt, so they are not physical capture-to-transcript latency. The optional batch stale-window policy measures age from enqueue time; live mode measures stale unsent packets from receipt time.

## Audio and transcripts

| Output | Location |
| --- | --- |
| Last minute as WAV; up to five minutes retained in RAM | `http://127.0.0.1:8787/audio/last?seconds=60` |
| Immediate timestamped transcripts | `logs/omi_live_transcript.log` |
| Transcript text plus audio/receipt/processing times | `logs/omi_transcript_events.jsonl` |
| Pause-grouped transcript paragraphs (updated after each line) | `logs/YYYY-MM-DD.md` |
| Bridge events, errors, retries | `logs/omi_bridge.log` |
| Launcher stdout/stderr | `logs/omi_bridge.console.log` |

The five-minute audio ring buffer captures audio before silence gating and holds about 9.6 MB at 16 kHz. It is separate from pending STT work. Bridge, immediate transcript, and JSON event logs rotate at 5 MiB with three backups; daily Markdown files remain available until removed. Runtime files, logs, `.env`, and `.venv` are Git-ignored.

Transcript timestamps estimate the audio's start time from packet receipt and PCM duration, capped at actual receipt time for burst or backfilled uploads. Without upstream timestamps, they cannot reconstruct true physical capture times or remove network delay. JSON events persist `audio_start_at`, `received_at`, and `processed_at` as epoch timestamps, with `time_basis: "estimated_from_receipt"`, plus `session_id` and `sample_offset` to identify each window's position; terminal caption events carry the same timing metadata. `received_at` records when the complete window arrived, while `processed_at` records transcription completion so a backlog's processing delay remains visible. The plain transcript keeps its `[YYYY-MM-DD HH:MM:SS] text` format using the estimated audio time.

JSONL is the authoritative record: each new transcript or gap has a unique `event_id` and `type`. A JSONL write failure prevents secondary output for that event; plain text and Markdown errors are isolated from each other and from Gemini success/context. `output_errors` and `last_output_error` expose failures, and local health becomes degraded until a later event writes all outputs successfully. This is synchronous append logging, not a crash-safe transaction or automatic repair system; rotated JSONL backups must be retained to rebuild historical views.

Lost transcription coverage is counted in seconds as `lost_audio_sec_failed`, `lost_audio_sec_stale`, `lost_audio_sec_queue_evicted`, and `lost_audio_sec_buffer_evicted`, excluding overlap. Each loss also attempts to write a structured gap event and a `[GAP: ...]` marker to plain text, Markdown, and the terminal viewer. These report untranscribed audio duration, not a measurement of how much intelligible speech was present. Gaps are written when detected, so file order can differ from source-time order when older queued audio completes later; use source timestamps and offsets to locate them. Silence gating does not count as lost audio. Gap events are diagnostic records.

The existing Funnel configuration exposes the monitor and audio-download routes publicly, and ingestion authentication has not been added in this update.

For terminal captions, run `~/omi/tui.sh`; see [terminal viewer instructions](TUI.md) for controls and display timing.


## Verification

For an isolated end-to-end streaming check using real Gemini, run `.venv/bin/python3.15 scripts/live_smoke_test.py`. It synthesizes a known phrase, starts a temporary local bridge, sends one-second uploads, and verifies partial captions plus final-only persistence. It uses temporary logs and does not touch the running wearable stream; normal API charges apply.

```bash
cd ~/omi
.venv/bin/python3.15 -m unittest discover -s tests -v
```

The consolidated tests use synthetic audio and mocked Gemini responses, without paid API calls. They check hard processing deadlines, live-only stale skipping, duration/gap accounting, output failures, configuration, and graceful shutdown. Mutable runtime state lives in one `BridgeState` instance; configuration stays separate.

For an **opt-in integration check using real Gemini**, turn Omi off, let pending work drain, and run:

```bash
.venv/bin/python3.15 scripts/smoke_test.py
# Include public HTTPS/Funnel ingress:
.venv/bin/python3.15 scripts/smoke_test.py --endpoint https://mini.tail287a74.ts.net/omi
# Supply your own short mono PCM16/16 kHz recording:
.venv/bin/python3.15 scripts/smoke_test.py --wav /absolute/path/speech.wav --expect "blue lantern seven"
```

The default check uses macOS `say`/`afconvert` to generate “The bridge test is working. Blue lantern seven.” It sends one-second PCM uploads, waits at least eight seconds for the idle tail, and allows up to 120 seconds for the expected phrase to appear in **newly appended** transcript text. It also requires a new successful Gemini window. Normal Gemini API charges apply; this check makes real requests and writes its test speech to the usual transcript/daily files. It prints no unrelated transcript text. `--health-url` and `--transcript-log` select the bridge's local status endpoint and log; run this command on the bridge host. Concurrent audio or pending work causes refusal unless `--allow-active` explicitly permits mixing.

This controlled injection verifies the selected HTTP ingress, bridge, Gemini, and transcript output. To verify the **physical wearable → BLE → phone → Funnel** path, enable the Omi Audio Bytes webhook, note the current end of the transcript, and speak a distinct phrase into the wearable. Leave it on for at least 15 seconds after finishing so the phone uploads can deliver the phrase, then confirm a new transcript entry. Power Omi off and confirm its final received tail appears after the idle flush. The bridge can flush received audio; it cannot recover audio still buffered upstream when the device disconnects. A controlled injection alone does not verify those physical/device hops.

The browser, magazine, image generation, and browser research tools have been removed. Use `~/omi/tui.sh` for live captions and `tail -f ~/omi/logs/omi_live_transcript.log` for the transcript log.

`./run.sh clear` gracefully stops the bridge, deletes its transcript history, daily notes, generated editions, audio captures, and rotated runtime logs, then exits without restarting. Use `./run.sh start` when ready to record again. It preserves `.env`, `.venv`, the launcher lock, and files belonging to separate services. Already-open TUI windows can retain text in memory; reopen the viewer for a blank display. Upstream phone audio buffers are outside this reset.
