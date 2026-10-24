---
name: omi-live-streaming
description: "Use when streaming Omi wearable audio or live transcripts. Real-time STT, WebSocket, and conference tracking."
version: 1.1.0
platforms: [macos, linux]
metadata:
  hermes:
    tags: [omi, wearable, transcription, stt, gemini, tailscale, webhooks, conference]
---

# Omi Live Streaming & Real-Time Transcription

Use when connecting an Omi wearable to Hermes for live audio/transcript streaming, real-time STT, or live conference/meeting tracking.

## Architecture & Data Flow

1. **Ingress Tunnel (Public HTTPS on Port 443)**:
   - Omi's backend (`api.omi.me`) dispatches developer webhooks from Google Cloud, not directly from the phone.
   - Omi's SSRF validator (`safe_request_target`) rejects RFC 1918 private IPs, loopback, and Tailscale CGNAT IPs (`100.64.0.0/10`), and pins the resolved public DNS IP at delivery time.
   - Expose the local bridge on standard HTTPS port 443 via **Tailscale Funnel** (`tailscale funnel --bg --https=443 http://127.0.0.1:8787`) or a single `cloudflared tunnel --protocol http2 --url http://127.0.0.1:8787` process. Avoid running redundant tunnel processes.

2. **Local Bridge (`~/omi_bridge.py` on `0.0.0.0:8787` — starter template in `templates/omi_bridge.py`)**:
   - Bind the async `aiohttp` server to `0.0.0.0:8787` (run via Hermes's bundled Python environment) so it is reachable via local loopback, Tailscale IP (`100.75.54.79`), and public Tailscale Funnel (`https://<tailnet-host>/omi` on port 443) simultaneously.
   - Return `200 OK` (`{"status":"ok","is_setup_completed":true,"message":""}`) immediately before any downstream processing so Omi's 3-failure circuit breaker never trips and auto-disables the webhook.
   - Support three ingestion modes and live monitoring on the same port:
     - **Raw PCM16 Audio Bytes Webhook** (`Developer Settings -> Webhooks -> Audio Bytes`, `application/octet-stream` with `?sample_rate=16000&uid=...`): Fires continuously even when the user's 300-minute Omi cloud transcription quota is exhausted. Set the Omi app interval to `5` seconds (avoids 1s request storms that can trip the circuit breaker on brief network blips while matching the 5s real-time window).
     - **Continuous WebSocket Audio Stream** (`Settings -> Transcription -> Custom (Real-time streaming)` pointed to `wss://<funnel-host>/omi`): Omi's `PureStreamingSttSocket` streams raw 16kHz PCM16 binary frames continuously over WebSocket with zero batching delay and expects OpenAI-schema JSON (`{"text": "...", "segments": [...]}`) back to display live words in the Omi mobile app while feeding the local transcript log.
     - **JSON Transcript Segments Webhook** (`Developer Settings -> Webhooks -> Real-time Transcript`): Omi sends `{"segments": [...]}` when Omi cloud or Gemini Live (`gemini-2.5-flash-native-audio-preview-12-2025` via `v1alpha BidiGenerateContent`) is active.
     - **Live Browser Audio & Transcript Monitor (`/listen` and `/listen-ws`)**: Broadcast incoming PCM16 frames and transcript JSON over WebSocket to browser clients (`AudioContext` at `16000` Hz) for live audio monitoring.

3. **Dynamic `[1s Pre + 5s Core + 1s Post]` Sliding-Window Gemini STT**:
   - Wait until `6` seconds (`5s` core + `1s` post-lookahead) of new 16kHz PCM16 audio (`32,000 bytes/sec`) is in the buffer.
   - Scan `±0.8s` around the `5.0s` mark in `50ms` steps (`find_dynamic_cut_offset`) to find the lowest-RMS `100ms` micro-pause (breath/syllable valley) so cuts never slice through a word.
   - Construct the `~7s` WAV window as `[1s pre-overlap from end of previous core] + [~5s core up to micro-pause] + [1s lookahead after micro-pause]`, and advance the audio buffer by only `core_cut` (so the `1s` lookahead becomes the start of the next `5s` core window, and the last `1s` before `core_cut` becomes the next `1s` pre-overlap).
   - Check RMS energy (`rms < 80`) on the core slice to skip silent windows at zero API cost.
   - Send the WAV buffer to `gemini-3.8-flash` (or `gemini-3.5-flash-lite`) with the last 24 words of the prior transcript in the prompt so boundary words are never clipped, then deduplicate any repeated boundary n-grams in Python.
   - **Idle tail flush on disconnect/power-off**: Track `last_packet_time` on every incoming audio packet. If no packet arrives for `>8` seconds while `>=1` second of audio remains in `audio_pcm_buffer`, automatically flush and transcribe that final tail and clear `audio_pre_tail` so powering off the wearable never strands the final sentence or leaks stale audio into the next session.

4. **Permanent Daemonization (macOS `LaunchAgent`)**:
   - Run `~/omi_bridge.py` under a user `LaunchAgent` (`~/Library/LaunchAgents/com.alhinai.omi-bridge.plist` with `RunAtLoad=true` and `KeepAlive=true`) alongside persistent Tailscale Funnel (`tailscale funnel --bg --https=443 http://127.0.0.1:8787`). Keeping the local bridge up 24/7 across reboots and crashes prevents `502 Bad Gateway` responses that would otherwise trip Omi's 3-failure webhook circuit breaker.

5. **Uniform Transcript Log Format**:
   - Write lines to `~/.hermes/logs/omi_live_transcript.log` strictly as:
     `[YYYY-MM-DD HH:MM:SS] <clean single-paragraph text>`
   - Do not include `TRANSCRIPT:` or `GEMINI_STT:` prefixes in the transcript log.

## Omi Mobile App Configuration Rules & Pitfalls

- **Populate or clear all 4 Developer Webhook fields before saving**: `DeveloperModeProvider.saveSettings()` executes `Future.wait([w1, w2, w3, w4])` across `Conversation Events`, `Real-time Transcript`, `Audio Bytes`, and `Day Summary` simultaneously — even for toggles that are turned OFF. If any disabled field holds a stale or private URL in its hidden text controller, the entire Save fails with `"Failed to save. Please check your connection."` Toggle each field on briefly to clear or set a valid public `https://` URL in all four, then turn off unused toggles and tap Save.
- **Edit a character in the target URL field if a prior read failed**: Omi tracks `_unreadWebhooks` on page open and skips saving any field whose text matches its initial cached value. Delete and re-type the last character of the URL so `hasUnsavedWebhookChanges` turns true and the `Save` button appears in the top-right app bar.
- **Light Mode Save button visibility**: In Light Mode, the `Save` button in the top-right `AppBar` of `Developer Settings` renders white-on-white; instruct the user to tap the top-right corner of the header bar (to the right of the "Developer Settings" title, below the battery icon) and confirm the `"Settings saved!"` snackbar.
- **Keep `Custom Backend URL` empty**: Never paste the webhook URL into `Custom Backend URL` above the Webhooks section — doing so redirects all Omi app API calls away from `https://api.omi.me` and breaks authentication and live listening.
- **Reconnect the live-listen socket after enabling webhooks**: Omi's backend (`routers/listen/runtime.py`) evaluates `transcript_webhook` and `mobile_standard_batch` only when the `/v4/listen` WebSocket first connects. Force-close and reopen the Omi app after saving webhook settings so a fresh session starts.
- **Reset Omi's webhook circuit breaker after local bridge restarts**: Restarting `~/omi_bridge.py` while Omi is mid-POST causes connection/502 errors that trip Omi's automatic circuit breaker (`_DEV_FAILURE_THRESHOLD = 3`), silently disabling the webhook (`HTTP POST /omi` stops arriving). When `omi_bridge.log` shows no new POST requests after a bridge restart, instruct the user to toggle `Audio Bytes` (or `Real-time Transcript`) OFF and back ON in `Developer Settings -> Webhooks`, which calls `/enable` and resets `record_dev_webhook_success` immediately.

## Live Query & Conference Tracking Rules

- **Clone external repos locally instead of fetching files one-by-one**: When inspecting an upstream codebase (such as `BasedHardware/omi`) across multiple files, run `git clone --depth 1` once to a local directory and use `read_file` / `search_files` rather than issuing repeated `curl` calls to `raw.githubusercontent.com`.
- **Never sleep when answering live "what was just said?" queries**: Read the tail of `~/.hermes/logs/omi_live_transcript.log` immediately and respond on the first turn without calling `sleep`. Waiting for the next 30-second audio flush stalls the user's terminal response.
- **Cache the full Sessionize schedule JSON locally**: When assisting at a conference on Sessionize, fetch `https://<event>.sessionize.com/api/schedule` once to disk and query it with Python to filter by current timestamp, room, and speaker rather than scraping HTML repeatedly.
- **Narrow down session choices one question at a time**: When helping the user choose between concurrent breakout talks, use `clarify` with sequential single-question tradeoffs tailored to their technical stack.
