# Operation and recovery

Run commands from the repository directory. `./omi` uses this repository's Python environment and configuration, regardless of the terminal's current environment. Explicit process environment variables override `.env`.

## Connect and read

1. Run `./omi start` and `./omi status`.
2. Connect the wearable in the Omi phone app.
3. Run `./omi captions`. Use `d` for diagnostics, `g` to follow live text, `/` to search, and `q` to close the viewer. `b` bookmarks a passage, `[` and `]` navigate bookmarks, `B` copies bookmarks, and `e` exports Markdown.

Service status distinguishes service health, phone connection, recently received audio, recognizer state and saving. A healthy idle service does not prove microphone quality or recognition accuracy. For structured output use `./omi status --json`.

The named Cloudflare tunnel is managed separately by the `com.alhinai.omi.tunnel` LaunchAgent. Check it with `launchctl print gui/$(id -u)/com.alhinai.omi.tunnel`. Tunnel logs are under `~/Library/Logs/OmiCloudflare/`; its credentials remain outside the repository. Tunnel setup on a new Mac is a separate provisioning step.

In the Omi app, choose Settings → Transcription → Cloud Provider → Custom (Live):

- URL: `wss://omi.alhinai.dev/v1/listen?token=<your stream token>`.
- Streaming parameters: `encoding: linear16`, `sample_rate: 16000`, `channels: 1`.
- Response mapping: `segments_path: segments`, fields `text`, `start`, `end`, and `text_path: text`.
- Disable Developer Settings → Webhooks → Audio Bytes.

Use the token from your existing `.env`; do not publish it. Other device sources can use the same PCM contract, described in [architecture](ARCHITECTURE.md).

## Select a recognizer

```bash
./omi backend deepgram
./omi backend whisper
```

The switcher validates the target and recovery configuration, serializes updates with the launcher, preserves unrelated `.env` bytes, restarts gracefully, and verifies health. Failed activation attempts recover the previous backend. Cloud switching with an active phone also verifies provider connection. Without phone audio, cloud recognition remains unverified until audio arrives. Power loss or forced termination cannot guarantee recovery.

Deepgram currently supports fixed English with an English-only publication allowlist. Audio leaves the Mac for the cloud. Whisper runs locally and requires native arm64 macOS, Python 3.14, its pinned requirements and the installed MLX Turbo/Silero assets. Model paths are configurable in `.env.example`. On a new Mac, create the optional worker with `python3.14 -m venv .venv-whisper`, then install `requirements-whisper.txt` using that environment's Python.

Restore the pinned assets into `~/Library/Application Support/OmiWhisper/`: Whisper `config.json` and `weights.safetensors` go in `whisper-turbo/`; Silero goes in `silero_vad.onnx`.

| Asset | Pinned source |
| --- | --- |
| Whisper Turbo FP16 | `mlx-community/whisper-large-v3-turbo`, revision `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb` |
| Silero VAD | `snakers4/silero-vad`, revision `066409f3d9db45a30d21dfcbf9147826ffdd7052`, file `src/silero_vad/data/silero_vad.onnx` |

The Silero SHA-256 is `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`. Existing installations need no model download.

## Optional Pipecat orchestration

Pipecat is opt-in within the Deepgram backend. Whisper remains unchanged. No Daily room, WebRTC server, LLM, TTS service or phone-app modification is required for this caption pipeline.

On this Mac, prepare a separate environment using the existing Python 3.14 worker interpreter:

```bash
.venv-whisper/bin/python -m venv .venv-pipecat
.venv-pipecat/bin/python -m pip install -r requirements-pipecat.txt -c requirements-pipecat.lock
```

On a new installation, use a standard Python 3.14 interpreter to create that environment. The Pipecat dependencies stay separate from the bridge and local Whisper environments.

Test on temporary ports with synthesized audio:

```bash
./omi dev smoke --backend deepgram --pipeline pipecat
.venv-pipecat/bin/python -B -m unittest discover -s tests -p test_pipecat_runtime.py -q
```

The smoke report identifies `pipeline: pipecat`, library versions, processed audio/transcription frame counts, provider model and successful JSONL/Markdown verification. This is a controlled pipeline check, not a live-wearable accuracy claim.

For a future live activation, set `OMI_DEEPGRAM_PIPELINE=pipecat` in `.env` while keeping `OMI_STT_BACKEND=deepgram`, then run `./omi restart` when interruption is acceptable. Set the pipeline back to `direct` and restart to return to the direct adapter. `./omi backend whisper` remains the local fallback. `./omi check`, health and viewer diagnostics show the selected orchestration after activation. No automatic provider/pipeline fallback is enabled.

Pipecat is pinned to 1.16.0 and Deepgram SDK to 7.19.0. The worker uses a small pinned-service subclass to forward raw result provenance and drain metadata in pipeline order and to prevent hidden reconnects from resetting source timestamps. Omi owns retry, bounded replay and immutable save recovery. Upgrade those packages together with the optional runtime tests and a fresh smoke check. `pipeline_start_sec` includes optional-process/framework startup; `provider_connect_sec` measures the SDK connection separately. Cold pipeline startup can delay the first caption, so this integration is not presented as a speed improvement.

[Pipecat Deepgram service documentation](https://docs.pipecat.ai/api-reference/server/services/stt/deepgram) describes the upstream service. The CLI/context-hub shown on the hackathon slide is development tooling and is not needed at runtime.

## Diagnose a problem

```bash
./omi check
./omi verify --public
```

`check` reads configuration, package metadata and local health without sending audio. `verify` observes authenticated captions and packet progress; `--public` checks that unauthenticated public caption access is denied. Its report distinguishes observed, failed and untested stages.

Viewer diagnostics (`d`) distinguish segment receipt age from estimated word-end delay. The starred word-end values use the bridge receipt clock and exclude upstream wearable delay. A growing `phone_delivery_errors` health counter means phone replies stalled; that connection closes for reconnection while the journal and caption viewers retain their own delivery path.

For a deliberate phrase check:

```bash
./omi verify --phrase "Blue lantern seven" --timeout 60
```

Wait for the subscription-ready message, then speak. Only a new matching final is checked in the journal, Markdown and a fresh Viewer. Existing terminal painting, physical microphone identity and pre-bridge delay require separate observation.

If the intended words are missing, first verify input with an intentional recording:

```bash
./omi dev capture --record --wait-for-enter
```

In a real terminal, press Enter and speak after the countdown and Recording message. Capture requires fresh advancing phone audio and ends after at most ten seconds. A private WAV is saved under `~/Library/Application Support/OmiDiagnostics/` with mode `0600`. Listen for the intended words before scoring recognition. There is no public audio-download endpoint.

The earlier diagnostic clip lacked the intended phrase, as confirmed by the user. Its recognition comparison is inconclusive; it does not establish a microphone fault or recognition failure.

## Stop, restart and history

`./omi stop` drains accepted audio and stops the bridge. `./omi restart` preserves history. Finals live in `logs/omi_transcript_events.jsonl` and `logs/YYYY-MM-DD.md`. Diagnostic process logs are separate. Keep `.env`, transcripts and model assets out of source control.

The journal is authoritative; Markdown is a readable projection. Unsaved finals and pending PCM are memory-only, so a complete process crash or power loss can lose them. Avoid history-reset commands during routine operation.

## Explicit developer tools

- `./omi dev test -q`: offline regressions.
- `./omi dev faults`: eight deterministic fault/corruption scenarios, without model/audio-fixture generation.
- `./omi dev smoke --backend whisper|deepgram`: temporary complete recognition pipeline; no production ports or notes are used. Cloud selection uploads synthesized or supplied test audio.
- `./omi dev benchmark --backends whisper,deepgram`: explicit paired synthetic benchmark; uploads test audio to the selected cloud provider.
- `./omi dev demo --recorded-audio`: labeled recorded-audio fallback through real local Whisper, using temporary ports and notes. A plain `demo` uses synthetic caption fixtures and performs no recognition.

The superseded extended Whisper fixture generator is retired in the [recovery archive](archive/README.md). The maintained paired benchmark is `./omi dev benchmark`; recorded demonstrations use the bundled synthesized `assets/demo.wav`. Reports describe their exact source/configuration and input; historical timings are not current performance guarantees.
