# Omi transcription status

Updated 2026-10-31. **Whisper-only cleanup is active and healthy; saved history was preserved.**

## Current design

Omi wearable → phone → Cloudflare → `/v1/listen` → CPU speech detection + GPU Whisper → JSONL, daily Markdown, and `/listen` → `tui.sh`.

| File | Responsibility |
| --- | --- |
| `bridge.py` | Audio/caption connections, health and application lifecycle |
| `stt.py` | Private worker protocol, bounded buffering, save retries and worker recovery |
| `whisper.py` | One prewarmed MLX model, CPU Silero, speech boundaries and draft/final decoding |
| `common.py` | Shared audio types, queue limits and validated recognition policy |
| `storage.py` | Journal writing/rotation, uncertain-write deduplication, replay and Markdown |
| `tui.py` | Caption display, stable paragraph placement, selection, search and replay |

Recognition is English-only, original Turbo FP16, fused attention and greedy decoding. Drafts start after a 250 ms minimum snapshot; subsequent drafts require 200 ms of new audio and a 200 ms start interval. Received low-VAD audio of about 512 ms closes a turn. Approximately one second of pre-roll protects opening words. A continuous turn is capped at 20 seconds.

The bridge/viewer use Python 3.15; the isolated MLX worker uses Python 3.14. Models stay outside Git in `~/Library/Application Support/OmiWhisper/`. Gemini and its configuration have been removed.

## Why this model and policy

The earlier 18-clip comparison favored original FP16 for correct-word latency and final accuracy. Its selected paced profile produced a median first draft of 546 ms and retrospectively stable first two correct words at 841 ms, with 2/214 normalized word errors. The fastest pruned profile drafted at 466 ms but stabilized those words at 944 ms and made 4/214 errors. These are small-corpus observations, not general accuracy guarantees or immutable-final timings.

The prior integrated adapter evaluation saved all 16 speech clips, produced no text on two controls, and reported 2/214 normalized errors. Median first caption was 444 ms from first bridge ingress. The actual wearable phrase was verified in JSONL and Markdown. These historical timings exclude wearable/BLE/phone delay, public transport, and terminal painting.

Research and raw evidence: [deployment comparison](/Users/alhinai/omi-model-bench/MAC_DEPLOYMENT.md), [integrated results](/Users/alhinai/omi-model-bench/integrated-whisper-summary.json).

## Latest cleanup

- Removed unused draft-session state, a duplicate turn-limit export and dummy lists used as counters.
- Centralized unsent audio, packet count, replay memory and packet-size limits.
- Aligned the launcher dependency check with `aiohttp>=3.14.5,<4`.
- Worker stderr is drained into a bounded 8 KiB memory buffer. Diagnostic logs contain allowlisted runtime failure metadata, never raw stderr, PCM or recognized text.
- Flush and shutdown expose completion/timeout results. A phone close with a timed-out drain returns WebSocket code 1013; accepted audio remains retained subject to the documented queue/recovery limits.
- Extracted persistence into `storage.py`. JSONL is written before caption publication or Markdown; uncertain writes reuse event IDs to avoid duplicate journal lines.
- Drafts after a pause keep their timestamp, paragraph identity and position when confirmed.

## Verification

**124 tests passed**, along with shell syntax checks and the real-model isolated smoke check (draft captions, final JSONL and daily Markdown). The one-phrase smoke check observed a first draft at 384 ms and a final 711 ms after the last VAD-active frame receipt; these are Mac-side observations, not a new end-to-end benchmark. Stop/start activation reported HTTP 200/ok with the local worker ready and no pending-save or output errors. Existing-history prefix checksums matched across startup. `bridge.py` is now 705 lines, down from 822 before this pass. Tests cover source continuity, short/long turns, backpressure, worker restarts, failed saves, partial writes, flush timeouts, bounded private diagnostics, dependency versions and viewer behavior.

Manual check: `.venv/bin/python3.15 scripts/smoke_test.py`. It uses a temporary bridge and leaves normal saved history untouched.

## Operation and limits

Use `./run.sh stop` followed by `./run.sh start` to preserve history when deploying. `restart` intentionally clears history/logs; this behavior remains as previously requested. Reopen `./tui.sh` after viewer changes. Model revisions and download instructions are listed below.

Unsaved finals and replay audio remain memory-only. They survive the tested worker-restart paths, but a full bridge crash, forced termination or power loss can lose them. This cleanup does not add a durable outbox or change transcript formats. JSONL is authoritative; Markdown is a best-effort readable projection. The 20-second speech cap can split a word; word-aligned stitching is not implemented. Broader noisy-wearable, long-session and sustained thermal testing remains. `/listen` is still unauthenticated.

## Pinned models

- Whisper: `mlx-community/whisper-large-v3-turbo`, revision `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`; `config.json` and `weights.safetensors`.
- Silero: `snakers4/silero-vad`, revision `066409f3d9db45a30d21dfcbf9147826ffdd7052`; `src/silero_vad/data/silero_vad.onnx`; SHA-256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.

After preparing `.venv-whisper` as described in README, download the pinned files:

```bash
.venv-whisper/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
from urllib.request import urlretrieve
import hashlib
root = Path.home() / 'Library/Application Support/OmiWhisper'
(root / 'whisper-turbo').mkdir(parents=True, exist_ok=True)
for name in ('config.json', 'weights.safetensors'):
    hf_hub_download('mlx-community/whisper-large-v3-turbo', name,
        revision='a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb', local_dir=root/'whisper-turbo')
urlretrieve('https://raw.githubusercontent.com/snakers4/silero-vad/066409f3d9db45a30d21dfcbf9147826ffdd7052/src/silero_vad/data/silero_vad.onnx', root/'silero_vad.onnx')
assert hashlib.sha256((root/'silero_vad.onnx').read_bytes()).hexdigest() == '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'
PY
```
