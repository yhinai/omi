# Architecture and ownership

One bridge serves one active audio source. Recognition is replaceable; ingress, saving and caption delivery are shared.

```text
/v1/listen → validated PCM + session/sample clock → selected backend
                                                     ↓
                             finalized event → JSONL journal
                                                     ├→ daily Markdown
                                                     └→ /listen → terminal viewer
```

## Component boundaries

| Component | Responsibility |
| --- | --- |
| `config.py` | One non-executing dotenv parser; process values override file values |
| `backend_settings.py` | Backend protocol and validated Deepgram settings |
| `bridge.py` | Authentication, source ownership, backend selection, event creation, persistence ordering and caption fan-out |
| `deepgram_stt.py` | Cloud transport, provider timestamps, bounded recovery and pending-save retry |
| `pipecat_stt.py` + `pipecat_worker.py` | Optional private-process Pipecat frame pipeline and official Deepgram STT service; reuses the existing Deepgram source/save state machine |
| `stt.py` + `whisper.py` | Optional local worker lifecycle, replay, speech segmentation and MLX decoding |
| `common.py` | Audio types, bounded queues, quality measurements and local recognition policy |
| `storage.py` | Authoritative journal, idempotent commit, replay and daily projection |
| `tui.py` | Caption consumption, display, search, selection and bookmarks |
| `diagnostics.py` | Explicit bounded local recording; inactive by default |
| `cli.py` + `omi` | Consistent command surface; delegates to existing lifecycle and verification implementations |

Deepgram mode does not import the local Whisper adapter or load its model. The viewer reads shared configuration without importing the bridge. Provider-specific reconnect/drain logic remains in each adapter because the transports have different semantics. The optional Pipecat process acknowledges audio only after service processing and returns complete terminal metadata before an epoch drains. Its SDK cannot reconnect silently under an existing source clock. There is no additional hosted service, queue server, database or automatic provider fallback.

## Audio and caption contract

- Public bridge port: `8787`. Audio ingress: `/v1/listen`. Caption JSON: `/listen`.
- Loopback-only health: `http://127.0.0.1:8788/health`.
- Binary audio is raw little-endian PCM16, mono, 16 kHz, with complete even-length samples. A WAV container, compressed Opus or different advertised format is rejected. Arbitrary raw bytes cannot independently prove their actual format or physical source.
- Use bearer authorization or the phone-compatible stream-token query parameter. Captions use a separate optional reader token, defaulting to the stream token. Never forward local credentials automatically to a remote viewer URL.
- Send `{"type":"CloseStream"}` to drain. A same-remote reconnect can replace the existing source; stale frames and closes cannot affect its replacement. This is a single-device policy, not multi-tenant identity isolation.
- Phone results retain the Omi `Results`/`segments` compatibility format. Its speaker label is not speaker identification. Caption events include partials, immutable finals, gaps, status and replay boundaries.

## Guarantees and limits

Accepted audio carries source-session and sample positions. Unsent PCM is bounded to ten seconds / 320,000 bytes and 2,048 packet records; overload records explicit gaps. Partials are provisional and are not saved as final transcripts.

The journal commits before publication and Markdown projection. Recognition results awaiting a failed save are retried without decoding again. Original provider provenance and source identity remain attached. The journal is authoritative; filesystem durability across power loss is not guaranteed.

Caption clients have bounded queues. Slow readers reconnect and replay reliable events rather than silently losing finals. Viewer replay uses the same paragraph rules as live consumption. Source offsets prevent duplicate finals across supported worker recovery paths.

Phone reply overflow or send failure closes that connection with code 1013 and increments `phone_delivery_errors`; journal and caption-viewer delivery continue. Reconnecting the official phone app does not guarantee replay of every missed phone reply.

The Whisper fallback uses a separate Python 3.14 process and one prewarmed MLX GPU model. Its eight-second turn cap can split a word. Speculative quiet-settle decoding is reused only after the turn closes with matching speech bounds and no resumed voice. Both recognizers retain pending state in RAM; full bridge crashes can lose unsaved work.

Operational status separates service availability, phone connectivity, audio freshness, provider readiness and saving. Latencies distinguish provider connection, partial/final arrival and persistence. They start from observable Mac/provider stages and exclude unknown wearable/BLE/phone/tunnel delays.

`receipt_to_final_sec` includes segment duration. Provider word-end metrics estimate delay using the original packet receipt clock; `final_word_packet_to_result_sec` measures the containing packet's observed receipt to result handling. None establishes acoustic-to-screen latency. See the [stage audit](OPTIMIZATION.md) for measurements and limits.

## Verification and change process

`./omi dev test -q` runs offline regressions. CI checks those tests plus shell syntax and whitespace. Real-model/cloud checks are separate, explicit and use temporary bridges. The fault runner exercises eight distinct scenarios; it no longer nests another full unit run or synthesizes unused WAVs.

Dependency installs use the pinned bridge requirements and constraints. The optional Whisper environment is pinned separately. Runtime compatibility checks still permit supported aiohttp 3.x versions; fresh installs use the verified lock.

`/Users/alhinai/vexo-captions` on the Mac is the active deployment. The former Mini deployment is stopped and its Omi autostart/tunnel jobs are disabled. The original checkout and uncommitted state are preserved in the private Mac recovery package described in the [Mac handoff](HANDOFF_MAC.md). Earlier external merged trees and extended experiments remain in the recovery archive. [Archived narratives](archive/README.md) and [evidence receipts](evidence/README.md) retain their original scope. A receipt from a different source hash is historical evidence, not verification of the current build.
