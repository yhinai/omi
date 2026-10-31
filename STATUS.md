# Whisper Turbo deployment status and decision

Updated 2026-10-31 15:57 PDT. **Whisper Turbo is active, healthy, and verified through the actual wearable/phone path.** Existing saved history was preserved using stop/start. The phone URL and public caption endpoint work with the local backend.

## Why this configuration

We compared the same English recordings across Gemini, Nemotron, FluidAudio, Whisper, and then several Apple Silicon runtimes and Whisper optimizations. We selected **the original Whisper large-v3 Turbo FP16 checkpoint on MLX GPU, fused encoder/decoder attention, CPU Silero VAD, fixed English, greedy decoding, a 250 ms minimum first snapshot and 200 ms target decode starts**.

The decision favors quickly obtaining correct words while retaining the full model's accuracy. Filling every CPU/GPU/Neural Engine is not the goal: the allocation with the lowest measured useful latency wins.

| Paced configuration | Median first draft | First two correct words stopped changing* | Normalized word errors |
| --- | ---: | ---: | ---: |
| Full model, 500 ms buffer / 500 ms updates | 703 ms | 1,201 ms | 2/214 |
| **Selected full model, 250 ms / 200 ms, early voice gate** | **546 ms** | **841 ms** | **2/214** |
| Full model, event-driven early voice gate | 497 ms | 888 ms | 2/214 |
| Pruned model, event-driven early voice gate | 466 ms | 944 ms | 4/214 |

*Calculated retrospectively from recorded update traces; this is not a live confirmation algorithm or final-transcript latency. Nine profiles ×18 clips produced 162 paced results. There were 16 speech clips (one human speaker plus synthetic English) and two silence/noise controls. All gated profiles produced no text on those two controls. The selected profile had raw WER 6/215 (2.8%) and normalized WER 2/214 (0.9%); normalization additionally equates WebSocket/web socket and Mister/Mr. This is a small pilot, not a general accuracy guarantee.

A separate matched-three-speech warm inference screen, after one second of audio was already available, favored MLX GPU (179 ms) over public WhisperKit with GPU encoder/ANE-eligible decoder (242 ms), whisper.cpp Metal (354 ms), and FP16 WhisperKit with ANE-eligible encoder/decoder (379 ms). These are inference microbenchmarks, not caption delays; Core ML device eligibility does not prove per-operator placement. Some Core ML cold starts took minutes. The mixed whisper.cpp Core ML startup timed out, so it supplied no warm comparison.

Four-bit quantization saved little latency and added a word error. Pruning delivered earlier unstable drafts but more final errors. Arbitrarily shortening encoder context caused severe recognition errors and was rejected. Whole-decoder graph compilation failed warmup and was excluded. The selected worker keeps full attention context. Argmax Pro remains optional and untested; its ready-made streaming/confirmed-text API might reduce custom code, but no measured M6 advantage justifies requiring it.

Full research, alternatives, raw results and reproduction scripts: [benchmark report](/Users/alhinai/omi-model-bench/MAC_DEPLOYMENT.md), [paced results](/Users/alhinai/omi-model-bench/m6-streaming-summary.json), [native runtime screen](/Users/alhinai/omi-model-bench/native-runtime-summary.json).

## What changed

- `stt.py` connects the existing async bridge to a private worker process, retains unresolved PCM for restart replay, bounds unsent audio, and preserves source offsets and phone-session ownership.
- `whisper.py` keeps one GPU model warm, runs causal speech detection on one CPU thread, and decodes immutable snapshots serially. Audio intake and speech segmentation continue while inference runs.
- Drafts go only to live captions. Only a separate decode of a closed, VAD-confirmed turn reaches the existing final-transcript writer, phone replies, JSONL and daily Markdown.
- About 512 ms of received low-VAD audio closes short turns without a minimum speech-duration requirement. Missing packets alone do not imply silence. Explicit close and idle timeout also flush accepted audio.
- Approximately one second of pre-roll protects opening words. Continuous speech is split at 20 seconds to bound memory/context. This hard limit can split a word; the adapter preserves samples but does not implement word-aligned overlap stitching.
- A separate `.venv-whisper` uses native Python 3.14 and pinned MLX/ONNX dependencies; the bridge stays on Python 3.15. Model assets live outside the repository. Cloud transcription support has been removed; only local Whisper is supported.
- The phone URL, Cloudflare hostname, `/v1/listen`, `/listen`, `tui.sh`, and saved-file formats remain the same.

## Verification so far

- Existing baseline: 119 tests passed before edits.
- Regression suite including new short-turn, source continuity, pre-roll, concurrent-final, duplicate-final, worker-crash replay and backend-configuration tests: **131 passed**.
- Real-model isolated bridge smoke check passed: live captions, phone ingress, saved final JSONL and daily Markdown. First draft 386 ms; last VAD-active frame to saved final 745 ms; turn closure to final 244 ms. These are one-phrase Mac-side observations, not end-to-end wearable measurements.
- Fresh **18-case integrated evaluation passed**: all 16 speech clips produced finals and phone replies; both controls produced no drafts/finals; zero gaps; JSONL and daily Markdown verified. Final normalized errors: **2/214 (0.9%)**. Error locations differ from the isolated benchmark; the noisy name “James” became “Jane.”
- Integrated median first caption: **444 ms from the first ingress packet**. Health p50 diagnostics: GPU decode 203 ms; turn closure to final 356 ms; last VAD-active frame receipt to final 911 ms. These use different timing origins from the earlier model-only table: 100 ms packets contain audio already captured before bridge receipt. They exclude microphone/BLE/phone delay, public transport and terminal painting. Receipt-to-final includes the duration of the utterance and is not a speech-end delay.
- Actual wearable verification passed after activation. The requested phrase was saved as two consecutive finals, “Blue Lantern 7” and “Local whisper is working,” matching the spoken text after digit normalization. Both finals are in JSONL and daily Markdown. Pauses can create separate turns; paragraphs group those turns for reading.
- Phone reconnected automatically; production health is 200/ok with backend `whisper`, worker ready, zero reconnects, zero audio gaps and zero output errors at verification. Saved-history prefix checksums matched across activation. The public `wss://omi.alhinai.dev/listen` caption WebSocket connected successfully.
- Raw integration evidence: [corpus summary](/Users/alhinai/omi-model-bench/integrated-whisper-summary.json), [per-clip results](/Users/alhinai/omi-model-bench/integrated-whisper-results.json), [wearable verification](/Users/alhinai/omi-model-bench/wearable-whisper-verification.json).

## Models and reproduction

The installed model assets match the benchmark checkpoint hashes. Local asset metadata is in `~/Library/Application Support/OmiWhisper/manifest.json`.

- Whisper: `mlx-community/whisper-large-v3-turbo`, revision `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb` (`config.json`, `weights.safetensors`).
- Silero VAD: `snakers4/silero-vad`, revision `066409f3d9db45a30d21dfcbf9147826ffdd7052`, `src/silero_vad/data/silero_vad.onnx`, SHA-256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.

On a new machine, after creating the worker environment as described in README:

```bash
.venv-whisper/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
from urllib.request import urlretrieve
root = Path.home() / 'Library/Application Support/OmiWhisper'
(root / 'whisper-turbo').mkdir(parents=True, exist_ok=True)
for name in ('config.json', 'weights.safetensors'):
    hf_hub_download('mlx-community/whisper-large-v3-turbo', name,
        revision='a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb', local_dir=root/'whisper-turbo')
urlretrieve('https://raw.githubusercontent.com/snakers4/silero-vad/066409f3d9db45a30d21dfcbf9147826ffdd7052/src/silero_vad/data/silero_vad.onnx', root/'silero_vad.onnx')
import hashlib
assert hashlib.sha256((root/'silero_vad.onnx').read_bytes()).hexdigest() == '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'
PY
```

Manual verification uses one command:

```bash
.venv/bin/python3.15 scripts/smoke_test.py
```

## Operating and rollback

Use `./run.sh stop` followed by `./run.sh start` to activate backend changes while preserving saved history. **`restart` intentionally clears history and logs**, as previously requested. Phone configuration is unchanged. Rollback requires a prior Git revision; there is no backend selector.

The short wearable phrase is verified. Broader validation remains: noisy real conversation, rapid consecutive turns, long uninterrupted speech, repeated real reconnects, sustained power and thermal behavior. The automated long-turn test proves PCM continuity at the hard boundary, not word-perfect recognition there. The earlier microbenchmarks exclude wearable/BLE/phone/tunnel/UI delay.

## Caption placement update (2026-10-31)

The viewer now chooses the draft's paragraph on its first update. After more than eight seconds since the previous audio ends, a new session, or a paragraph-length boundary, the draft begins its own timestamped paragraph. Its row key, timestamp, placement and trailing spacing remain the same when its final arrives; it no longer appears at the end of an old paragraph and then jumps to a new one. Short continuations stay inline. Audio-start metadata is used when available, otherwise the first draft arrival supplies the timestamp. This is a presentation fix; it does not declare draft words confirmed or persist them early.

Regression coverage verifies timestamp/row identity through revisions and finalization, stable positioning in a full following viewport, long-utterance end timing, wrapping, hidden timestamps and existing replay ordering. Full suite: 134 tests passed. Reopen `./tui.sh` to load the viewer update. The transcription backend was not restarted and saved history was not modified by this update.

## Current simplification (2026-10-31)

Removed the optional Gemini implementation, its provider-specific tests, API-key aliases, cloud settings and backend selection. Whisper is the only runtime recognizer. Provider-independent partial/persistence regressions were retained in the bridge tests; the smoke-check regression now uses a fake local worker. The two environments remain necessary for the bridge and MLX worker. Saved history and model assets are retained.

Recent save-retry changes retain unsaved finals and their source PCM until the journal accepts them. Full bridge/process-crash durability, word-safe long-speech boundaries, complete end-to-end timing and authenticated caption access remain follow-up work. Earlier benchmark measurements are historical, not fresh measurements of this revision.

Whisper-only verification: 119 tests passed after removing cloud-specific tests and migrating shared coverage. A real local smoke check passed (draft captions, final JSONL and daily Markdown); first draft 400 ms and VAD-active-frame receipt to final 709 ms for that one synthetic phrase. Activated with stop/start; live health is 200/ok with local Whisper, and saved-history prefix checksums are preserved. No Gemini source, tests, API settings or cloud provider options remain in executable project code.

## Script names (2026-10-31)

Renamed `omi_bridge.py` to `bridge.py` and `omi_tui.py` to `tui.py`; imports, launchers and tests follow the new names. Log/PID filenames retain their existing names to preserve history and process tracking. There are no compatibility wrappers for the old script names. User commands remain `./run.sh` and `./tui.sh`.

Rename verification: 119 tests passed in an isolated copy of the renamed tree; shell syntax checks passed. The service was activated with stop/start and health is good. Saved-history prefix checksums match the files recorded before startup.

Short module names: `stt.py` handles worker lifecycle and recovery; `whisper.py` runs recognition; `common.py` contains shared audio types and policy. The corresponding regression file is `tests/test_stt.py`. Imports and subprocess launch paths use these names.

Short-module verification: 119 tests passed against the renamed tree. Stop/start activated the new imports and worker launch path; health is good, and saved-history prefixes are unchanged.

Recognition script renamed to `whisper.py`. Updated the subprocess launch path, test imports and documentation. All 119 tests passed in isolation; the renamed worker warmed successfully after stop/start, health is good, and saved-history checksums are preserved.
