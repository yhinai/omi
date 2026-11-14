# Omi DevKit captions on this Mac

This independent deployment captures an Omi DevKit directly over Bluetooth, decodes
its audio, and feeds the existing authenticated PCM bridge. It does not require the
Omi phone app or route audio through Mini. Deepgram remains a cloud service; Whisper
is the explicit local alternative. No raw microphone recording is enabled.

## Independent deployment

Checkout: `~/omi-captions`, local branch `omi-devkit`.
Audio/captions: loopback port **8797**. Health: loopback port **8798**.
The Vexo deployment keeps ports 8787/8788 and is not changed.
Private credentials and transcripts are separate. Immutable Whisper/Silero model
assets are shared from the existing local OmiWhisper asset directory. Each deployment
has its own newly created Python environments; environments were not copied.

The fixed defaults were changed consistently in the copied bridge, launcher, viewer,
verifier, doctor and backend switcher. Use this checkout's commands for this deployment.
The same source/session ownership, bounded queues, finalized save retries, journal,
Markdown, authenticated captions and Pipecat/Whisper engines are retained.

## Connect

Disconnect your own DevKit from the phone app, keep it powered on and near the Mac.
Set `OMI_DEVICE_ID` in the private `.env` to its verified macOS Bluetooth UUID.
Never choose another person's wearable or connect to the first scan result.

```sh
cd ~/omi-captions
./omi start
./omi source omi
```

Or specify the verified UUID using `./omi source omi --device UUID`.
The source runs in the foreground. Ctrl+C stops notifications, disconnects BLE,
drains accepted PCM, sends CloseStream and waits for transcription drain.
In another terminal run `./omi captions`, `./omi status` or `./omi check`.
Stop capture before `./omi stop` or selecting `./omi backend whisper|deepgram`.
No login auto-start or implicit recognizer fallback is installed.

The microphone is active while audio notifications are subscribed. With Deepgram
selected, audio is sent to that company; tell people nearby before enabling it.
Disconnecting stops Mac capture; device-side behavior depends on its firmware.

## Device adapter

`omi_device.py` follows Omi's BLE protocol with Bleak. It verifies the configured
peripheral UUID and audio service, reads codec and available device information before
subscribing, and supports PCM16/16 kHz (codec 0) and Opus (20/21).
Older 8 kHz/mu-law firmware is rejected instead of being mislabelled as 16 kHz audio.

A notification has a little-endian 16-bit counter and an 8-bit fragment index.
Counters advance per notification; index 0 starts a codec frame. Reassembly is bounded
to 4096 bytes. Duplicates are counted, missing notifications or invalid fragment order
end the source session, and frames are never silently filled with invented speech.
The final pending frame is flushed on normal stop. Opus decoding uses the existing
PyAV-backed decoder, preserving actual sample counts for 10/20 ms packets.
The Omi shim reuses the capture queue/decoder/sender engine in `vexo_source.py`.

## Validation

Offline core tests, capture tests, fault checks and isolated recognizer smokes verify
software behavior. A physical device test must separately establish identity, received
audio, an intentional spoken phrase, matching caption/journal/Markdown output, source
provenance, loss counters and graceful stop. Do not call this deployment hardware-verified
until that check has completed. A device not advertising may still be connected to the
phone, switched off, discharged, or out of range.

## Upstream protocol references

Inspected BasedHardware/omi revision: `c4e2466bb7131b34f0b97a3ddb867a73fdf05ed6`.

- https://github.com/BasedHardware/omi/blob/c4e2466bb7131b34f0b97a3ddb867a73fdf05ed6/sdks/device/PROTOCOL.md
- https://github.com/BasedHardware/omi/blob/c4e2466bb7131b34f0b97a3ddb867a73fdf05ed6/omi/firmware/devkit/src/transport.c
- https://github.com/BasedHardware/omi/blob/c4e2466bb7131b34f0b97a3ddb867a73fdf05ed6/sdks/python/omi/decoder.py

The adapter is a local protocol integration, not a claim that Omi's Python SDK validates
fragmented packets or provides this application's save/recovery behavior.

## Current setup checks

Core discovery: 444 tests passed, two optional Pipecat runtime tests skipped. The two
runtime tests passed separately. Eleven capture checks passed, including synthetic
Omi PCM/Opus through the shared capture engine with no serial access. Eight fault
scenarios passed. Isolated Pipecat/Deepgram and Whisper smokes each produced a final
verified in JSONL and Markdown. A physical DevKit 1 was subsequently verified: model DEVKIT, Based Hardware,
firmware 1.0.4, Opus codec 20 (10 ms frames), battery 67% at connection. A deliberately
spoken test sentence was saved exactly and verified in authenticated caption replay,
JSONL, daily Markdown and viewer layout, with Omi BLE source and Pipecat provenance.
This verifies this input/save path, not broad recognition accuracy or battery endurance.

The first physical runs hit capture queue pressure and explicitly broke the source
session rather than concealing the discarded audio. The Omi path now permits bounded
20% catch-up when more than five PCM frames wait, preserving every sample and keeping
both queues bounded. A subsequent check of the initial 5% catch-up still observed an overflow. The final
adjustment gives the 10 ms DevKit 100-frame queues (Vexo uses 50 for its 20 ms frames)
and allows bounded 20% backlog catch-up. The oldest-frame limit remains two seconds;
sequence loss still breaks the session. Physical stability after this adjustment is
reported separately from the successful test-phrase check. The Vexo deployment
and its active capture were left unchanged.
