# Wearable pipeline audit — 14 November 2026

For this captioning product, the strongest current design is **Vexo USB → capture and PCM decode on the attached Mac → local bridge → Nova-3 → captions and journal**, with Whisper as an explicit local alternative. Keep Pipecat when its frame pipeline is useful for the hackathon or future interaction; the evidence does not show that adding Pipecat accelerates transcription.

This audit covers source, controlled synthetic replay, and read-only observation of both deployments. It does not establish an absolute fastest configuration, physical microphone accuracy, wireless battery life, or production capacity.

## Targets and deployment boundaries

- Audited base: Omi `5ebaf98fb3f934ee0c9d4df8a0d2b9ad1ff269ce`. Final changes were reconciled with the published Vexo integration `ed7cd88` and its migration verification update `e1ac8e2`.
- Vexo SDK: [`1e4291e2fa4896b1f603d3d4f0b009c483425da0`](https://github.com/VexoAI-Org/vexo-open-kit/tree/1e4291e2fa4896b1f603d3d4f0b009c483425da0).
- Original Mini deployment: `/Users/alhinai/omi`, Pipecat/Deepgram. It stayed running; this audit did not replace its files or configuration.
- New Mac deployment, owned by the migration session: `/Users/alhinai/vexo-captions`. Its chat metadata says `/Users/alhinai/repo`, but that directory did not exist during inspection. Its integration was changing during the captured observation and was subsequently published; source hashes accompany the runtime receipt.
- Mac USB host: Apple M4 Pro, 24 GiB. The Mini reports Apple M6, 16 GiB. Hardware names alone do not establish either recognition latency or capacity.
- Chrome released `/dev/cu.usbmodem1101`; the other session's Python capture process subsequently owned it. This audit never opened the physical serial port, restarted that deployment, or uploaded its microphone audio.

```text
Vexo band 43 -- USB / sequenced Opus --> Mac capture worker
                                          |
                                  decode into PCM16
                                          |
                           authenticated local /v1/listen
                                          |
                      selected recognizer: Deepgram or Whisper
                                          |
                       journal acceptance → captions + Markdown
```

The PCM boundary preserves both recognizers. Raw Opus sent directly to Deepgram is a valid alternative, but it needs its own packet-loss and timestamp handling and cannot be inserted into this PCM pipeline by changing an encoding flag.

## Every stage: evidence and remaining validation

| Stage | What is established | Optimization and next validation | Status |
| --- | --- | --- | --- |
| Microphone and placement | Real audio reaches the Mac deployment; longer human test phrases were only partly recognized in the migration session | Compare near/far placement and natural speech in quiet/noise against a held-out, manually checked reference. Measure clipping and level before changing gain | Partial |
| Firmware and power | Earlier device session reported firmware 1.1.0 and a nearly empty charging battery | Keep USB power for the demo. Measure actual wireless runtime separately; no firmware image or power trace was analyzed | Partial |
| Band encoding | SDK contract: mono 16 kHz, 20 ms Opus packets, 16-bit sequence counter | Keep native packet cadence; higher motion rates do not improve audio recognition. Sequence wraps every 1,310.72 seconds | Source verified |
| USB transport | Python owned the correct Vexo port; runtime grew by 100 packets over a two-second observation | One device owner; detect loss, age, stream takeover, disconnect and stalled reads. Never infer USB throughput from the nominal serial baud rate | Runtime verified, fault tests partial |
| Bluetooth | SDK supports BLE and shared USB/BLE settings | Measure body occlusion, distance, reconnection, packet loss and battery before choosing BLE for the demo | Unmeasured |
| Host buffering | SDK serial reader permits 5,000 queued frames; new frames can be dropped when full | Maintain a much shorter downstream queue and oldest-frame age limit. A large queue can hide stale captions rather than provide useful resilience | Source verified |
| Opus decode | SDK can conceal gaps, ignore late packets and skip large holes; these policies change the apparent timeline | Reset decoder per source generation; count corruption, loss and concealment separately. The Mac adapter currently ends a session on loss and performs a second validation decode | Source verified; offline codec receipt separate |
| PCM ingress | Existing endpoint validates declared PCM16/16 kHz/mono and owns one active source | Preserve sample positions and source identity; carry source capture timing if physical latency is required. PCM bytes cannot prove which microphone generated them | Tested contract |
| Host → recognizer | Mac receipt-to-send p50 1 ms; p95 4–7 ms in captured rolling windows | Keep the recognizer on the USB host for this demo. Measure the complete route before relocating to a remote worker | Runtime verified |
| Pipecat orchestration | Real official Deepgram service, ordered frames and acknowledged private IPC | Avoid loading unrelated VAD/LLM/TTS services. Cold subprocess initialization matters at session start; keep the active session healthy rather than optimizing an idle benchmark alone | Synthetic/runtime verified |
| Recognition | Nova-3, interim results, `no_delay=true`, endpointing 300 ms | Benchmark on actual band speech before changing model or endpointing. Nova-3 fits captions; Flux changes both protocol and turn semantics and is appropriate to evaluate for a speaking agent | Controlled comparison; acoustic quality unresolved |
| Provider recovery | Bounded PCM replay and explicit epochs prevent silent timestamp resets | Preserve allowlisted provider error/status classification through Pipecat IPC; introduce jitter and admission control before many simultaneous reconnects | Existing tests; scaling unresolved |
| Saving | Live final-to-persist p50 1–2 ms, with no pending saves/errors in observations | Keep journal-before-publication. Normal storage cost does not justify a broad rewrite; failed gap writes still need a bounded retry design | Runtime/tested, known gap |
| Caption delivery | Slow external caption readers reconnect/replay; phone replies had a silent queue-drop path | New patch explicitly closes a failed phone connection with 1013 and counts delivery errors. This makes loss visible; it does not replay all missed finals into the official phone app | New isolated regressions |
| Display/export | TUI has bounded history, replay, search and bookmarks; an artificial 8,000-character unbroken Unicode token took about 320 ms to wrap | New word-lag diagnostics and unique private exports. Replace quadratic wrapping before broad-language workloads; physical receipt-to-paint remains unmeasured | Partial |
| Whisper alternative | Original worker remains available; the migration session completed a new-Mac real-model smoke and an actual switch to Whisper and back | Retain explicit switching and history checks; fallback is not automatic seamless failover. This audit did not repeat the live switch | Verified by isolated checks and migration evidence |
| Multiple wearers | Current bridge owns one recognizer and one global source timeline | Introduce authenticated device/session isolation before increasing capacity. Running more callers against the present endpoint is not a multi-device implementation | Design limit established |

The serial source header mentions firmware 1.1.3's five-second stalled-reader behavior, while the contract describes a different session rule. That behavior was not established on this physical firmware, so it is not used as a reliability guarantee here.

An offline probe on the actual Mac's installed capture environment passed 150 protocol vectors, 108 USB byte-partition cases and sequence/decode checks. The public 796-packet fixture decoded to exactly 15.920 seconds including the decoder's flush tail. Its first output was 304 samples rather than 320, with the remaining 16 samples emitted at flush: requiring every individual decode call to return exactly 640 bytes would reject valid audio. Validation plus decoding took approximately 0.047–0.048 ms p50 and 0.056–0.059 ms p95 per packet. Removing the adapter's extra validation pass would save only about seven microseconds in this probe and is not a useful latency priority. This is codec validation, not a physical microphone test.

## Measured latency: what the numbers mean

The original `receipt_to_final_sec` starts at a result segment's initial packet. It includes speech duration. A five-second value therefore does **not** establish five seconds of caption delay.

New metrics preserve the old contract and add:

- `provider_word_end_to_partial_sec` and `provider_word_end_to_final_sec`: provider word-end positions interpolated against the containing packet's receipt clock. These are estimates and exclude unobserved capture, USB/BLE, and upstream buffering.
- `final_word_packet_to_result_sec`: observed receipt of the packet containing the final word to result handling, without word-position interpolation.
- TUI diagnostics expose the word-end estimates with an asterisk and a receipt-clock explanation.

Provider result handling, journal acceptance, socket delivery and screen paint are separate stages. A genuinely acoustic end-to-end number requires a known signal and a capture/display timing witness; health alone cannot supply it.

The updated build's isolated real-provider check emitted three interims and one final, verified both JSONL and Markdown, and exercised the new metrics: estimated word-end-to-partial p50 209 ms and word-end-to-final 641 ms for that one synthesized clip. A separate real-model Whisper check also passed. These are controlled checks on this checkout, not measurements of the Mac migration's live microphone.

### Controlled comparison

Four sequential configurations used the same three synthesized English clips (33 reference words) and a digital-silence control. Each row contains three speech trials, not a robust population estimate. All 16 trials drained successfully with no detected transcription gaps; each configuration made one normalized lexical error. No private microphone audio was used.

| Pipeline | Input packet | Endpointing | First partial p50 from first feed | Last final p50 relative to synthesized clip end |
| --- | ---: | ---: | ---: | ---: |
| Direct Deepgram | 100 ms | 300 ms | 1.120 s | +0.304 s |
| Pipecat/Deepgram | 100 ms | 300 ms | 1.312 s | +0.303 s |
| Pipecat/Deepgram | 20 ms | 300 ms | 1.186 s | +0.273 s |
| Pipecat/Deepgram | 20 ms | 100 ms | 1.434 s | +0.306 s |

These runs do not prove a winner: order, cold start, network variation and the small corpus are confounders. They support leaving endpointing at 300 ms pending a real-band comparison. Use 20 ms naturally arriving packets without intentionally waiting to aggregate 100 ms. Changing the setting to 100 ms did not improve final timing in this sweep.

Reproduce with `scripts/benchmark_stt.py --backends deepgram --pipeline pipecat --chunk-ms 20 --endpointing-ms 300 --report PATH`. Deepgram mode uploads synthetic clips and incurs provider usage. Configuration overrides apply only to that benchmark process; it does not connect to the live ingress or alter `.env`.

Five HTTPS-root handshake probes on each host gave median cumulative TLS-ready times of approximately 162 ms on Mini and 311 ms on Mac. These include DNS/TCP and describe startup connections, not authenticated STT or steady-state recognition. They show why “remove one hop” is a hypothesis to measure, not proof that one host has a faster route to Deepgram.

## Lean changes prepared in this checkout

1. Explicit phone delivery failure and reconnection instead of silently losing replies under queue pressure or send errors.
2. Word-relative latency measurements that preserve the original packet clock across splitting, replay epochs and committed prefixes.
3. Benchmark packet size, orchestration and endpointing overrides with real-time pacing and complete short tails.
4. Preserve URLs, email addresses and case-sensitive provider text when saving; normalize whitespace without inserting sentence boundaries.
5. Reserve unique 0600 caption exports, including two exports within the same second.

These changes are isolated from the running Mini and Mac deployments. The migration session should reconcile them against its source identity/USB adapter changes before any restart.

Verification after reconciliation: 443 distinct tests passed (433 in the standard environment, two optional real-Pipecat tests in its environment, and eight capture tests against matching source hashes on the Mac), all eight fault scenarios passed, and isolated Pipecat/Deepgram and real-model Whisper checks verified saved output again. The earlier audit-only build passed 432 distinct tests. Shell syntax and whitespace checks passed. Published results retain their source scope.

## Extreme efficiency and scale

Optimize usefulness first: show partials immediately; save immutable finals; use a tap/bookmark to select useful context; run any summarization or action generation after a deliberate trigger or a finalized boundary. The current product has no need for an LLM call per interim revision or audio packet. Gestures should reference source session/sample positions and be deduplicated; do not connect actions to provisional text.

For one wearer, keep the USB owner, decoder, bridge and UI on the attached Mac. Keep motion at the lowest rate needed by the chosen gesture. Decoder warmup should finish before the microphone is enabled. Avoid automatic tuning based on a single recognition error or fixed loudness threshold.

The current Deepgram adapter deliberately accepts Nova-3 English only. Supporting other languages requires coordinating provider settings, publication filters and recognition tests; changing one environment value does not make this build multilingual.

For multiple wearers, the next architecture change is a `DeviceSession` boundary: authenticated device identity, independent source generation/sample clock, decoder, bounded queue, recognizer, save retry state and replay cursor. Route each device to one owning worker. Keep caption readers separate from audio producers and enforce per-device admission and memory limits. Start with a small number of processes and measured capacity; there is no evidence that adding a broker or cluster scheduler now improves the one-band demo.

At 16 kHz PCM16 mono, each wearer contributes 32,000 audio bytes/second; 100 wearers contribute 3.2 MB/second before transport overhead. Opus reduces bandwidth substantially, but removing a sub-millisecond decode is unlikely to dominate a cloud result delay. Reusing compressed audio safely requires preserving frame durations, sequence gaps and fallback decoding.

Deepgram's published Nova-3 streaming limit is up to 150 concurrent requests per project on the listed self-serve plans; diarization can lower the limit. This is an external ceiling, **not** validated application capacity. Enforce quotas and bounded admission, monitor event-loop lag, RSS, oldest-frame age, p95/p99 result delay and save backlog; test reconnect bursts without load-testing the provider beyond agreed limits.

### Acceptance checks still required

- Natural human speech through this band: quiet/noisy room, normal wearing position, distance, names/numbers and intended languages; reference labels must actually be audible.
- Repeated and randomized paired runs using the same intentional clips: 20/40/100 ms packetization, endpointing 100/300/500 ms, direct versus Pipecat, and quality alongside latency.
- USB unplug/replug, takeover, packet corruption/loss, queue pressure, stale audio and clean stop/drain on the actual firmware.
- Maintain model/history verification during future cutovers; the migration session completed the current new-Mac Whisper and history checks. Cloud failure must not be described as automatic local failover.
- Mock-provider capacity runs at increasing device counts after session isolation, followed by a small authorized real-provider canary.
- Battery/wireless and acoustic-to-visible timing tests before making untethered or sub-100-ms claims.

## Evidence ledger and primary sources

| ID | Evidence and authority | Boundary |
| --- | --- | --- |
| E1 | REA `ev_4a37f002b3cca30b371fd64ed9a773a5062a792f2a0de3e04c588b37048291da`, controlled synthetic comparison | Callback timings; excludes wearable, saving, delivery and paint |
| E2 | REA `ev_2622772d99a4ae7564560a1393630b6d3ecc9c5a6c5103c25b20a70a009dd060`, HTTPS and original service observation | Handshake probes and safe health fields; no microphone content |
| E3 | REA `ev_b9ddec7144ef112673f6c72bad1b53b9d98c32e6ea895a4be4e16a5f16808294`, SSH Mac runtime observation | Two health samples, port owner and changing deployment source hashes; not firmware execution tracing |
| E4 | Repository regressions and isolated fault verification | Controlled failure behavior, not many-user production capacity |
| E5 | Pinned Vexo SDK/contract and current adapter source | Source behavior; unobserved hardware rules remain unknown |
| E6 | REA `ev_04fd3c24342148793bf74a37cc15239c8148a5b8e3d1851c6b9383671cddb379`, real-provider and real-model synthetic checks | Temporary bridges; journal/Markdown confirmed without reading live transcripts |
| E7 | Offline Mac codec/contract probe, pinned input digests | Public fixture and generated packets only; no serial port or microphone |

E4 is retained as REA `ev_afd57b08d44eabdcaa18957a34fcab60162769e3c0040288492cfc4459ec370b`.

Final combined-build checks are REA `ev_3c9ac1a9ca7d679ccb0671121aa768f0e02335e6f42487fadf6c35760ddbe787` (tests/faults/source identity) and `ev_e32d15dc8568ba723645d68a0033bcb13d045794340e51c52491471f5f6f5b62` (isolated Deepgram and Whisper saved-output checks). The final documentation-only migration commit does not change their verified executable source hashes.

Complete REA bundle and reusable hashes/index are local ignored files under `docs/evidence/rea-extreme-audit.json` and `docs/evidence/optimization-index.json`. Numeric receipts include `latency-*.json`, `mac-vexo-runtime.json`, `topology-audit.json` and the offline codec/ops receipts. No keys, recordings or private transcript text are published with this report.

- [Vexo speech integration](https://github.com/VexoAI-Org/vexo-open-kit/blob/1e4291e2fa4896b1f603d3d4f0b009c483425da0/docs/speech.md) and [Deepgram example and measured limitations](https://github.com/VexoAI-Org/vexo-open-kit/blob/1e4291e2fa4896b1f603d3d4f0b009c483425da0/docs/deepgram.md).
- [Deepgram model selection](https://developers.deepgram.com/docs/models-languages-overview), [endpointing versus final segments](https://developers.deepgram.com/docs/understand-endpointing-interim-results/), [formatting/no-delay tradeoff](https://developers.deepgram.com/docs/smart-format), [raw audio encoding](https://developers.deepgram.com/docs/encoding), and [current concurrency limits](https://developers.deepgram.com/reference/api-rate-limits).
- [Pipecat Deepgram service](https://docs.pipecat.ai/api-reference/server/services/stt/deepgram). The installed pinned service and actual pipeline determine which optimizations are active; documentation alone does not prove that a VAD finalization path runs here.

The linked hackathon pages did not expose readable event requirements through this research path. This report does not invent judging rules or claim submission acceptance.
