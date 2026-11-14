# Speaker separation and your voice

Deepgram diarization separates numbered speakers. Local voice verification compares a
speech segment with a deliberately enrolled profile; it never assumes Speaker 0 is you.
The same authenticated ingress, sample clock, recognizer and authoritative journal remain.

Set private configuration explicitly:

```dotenv
OMI_DEEPGRAM_DIARIZE=true
OMI_VOICE_IDENTIFICATION=true
OMI_DEEPGRAM_KEYTERMS='["Vexo","Pipecat"]'
```

The vocabulary list is optional and bounded to 100 terms / 500 characters. It is sent
only to the selected Deepgram pipeline. `redact_usage=true` remains set. The pinned
Pipecat service uses the supported `diarize=true` streaming path (v1 diarizer); batch v2
is not used for live captions. Speaker numbers are scoped by provider connection epoch.

## Local installation and enrollment

Create `.venv-speaker` with Python 3.14 and install `requirements-speaker.txt` with
`-c requirements-speaker.lock`. The bridge still runs in Python 3.15 and imports no
speaker inference dependencies. The optional worker uses CPU inference via Sherpa-ONNX.

Model: `wespeaker_en_voxceleb_resnet34.onnx`, from the
[Sherpa-ONNX speaker model release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models).
Authors and model terms are documented by [WeSpeaker](https://github.com/wenet-e2e/wespeaker)
and [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx).
Install under `~/Library/Application Support/OmiSpeaker/`.
The tested model SHA-256 is
`5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94`.
Never copy virtual environments between machines.

Use a deliberate 9–30 second sample of only your voice: uncompressed mono PCM16 at
16 kHz, with ordinary speech in a quiet setting. For a band sample, coordinate the
existing explicit diagnostic capture (`./omi dev capture --record`); it saves a private
WAV. Check it contains only your intended speech before enrolling. Never enroll an
ambient recording or a synthetic demo as the wearer's voice.

```sh
./omi voice enroll --wav /absolute/path/to/my-deliberate-sample.wav
./omi voice status
./omi voice forget
```

Enrollment extracts several three-second embeddings and rejects inconsistent samples.
Only the derived profile is written by enrollment, atomically with mode 0600, to
`~/Library/Application Support/OmiSpeaker/my-voice.json`. An existing profile is never
overwritten. Enrollment does not copy the supplied WAV or delete it; remove an enrollment
recording yourself after checking it if you do not want to retain it. `forget` removes
the profile only. No voice sample or embedding is sent by the local worker over a network.
Deepgram still receives the ordinary audio when it is the selected recognition backend.

## Output and uncertainty

Finals retain original provider text and per-word timing. `speaker_turns` adds scoped
numbered speakers, display labels and optional local identity/similarity metadata.
The original words are not rewritten to insert names. Captions and Markdown display
`You`, `Other speaker`, or an uncertain numbered speaker. Partials remain provisional
and do not assert a personal identity.

The local worker reads at most eight seconds of original PCM for a speaker turn and
requires at least 1.5 seconds. It checks audio level and profile/model compatibility.
Missing audio, short turns, worker failure, absent enrollment or ambiguous comparisons
stay uncertain. The whole final's verification step is bounded to four seconds, does
not run on device callbacks, and cannot prevent transcription from continuing after a
verification timeout. Final identity is fixed before persistence, so save retries do not
reclassify a previously constructed final. Provider-receipt timestamps are captured
before local matching so provider latency is not falsely increased by local inference.

The initial similarity thresholds are conservative and provisional. Validate on new
speech from the wearer and, with permission, other speakers near the band. Synthetic
same/other voice tests are integration checks, not evidence of accuracy on the wearer,
accents, noise, overlapping speakers or replayed speech. Short turns will often remain
uncertain; the system does not force every sentence to a named speaker.

Voice identification currently requires Deepgram diarization. Explicitly selecting
Whisper leaves the identity layer inactive while retaining the local transcription
fallback. Selecting Deepgram again restores the explicitly configured identity layer. No automatic provider fallback is added.

## Verify without touching live capture

```sh
python -m unittest discover -s tests -q
.venv-speaker/bin/python -m unittest scripts.test_speaker_runtime -q
```

The optional runtime check synthesizes two voices locally, tests a held-out sample and
private enrollment/no-overwrite/deletion behavior, then discards its temporary profiles.
An isolated Pipecat smoke with diarization and vocabulary enabled verifies final delivery,
JSONL and Markdown. Real wearer enrollment and a two-person live attribution check are
required before treating personal speaker labels as validated.


## Staged verification

446 core tests passed (two optional Pipecat runtime tests skipped), plus the eight existing
fault scenarios. The optional real-model enrollment test passed. A real isolated
Pipecat/Deepgram run with diarization, vocabulary hints and a temporary synthetic voice
profile produced two finals and verified local `You` attribution, JSONL and Markdown.
The wearer has not been enrolled or measured by these synthetic tests.

## Caption layout

The terminal viewer reserves a separate speaker row above each turn, and keeps the
recording timestamp beside the spoken text in its original column:

```text
          Speaker 1
12:00:00  Demo's in prison.


          Speaker 2
12:00:03  Another reply.
```

From the first partial, the speaker row is already reserved but blank. Confirmation
fills that row instead of moving the sentence. A new segment without a speaker ID
starts its own block, with a blank speaker row, instead of borrowing the previous
speaker. When identified, it stays in that block. A provisional segment can continue
the preceding turn only when it already carries the same scoped speaker ID; a
different ID starts a separate block immediately. Two blank lines separate turns, and the original
recording-start timestamp is retained. Session export/bookmarks use the same layout.
No bracketed speaker tags are inserted into sentence text.

These labels are anonymous provider groups, not verified identities. Local verification
uncertainty remains in the saved metadata; the reading margin omits the repeated
`(uncertain)` suffix. The rendering change does not rewrite the authoritative journal or
correct recognition/diarization errors. Speaker IDs remain scoped to their provider
connection; do not assume `Speaker 1` always identifies the same person after reconnects.
