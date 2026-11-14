# Omi live captions

Wearable speech becomes live terminal captions and saved notes. One bridge accepts audio from the Omi phone app, sends it to the selected recognizer, and saves finalized transcripts before publishing them.

```text
Wearable → phone → bridge → Deepgram or Whisper → journal → captions + daily notes
```

This Mac uses **Deepgram Nova-3 English**. Local Whisper remains an explicit fallback. Optional Pipecat orchestration uses the official Deepgram STT service in a separate process; the direct Deepgram path remains available. Switching is intentional; there is no automatic cloud/local fallback. The current deployment supports one wearer at a time.

## Daily use

From `/Users/alhinai/omi`:

```bash
./omi start
./omi status
./omi captions
```

Connect the wearable in the Omi phone app. The service runs in the background; closing the viewer leaves transcription running. Finals are saved under `logs/` as an authoritative JSONL journal and readable daily Markdown. Raw audio is not recorded by default.

| Command | Purpose |
| --- | --- |
| `./omi start`, `./omi stop`, `./omi restart` | Manage the service; restart preserves history |
| `./omi status` | Show service, phone, audio, recognizer and saving states |
| `./omi check` | Inspect configuration, dependencies and local health |
| `./omi captions` | Open the terminal viewer |
| `./omi backend deepgram` or `./omi backend whisper` | Validate, switch, verify, and recover from a failed switch |
| `./omi verify --phrase "Blue lantern seven" --timeout 60` | Observe a new intentional phrase through captions and saved notes |

`./omi --help` lists the commands. Existing `run.sh`, `tui.sh` and Python scripts remain supported entry points.

## Setup

The bridge requires standard CPython 3.15 with the GIL enabled. Create `.env` from `.env.example` only on a new installation, then set a private stream token and the selected recognizer's credentials. Deepgram requires `DEEPGRAM_API_KEY`; Whisper requires its separate Python 3.14 environment and local models. Never overwrite an existing `.env` during an upgrade.

`./omi start` prepares the bridge environment if needed. Bridge installation is pinned through `requirements.txt` and `requirements.lock`; the optional worker has `requirements-whisper.txt`. Phone and tunnel setup are in [operations](docs/OPERATIONS.md).

## Development

```bash
./omi dev test -q
./omi dev faults
bash -n omi run.sh tui.sh
git diff --check
```

Offline checks need no microphone, cloud key, local model or production connection. CI runs unit tests and syntax/diff checks. Real-model checks are explicit: `./omi dev smoke --backend whisper` or `./omi dev smoke --backend deepgram`. The latter sends the test audio to Deepgram. Benchmarking, demos and opt-in recording are under `./omi dev`.

To test Pipecat without interrupting the live stream, install its optional Python 3.14 environment as described in [operations](docs/OPERATIONS.md), then run:

```bash
./omi dev smoke --backend deepgram --pipeline pipecat
```

This sends synthesized test audio through a real Pipecat pipeline and Deepgram on temporary ports, verifies saved output and reports actual frame counts/provenance. It does not switch or restart the running service.

[Operations](docs/OPERATIONS.md) covers use and recovery. [Architecture](docs/ARCHITECTURE.md) defines ownership and guarantees. [Recovery archive](docs/archive/README.md) preserves retired research, experiments and the hackathon packet; [evidence](docs/evidence/README.md) records verification scope and limits.
