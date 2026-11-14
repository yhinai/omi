# Omi DevKit captions

This independent checkout lives at `/Users/alhinai/omi-captions` and uses local ports
**8797/8798**, separate from the running Vexo deployment.

```sh
cd ~/omi-captions
./omi start
./omi source omi --device YOUR_VERIFIED_BLUETOOTH_UUID
# In another terminal:
./omi captions
./omi status
```

Configure `OMI_DEVICE_ID` in private `.env` to omit `--device` on later runs.
Disconnect Omi from its phone app before connecting directly to this Mac.
Ctrl+C stops capture and drains accepted audio. Stop capture before `./omi stop`
or `./omi backend whisper|deepgram`.

See [Omi DevKit setup, protocol and verification limits](docs/OMI-DEVKIT.md).
DevKit 1 was verified over Bluetooth with a deliberately spoken sentence saved through
Pipecat/Deepgram into authenticated captions, JSONL and daily Markdown. See the
verification limits and initial queue-pressure correction in the device guide.

---

# Omi live captions

Wearable speech becomes live terminal captions and saved notes. One bridge accepts audio from the Omi phone app, sends it to the selected recognizer, and saves finalized transcripts before publishing them.

```text
Wearable → phone → bridge → Deepgram or Whisper → journal → captions + daily notes
```

This Mac uses **Deepgram Nova-3 English**. Local Whisper remains an explicit fallback. Optional Pipecat orchestration uses the official Deepgram STT service in a separate process; the direct Deepgram path remains available. Switching is intentional; there is no automatic cloud/local fallback. The current deployment supports one wearer at a time.

## Daily use

From `/Users/alhinai/omi-captions` on the Mac:

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

For Vexo band 43 over Bluetooth or USB, see [Vexo setup and operation](docs/VEXO.md).

The Mac is the sole active host; the Mini deployment is retired. [Continue on the Mac](docs/HANDOFF_MAC.md) contains the complete operating handoff, recovery locations, completed work and remaining priorities.

[Wearable optimization audit](docs/OPTIMIZATION.md) traces the Vexo/Omi stages, measured latency, deployment boundaries and remaining validation for multiple devices.
