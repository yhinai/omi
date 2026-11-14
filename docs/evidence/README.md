# Current verification

The published `pipecat-smoke.json` and `pipecat-whisper-fallback.json` record isolated synthetic-audio checks without transcript text or credentials. Other receipts listed below are local-only and excluded from Git because they contain machine-specific paths and process/history metadata.

- `cleanup-verification.json`: source fingerprints, checks, live status and limits.
- `cleanup-faults.json`: the maintained offline fault/corruption checks.
- `cleanup-deepgram-smoke.json` and `cleanup-whisper-smoke.json`: isolated complete-pipeline checks.
- `cleanup-recorded-demo.json`: real local Whisper verification of the bundled recorded-audio fallback.
- `cleanup-live.json`: bounded live transport observation, with unobserved stages labeled.
- `cleanup-history.json`: journal-prefix and configuration preservation.
- `cleanup-integration.json`: canonical-source reconciliation provenance.
- `temp-cleanup.json`: retired material and the verified private recovery archive.

Older receipts and REA bundles are preserved in the [recovery archive](../archive/README.md). Match source hashes, backend, configuration and input before reusing a claim. A current service-health check does not validate microphone identity or recognition accuracy.

The Mac-only cutover copies the Mini's ignored receipts, full REA bundle, source snapshot, Git history and older recovery archives into a private package on the Mac. [Mac handoff](../HANDOFF_MAC.md) records its location. `mac-only-cutover.json` is an ignored local receipt with retirement, transfer and destination verification. Original capture paths remain unchanged in historical evidence; the private transfer manifest and destination index identify their new location.

The earlier wearable comparison was inconclusive: the user confirmed that the intended phrase was absent from the private recording. Raw diagnostic audio remains outside the repository. The maintained demo fixture is synthesized speech at `assets/demo.wav`; it is unrelated to that private recording.
