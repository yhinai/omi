# Fieldnotes — a personal conference-day magazine

An isolated prototype: it does not edit, restart, or replace the live Omi bridge or Focus page.

```bash
cd /Users/alhinai/omi
.venv/bin/python3.15 magazine/engine.py
```

Open **http://127.0.0.1:8790**. It starts with a clearly labeled illustrative edition. No paid API call runs until you request generation.

## Two engines

1. **Editorial engine:** reads a selected day from the existing JSONL transcript and rotated backups; analyzes long days in chunks; writes a personal cover story, 3–5 features, takeaways and suggested/recorded follow-ups. Each section cites original passage IDs and the page opens those passages on demand. It validates references, not the truth of model prose. Readers must review the result. Transcript gaps remain visible.
2. **Visual engine:** searches Wikimedia Commons for credited, permissively licensed reference images, or calls Gemini for a conceptual AI illustration. Photographs are never represented as images of your actual conference. Generated images are labeled. Check relevance and individual source/license requirements before publishing.

Uses the project's existing key precedence (process environment → project `.env` → Hermes fallback) without starting the bridge. Optional configuration in `.env`:

```text
OMI_MAGAZINE_MODEL=gemini-3.8-flash
OMI_MAGAZINE_IMAGE_MODEL=gemini-3.1-flash-image
```

The selected day's transcript and optional interests are sent to your Gemini account. Large days require multiple analysis calls. Image generation is a separate paid call; only the illustration prompt is sent. Wikimedia receives the public subject query you explicitly submit, not the transcript. API keys remain on the local server. The server binds only to `127.0.0.1`; do not expose this private prototype through Funnel.

Editions and generated illustrations are stored under Git-ignored `logs/magazine/`. Generated editions can be reopened by date. Image placement is currently session-only: printing captures the current design; reloading an edition restores its text but not selected images. A regenerated edition replaces that day's JSON only after successful generation and validation. There is no scheduled generation or automatic publishing.

Current boundaries: summary generation is subject to model errors; captions may contain transcription mistakes; source IDs only establish traceability. Actual attendee identification, biographies, invented quotes and fabricated commitments are excluded by the editorial instructions. AI illustrations are conceptual, not documentary. Image candidates require human selection. No end-to-end paid generation has been run as part of scaffolding this prototype. The local preview and photo search were exercised against the running app; offline tests cover citation validation, hostile origins/hosts, malformed transcript/model data, saved editions, and image-label/storage behavior.

Provider documentation: [Gemini image generation](https://ai.google.dev/gemini-api/docs/generate-content/image-generation), [Commons reuse guidance](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia).

Run the isolated checks with `.venv/bin/python3.15 -m unittest discover -s magazine -p "test_*.py" -q`.
