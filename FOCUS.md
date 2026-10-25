# Omi Focus

Open **http://127.0.0.1:8788/focus** on the bridge host. The reader and personal edition are served by the existing local-only service; the public `/omi` validation response is unchanged.

The page is strictly light, with a warm paper background, a narrow monospaced column, sentence or paragraph emphasis, and quiet controls. It follows [iA Writer's focus and typewriter-scrolling concepts](https://ia.net/writer/support/editor/focus-mode?platform=windows), with original styling and no external fonts or frontend runtime dependencies.

- **Listen:** captions from the existing caption-only WebSocket, with replay and reconnection. Live partial hypotheses update immediately in a single temporary sentence, with no reveal delay or synthetic typing. A final result replaces that draft and is saved once. Older/replayed batch captions can still use a short layout-stable opacity reveal; real live finals appear immediately. Network and upstream phone buffering still affect arrival.
- **Reading:** choose Sentence, Paragraph or Off; toggle typewriter scrolling, reveal, optional completed-phrase screen-reader announcements, and first-word receipt timing. Reduced-motion preferences disable reveal/scroll animation.
- **Look back:** scrolling up, selecting words, or pressing Space pauses following while captions continue. **Back to live** resumes. Press `F` to cycle focus styles; `End` follows live. Pause following does not stop Omi recording.
- **Status:** distinguishes audio arriving, transcription, finishing buffered speech, waiting for audio, and reconnecting. Gaps say *not transcribed*, since recent audio may still be in the playback ring.
- **Copy:** copies the visible, bounded transcript history. The page retains at most 160 chunks and deduplicates the last 400 IDs; reconnect replay is bounded by the bridge's retained caption history.
- **The daily edition:** a separate magazine view. Choose a date and Web photograph, AI illustration, or Words only, then **Create edition**. This makes paid Gemini requests using the configured key. Generation uses a separate subprocess; only one build can run at a time. Edition generation does not change live STT settings.

Use **http://127.0.0.1:8788/focus?demo=1** for explicitly labeled sample captions; add `&view=edition` to see the illustrative magazine design. Demo text does not represent your day. Normal mode loads only real saved editions.

## Conversation and image engines

`magazine_engine.py` reads the chosen day from the current JSONL and three rotated backups, deduplicates events, counts known gaps, and asks Gemini for a structured editorial summary. Every story/takeaway needs valid source IDs; a pull quote must be an exact source excerpt. The page lets you open cited original passages. These checks establish source references, not the truth of every generated claim. Transcription and summarization can contain errors, and expired log history is unavailable.

```bash
cd ~/omi
.venv/bin/python3.15 magazine_engine.py --date 2026-10-24 --images web
.venv/bin/python3.15 magazine_engine.py --date 2026-10-24 --images ai
.venv/bin/python3.15 magazine_engine.py --date 2026-10-24 --images none
# Regenerate only the cover, preserving the existing analysis:
.venv/bin/python3.15 image_engine.py --date 2026-10-24 --mode ai
```

`OMI_EDITION_MODEL` defaults to the STT model; `OMI_IMAGE_MODEL` defaults to `gemini-3.1-flash-image`. Both use the project's existing Gemini key. Summary evidence goes to Gemini; web-photo lookup sends generic subject keywords to Wikimedia Commons. The image engine accepts raster downloads only from Wikimedia's image host, retains creator/license/source credits, and explicitly labels generated illustrations. A related web image is illustrative, not proof of conference attendance. No image is substituted when lookup fails: the text edition remains available with a short notice.

The engine generates one cover per edition to bound cost. Builds have a 270-second process deadline; an edition input is limited to 750,000 serialized source characters and fails explicitly rather than silently cutting the day. Analysis and image calls each have bounded timeouts. Generation shares the account's Gemini quota, although it does not share the audio worker or HTTP session. The text edition is published atomically after validation. Failed analysis leaves the previous edition untouched; failed image preparation still publishes the newly validated text edition without a cover. Assets and personal editions live under Git-ignored `logs/editions/YYYY-MM-DD/`, with files saved mode `0600`. Old covers remain until you remove them.

The edition UI, data, assets and generation API exist only on port 8788, not the Funnel-backed ingestion app. Builds require same-origin JSON requests. No automatic schedule, extra service, database, or image subscription is created. Rebuild an edition to include newer conversation. Edits to the reader HTML appear on reload; adding backend routes requires the usual graceful bridge restart.

## Verification

```bash
.venv/bin/python3.15 -Werror::ResourceWarning -m unittest discover -s tests -q
# Optional development-only DOM interaction checks (not needed to run the reader):
npm ci --ignore-scripts
npm test
```

The Node tests use jsdom for DOM/interaction logic; they do not verify browser layout or produce rendered screenshots. Python tests check source validation, image provenance/bounds, local routing, build isolation and existing bridge behavior. Real summary generation, image generation and public image lookup are opt-in network operations, separate from offline tests.
