# Live conversation research

Refresh **http://127.0.0.1:8788/focus** and click **Search ›**. The search panel is collapsed by default. Opening it creates a narrow research column on the left and keeps speech on the right. On a small screen it opens as a dismissible drawer.

The panel follows the latest speech automatically, including real partial transcripts when available. No query typing is needed. It searches meaningful topic changes, rather than every word, and shows source links with short grounded summaries. The visible query is Google's actual search query, not an invented title. New results do not replace text you are selecting or results you have scrolled down to read; use **New results ready · Update** to load them.

**Pause research** stops new lookups while keeping the latest context. **Resume research** catches up. Collapsing the panel, switching to the edition, or hiding the browser tab stops new searches. A request already sent may finish. Speech recording/transcription is unaffected.

## Run

```bash
cd ~/omi
./research.sh
```

This starts only the research companion on loopback port **8791** using the existing Python 3.15 environment. It does not start, stop, restart or reconfigure the audio bridge. The companion's PID is saved in `logs/omi_research.pid`; logs go to `logs/omi_research.log`. Run the command again safely to check/start it after a machine restart. The sidebar script is added to `omi_focus.html` with one script tag; the main caption code is independent.

No API key is sent to the browser. The server uses the project's existing Gemini key precedence and Google Search grounding, documented at https://ai.google.dev/gemini-api/docs/generate-content/google-search. `OMI_RESEARCH_MODEL` optionally overrides `gemini-3.8-flash`; no configuration file was changed for this feature.

While open, recent conversation text (up to 1,400 characters from the latest two final phrases and current partial) goes to Gemini to select and search the current topic. Google's source-backed summaries are AI-generated and can be wrong; open the original source for verification. Results come only from returned grounding citations, not URLs invented in model prose. The required Google Search suggestions are displayed in a sandboxed, script-disabled frame. No captions or queries are written to research logs.

Queries are debounced, topic-change filtered, globally limited to one new lookup per eight seconds, and bounded to one provider request at a time. The last 24 context hashes/results are cached in RAM for up to three minutes. Search latency depends on Google; this is live topic-following research, not a guarantee of instant results for every word. Google/Gemini charges apply while research is active. By default, a collapsed panel does not initiate searches.

The companion accepts only local Focus origins on port 8788, verifies JSON request size, keeps credentials server-side, and returns only HTTPS source links. It is not added to Tailscale Funnel. The parent thread's streaming implementation and process are left alone.

## Offline checks

```bash
.venv/bin/python3.15 -Werror::ResourceWarning -m unittest discover -s tests -p test_research.py -q
# Uses the existing jsdom development dependency:
node --test tests/test_research_ui.cjs
```
