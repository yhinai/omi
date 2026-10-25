# Omi terminal captions

```bash
~/omi/tui.sh
```

A centered, unboxed page of live writing. Wide windows use 25% left margin,
50% transcript, and 25% right margin; compact windows use most of their width.
Live following leaves a four-line reading cushion; reading history or holding
the display reclaims that space down to the two-row footer area. Body text inherits your terminal's foreground, background, and font.
Finalized words appear immediately; none are hidden by a word-reveal timer.

## The developing thought

The newest received chunk uses a softer shade for up to two seconds. Older words
stay neutral; the next chunk or a reported idle pipeline settles the previous tail.
This is styling of **completed transcription**, not speculative partial speech.

When the caption feed reports buffered audio or an in-flight request for the
current session, a quiet trailing `…` shows that more work is pending. The ellipsis
is added after word wrapping, so it cannot move existing words onto another line.
It disappears when the pipeline settles, its status expires, or the connection
closes. It never invents words. Live `partial` events replace an ephemeral draft
that flows into the active paragraph in a softer shade. Drafts from a different
stream/session stay separate. Completed-utterance IDs and revision numbers keep
late clears, stale revisions, and older final events from erasing newer drafts.
Only finalized text becomes retained/copyable paragraph history.

Related chunks from one bridge session join into paragraphs. Sentence-ending
punctuation is preferred once a paragraph reaches about 90 words; a hard 150-word
cap keeps long unpunctuated speech manageable. The hard cap can split a sentence.
Session boundaries and gaps always separate passages. Muted `HH:MM:SS` times sit in a left gutter on the same row as the first
line of a passage, without taking a separate text row. Times appear at the
first passage, a new session, after interruptions, or a paragraph starting at least
three minutes after the last timestamp. `t` hides or restores those editorial times.
Missing audio appears as a separate note such as `2.4 seconds unavailable.`

## Reading controls

The normal view shows only transcript text and timestamps, without a title,
connection status, or shortcut bar. Press `?` for help or `d` for connection details.

| Key | Action |
| --- | --- |
| `?` / Esc | Show / dismiss help or details |
| ↑ / ↓ / `k` / `j` / Page Up / Page Down | Read the conversation |
| Mouse wheel / trackpad | Scroll when supported by the terminal |
| `m` | Toggle wheel scrolling / native text selection (wheel is the startup default) |
| Space | Freeze the display and release the mouse for selection; resume restores its previous mode |
| Home | Go to the oldest retained text |
| Ctrl+U / Ctrl+D | Scroll half a page up / down |
| `g` / End | Follow the newest words |
| `/` | Search retained text; Enter keeps highlights, Esc clears them |
| `n` / `N` | Next / previous search match |
| `y` | Copy the latest paragraph when following, or matched/top visible paragraph when reading history |
| `f` | Focus the active/selected passage and dim older paragraphs |
| `t` | Show / hide editorial times |
| `d` | Show connection, buffering, queue, and replay details on demand |
| Ctrl+L | Redraw after external terminal corruption |
| `q` / Ctrl+C | Quit and restore the terminal |

Copy uses local macOS `/usr/bin/pbcopy` only when invoked, excludes interface text,
and briefly confirms `Copied`. Blue is reserved for search and returning to live.
Mouse selection and wheel behavior depend on terminal support. Wheel scrolling
is enabled at startup. Press Space to freeze the display and temporarily release
mouse capture, then drag to select and use your terminal’s normal Copy command.
Space or `g` resumes live text and restores the previous mouse mode. Reception
continues while held; `y` copies the held paragraph without adding newly received,
hidden words. `m` makes an explicit mouse-mode choice. Keyboard scrolling works
in both modes. Clipboard behavior over SSH is not extended with OSC 52.

Use `~/omi/tui.sh --mouse select` to start with native selection instead.


Scrolling anchors the viewport to a paragraph and character position. Incoming
words and resizing preserve the passage rather than its distance from the bottom.
A restrained `↓ N new passages` notice counts distinct updated/new passages while
you read history. Returning to the bottom resumes following. The viewer retains
200 paragraphs and marks expiry if the passage you were reading leaves that history.

## Caption feed and recovery

The viewer requests `/listen-ws?mode=captions&replay=100`, then includes
`after=<last_event_id>` on reconnect. It deduplicates by event ID. Replayed text is
neutral, not treated as newly developing words. An opaque cursor identifies an
exact retained event; it is not compared lexicographically or treated as a sequence
number. If the cursor has expired, the feed explicitly resets and the viewer shows
a note that earlier words are outside the available history.

The bridge retains 200 authoritative transcript/gap events. Startup seeds them
from bounded tail reads of the current JSONL file and rotation backups, skipping
malformed lines and assigning stable IDs to legacy entries without IDs. New events
carry a process stream ID so sessions reused after restart do not merge accidentally.
A replay snapshot and live subscription are registered without an intervening await.

Caption clients receive JSON only: transcript/gap events, replay boundaries, and
small status messages. The live caption queue holds 64 events. Status uses a
separate replaceable slot, so frequent status cannot evict completed captions.
A slow client whose caption queue fills is disconnected and can recover retained
events by cursor; recovery beyond the retained range is explicitly incomplete.
The existing browser audio monitor keeps its original binary-audio WebSocket mode.

Status distinguishes connection readiness, recent audio, and actual in-flight
transcription. It includes no API keys, process IDs, or raw audio. Busy pipelines
refresh status once per second; idle pipelines send transitions. The viewer wakes
for messages, keys, resizing, and its next styling/status expiry; it does not run a
constant repaint loop. A polling fallback is used if terminal readiness callbacks
are unavailable. This is bounded, event-driven rendering, not a promise of exactly
zero CPU usage.

## Runtime and verification

No extra packages are required: Python 3.15, existing `aiohttp`, and built-in
`curses` suffice. The viewer neither makes Gemini calls nor changes ingestion/STT
settings. Backend replay/status changes require restarting the bridge; reopening
the viewer loads its updated display code. It remains compatible with an older
bridge, using the old feed and marking uncertain reconnection gaps.

```bash
~/omi/tui.sh --url wss://mini.tail287a74.ts.net/listen-ws
```

The old `--instant` option remains accepted; display is always instant. Unicode
wrapping handles combining marks and wide characters, but does not implement a
complete emoji/grapheme shaping engine. Terminal control characters are removed.

Offline verification uses synthetic captions, local fake WebSocket servers, and
real pseudo-terminals, without wearable access or Gemini calls:

```bash
cd ~/omi
.venv/bin/python3.15 -B -m unittest discover -s tests -q
```

## Rendering compatibility

The viewer compares composed frames and updates only changed terminal rows. An
unchanged frame produces no terminal refresh. Partial revisions do not repaint
unrelated paragraphs. Resizing or Ctrl+L requests a full redraw.

`--sync-output auto` additionally uses DECSET 2026 for recognized Ghostty and
WezTerm environments, with plain incremental curses output elsewhere. Auto mode
is conservative under tmux/screen; it does not send capability queries that might
be mistaken for input. Use `--sync-output on` to enable it explicitly on another
compatible terminal, or `--sync-output off` if rendering behaves incorrectly.
Synchronized output is always ended in a `finally` block, including refresh errors.

References: [Ghostty rendering guidance](https://ghostty.org/docs/help/synchronized-output),
[WezTerm mode support](https://wezterm.org/escape-sequences.html#mode-functions),
and [Python curses refresh behavior](https://docs.python.org/3/library/curses.html#curses.doupdate).
