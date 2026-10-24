# Omi terminal captions

Run on the Mac mini:

```bash
~/omi/tui.sh
```

A quiet, centered terminal display with live audio activity, word reveal, scrollback,
connection status, and automatic reconnection. It uses the bridge's existing outgoing
WebSocket. It does not start or restart the bridge, make Gemini calls, record audio,
or write transcript files. Requires the existing Python 3.15 environment and `aiohttp`;
the terminal UI uses Python's built-in `curses` module. No extra packages are needed.

**Word reveal is a display animation of already completed transcripts.** It adds at
most 1.2 seconds of visual reveal for the newest chunk, finishes older chunks as new
ones arrive, and never builds another transcription queue. The current bridge's
roughly eight-second windows, phone uploads, and API latency still determine when
text first becomes available. It is not streaming recognition or model token output.

| Key | Action |
| --- | --- |
| `q` / Ctrl+C | Quit; restore the terminal |
| Space | Pause/resume display; reception continues |
| ↑ / ↓ / Page Up / Page Down | Browse the paused snapshot |
| `g` / End | Return to the newest captions |
| `i` | Toggle instant display / word reveal |

```bash
~/omi/tui.sh --instant
~/omi/tui.sh --url wss://mini.tail287a74.ts.net/listen-ws
```

The viewer keeps 200 captions in memory and starts with new events only. It marks
reconnections because the existing WebSocket cannot replay text missed while
disconnected. Transcript gap events appear in amber. Any escaped terminal controls
in received captions are removed; captions are displayed as literal text.

Design references: [Python curses](https://docs.python.org/3.15/library/curses.html),
[Rich Live](https://github.com/Textualize/rich/blob/main/rich/live.py), and
[realtime-context-tui](https://github.com/darwashiOM/realtime-context-tui).
The implementation is separate from the bridge so terminal interaction does not
change audio ingestion or transcription.
