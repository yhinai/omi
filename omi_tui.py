#!/usr/bin/env python3.15
"""A quiet, read-only page of Omi captions; finalized words appear immediately."""
import argparse
import asyncio
from collections import deque
import curses
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
import json
import locale
import math
import os
import signal
import re
import sys
import termios
import time
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import aiohttp


def clean(text):
    """Keep captions literal: strip terminal escapes and control characters."""
    return " ".join("".join(c for c in str(text) if c.isprintable() or c.isspace()).split())


def cells(text):
    if text.isascii() and text.isprintable():
        return len(text)
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1
               for c in text)


def fit(text, width):
    if width >= 0 and text.isascii() and text.isprintable():
        return text[:width]
    result, used = [], 0
    for char in text:
        size = cells(char)
        if used + size > width:
            break
        result.append(char)
        used += size
    return "".join(result)


@lru_cache(maxsize=512)
def wrap(text, width):
    """Word wrapping with safe splitting of long words and wide Unicode glyphs."""
    if width < 2:
        return [fit(text, width)]
    lines, line, line_cells = [], "", 0
    for word in text.split():
        w_cells = cells(word)
        if line and line_cells + 1 + w_cells <= width:
            line += " " + word
            line_cells += 1 + w_cells
            continue
        if line:
            lines.append(line)
            line, line_cells = "", 0
        while w_cells > width:
            head = fit(word, width)
            lines.append(head)
            word = word[len(head):]
            w_cells = cells(word)
        line, line_cells = word, w_cells
    if line:
        lines.append(line)
    return lines or [""]


@dataclass
class Caption:
    key: int
    timestamp: str
    words: list[str]
    kind: str = "transcript"
    session: object = None
    source_at: float | None = None
    show_stamp: bool = False
    soft_from: int = 0
    soft_until: float = 0

    @property
    def text(self):
        return " ".join(self.words)


@dataclass(frozen=True)
class Row:
    key: int
    offset: int
    text: str
    style: str = "text"
    soft_from: int | None = None
    suffix: str = ""
    timestamp: str = ""


@dataclass
class Viewer:
    captions: deque = field(default_factory=lambda: deque(maxlen=200))
    seen: deque = field(default_factory=lambda: deque(maxlen=400))
    connected: bool = False
    connections: int = 0
    last_audio: float | None = None
    last_status: float = 0
    transcribing: bool = False
    draft: str = ""
    draft_identity: tuple | None = None
    draft_session: tuple | None = None
    draft_revision: int = 0
    finalized: deque = field(default_factory=lambda: deque(maxlen=200))
    following: bool = True
    anchor: tuple[int, int] | None = None
    unread: int = 0
    show_times: bool = True
    focus: bool = False
    diagnostics: bool = False
    mouse_enabled: bool = True
    holding: bool = False
    held_mouse_enabled: bool | None = None
    held_text: dict = field(default_factory=dict, repr=False)
    sync_output: str = "auto"
    paint_cache: object = field(default=None, repr=False)
    pipeline: dict = field(default_factory=dict)
    replaying: bool = False
    replay_supported: bool = False
    last_event_id: str | None = None
    last_stamp_at: float | None = None
    last_unread_key: int | None = None
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    help_open: bool = False
    searching: bool = False
    query: str = ""
    match: tuple[int, int] | None = None
    toast: str = ""
    toast_until: float = 0
    revision: int = 0
    next_key: int = 0

    def notify(self, text):
        self.toast, self.toast_until = text, time.monotonic() + 2
        self.revision += 1
        self.wake.set()

    def clear_draft(self):
        self.draft, self.draft_identity, self.draft_session, self.draft_revision = "", None, None, 0

    def add(self, event):
        if not isinstance(event, dict):
            return
        kind = event.get("type", "transcript")
        self.wake.set()
        if kind == "replay":
            self.replay_supported = True
            self.replaying = event.get("phase") == "start"
            if self.replaying and event.get("reset"):
                self.last_event_id = None
                self.clear_draft()
                self.add({"type": "gap", "text": "Earlier words are outside the available history."})
            self.revision += 1
            return
        if kind == "status":
            self.pipeline = {k: event.get(k) for k in ("buffer_sec", "core_window_sec", "queue_depth",
                "stt_in_flight", "last_packet_age_sec", "audio_active", "stream_id", "session_id")}
            self.transcribing = event.get("stt_in_flight") is True
            self.last_status = time.monotonic()
            age = event.get("last_packet_age_sec")
            if isinstance(age, (float, int)) and not isinstance(age, bool) and math.isfinite(age) and age >= 0:
                self.last_audio = self.last_status - age
            elif age is None:
                self.last_audio = None
            if not self.transcribing and not event.get("buffer_sec") and not event.get("queue_depth"):
                for caption in self.captions:
                    caption.soft_until = 0
            if self.diagnostics:
                self.revision += 1
            return
        if kind == "partial":
            identity = ((event.get("stream_id"), event["utterance_id"])
                        if event.get("utterance_id") is not None else None)
            text = clean(event.get("text", ""))[:8000]
            revision = event.get("revision", 0)
            revision = revision if isinstance(revision, int) and not isinstance(revision, bool) else 0
            if identity is not None and (identity in self.finalized
                    or (not text and identity != self.draft_identity)
                    or (identity == self.draft_identity and text and revision and revision <= self.draft_revision)):
                return
            self.draft = text
            self.draft_identity = identity if text else None
            self.draft_session = ((event.get("stream_id"), event["session_id"])
                                  if text and event.get("session_id") is not None else None)
            self.draft_revision = revision if text else 0
            self.revision += 1
            return
        if kind not in ("transcript", "gap"):
            return
        text = clean(event.get("text", ""))[:8000]
        if not text:
            return
        identity = event.get("event_id")
        if isinstance(identity, str):
            if identity in self.seen:
                if self.replaying:
                    self.last_event_id = identity
                return
            self.seen.append(identity)
            self.last_event_id = identity
        for caption in self.captions:
            caption.soft_until = 0
        if kind == "gap":
            duration = event.get("duration_sec")
            if isinstance(duration, (int, float)) and not isinstance(duration, bool) and 0 < duration < 86400:
                text = f"{duration:.1f} seconds unavailable."
        stamp = clean(event.get("timestamp", ""))
        stamp = stamp[11:19] if len(stamp) >= 19 else datetime.now().strftime("%H:%M:%S")
        source = event.get("audio_start_at")
        source = source if isinstance(source, (int, float)) and not isinstance(source, bool) and math.isfinite(source) else None
        if source is None:
            try:
                source = datetime.fromisoformat(clean(event.get("timestamp", ""))).timestamp()
            except (ValueError, OverflowError):
                pass
        session_id = event.get("session_id")
        session = (event.get("stream_id"), session_id) if session_id is not None else None
        previous = self.captions[-1] if self.captions else None
        words = text.split()
        # The same bridge session closes after its actual idle timeout. Source timing
        # prevents delayed network delivery from creating false paragraph breaks.
        continuous = (kind == "transcript" and previous is not None and previous.kind == kind
                      and session is not None and session == previous.session
                      and (source is None or previous.source_at is None or 0 <= source - previous.source_at <= 16))
        while words:
            previous = self.captions[-1] if self.captions else None
            sentence_end = previous and len(previous.words) >= 90 and re.search(r"[.!?][^\w]*$", previous.words[-1])
            append = continuous and previous is not None and len(previous.words) < 150 and not sentence_end
            if not append:
                meaningful = (kind == "transcript" and (previous is None or previous.kind == "gap"
                    or (session is not None and previous.session != session)
                    or (source is not None and (self.last_stamp_at is None or source - self.last_stamp_at >= 180))))
                self.next_key += 1
                previous = Caption(self.next_key, stamp, [], kind, session, source, meaningful)
                self.captions.append(previous)
                if meaningful:
                    self.last_stamp_at = source
            prefix = len(previous.text) + (1 if previous.words else 0)
            take = min(len(words), 150 - len(previous.words))
            if kind == "transcript":
                for index, word in enumerate(words[:take], 1):
                    if len(previous.words) + index >= 90 and re.search(r"[.!?][^\w]*$", word):
                        take = index
                        break
            previous.words.extend(words[:take])
            previous.source_at = source
            previous.soft_from = prefix
            previous.soft_until = time.monotonic() + 2 if kind == "transcript" and not self.replaying else 0
            if not self.following and previous.key != self.last_unread_key:
                self.unread += 1
                self.last_unread_key = previous.key
            words = words[take:]
            continuous = False  # Remaining words start the next capped paragraph.
        identity = ((event.get("stream_id"), event["utterance_id"])
                    if event.get("utterance_id") is not None else None)
        if identity is not None:
            self.finalized.append(identity)
        if kind == "gap" or identity is None or self.draft_identity in (None, identity):
            self.clear_draft()
        self.revision += 1

    def pending(self, now):
        if not self.connected or now - self.last_status >= 5 or not self.captions or self.draft:
            return False
        caption = self.captions[-1]
        session_id = self.pipeline.get("session_id")
        same_session = session_id is None or caption.session == (self.pipeline.get("stream_id"), session_id)
        buffer_sec = self.pipeline.get("buffer_sec")
        buffering = isinstance(buffer_sec, (float, int)) and buffer_sec > 0
        return caption.kind == "transcript" and same_session and (self.transcribing or buffering)

    def audio(self, raw):
        if raw and not len(raw) % 2:
            self.last_audio = time.monotonic()
            self.wake.set()

    def live(self):
        self.following, self.anchor, self.unread, self.match = True, None, 0, None
        self.holding = False
        self.held_text.clear()
        if self.held_mouse_enabled is not None:
            self.mouse_enabled, self.held_mouse_enabled = self.held_mouse_enabled, None
        self.last_unread_key = None
        self.revision += 1

    def status(self, now):
        if not self.connected:
            return "Reconnecting" if self.connections else "Connecting"
        if self.transcribing and now - self.last_status < 5:
            return "Transcribing"
        if self.last_audio is not None and now - self.last_audio < 3:
            return "Listening"
        return "Ready"


def rows_for(viewer, width):
    rows, now = [], time.monotonic()
    focused = viewer.match[0] if viewer.match else (viewer.anchor[0] if not viewer.following and viewer.anchor else
              next((c.key for c in reversed(viewer.captions) if c.kind == "transcript"), None))
    draft_inlined = False
    for caption in viewer.captions:
        offset, text = 0, caption.text
        inline_draft = bool(viewer.draft and caption is viewer.captions[-1]
                            and caption.kind == "transcript" and len(caption.words) < 150
                            and (viewer.draft_session is None or viewer.draft_session == caption.session)
                            and not (len(caption.words) >= 90 and re.search(r"[.!?][^\w]*$", caption.words[-1])))
        if inline_draft:
            draft_inlined = True
            draft_start = len(text) + 1
            full_text = f"{text} {viewer.draft}"
        else:
            draft_start = None
            full_text = text
        for line in wrap(full_text, width):
            offset = full_text.find(line, offset)
            muted = caption.kind == "gap" or (viewer.focus and caption.key != focused) or (draft_start is not None and offset >= draft_start)
            soft = caption.soft_from if not muted and draft_start is None and now < caption.soft_until else None
            if not muted and draft_start is not None and offset + len(line) > draft_start:
                soft = draft_start
            stamp = caption.timestamp if viewer.show_times and caption.show_stamp and offset == 0 else ""
            rows.append(Row(caption.key, offset, line, "muted" if muted else "text", soft, timestamp=stamp))
            offset += len(line)
        if caption is viewer.captions[-1] and viewer.pending(now):
            # Decorate after wrapping: the ellipsis cannot push an existing word to another line.
            if cells(rows[-1].text) + 2 <= width:
                row = rows[-1]
                rows[-1] = Row(row.key, row.offset, row.text, row.style, row.soft_from, " …", row.timestamp)
            else:
                rows.append(Row(caption.key, len(full_text), "…", "muted"))
        rows.append(Row(caption.key, len(full_text) + 1, ""))
    if viewer.draft and not draft_inlined:
        for line in wrap(viewer.draft, width):
            rows.append(Row(-1, 0, line, "muted"))
    return rows


def transitions(viewer, now):
    return tuple(c.key for c in viewer.captions if now < c.soft_until), viewer.pending(now)


def next_transition(viewer, now):
    deadlines = [viewer.toast_until, viewer.last_status + 5,
                 *[c.soft_until for c in viewer.captions]]
    if viewer.last_audio is not None:
        deadlines.append(viewer.last_audio + 3)
    if viewer.diagnostics:
        deadlines.append(now + 1)
    future = [deadline - now for deadline in deadlines if deadline > now]
    return max(.001, min(future)) if future else None


def viewport(viewer, rows, available):
    maximum = max(0, len(rows) - available)
    if viewer.following:
        return maximum
    if viewer.anchor is None:
        return maximum
    key, offset = viewer.anchor
    candidates = [(i, row.offset) for i, row in enumerate(rows) if row.key == key and row.offset <= offset]
    if candidates:
        return min(maximum, candidates[-1][0])
    same_passage = next((i for i, row in enumerate(rows) if row.key == key), None)
    if same_passage is not None:
        return min(maximum, same_passage)
    # History is bounded. If the anchored paragraph expires, stay at the oldest retained text.
    return 0


def move(viewer, rows, available, amount):
    start = viewport(viewer, rows, available)
    target = min(max(0, len(rows) - available), max(0, start + amount))
    if amount > 0 and target == max(0, len(rows) - available):
        viewer.live()
        return
    viewer.following = False
    if rows:
        viewer.anchor = rows[target].key, rows[target].offset
    viewer.revision += 1


def search(viewer, backwards=False, advance=False):
    if not viewer.query:
        viewer.match = None
        return
    matches = [(caption.key, match.start()) for caption in viewer.captions if caption.kind == "transcript"
               for match in re.finditer(re.escape(viewer.query), caption.text, re.IGNORECASE)]
    if not matches:
        viewer.match = None
        return
    index = matches.index(viewer.match) if viewer.match in matches else 0
    if advance:
        index = (index + (-1 if backwards else 1)) % len(matches)
    viewer.match = matches[index]
    viewer.following, viewer.anchor = False, viewer.match
    viewer.revision += 1


def copy_text(viewer, rows, available):
    if viewer.following:
        return next((c.text for c in reversed(viewer.captions) if c.kind == "transcript"), "")
    key = viewer.match[0] if viewer.match else (rows[viewport(viewer, rows, available)].key if rows else None)
    if viewer.holding and key in viewer.held_text:
        return viewer.held_text[key]
    return next((c.text for c in viewer.captions if c.key == key), "")


async def copy_paragraph(viewer, text):
    if not text:
        viewer.notify("Nothing to copy")
        return
    try:
        process = await asyncio.create_subprocess_exec('/usr/bin/pbcopy', stdin=asyncio.subprocess.PIPE)
        try:
            await asyncio.wait_for(process.communicate(text.encode("utf-8")), 2)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        viewer.notify("Copied" if process.returncode == 0 else "Could not copy")
    except (OSError, TimeoutError):
        viewer.notify("Could not copy")


def caption_url(url, after=None):
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(mode="captions", replay="100")
    if after:
        query["after"] = after
    else:
        query.pop("after", None)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


async def receive(viewer, url):
    delay = 1
    timeout = aiohttp.ClientTimeout(total=None, connect=5, sock_connect=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            awaiting_replay, reconnected = False, False
            try:
                async with session.ws_connect(caption_url(url, viewer.last_event_id), heartbeat=20, max_msg_size=1024 * 1024,
                        timeout=aiohttp.ClientWSTimeout(ws_close=1)) as ws:
                    viewer.connected = True
                    reconnected, awaiting_replay = bool(viewer.connections), True
                    viewer.replaying = True
                    viewer.connections += 1
                    viewer.wake.set()
                    delay = 1
                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.BINARY:
                            if awaiting_replay:
                                awaiting_replay, viewer.replaying = False, False
                                if reconnected:
                                    viewer.add({"type": "gap", "text": "Connection restored · some words may be missing"})
                            viewer.audio(message.data)
                        elif message.type == aiohttp.WSMsgType.TEXT:
                            try:
                                event = json.loads(message.data)
                                if awaiting_replay:
                                    awaiting_replay = False
                                    if not isinstance(event, dict) or event.get("type") != "replay":
                                        viewer.replaying = False
                                        if reconnected:
                                            viewer.add({"type": "gap", "text": "Connection restored · some words may be missing"})
                                viewer.add(event)
                            except (ValueError, TypeError):
                                pass
                        elif message.type == aiohttp.WSMsgType.ERROR:
                            break
            except (aiohttp.ClientError, TimeoutError, OSError):
                pass
            if awaiting_replay and reconnected:
                viewer.add({"type": "gap", "text": "Connection restored · some words may be missing"})
            viewer.connected, viewer.transcribing = False, False
            viewer.clear_draft()
            viewer.replaying = False
            viewer.revision += 1
            viewer.wake.set()
            await asyncio.sleep(delay)
            delay = min(10, delay * 2)


def put(screen, y, x, text, attr=0):
    height, width = screen.getmaxyx()
    if 0 <= y < height and 0 <= x < width - 1:
        try:
            screen.addstr(y, x, fit(text, width - x - 1), attr)
        except curses.error:
            pass


def geometry(height, width, following=True):
    # Wide windows use 25% / 50% / 25%; compact windows keep a usable line length.
    column = max(1, width // 2 if width >= 80 else width - 16)
    top = 1
    bottom_cushion = 4 if following and height >= 18 else 2
    available = max(1, height - top - bottom_cushion)
    left = (width - column) // 2 if width >= 80 else 13
    return column, left, top, available


def synchronized_output(mode, env=None):
    """Be conservative without consuming terminal replies as keyboard input."""
    env = os.environ if env is None else env
    if mode != "auto":
        return mode == "on"
    if env.get("TMUX") or env.get("STY"):
        return False  # Multiplexer capability/passthrough depends on its configuration.
    return env.get("TERM_PROGRAM") in {"ghostty", "WezTerm"} or env.get("TERM") == "xterm-ghostty"


class Frame:
    """Compose a frame, then touch only terminal rows whose content/style changed."""
    def __init__(self, screen):
        self.screen = screen
        self.size = screen.getmaxyx()
        self.erase()

    def getmaxyx(self):
        return self.size

    def erase(self):
        self.rows = [[] for _ in range(self.size[0])]

    def addstr(self, y, x, text, attr=0):
        self.rows[y].append((x, text, attr))

    def flush(self, viewer):
        cached = viewer.paint_cache
        previous = cached[2] if cached and cached[0] is self.screen and cached[1] == self.size else None
        changed = [y for y, row in enumerate(self.rows) if previous is None or row != previous[y]]
        if not changed:
            return
        sync = sys.stdout.isatty() and synchronized_output(viewer.sync_output)
        viewer.paint_cache = None  # A failed refresh must be fully redrawn next time.
        try:
            if sync:
                sys.stdout.write("\x1b[?2026h")
                sys.stdout.flush()
            if previous is None:
                self.screen.clearok(True)
            for y in changed:
                self.screen.move(y, 0)
                self.screen.clrtoeol()
                for x, text, attr in self.rows[y]:
                    put(self.screen, y, x, text, attr)
            self.screen.noutrefresh()
            curses.doupdate()
            viewer.paint_cache = (self.screen, self.size, self.rows)
        finally:
            if sync:
                sys.stdout.write("\x1b[?2026l")
                sys.stdout.flush()


def paint(screen, viewer, palette):
    screen = Frame(screen)
    height, width = screen.getmaxyx()
    screen.erase()
    if height < 10 or width < 32:
        put(screen, 1, 2, "A little more room, please.", palette["muted"])
    else:
        column, left, top, available = geometry(height, width, viewer.following and not viewer.holding)
        rows = rows_for(viewer, column)
        start = viewport(viewer, rows, available)
        if not viewer.following and viewer.anchor is not None and not any(r.key == viewer.anchor[0] for r in rows):
            viewer.anchor = (rows[0].key, rows[0].offset) if rows else None
            viewer.notify("Earlier text has left the reading history")
        if viewer.diagnostics:
            details = ["Live connection", "", "Viewer: " + viewer.status(time.monotonic()),
                       "Retained passages: " + str(len(viewer.captions)),
                       "Replay: " + ("available" if viewer.replay_supported else "not negotiated")]
            for key, label in (("buffer_sec", "Buffer (seconds)"), ("queue_depth", "Queued windows"),
                               ("stt_in_flight", "Transcribing"), ("last_packet_age_sec", "Last packet age (seconds)")):
                value = viewer.pipeline.get(key, "unavailable")
                if key == "last_packet_age_sec" and isinstance(value, (int, float)):
                    value = round(value + max(0, time.monotonic() - viewer.last_status), 1)
                details.append(f"{label}: {value}")
            details += ["", "d / Esc to return"]
            for y, line in enumerate(details[:available], top):
                put(screen, y, left, line, palette["muted"])
        elif viewer.help_open:
            help_lines = ["A quiet place to follow the conversation.", "", "↑ ↓ / j k      Read at your own pace",
                          "PgUp / PgDn    Scroll a page", "Home           Oldest retained words",
                          "g / End        Return to live", "Space          Freeze / resume display",
                          "/              Find words", "n / N          Next / previous match",
                          "y              Copy this paragraph", "t              Show / hide times", "f              Focus this passage",
                          "m              Wheel / text selection", "d              Connection details",
                          "? / Esc        Close help", "q              Quit"]
            for y, line in enumerate(help_lines[:available], top):
                put(screen, y, left, line, palette["text"] if y == top else palette["muted"])
        else:
            if not rows:
                put(screen, top, left, "Your next words will appear here.", palette["muted"])
            for y, row in enumerate(rows[start:start + available], top):
                if row.timestamp:
                    put(screen, y, left - 10, row.timestamp, palette["muted"])
                put(screen, y, left, row.text, palette[row.style])
                if row.soft_from is not None:
                    start_soft = max(0, row.soft_from - row.offset)
                    if start_soft < len(row.text):
                        put(screen, y, left + cells(row.text[:start_soft]), row.text[start_soft:], palette["muted"])
                if row.suffix:
                    put(screen, y, left + cells(row.text), row.suffix, palette["muted"])
                if viewer.query and row.style == "text":
                    for match in re.finditer(re.escape(viewer.query), row.text, re.IGNORECASE):
                        put(screen, y, left + cells(row.text[:match.start()]), match.group(), palette["accent"] | curses.A_UNDERLINE)
        footer = ""
        if viewer.searching:
            footer = "/ " + viewer.query + ("  · no match" if viewer.query and viewer.match is None else "")
        elif time.monotonic() < viewer.toast_until:
            footer = viewer.toast
        elif viewer.holding:
            footer = "Held · select text · y copy · Space resume"
        elif not viewer.following:
            footer = f"↓ {viewer.unread} new passages · g to follow" if viewer.unread else "Reading · g to follow"
        if footer:
            footer = fit(footer, column)
            put(screen, height - 2, left if viewer.searching else left + column - cells(footer), footer,
                palette["accent"] if not viewer.searching and not viewer.following else palette["muted"])
    screen.flush(viewer)


def handle_key(viewer, key, rows, available):
    if viewer.searching:
        if key in ("\n", "\r", curses.KEY_ENTER):
            viewer.searching = False
        elif key == "\x1b":
            viewer.searching, viewer.query, viewer.match = False, "", None
        elif key in ("\b", "\x7f", curses.KEY_BACKSPACE):
            viewer.query = viewer.query[:-1]
            search(viewer)
        elif isinstance(key, str) and key.isprintable():
            viewer.query = (viewer.query + key)[:100]
            search(viewer)
    elif key in ("q", "Q", "\x03"):
        return "quit"
    elif key in ("?", "\x1b"):
        viewer.help_open = not viewer.help_open if key == "?" else False
        viewer.diagnostics = False
    elif key == "/":
        viewer.help_open, viewer.searching, viewer.query, viewer.match = False, True, "", None
        viewer.diagnostics = False
    elif key in ("g", "G", curses.KEY_END):
        viewer.live()
    elif key == " ":
        if viewer.holding:
            viewer.live()
        else:
            move(viewer, rows, available, 0)
            viewer.holding = True
            viewer.held_text = {caption.key: caption.text for caption in viewer.captions}
            viewer.held_mouse_enabled = viewer.mouse_enabled
            viewer.mouse_enabled = False
    elif key == curses.KEY_HOME:
        move(viewer, rows, available, -len(rows))
    elif key in ("\x15", "\x04"):
        move(viewer, rows, available, max(1, available // 2) * (-1 if key == "\x15" else 1))
    elif key in (curses.KEY_UP, curses.KEY_PPAGE, "k"):
        move(viewer, rows, available, -available if key == curses.KEY_PPAGE else -1)
    elif key in (curses.KEY_DOWN, curses.KEY_NPAGE, "j"):
        move(viewer, rows, available, available if key == curses.KEY_NPAGE else 1)
    elif key in ("n", "N"):
        search(viewer, backwards=key == "N", advance=True)
    elif key == "t":
        viewer.show_times = not viewer.show_times
    elif key == "f":
        viewer.focus = not viewer.focus
    elif key == "d":
        viewer.diagnostics, viewer.help_open = not viewer.diagnostics, False
    elif key == "m":
        viewer.held_mouse_enabled = None  # An explicit choice overrides automatic restoration.
        viewer.mouse_enabled = not viewer.mouse_enabled
        viewer.notify("Wheel mode · m for selection" if viewer.mouse_enabled else "Text selection · Space to hold")
        return "mouse"
    elif key == "y":
        return "copy"
    elif key == "\x0c":
        viewer.paint_cache = None
    viewer.revision += 1
    return None


async def run(screen, args):
    screen.nodelay(True)
    screen.keypad(True)
    with_curses_mouse = False
    try:
        curses.curs_set(0)
        curses.mousemask(curses.ALL_MOUSE_EVENTS if args.mouse == "wheel" else 0)
        with_curses_mouse = True
    except curses.error:
        pass
    palette = {k: 0 for k in ("muted", "text", "accent")}
    if curses.has_colors():
        curses.start_color()
        try:
            curses.use_default_colors()
            foreground, background = -1, -1
        except curses.error:
            foreground, background = curses.COLOR_WHITE, curses.COLOR_BLACK
        # Body and background follow the terminal, including light themes.
        for index, (key, color) in enumerate((("muted", foreground), ("text", foreground), ("accent", curses.COLOR_BLUE)), 1):
            curses.init_pair(index, color, background)
            palette[key] = curses.color_pair(index)
        palette["muted"] |= curses.A_DIM
    viewer, rendered = Viewer(mouse_enabled=args.mouse == "wheel", sync_output=args.sync_output), None
    loop = asyncio.get_running_loop()
    input_ready = resize_ready = False
    try:
        loop.add_reader(sys.stdin.fileno(), viewer.wake.set)
        input_ready = True
    except (ValueError, OSError, NotImplementedError):
        pass
    try:
        loop.add_signal_handler(signal.SIGWINCH, viewer.wake.set)
        resize_ready = True
    except (ValueError, OSError, NotImplementedError):
        pass
    reader = asyncio.create_task(receive(viewer, args.url))
    reader.add_done_callback(lambda task: viewer.wake.set())
    try:
        while True:
            viewer.wake.clear()
            if reader.done():
                await reader
            if resize_ready:
                size = os.get_terminal_size(sys.stdin.fileno())
                if screen.getmaxyx() != (size.lines, size.columns):
                    curses.resizeterm(size.lines, size.columns)
            height, width = screen.getmaxyx()
            column, _, _, available = geometry(height, width, viewer.following and not viewer.holding)
            rows = rows_for(viewer, column)
            had_input = False
            # Drain curses' own input buffer, which can outlive readiness of the fd.
            for _ in range(128):
                try:
                    key = screen.get_wch()
                except curses.error:
                    break
                had_input = True
                previous_mouse_enabled = viewer.mouse_enabled
                if key == curses.KEY_MOUSE and with_curses_mouse and viewer.mouse_enabled:
                    try:
                        mouse = curses.getmouse()[4]
                        if mouse & getattr(curses, "BUTTON4_PRESSED", 0):
                            move(viewer, rows, available, -3)
                        elif mouse & getattr(curses, "BUTTON5_PRESSED", 0):
                            move(viewer, rows, available, 3)
                    except curses.error:
                        pass
                else:
                    action = handle_key(viewer, key, rows, available)
                    if action == "quit":
                        return
                    if action == "copy":
                        await copy_paragraph(viewer, copy_text(viewer, rows, available))
                if with_curses_mouse and viewer.mouse_enabled != previous_mouse_enabled:
                    curses.mousemask(curses.ALL_MOUSE_EVENTS if viewer.mouse_enabled else 0)
                column, _, _, available = geometry(height, width, viewer.following and not viewer.holding)
                rows = rows_for(viewer, column)
            else:
                viewer.wake.set()
            now = time.monotonic()
            token = (viewer.revision, (height, width), viewer.status(now), now < viewer.toast_until,
                     transitions(viewer, now), int(now) if viewer.diagnostics else None)
            # Holding leaves the actual terminal pixels alone for native selection.
            # Reception continues; keyboard input and resizing still redraw the view.
            frozen = viewer.holding and not had_input and rendered is not None and rendered[1] == (height, width)
            if token != rendered and not frozen:
                paint(screen, viewer, palette)
                rendered = token
            timeout = next_transition(viewer, now)
            if not input_ready or not resize_ready:
                timeout = min(timeout, .1) if timeout is not None else .1
            try:
                await asyncio.wait_for(viewer.wake.wait(), timeout=timeout)
            except TimeoutError:
                pass
    finally:
        if with_curses_mouse:
            try:
                curses.mousemask(0)
            except curses.error:
                pass
        if input_ready:
            loop.remove_reader(sys.stdin.fileno())
        if resize_ready:
            loop.remove_signal_handler(signal.SIGWINCH)
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


def main():
    if sys.version_info[:2] != (3, 15):
        sys.exit("Run this viewer with ~/omi/.venv/bin/python3.15.")
    parser = argparse.ArgumentParser(description=__doc__, epilog="Press ? in the viewer for controls.")
    parser.add_argument("--url", default="ws://127.0.0.1:8787/listen-ws", help="Omi monitor WebSocket URL")
    parser.add_argument("--mouse", choices=("wheel", "select"), default="wheel",
                        help="Startup mouse mode (default: wheel); Space releases the mouse for copying")
    parser.add_argument("--sync-output", choices=("auto", "on", "off"), default="auto",
                        help="Synchronized terminal updates (default: auto); use off for incompatible terminals")
    parser.add_argument("--instant", action="store_true", help=argparse.SUPPRESS)  # Backward-compatible; always instant now.
    args = parser.parse_args()
    if urlsplit(args.url).scheme not in ("ws", "wss") or not urlsplit(args.url).hostname:
        parser.error("--url must be a ws:// or wss:// URL")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("Open a real terminal to use the viewer.")
    locale.setlocale(locale.LC_ALL, "")
    terminal_state = termios.tcgetattr(sys.stdin.fileno())
    try:
        curses.wrapper(lambda screen: asyncio.run(run(screen, args)))
    except KeyboardInterrupt:
        pass
    except (curses.error, OSError):
        sys.exit("Could not open the terminal display. Try TERM=xterm-256color ~/omi/tui.sh")
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal_state)


if __name__ == "__main__":
    main()
