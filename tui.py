#!/usr/bin/env python3.15
"""A quiet, read-only page of Omi captions; finalized words appear immediately."""
import argparse
import asyncio
import base64
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
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:
    import aiohttp
except ImportError:
    sys.exit("The caption viewer requires aiohttp. Install it with: python3 -m pip install 'aiohttp>=3.14.5,<4'")


PAUSE_SECONDS = 2.0             # a silence at least this long starts a new paragraph
NO_TIMING_CONTINUE_SEC = 16.0   # fallback window when a final carries no sample offsets
DRAFT_NO_TIMING_SEC = 8.0
READING_MEASURE = 72            # Bringhurst: 45-75 characters, 66 ideal
DEFAULT_CAPTION_URL = "ws://127.0.0.1:8787/listen"


def clean(text):
    """Keep captions literal: strip terminal escapes and control characters."""
    # Joiners are formatting controls, but removing them changes words in scripts
    # such as Persian and splits family emoji into unrelated glyphs. Keep those
    # two semantic characters while still rejecting terminal and bidi controls.
    return " ".join("".join(c for c in str(text)
                            if c.isprintable() or c.isspace() or c in {"\u200c", "\u200d"}).split())


def within_paragraph(source, previous, fallback=NO_TIMING_CONTINUE_SEC, *, speech_start_offset=None, same_session=False):
    """Whether audio starting at `source` continues the previous paragraph.

    When both events carry VAD sample offsets in the same session, sample-clock
    silence is exact and immune to pre-roll/post-roll framing. Otherwise, when
    the previous final reported sample offsets we compare against source_end_at,
    and legacy events without offsets use the start-to-start fallback window.
    """
    if previous is None:
        return True
    if (same_session and isinstance(speech_start_offset, int) and not isinstance(speech_start_offset, bool)
            and previous.voice_end_offset is not None and speech_start_offset >= previous.voice_end_offset):
        return (speech_start_offset - previous.voice_end_offset) / 16000 < PAUSE_SECONDS
    if source is None or previous.source_at is None:
        return True
    if previous.timed and previous.source_end_at is not None:
        return 0 <= source - previous.source_end_at < PAUSE_SECONDS
    return 0 <= source - previous.source_at <= fallback


def clusters(text):
    """Split text into user-perceived characters (UAX #29 approximation).

    Terminals draw a ZWJ sequence, a regional-indicator flag, a skin-tone modifier
    or a keycap as one glyph, so measuring codepoints makes every column to its
    right drift. This groups the sequences the same way before any width math.
    """
    out, index, length = [], 0, len(text)
    while index < length:
        start = index
        char = text[index]
        index += 1
        if 0x1F1E6 <= ord(char) <= 0x1F1FF and index < length and 0x1F1E6 <= ord(text[index]) <= 0x1F1FF:
            index += 1  # regional indicator pair = one flag
        else:
            while index < length and _trailing_mark(text[index]):
                index += 1
            while index < length and text[index] == "\u200d":
                index += 1
                if index < length:
                    index += 1
                    while index < length and _trailing_mark(text[index]):
                        index += 1
        out.append(text[start:index])
    return out


def _trailing_mark(char):
    return (unicodedata.combining(char) or char in {"\ufe0e", "\ufe0f", "\u20e3", "\u200c"}
            or 0x1F3FB <= ord(char) <= 0x1F3FF)


def cluster_width(cluster):
    base = cluster[0]
    if 0x1F1E6 <= ord(base) <= 0x1F1FF:
        return 2
    if unicodedata.combining(base) or base in {"\ufe0e", "\ufe0f", "\u200c", "\u200d", "\u20e3"}:
        return 0
    if 0x1F3FB <= ord(base) <= 0x1F3FF:
        return 0
    if 0x1F300 <= ord(base) <= 0x1FAFF:
        return 2
    return 2 if unicodedata.east_asian_width(base) in "WF" else 1


def cells(text):
    if text.isascii() and text.isprintable():
        return len(text)
    return sum(cluster_width(cluster) for cluster in clusters(text))


def fit(text, width):
    if width <= 0:
        return ""
    if text.isascii() and text.isprintable():
        return text[:width]
    result, used = [], 0
    for cluster in clusters(text):
        size = cluster_width(cluster)
        if used + size > width:
            break
        result.append(cluster)
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
    source_end_at: float | None = None
    voice_end_offset: int | None = None
    rev: int = 0
    timed: bool = False
    _cache: tuple | None = None

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
    timestamp: str = ""
    bookmarked: bool = False


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
    draft_revision: int = 0
    draft_source_at: float | None = None
    draft_timestamp: str = ""
    draft_target_key: int | None = None
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
    selection: tuple | None = None
    dragging: bool = False
    match_total: int = 0
    match_number: int = 0
    model: str = ""
    backend: str = ""
    orchestration: str = ""
    provider_state: str = ""
    resolved_model: str = ""
    backend_ready: bool | None = None
    p50: float | None = None
    speech_draft_p50: float | None = None
    speech_final_p50: float | None = None
    turn_close_p50: float | None = None
    connect_p50: float | None = None
    persist_p50: float | None = None
    bookmarks: set = field(default_factory=set)
    duty: float | None = None
    workers_alive: bool | None = None
    bridge_error: str | None = None
    degraded_notified: bool = False
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
        self.draft, self.draft_identity, self.draft_revision = "", None, 0
        self.draft_source_at, self.draft_timestamp, self.draft_target_key = None, "", None

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
            self.pipeline = {k: event.get(k) for k in ("buffer_sec", "queue_depth",
                "stt_in_flight", "last_packet_age_sec", "audio_active", "stream_id", "session_id")}
            self.transcribing = event.get("stt_in_flight") is True
            self.last_status = time.monotonic()
            model = event.get("model")
            if isinstance(model, str) and model:
                self.model = model
            for name, key in (("p50", "latency_final_p50_sec"),
                              ("speech_draft_p50", "latency_speech_start_to_first_partial_p50_sec"),
                              ("speech_final_p50", "latency_speech_end_to_final_p50_sec"),
                              ("turn_close_p50", "latency_turn_close_to_final_p50_sec")):
                value = event.get(key)
                if key in event and value is None:
                    setattr(self, name, None)
                elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                    setattr(self, name, float(value))
            duty = event.get("audio_duty")
            if isinstance(duty, (int, float)) and not isinstance(duty, bool) and math.isfinite(duty):
                self.duty = float(duty)
            if "workers_alive" in event:
                self.workers_alive = bool(event.get("workers_alive"))
            for source, target in (("stt_backend", "backend"), ("provider_connection_state", "provider_state"),
                                   ("provider_resolved_model", "resolved_model"), ("stt_pipeline", "orchestration")):
                if source in event:
                    value = event[source]
                    setattr(self, target, clean(value)[:120] if isinstance(value, str) else "")
            for source, target in (("latency_provider_connect_p50_sec", "connect_p50"),
                                   ("latency_final_to_persist_p50_sec", "persist_p50")):
                if source in event:
                    value = event[source]
                    setattr(self, target, float(value) if isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value) and value >= 0 else None)
            if "backend_ready" in event:
                self.backend_ready = event["backend_ready"] if isinstance(event["backend_ready"], bool) else None
            error = event.get("last_error") or event.get("save_error")
            self.bridge_error = str(error) if error else None
            if self.bridge_error or self.workers_alive is False or self.backend_ready is False:
                if not self.degraded_notified:
                    self.degraded_notified = True
                    self.notify("Bridge degraded · d for details")
            else:
                self.degraded_notified = False
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
            if text and (not self.draft or identity != self.draft_identity):
                source = event.get("audio_start_at")
                self.draft_source_at = (source if isinstance(source, (int, float))
                    and not isinstance(source, bool) and math.isfinite(source) else time.time())
                try:
                    self.draft_timestamp = datetime.fromtimestamp(self.draft_source_at).strftime("%H:%M:%S")
                except (ValueError, OverflowError, OSError):
                    self.draft_source_at = time.time()
                    self.draft_timestamp = datetime.fromtimestamp(self.draft_source_at).strftime("%H:%M:%S")
                session = ((event.get("stream_id"), event["session_id"])
                           if event.get("session_id") is not None else None)
                previous = self.captions[-1] if self.captions else None
                same_session = bool(previous is not None and session is not None and session == previous.session)
                continuous = (previous is not None and previous.kind == "transcript"
                    and (session is None or session == previous.session)
                    and within_paragraph(self.draft_source_at, previous, DRAFT_NO_TIMING_SEC,
                                         speech_start_offset=event.get("speech_start_offset"),
                                         same_session=same_session)
                    and len(previous.words) < 150
                    and not (len(previous.words) >= 90 and re.search(r"[.!?][^\w]*$", previous.words[-1])))
                if continuous:
                    self.draft_target_key = previous.key
                else:
                    self.next_key += 1
                    self.draft_target_key = self.next_key
            self.draft = text
            self.draft_identity = identity if text else None
            self.draft_revision = revision if text else 0
            if not text:
                self.clear_draft()
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
            # Gap records mark paragraph boundaries without occupying the reading
            # column. A transient footer makes lost/replay-expired words visible.
            self.notify(text)
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
        utterance = ((event.get("stream_id"), event["utterance_id"])
                     if event.get("utterance_id") is not None else None)
        matching_draft = bool(kind == "transcript" and self.draft
                              and self.draft_identity == utterance)
        # The same bridge session closes after its actual idle timeout. Source timing
        # prevents delayed network delivery from creating false paragraph breaks.
        same_session = bool(previous is not None and session is not None and session == previous.session)
        speech_start = event.get("speech_start_offset")
        voice_end = event.get("voice_end_offset")
        valid_voice_span = (isinstance(speech_start, int) and not isinstance(speech_start, bool)
                            and isinstance(voice_end, int) and not isinstance(voice_end, bool)
                            and voice_end >= speech_start)
        continuous = (kind == "transcript" and previous is not None and previous.kind == kind
                      and same_session
                      and within_paragraph(source, previous, speech_start_offset=speech_start,
                                           same_session=same_session))
        reserved_key = None
        if matching_draft:
            continuous = previous is not None and previous.key == self.draft_target_key
            if not continuous:
                reserved_key = self.draft_target_key
                stamp = self.draft_timestamp
        source_end = source
        timed = False
        start_sample, end_sample = event.get("sample_offset"), event.get("sample_end_offset")
        if source is not None and valid_voice_span:
            source_end = source + (voice_end - speech_start) / 16000
            timed = True
        elif (source is not None and isinstance(start_sample, int) and not isinstance(start_sample, bool)
                and isinstance(end_sample, int) and not isinstance(end_sample, bool)
                and end_sample >= start_sample):
            source_end = source + (end_sample - start_sample) / 16000
            timed = True
        while words:
            previous = self.captions[-1] if self.captions else None
            sentence_end = previous and len(previous.words) >= 90 and re.search(r"[.!?][^\w]*$", previous.words[-1])
            append = continuous and previous is not None and len(previous.words) < 150 and not sentence_end
            if not append:
                meaningful = (kind == "transcript" and (previous is None or previous.kind == "gap"
                    or (session is not None and previous.session != session)
                    or (source is not None and (self.last_stamp_at is None or source - self.last_stamp_at >= 180))))
                if reserved_key is not None:
                    key, reserved_key, meaningful = reserved_key, None, True
                else:
                    self.next_key += 1
                    key = self.next_key
                previous = Caption(key, stamp, [], kind, session, source, meaningful)
                self.captions.append(previous)
                if self.bookmarks:
                    self.bookmarks.intersection_update({c.key for c in self.captions})
                if meaningful:
                    self.last_stamp_at = source
            assert previous is not None
            prefix = len(previous.text) + (1 if previous.words else 0)
            take = min(len(words), 150 - len(previous.words))
            if kind == "transcript":
                for index, word in enumerate(words[:take], 1):
                    if len(previous.words) + index >= 90 and re.search(r"[.!?][^\w]*$", word):
                        take = index
                        break
            previous.words.extend(words[:take])
            previous.rev += 1
            previous.source_at = source
            if timed:
                previous.source_end_at = source_end
                previous.timed = True
            if valid_voice_span:
                previous.voice_end_offset = voice_end
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

    def live(self):
        self.following, self.anchor, self.unread, self.match = True, None, 0, None
        self.selection, self.dragging = None, False
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


def cached_lines(caption, width):
    """Wrapped rows for one caption, reused until its words or the width change.

    rows_for runs on every keypress and every frame; without this, each pass
    re-wrapped the entire retained history.
    """
    key = (width, caption.rev)
    if caption._cache is not None and caption._cache[0] == key:
        return caption._cache[1]
    lines = wrap(caption.text, width)
    caption._cache = (key, lines)
    return lines


def rows_for(viewer, width):
    rows, now = [], time.monotonic()
    focused = viewer.match[0] if viewer.match else (viewer.anchor[0] if not viewer.following and viewer.anchor else
              next((c.key for c in reversed(viewer.captions) if c.kind == "transcript"), None))
    draft_inlined = False
    for caption in viewer.captions:
        if caption.kind == "gap":
            continue
        offset, text = 0, caption.text
        inline_draft = bool(viewer.draft and caption is viewer.captions[-1]
                            and caption.key == viewer.draft_target_key)
        if inline_draft:
            draft_inlined = True
            draft_start = len(text) + 1
            full_text = f"{text} {viewer.draft}"
        else:
            draft_start = None
            full_text = text
        lines = wrap(full_text, width) if draft_start is not None else cached_lines(caption, width)
        is_bookmarked = caption.key in viewer.bookmarks
        for line in lines:
            offset = full_text.find(line, offset)
            muted = caption.kind == "gap" or (viewer.focus and caption.key != focused) or (draft_start is not None and offset >= draft_start)
            soft = caption.soft_from if not muted and draft_start is None and now < caption.soft_until else None
            if not muted and draft_start is not None and offset + len(line) > draft_start:
                soft = draft_start
            stamp = caption.timestamp if viewer.show_times and (caption.show_stamp or is_bookmarked) and offset == 0 else ""
            rows.append(Row(caption.key, offset, line, "muted" if muted else "text", soft,
                            timestamp=stamp, bookmarked=is_bookmarked and offset == 0))
            offset += len(line)
        rows.append(Row(caption.key, len(full_text) + 1, ""))
    if viewer.draft and not draft_inlined:
        offset = 0
        for line in wrap(viewer.draft, width):
            offset = viewer.draft.find(line, offset)
            stamp = viewer.draft_timestamp if viewer.show_times and offset == 0 else ""
            rows.append(Row(viewer.draft_target_key, offset, line, "muted", timestamp=stamp))
            offset += len(line)
        # Reserve the same trailing space as a final paragraph. Otherwise a
        # full, following viewport would shift up one row on confirmation.
        rows.append(Row(viewer.draft_target_key, len(viewer.draft)+1, ""))
    return rows


def transitions(viewer, now):
    return tuple(c.key for c in viewer.captions if now < c.soft_until)


def next_transition(viewer, now):
    deadlines = [viewer.toast_until, viewer.last_status + 5,
                 now + (1.0 - (time.time() % 1.0)),  # the wall clock ticks once a second
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
        viewer.match, viewer.match_total, viewer.match_number = None, 0, 0
        return
    matches = [(caption.key, match.start()) for caption in viewer.captions if caption.kind == "transcript"
               for match in re.finditer(re.escape(viewer.query), caption.text, re.IGNORECASE)]
    if not matches:
        viewer.match, viewer.match_total, viewer.match_number = None, 0, 0
        return
    index = matches.index(viewer.match) if viewer.match in matches else 0
    if advance:
        index = (index + (-1 if backwards else 1)) % len(matches)
    viewer.match = matches[index]
    viewer.match_total, viewer.match_number = len(matches), index + 1
    viewer.following, viewer.anchor = False, viewer.match
    viewer.revision += 1


def focused_caption(viewer, rows, available):
    if viewer.following:
        return next((c for c in reversed(viewer.captions) if c.kind == "transcript"), None)
    key = (viewer.match[0] if viewer.match else
           viewer.anchor[0] if viewer.anchor else
           (rows[viewport(viewer, rows, available)].key if rows else None))
    return next((c for c in viewer.captions if c.key == key and c.kind == "transcript"), None)


def toggle_bookmark(viewer, rows, available):
    caption = focused_caption(viewer, rows, available)
    if caption is None:
        viewer.notify("Nothing to bookmark")
        return
    if caption.key in viewer.bookmarks:
        viewer.bookmarks.discard(caption.key)
        viewer.notify(f"Removed bookmark · {caption.timestamp}")
    else:
        viewer.bookmarks.add(caption.key)
        viewer.notify(f"Bookmarked · {caption.timestamp}")


def jump_bookmark(viewer, rows, available, backwards=False):
    bookmarked = [c for c in viewer.captions if c.kind == "transcript" and c.key in viewer.bookmarks]
    if not bookmarked:
        viewer.notify("No bookmarks yet · press b to mark")
        return
    current = focused_caption(viewer, rows, available)
    keys = [c.key for c in bookmarked]
    if current is not None and current.key in keys and not viewer.following:
        index = (keys.index(current.key) + (-1 if backwards else 1)) % len(bookmarked)
    else:
        index = len(bookmarked) - 1 if backwards else 0
    target = bookmarked[index]
    viewer.following = False
    viewer.match = None
    viewer.anchor = (target.key, 0)
    viewer.notify(f"Bookmark {index + 1}/{len(bookmarked)} · {target.timestamp}")


def bookmarked_text(viewer):
    return "\n\n".join(
        f"[{c.timestamp}] {c.text}"
        for c in viewer.captions
        if c.kind == "transcript" and c.key in viewer.bookmarks
    )


def copy_text(viewer, rows, available):
    if viewer.following:
        return next((c.text for c in reversed(viewer.captions) if c.kind == "transcript"), "")
    key = viewer.match[0] if viewer.match else (rows[viewport(viewer, rows, available)].key if rows else None)
    if viewer.holding and key in viewer.held_text:
        return viewer.held_text[key]
    return next((c.text for c in viewer.captions if c.key == key), "")


OSC52_LIMIT = 100_000


def clipboard_mode(requested, env=None):
    """OSC 52 travels over SSH and through tmux; pbcopy only reaches this machine."""
    env = os.environ if env is None else env
    if requested != "auto":
        return requested
    if env.get("SSH_CONNECTION") or env.get("SSH_TTY") or env.get("TMUX"):
        return "osc52"
    return "pbcopy"


async def copy_paragraph(viewer, text, mode="pbcopy", toast="Copied"):
    if not text:
        viewer.notify("Nothing to copy")
        return
    payload = text.encode("utf-8")
    if mode == "osc52" and len(payload) <= OSC52_LIMIT:
        sys.stdout.write("\x1b]52;c;" + base64.b64encode(payload).decode("ascii") + "\x07")
        sys.stdout.flush()
        viewer.notify(toast)
        return
    if len(payload) > OSC52_LIMIT:
        text = text[:OSC52_LIMIT // 4]
    try:
        process = await asyncio.create_subprocess_exec('/usr/bin/pbcopy', stdin=asyncio.subprocess.PIPE)
        try:
            await asyncio.wait_for(process.communicate(text.encode("utf-8")), 2)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        viewer.notify(toast if process.returncode == 0 else "Could not copy")
    except (OSError, TimeoutError):
        viewer.notify("Could not copy")


def session_text(viewer):
    parts = []
    for caption in viewer.captions:
        if caption.kind != "transcript":
            continue
        parts.append(f"[{caption.timestamp}] {caption.text}" if caption.show_stamp else caption.text)
    return "\n\n".join(parts)


def export_session(viewer, directory):
    text = session_text(viewer)
    if not text:
        return "Nothing to export"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(directory) / f"omi-captions-{stamp}.md"
    header = f"# Omi captions — {datetime.now():%Y-%m-%d %H:%M}\n\n"
    marked = [c for c in viewer.captions if c.kind == "transcript" and c.key in viewer.bookmarks]
    if marked:
        pin_lines = "\n".join(f"- **[{c.timestamp}]** {c.text}" for c in marked)
        body = f"## Bookmarked Passages\n\n{pin_lines}\n\n## Full Transcript\n\n{text}\n"
    else:
        body = text + "\n"
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        path.write_text(header + body, encoding="utf-8")
    except OSError as error:
        return f"Export failed: {error.__class__.__name__}"
    return f"Exported {path.name}"


def caption_url(url, after=None):
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.pop("mode", None)
    query.update(replay="100")
    if after:
        query["after"] = after
    else:
        query.pop("after", None)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def caption_token(url, token_file=None, environ=None):
    """Resolve viewer credentials without sharing project secrets with remote URLs."""
    env = os.environ if environ is None else environ
    if token_file is not None:
        with Path(token_file).expanduser().open(encoding="utf-8") as source:
            token = source.read(4097).strip()
        if not token:
            raise ValueError("Caption token file is empty")
    elif "OMI_CAPTION_TOKEN" in env:
        token = env["OMI_CAPTION_TOKEN"].strip()
    else:
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        if query.get("token") or query.get("key"):
            return ""  # Preserve explicitly supplied URL credentials.
        local = (parts.scheme == "ws" and parts.hostname in {"127.0.0.1", "localhost", "::1"}
                 and parts.port == 8787 and parts.path == "/listen"
                 and parts.username is None and parts.password is None)
        if not local:
            return ""
        try:
            from config import load_env
        except ImportError:
            values = env  # A standalone copy of the viewer can use process env.
        else:
            values = load_env(environ=env)
        token = values.get("OMI_CAPTION_TOKEN", values.get("OMI_STREAM_TOKEN", "")).strip()
    if len(token) > 4096 or any(not c.isprintable() or c.isspace() for c in token):
        raise ValueError("Caption token must not contain whitespace or control characters")
    return token


async def receive(viewer, url, token=""):
    delay = 1
    outage_notice = None
    headers = {"Authorization": "Bearer " + token} if token else None
    timeout = aiohttp.ClientTimeout(total=None, connect=5, sock_connect=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            awaiting_replay, reconnected = False, False
            failure_notice = "Reconnecting · waiting for captions"
            try:
                async with session.ws_connect(caption_url(url, viewer.last_event_id), heartbeat=20, max_msg_size=1024 * 1024,
                        timeout=aiohttp.ClientWSTimeout(ws_close=1), headers=headers) as ws:
                    viewer.connected = True
                    reconnected, awaiting_replay = bool(viewer.connections), True
                    viewer.replaying = True
                    viewer.connections += 1
                    viewer.wake.set()
                    delay = 1
                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.TEXT:
                            try:
                                event = json.loads(message.data)
                                useful = isinstance(event, dict) and (event.get("type", "transcript") in
                                    {"transcript", "partial", "status", "gap"} or
                                    (event.get("type") == "replay" and event.get("phase") == "end"))
                                if useful and outage_notice is not None:
                                    if viewer.toast == outage_notice:
                                        viewer.toast_until = 0
                                        viewer.revision += 1
                                    outage_notice = None
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
            except aiohttp.WSServerHandshakeError as error:
                if error.status in (401, 403):
                    failure_notice = "Caption access denied · check token"
            except (aiohttp.ClientError, TimeoutError, OSError):
                pass
            if awaiting_replay and reconnected:
                viewer.add({"type": "gap", "text": "Connection restored · some words may be missing"})
            viewer.connected, viewer.transcribing = False, False
            viewer.clear_draft()
            viewer.replaying = False
            viewer.revision += 1
            viewer.wake.set()
            if outage_notice != failure_notice:
                viewer.notify(failure_notice)
                outage_notice = failure_notice
            await asyncio.sleep(delay)
            delay = min(10, delay * 2)


def theme_from_env(env=None):
    """Light or dark, from COLORFGBG ('fg;bg') when the terminal sets it.

    Deliberately not an OSC 11 query: the reply arrives on stdin and can be
    consumed as a keystroke or leak into the event reader, and it costs ~120 ms
    at startup on terminals that never answer.
    """
    env = os.environ if env is None else env
    value = str(env.get("COLORFGBG") or "").strip()
    if ";" in value:
        background = value.rsplit(";", 1)[1].strip()
        try:
            return "light" if int(background) >= 7 else "dark"
        except ValueError:
            return "dark"
    return "dark"


def monochrome_requested(flag="auto", env=None):
    env = os.environ if env is None else env
    if flag == "never":
        return True
    if flag == "auto" and env.get("NO_COLOR"):
        return True
    return env.get("TERM") == "dumb"


def palette_spec(theme="dark", mono=False):
    """Role -> (ANSI slot, attribute). Pure, so the choices are testable.

    Muted uses slot 8 (bright black) rather than A_DIM, because dim is invisible
    on light themes and collides with user palettes; accent leans cyan on dark and
    blue on light for contrast.
    """
    if mono:
        return {"text": (-1, 0), "muted": (-1, curses.A_DIM), "accent": (-1, curses.A_BOLD)}
    return {"text": (-1, 0), "muted": (8, 0), "accent": (4 if theme == "light" else 6, 0)}


def build_palette(spec):
    palette = {}
    curses.start_color()
    try:
        curses.use_default_colors()
        background = -1
    except curses.error:
        background = curses.COLOR_BLACK
    for index, (role, (foreground, attr)) in enumerate(spec.items(), 1):
        try:
            curses.init_pair(index, foreground, background)
            palette[role] = curses.color_pair(index) | attr
        except curses.error:
            palette[role] = attr
    return palette


def put(screen, y, x, text, attr=0, reserve=1):
    """Draw text, clipping to the terminal; `reserve` keeps columns free on the right."""
    height, width = screen.getmaxyx()
    limit = width - reserve
    if 0 <= y < height and 0 <= x < max(1, limit):
        try:
            screen.addstr(y, x, fit(text, max(0, limit - x)), attr)
        except curses.error:
            pass


def cell_to_char(text, offset_cells):
    """Character index at a display-cell offset, so wide glyphs select cleanly."""
    if offset_cells <= 0:
        return 0
    used, index = 0, 0
    for cluster in clusters(text):
        width = cluster_width(cluster)
        if used + width > offset_cells:
            return index
        used += width
        index += len(cluster)
    return len(text)


def position_at(rows, available, top, left, y, x, start):
    """Map a screen cell to a (caption key, character offset) reading position."""
    index = start + (y - top)
    if not 0 <= index < len(rows):
        return None
    row = rows[index]
    if row.offset is None:
        return None
    return (row.key, row.offset + cell_to_char(row.text, max(0, x - left)))


def ordered_selection(viewer):
    if not viewer.selection:
        return None
    first, second = viewer.selection
    if first is None or second is None:
        return None
    return (first, second) if first <= second else (second, first)


def selection_text(viewer):
    span = ordered_selection(viewer)
    if span is None:
        return ""
    (start_key, start_offset), (end_key, end_offset) = span
    parts = []
    for caption in viewer.captions:
        if caption.kind != "transcript" or caption.key < start_key or caption.key > end_key:
            continue
        text = caption.text
        first = start_offset if caption.key == start_key else 0
        last = end_offset if caption.key == end_key else len(text)
        chunk = text[first:last].strip()
        if chunk:
            parts.append(chunk)
    return "\n\n".join(parts)


def mouse_protocol(enabled):
    """Ask the terminal for button-event tracking in SGR coordinates.

    ncurses on this macOS stops at `?1000h` (clicks only), its mouse mask has no
    BUTTON5, and it cannot parse SGR reports at all — it hands them to the app as
    the characters ESC [ < ... . So we request `?1002h` plus `?1006h` and decode
    SGR ourselves (`sgr_mouse`); terminals that ignore 1006 keep sending X10
    reports, which ncurses does parse.
    """
    suffix = "h" if enabled else "l"
    try:
        sys.stdout.write(f"\x1b[?1002{suffix}\x1b[?1006{suffix}")
        sys.stdout.flush()
    except OSError:
        pass


SGR_MOUSE = re.compile(r"\[<(\d+);(\d+);(\d+)([Mm])")


def sgr_mouse(screen, pending, timeout=0.02):
    """Decode an SGR-1006 report that ncurses delivered as individual characters.

    Returns (x, y, bstate) in curses terms, or None — in which case the consumed
    characters are pushed back so a plain Escape keeps working.
    """
    chars = []
    deadline = time.monotonic() + timeout
    while len(chars) < 24:
        try:
            ch = screen.get_wch()
        except curses.error:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.001)
            continue
        chars.append(ch)
        if ch in ("M", "m") and len(chars) > 2:
            break
    match = SGR_MOUSE.fullmatch("".join(chars))
    if match is None:
        pending.extend(chars)
        return None
    button, column, row, final = (int(match.group(1)), int(match.group(2)),
                                  int(match.group(3)), match.group(4))
    if button & 64:                                   # wheel
        state = (getattr(curses, "BUTTON4_PRESSED", 0) if not button & 1
                 else getattr(curses, "REPORT_MOUSE_POSITION", 0))
    elif button & 32:                                 # drag with a button held
        state = getattr(curses, "REPORT_MOUSE_POSITION", 0)
    elif final == "M":
        state = getattr(curses, "BUTTON1_PRESSED", 0)
    else:
        state = getattr(curses, "BUTTON1_RELEASED", 0)
    return column - 1, row - 1, state


async def apply_mouse(viewer, rows, available, top, left, mx, my, bstate, clipboard):
    wheel_up = getattr(curses, "BUTTON4_PRESSED", 0)
    wheel_down = getattr(curses, "BUTTON5_PRESSED", 0)  # absent in the macOS _curses build
    pressed = getattr(curses, "BUTTON1_PRESSED", 0)
    released = getattr(curses, "BUTTON1_RELEASED", 0)
    motion = getattr(curses, "REPORT_MOUSE_POSITION", 0)
    start = viewport(viewer, rows, available)
    position = position_at(rows, available, top, left, my, mx, start)
    if os.environ.get("OMI_TUI_MOUSE_LOG"):
        try:
            with open(os.environ["OMI_TUI_MOUSE_LOG"], "a") as log:
                log.write(f"bstate={bstate:#x} x={mx} y={my} pos={position} "
                          f"rows={len(rows)} top={top} left={left} "
                          f"following={viewer.following} sel={viewer.selection}\n")
        except OSError:
            pass
    if bstate & wheel_up:
        move(viewer, rows, available, -3)
    elif wheel_down and bstate & wheel_down:
        move(viewer, rows, available, 3)
    elif position is not None and bstate & pressed:
        viewer.following = False
        viewer.anchor = position   # keep the page still while dragging
        viewer.selection = (position, position)
        viewer.dragging = True
        viewer.revision += 1
    elif viewer.dragging and viewer.selection and bstate & motion and position is not None:
        viewer.selection = (viewer.selection[0], position)
        viewer.revision += 1
    elif bstate == motion and not viewer.dragging:
        # Wheel-down arrives as a bare position event: this ncurses
        # mask stops at BUTTON4, so button 5 has no bit of its own.
        move(viewer, rows, available, 3)
    elif bstate & released:
        viewer.dragging = False
        text = selection_text(viewer)
        if text:
            await copy_paragraph(viewer, text, clipboard)
        elif position is not None:
            viewer.anchor = position
        viewer.revision += 1


def geometry(height, width, following=True, measure: int | None = READING_MEASURE):
    # Wide windows get a centred column, but never wider than a readable measure.
    natural = max(1, width // 2 if width >= 80 else width - 16)
    column = min(natural, measure) if measure else natural
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


def complex_graphemes(text):
    """Runs old ncurses cannot preserve or measure as one terminal glyph."""
    if text.isascii():
        return False
    return ("\u200d" in text or any("\U0001f3fb" <= char <= "\U0001f3ff" for char in text)
            or any(len(cluster) > 5 or (len(cluster) > 1 and "\U0001f1e6" <= cluster[0] <= "\U0001f1ff")
                   for cluster in clusters(text)))


def terminal_style(attr):
    """Translate an existing curses style for a small raw UTF-8 overlay."""
    output = bytearray(curses.tigetstr("sgr0") or b"")
    for flag, capability in ((curses.A_BOLD, "bold"), (curses.A_DIM, "dim"),
                             (curses.A_UNDERLINE, "smul"), (curses.A_REVERSE, "rev"),
                             (curses.A_STANDOUT, "smso"), (curses.A_BLINK, "blink"),
                             (curses.A_INVIS, "invis"), (getattr(curses, "A_ITALIC", 0), "sitm")):
        if flag and attr & flag:
            output.extend(curses.tigetstr(capability) or b"")
    pair = curses.pair_number(attr)
    if pair:
        for color, capability in zip(curses.pair_content(pair), ("setaf", "setab")):
            command = curses.tigetstr(capability)
            if color >= 0 and command:
                output.extend(curses.tparm(command, color))
    return output


def overlay_grapheme_rows(screen, rows, selected):
    """Preserve full grapheme bytes on affected rows, retaining composed style runs.

    macOS ncurses 6.0 drops ZWJ and expands flags/skin tones into separate cells.
    The terminal can render those clusters correctly when it receives the bytes
    directly. Cursor save/restore leaves curses' cursor and style state intact.
    """
    cup, clear, save, restore, reset = (curses.tigetstr(name) for name in ("cup", "el", "sc", "rc", "sgr0"))
    if not all((cup, clear, save, restore, reset)):
        return
    width = screen.getmaxyx()[1]
    output = bytearray(save)
    for y in sorted(selected):
        output.extend(reset + curses.tparm(cup, y, 0) + clear)
        for x, text, attr in rows[y]:
            if 0 <= x < width:
                output.extend(curses.tparm(cup, y, x))
                output.extend(terminal_style(attr))
                # A remote diagnostic field must never become raw terminal code.
                output.extend(fit(clean(text), width - x).encode("utf-8"))
    output.extend(reset + restore)
    sys.stdout.flush()
    remaining = memoryview(output)
    while remaining:
        remaining = remaining[os.write(sys.stdout.fileno(), remaining):]


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
        native_terminal = isinstance(self.screen, curses.window) and sys.stdout.isatty()
        raw_rows = {y for y, row in enumerate(self.rows) if any(complex_graphemes(text) for _, text, _ in row)} if native_terminal else set()
        old_raw_rows = {y for y, row in enumerate(previous) if any(complex_graphemes(text) for _, text, _ in row)} if native_terminal and previous else set()
        sync = sys.stdout.isatty() and synchronized_output(viewer.sync_output)
        viewer.paint_cache = None  # A failed refresh must be fully redrawn next time.
        try:
            if sync:
                sys.stdout.write("\x1b[?2026h")
                sys.stdout.flush()
            if previous is None:
                self.screen.clearok(True)
            # Raw overlays are outside curses' screen image. Force those lines to
            # refresh next time, including when a grapheme disappears or moves.
            for y in raw_rows | old_raw_rows:
                self.screen.redrawln(y, 1)
            for y in changed:
                self.screen.move(y, 0)
                self.screen.clrtoeol()
                for x, text, attr in self.rows[y]:
                    put(self.screen, y, x, text, attr, reserve=0)
            self.screen.noutrefresh()
            curses.doupdate()
            if raw_rows:
                overlay_grapheme_rows(self.screen, self.rows, raw_rows)
            viewer.paint_cache = (self.screen, self.size, self.rows)
        finally:
            if sync:
                sys.stdout.write("\x1b[?2026l")
                sys.stdout.flush()


def wall_clock():
    """Local time, seconds included: the page's only chrome, and it earns its row."""
    return datetime.now().strftime("%H:%M:%S")


def set_title(text):
    """Set the terminal window title (OSC 2). Empty text restores the shell's own."""
    try:
        sys.stdout.write(f"\x1b]2;{text}\x07")
        sys.stdout.flush()
    except OSError:
        pass


def measure_value(spec):
    """Parse --measure: 'auto' (default), 'full', or an explicit column count."""
    if spec in ("auto", ""):
        return READING_MEASURE
    if spec == "full":
        return None
    try:
        value = int(spec)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("--measure must be auto, full, or a column count")
    return max(30, min(200, value))


def clock_attr(viewer, palette):
    """The clock carries one bit of state: accent while the page is not following live."""
    return palette["accent"] if not (viewer.following and not viewer.holding) else palette["muted"]


def paint(screen, viewer, palette, measure: int | None = READING_MEASURE):
    screen = Frame(screen)
    height, width = screen.getmaxyx()
    screen.erase()
    if height < 10 or width < 32:
        put(screen, 1, 2, "A little more room, please.", palette["muted"])
    else:
        column, left, top, available = geometry(height, width, viewer.following and not viewer.holding, measure)
        rows = rows_for(viewer, column)
        # The clock belongs to the terminal, not the column: pin it to the top right corner.
        stamp = wall_clock()
        put(screen, 0, max(0, width - cells(stamp)), stamp, clock_attr(viewer, palette), reserve=0)
        start = viewport(viewer, rows, available)
        if not viewer.following and viewer.anchor is not None and not any(r.key == viewer.anchor[0] for r in rows):
            viewer.anchor = (rows[0].key, rows[0].offset) if rows else None
            viewer.notify("Earlier text has left the reading history")
        if viewer.diagnostics:
            details = ["Live connection", "", "Viewer: " + viewer.status(time.monotonic()),
                       "Retained passages: " + str(len(viewer.captions)),
                       "Bookmarks: " + str(len(viewer.bookmarks)),
                       "Replay: " + ("available" if viewer.replay_supported else "not negotiated")]
            for key, label in (("buffer_sec", "Buffer (seconds)"), ("queue_depth", "Queued packets"),
                               ("stt_in_flight", "Transcribing"), ("last_packet_age_sec", "Last packet age (seconds)")):
                value = viewer.pipeline.get(key, "unavailable")
                if key == "last_packet_age_sec" and isinstance(value, (int, float)):
                    value = round(value + max(0, time.monotonic() - viewer.last_status), 1)
                details.append(f"{label}: {value}")
            details.append("Recognizer: " + (viewer.backend or "unknown") + " · " + (viewer.provider_state or "unknown"))
            if viewer.orchestration:
                details.append("Pipeline: " + viewer.orchestration)
            details.append("Model: " + (viewer.model or "unknown")
                           + (" (" + viewer.resolved_model + ")" if viewer.resolved_model else ""))
            for label, value in (("Receipt → final p50", viewer.p50),
                                 ("Speech → draft p50", viewer.speech_draft_p50),
                                 ("Speech end → final p50", viewer.speech_final_p50),
                                 ("Turn close → final p50", viewer.turn_close_p50),
                                 ("Cloud connect p50", viewer.connect_p50),
                                 ("Final → saved p50", viewer.persist_p50)):
                details.append(label + ": " + (f"{value:.2f}s" if value is not None else "unavailable"))
            details.append("Audio duty: " + (f"{viewer.duty:.3f}" if viewer.duty is not None else "unavailable"))
            details.append("Workers: " + ("alive" if viewer.workers_alive else
                                          "down" if viewer.workers_alive is False else "unknown"))
            details.append("Last error: " + (viewer.bridge_error or "none"))
            details += ["", "d / Esc to return"]
            for y, line in enumerate(details[:available], top):
                put(screen, y, left, line, palette["muted"])
        elif viewer.help_open:
            help_lines = ["A quiet place to follow the conversation.", "", "↑ ↓ / j k      Read at your own pace",
                          "PgUp / PgDn    Scroll a page", "Home           Oldest retained words",
                          "g / End        Return to live", "Space          Freeze / resume display",
                          "b / B          Bookmark passage / copy bookmarks",
                          "[ / ]          Previous / next bookmark",
                          "/              Find words", "n / N          Next / previous match",
                          "y              Copy this paragraph", "Y              Copy the whole session",
                          "e              Export captions to ~/omi/logs",
                          "t              Show / hide times", "f              Focus this passage",
                          "drag           Select text · the wheel still scrolls",
                          "m              Hand the mouse back to the terminal", "d              Connection details",
                          "? / Esc        Close help", "q              Quit"]
            for y, line in enumerate(help_lines[:available], top):
                put(screen, y, left, line, palette["text"] if y == top else palette["muted"])
        else:
            if not rows:
                put(screen, top, left, "Your next words will appear here.", palette["muted"])
            span = ordered_selection(viewer)
            for y, row in enumerate(rows[start:start + available], top):
                if row.timestamp:
                    if row.bookmarked:
                        mark = f"* {row.timestamp}" if left >= 12 else row.timestamp
                        put(screen, y, left - (12 if left >= 12 else 10), mark, palette["accent"])
                    else:
                        put(screen, y, left - 10, row.timestamp, palette["muted"])
                put(screen, y, left, row.text, palette[row.style])
                if row.soft_from is not None:
                    start_soft = max(0, row.soft_from - row.offset)
                    if start_soft < len(row.text):
                        put(screen, y, left + cells(row.text[:start_soft]), row.text[start_soft:], palette["muted"])
                if span is not None and row.offset is not None and row.text:
                    low = max(span[0], (row.key, row.offset))
                    high = min(span[1], (row.key, row.offset + len(row.text)))
                    if low < high:
                        first, last = low[1] - row.offset, high[1] - row.offset
                        put(screen, y, left + cells(row.text[:first]), row.text[first:last],
                            palette.get("selection", curses.A_REVERSE))
                if viewer.query and row.style == "text":
                    for match in re.finditer(re.escape(viewer.query), row.text, re.IGNORECASE):
                        put(screen, y, left + cells(row.text[:match.start()]), match.group(), palette["accent"] | curses.A_UNDERLINE)
        footer, emphasize = "", False
        if viewer.searching:
            footer = "/ " + viewer.query
            if viewer.query:
                footer += ("  · no match" if viewer.match is None
                           else f"  · {viewer.match_number}/{viewer.match_total}")
        elif time.monotonic() < viewer.toast_until:
            footer, emphasize = viewer.toast, True
        elif viewer.holding:
            footer = "Held"
        elif not viewer.following and viewer.unread and len(rows) - (start + available) > 0:
            footer = f"↓{viewer.unread}"  # only when words are actually below the fold
        if footer:
            footer = fit(footer, column)
            put(screen, height - 2, left if viewer.searching else left + column - cells(footer),
                footer, palette["accent"] if emphasize else palette["muted"])
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
        if key == "\x1b":
            viewer.selection = None
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
    elif key == "b":
        toggle_bookmark(viewer, rows, available)
    elif key in ("[", "]"):
        jump_bookmark(viewer, rows, available, backwards=key == "[")
    elif key == "B":
        return "copy_bookmarks"
    elif key == "y":
        return "copy"
    elif key == "Y":
        return "copy_all"
    elif key == "e":
        return "export"
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
        curses.mousemask((curses.ALL_MOUSE_EVENTS | getattr(curses, "REPORT_MOUSE_POSITION", 0)) if args.mouse == "wheel" else 0)
        try:
            curses.mouseinterval(0)  # report presses immediately; click merging would eat drags
        except curses.error:
            pass
        if args.mouse == "wheel":
            mouse_protocol(True)
        with_curses_mouse = True
    except curses.error:
        pass
    theme = args.theme if args.theme != "auto" else theme_from_env()
    mono = monochrome_requested(args.color) or not curses.has_colors()
    spec = palette_spec(theme, mono)
    try:
        palette = {role: attr for role, (_, attr) in spec.items()} if mono else build_palette(spec)
    except curses.error:
        palette = {role: attr for role, (_, attr) in palette_spec(theme, True).items()}
    viewer, rendered = Viewer(mouse_enabled=args.mouse == "wheel", sync_output=args.sync_output), None
    clipboard = clipboard_mode(args.clipboard)
    pending = deque()
    title_state = None
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
    reader = asyncio.create_task(receive(viewer, args.url, args.caption_token))
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
            column, left, top, available = geometry(height, width, viewer.following and not viewer.holding, args.measure)
            rows = rows_for(viewer, column)
            had_input = False
            # Drain curses' own input buffer, which can outlive readiness of the fd.
            for _ in range(128):
                try:
                    key = pending.popleft() if pending else screen.get_wch()
                except (curses.error, IndexError):
                    break
                had_input = True
                previous_mouse_enabled = viewer.mouse_enabled
                if key == "\x1b" and with_curses_mouse and viewer.mouse_enabled:
                    report = sgr_mouse(screen, pending)
                    if report is not None:
                        try:
                            await apply_mouse(viewer, rows, available, top, left, *report, clipboard)
                        except curses.error:
                            pass
                        continue
                if key == curses.KEY_MOUSE and with_curses_mouse and viewer.mouse_enabled:
                    try:
                        _, mx, my, _, bstate = curses.getmouse()
                        await apply_mouse(viewer, rows, available, top, left, mx, my, bstate, clipboard)
                    except curses.error:
                        pass
                else:
                    action = handle_key(viewer, key, rows, available)
                    if action == "quit":
                        return
                    if action == "copy":
                        await copy_paragraph(viewer, selection_text(viewer) or copy_text(viewer, rows, available), clipboard)
                    elif action == "copy_all":
                        await copy_paragraph(viewer, session_text(viewer), clipboard)
                    elif action == "copy_bookmarks":
                        marked = [c for c in viewer.captions if c.kind == "transcript" and c.key in viewer.bookmarks]
                        if not marked:
                            viewer.notify("No bookmarks to copy")
                        else:
                            noun = "bookmark" if len(marked) == 1 else "bookmarks"
                            await copy_paragraph(viewer, bookmarked_text(viewer), clipboard,
                                                 toast=f"Copied {len(marked)} {noun}")
                    elif action == "export":
                        viewer.notify(export_session(viewer, Path(args.export_dir).expanduser()))
                if with_curses_mouse and viewer.mouse_enabled != previous_mouse_enabled:
                    curses.mousemask((curses.ALL_MOUSE_EVENTS | getattr(curses, "REPORT_MOUSE_POSITION", 0)) if viewer.mouse_enabled else 0)
                    mouse_protocol(viewer.mouse_enabled)
                column, left, top, available = geometry(height, width, viewer.following and not viewer.holding, args.measure)
                rows = rows_for(viewer, column)
            else:
                viewer.wake.set()
            now = time.monotonic()
            if args.title:
                label = viewer.status(now)
                if label != title_state:
                    set_title(f"Omi · {label}")
                    title_state = label
            token = (viewer.revision, (height, width), viewer.status(now), now < viewer.toast_until,
                     transitions(viewer, now), wall_clock(),
                     int(now) if viewer.diagnostics else None)
            # Holding leaves the actual terminal pixels alone for native selection.
            # Reception continues; keyboard input and resizing still redraw the view.
            frozen = viewer.holding and not had_input and rendered is not None and rendered[1] == (height, width)
            if token != rendered and not frozen:
                paint(screen, viewer, palette, args.measure)
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
                mouse_protocol(False)
            except curses.error:
                pass
        if args.title:
            set_title("")
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
    parser.add_argument("--url", default=DEFAULT_CAPTION_URL, help="Omi monitor WebSocket URL")
    parser.add_argument("--token-file", metavar="PATH",
                        help="Read the caption access token from a private file; remote viewers may also use OMI_CAPTION_TOKEN")
    parser.add_argument("--mouse", choices=("wheel", "select"), default="wheel",
                        help="Startup mouse mode (default: wheel); Space releases the mouse for copying")
    parser.add_argument("--sync-output", choices=("auto", "on", "off"), default="auto",
                        help="Synchronized terminal updates (default: auto); use off for incompatible terminals")
    parser.add_argument("--clipboard", choices=("auto", "osc52", "pbcopy"), default="auto",
                        help="Clipboard route (default: auto — OSC 52 over SSH/tmux, pbcopy locally)")
    parser.add_argument("--export-dir", default="~/omi/logs", help="Where 'e' writes exported captions")
    parser.add_argument("--measure", type=measure_value, default=READING_MEASURE, metavar="auto|full|N",
                        help="Reading column width in characters (default 72, inside Bringhurst's 45-75 range)")
    parser.add_argument("--title", action="store_true",
                        help="Put the pipeline state in the terminal window title (OSC 2)")
    parser.add_argument("--theme", choices=("auto", "dark", "light"), default="auto",
                        help="Colour theme (default: auto — from COLORFGBG, else dark)")
    parser.add_argument("--color", choices=("auto", "never"), default="auto",
                        help="Colour output (default: auto; NO_COLOR and TERM=dumb are honoured)")
    args = parser.parse_args()
    try:
        parts = urlsplit(args.url)
        valid_url = parts.scheme in ("ws", "wss") and parts.hostname and (parts.port is None or parts.port > 0)
    except ValueError:
        valid_url = False
    if not valid_url:
        parser.error("--url must be a ws:// or wss:// URL")
    try:
        args.caption_token = caption_token(args.url, args.token_file)
    except (OSError, ValueError):
        parser.error("Could not load caption credentials; check .env or --token-file.")
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
