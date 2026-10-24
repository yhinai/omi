#!/usr/bin/env python3.15
"""Read-only Omi terminal captions. Word reveal animates completed transcripts."""
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
import struct
import sys
import termios
import time
import unicodedata
from urllib.parse import urlsplit

import aiohttp


def clean(text):
    """Keep captions literal: never render terminal escapes or control sequences."""
    return " ".join("".join(c for c in str(text) if c.isprintable() or c.isspace()).split())


def cells(text):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1
               for c in text)


def fit(text, width):
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
    lines, line = [], ""
    for word in text.split():
        if line and cells(line + " " + word) <= width:
            line += " " + word
            continue
        if line:
            lines.append(line)
            line = ""
        while cells(word) > width:
            head = fit(word, width)
            lines.append(head)
            word = word[len(head):]
        line = word
    if line:
        lines.append(line)
    return lines or [""]


@dataclass
class Caption:
    timestamp: str
    words: list[str]
    kind: str = "transcript"
    arrived: float = field(default_factory=time.monotonic)

    def visible(self, now, instant=False):
        # A brief, bounded visual reveal, never an extra queue of delayed captions.
        duration = min(1.2, max(0.15, len(self.words) / 35))
        count = len(self.words) if instant else min(len(self.words), max(1, math.ceil(
            len(self.words) * max(0, now - self.arrived) / duration)))
        return " ".join(self.words[:count]), count < len(self.words)


@dataclass
class Viewer:
    captions: deque = field(default_factory=lambda: deque(maxlen=200))
    seen: deque = field(default_factory=lambda: deque(maxlen=400))
    connected: bool = False
    notice: str = "Connecting to Omi…"
    last_audio: float | None = None
    level: float = 0
    scroll: int = 0
    paused: bool = False
    snapshot: list | None = None
    connections: int = 0

    def add(self, event):
        if not isinstance(event, dict) or event.get("type", "transcript") not in ("transcript", "gap"):
            return
        text = clean(event.get("text", ""))[:8000]
        if not text:
            return
        identity = event.get("event_id")
        if isinstance(identity, str):
            if identity in self.seen:
                return
            self.seen.append(identity)
        stamp = clean(event.get("timestamp", ""))
        stamp = stamp[11:19] if len(stamp) >= 19 else datetime.now().strftime("%H:%M:%S")
        self.captions.append(Caption(stamp, text.split(), event.get("type", "transcript")))

    def audio(self, raw):
        if not raw or len(raw) % 2:
            return
        self.last_audio = time.monotonic()
        # Sample at most 800 values; the viewer must never slow the bridge.
        step = max(1, len(raw) // 1600) * 2
        samples = [struct.unpack_from("<h", raw, i)[0] for i in range(0, len(raw) - 1, step)]
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        self.level = min(1, math.log1p(rms) / math.log1p(8000))


async def receive(viewer, url):
    delay = 1
    timeout = aiohttp.ClientTimeout(total=None, connect=5, sock_connect=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            try:
                async with session.ws_connect(url, heartbeat=20, max_msg_size=1024 * 1024,
                        timeout=aiohttp.ClientWSTimeout(ws_close=1)) as ws:
                    viewer.connected, viewer.notice = True, "Listening"
                    if viewer.connections:
                        viewer.add({"type": "gap", "text": "[Connection restored; text during disconnection may be missing.]"})
                    viewer.connections += 1
                    delay = 1
                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.BINARY:
                            viewer.audio(message.data)
                        elif message.type == aiohttp.WSMsgType.TEXT:
                            try:
                                viewer.add(json.loads(message.data))
                            except (ValueError, TypeError):
                                pass
                        elif message.type == aiohttp.WSMsgType.ERROR:
                            break
            except (aiohttp.ClientError, TimeoutError, OSError):
                pass
            viewer.connected = False
            viewer.notice = f"Disconnected · reconnecting in {delay}s"
            await asyncio.sleep(delay)
            delay = min(10, delay * 2)


def rows_for(viewer, width, instant, captions=None):
    rows, now = [], time.monotonic()
    captions = viewer.captions if captions is None else captions
    for index, caption in enumerate(captions):
        text, revealing = caption.visible(now, instant or index < len(captions) - 1 or caption.kind == "gap")
        lines = wrap(text, width - 11)
        for i, line in enumerate(lines):
            rows.append((caption.timestamp if i == 0 else "", line,
                         "gap" if caption.kind == "gap" else "new" if revealing else "text"))
        rows.append(("", "", "text"))
    return rows


def put(screen, y, x, text, attr=0):
    height, width = screen.getmaxyx()
    if 0 <= y < height and 0 <= x < width - 1:
        try:
            screen.addstr(y, x, fit(text, width - x - 1), attr)
        except curses.error:
            pass  # Resizes may happen between measuring and painting.


def paint(screen, viewer, instant, palette):
    height, width = screen.getmaxyx()
    screen.erase()
    if height < 10 or width < 38:
        put(screen, 0, 0, "Resize terminal to at least 38 × 10.")
        screen.noutrefresh()
        curses.doupdate()
        return
    column = min(96, width - 4)
    left, top, available = (width - column) // 2, 4, height - 9
    put(screen, 1, left, "omi", palette["new"] | curses.A_BOLD)
    put(screen, 1, left + 5, "● live" if viewer.connected else "○ offline",
        palette["live"] if viewer.connected else palette["gap"])
    title = "COMPLETED TRANSCRIPTS"
    if column > 55:
        put(screen, 1, left + column - len(title), title, palette["muted"])
    put(screen, 2, left, "─" * column, palette["muted"])
    if viewer.paused:
        if viewer.snapshot is None:
            viewer.snapshot = list(viewer.captions)
        rows = rows_for(viewer, column, True, captions=viewer.snapshot)
    else:
        viewer.snapshot = None
        rows = rows_for(viewer, column, instant)
    maximum = max(0, len(rows) - available)
    viewer.scroll = min(viewer.scroll, maximum)
    end = max(0, len(rows) - viewer.scroll)
    visible = rows[max(0, end - available):end]
    if not visible:
        put(screen, top + 1, left, "Waiting for your next words…", palette["muted"])
        put(screen, top + 3, left, "Completed Omi chunks appear here as they arrive.", palette["muted"])
    for y, (stamp, text, style) in enumerate(visible, top):
        put(screen, y, left, stamp, palette["muted"])
        put(screen, y, left + 11, text, palette[style])
    age = time.monotonic() - viewer.last_audio if viewer.last_audio is not None else None
    active = age is not None and age < 8 and viewer.connected
    meter = "▰" * round(viewer.level * 8) if active else "·"
    audio = f"audio {meter:<8}" if active else "waiting for audio"
    put(screen, height - 4, left, "─" * column, palette["muted"])
    status = "Paused display · receiving continues" if viewer.paused else viewer.notice
    put(screen, height - 3, left, status, palette["live"] if viewer.connected else palette["gap"])
    if column >= 60:
        put(screen, height - 3, left + column - 24, audio, palette["muted"])
    hint = "q quit  space pause  ↑↓ scroll  g live  i instant"
    put(screen, height - 2, left, hint, palette["muted"])
    if column >= 75:
        mode = "instant" if instant else "word reveal"
        put(screen, height - 2, left + column - len(mode), mode, palette["muted"])
    screen.noutrefresh()
    curses.doupdate()


async def run(screen, args):
    screen.nodelay(True)
    screen.keypad(True)
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    palette = {k: 0 for k in ("muted", "text", "new", "live", "gap")}
    if curses.has_colors():
        curses.start_color()
        try:
            curses.use_default_colors()
            background = -1
        except curses.error:
            background = curses.COLOR_BLACK
        colors = (244, 252, 80, 114, 221) if curses.COLORS >= 256 else (7, 7, 6, 2, 3)
        for index, (key, color) in enumerate(zip(palette, colors), 1):
            curses.init_pair(index, color, background)
            palette[key] = curses.color_pair(index)
    viewer = Viewer()
    reader = asyncio.create_task(receive(viewer, args.url))
    try:
        while True:
            if reader.done():
                await reader  # Surface programming errors after the terminal is restored.
            key = screen.getch()
            if key in (ord("q"), ord("Q"), 3):
                return
            if key == ord(" "):
                viewer.paused = not viewer.paused
                if not viewer.paused:
                    viewer.scroll = 0
            elif key in (curses.KEY_UP, curses.KEY_PPAGE):
                viewer.paused = True
                viewer.scroll += 1 if key == curses.KEY_UP else max(1, screen.getmaxyx()[0] - 9)
            elif key in (curses.KEY_DOWN, curses.KEY_NPAGE):
                viewer.scroll = max(0, viewer.scroll - (1 if key == curses.KEY_DOWN else max(1, screen.getmaxyx()[0] - 9)))
            elif key in (ord("g"), ord("G"), curses.KEY_END):
                viewer.paused, viewer.scroll = False, 0
            elif key == ord("i"):
                args.instant = not args.instant
            paint(screen, viewer, args.instant, palette)
            await asyncio.sleep(0.04)
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


def main():
    if sys.version_info[:2] != (3, 15):
        sys.exit("Run this viewer with ~/omi/.venv/bin/python3.15.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8787/listen-ws", help="Omi monitor WebSocket URL")
    parser.add_argument("--instant", action="store_true", help="Show each completed chunk immediately, without word reveal")
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
