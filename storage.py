"""JSONL persistence, uncertain-write recovery, replay and daily Markdown output.

The journal is authoritative. Caption/phone delivery runs only after journal
acceptance; Markdown is a readable projection. Failed finals stay retryable.
"""
from speakers import display_text
from collections import deque
from datetime import datetime
import hashlib
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

JOURNAL_MAX_BYTES = 5 * 1024 * 1024
JOURNAL_BACKUPS = 3


def separate_journal_tail(path):
    """Preserve a crash's last bytes without letting them swallow the next event."""
    try:
        with open(path, "r+b") as journal:
            if journal.seek(0, os.SEEK_END):
                journal.seek(-1, os.SEEK_END)
                if journal.read(1) != b"\n":
                    journal.write(b"\n")
    except FileNotFoundError:
        pass


class OutputFileHandler(RotatingFileHandler):
    def handleError(self, record):
        # Logging normally swallows disk errors; output health must see them.
        raise sys.exc_info()[1]



class DailyNotes:
    """A paragraph projection; persist each update so restart does not lose a partial paragraph."""

    def __init__(self, directory, idle_flush=8):
        self.directory, self.idle_flush = directory, idle_flush
        self.day, self.words, self.last_time = None, [], None

    def close_paragraph(self):
        self.words = []

    def add(self, timestamp, text):
        dt = datetime.fromisoformat(timestamp)
        day = timestamp[:10]
        path = os.path.join(self.directory, f"{day}.md")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Omi Transcript — {day}\n\n")
            self.words = []
        if day != self.day or (self.last_time and (dt - self.last_time).total_seconds() > self.idle_flush):
            self.words = []
        size = os.path.getsize(path)
        if self.words and getattr(self, "offset", 0) > size:
            self.words = []
        if not self.words:
            self.start, self.offset = timestamp[11:], size
        self.day, self.last_time = day, dt
        self.words.extend(text.split())
        # Offset comes from getsize (bytes); binary I/O keeps Unicode offsets correct.
        with open(path, "r+b") as f:
            f.seek(self.offset)
            f.write(f"**[{self.start}]** {' '.join(self.words)}\n\n".encode("utf-8"))
            f.truncate()
        if len(self.words) >= 120:
            self.words = []



class TranscriptStore:
    def __init__(self, *, directory, append, create_event, on_commit, on_error, on_recovered):
        self._directory, self.append, self.create_event = directory, append, create_event
        self.on_commit, self.on_error, self.on_recovered = on_commit, on_error, on_recovered
        self.notes = None
        self.failed_finals = {}

    @property
    def directory(self):
        return self._directory() if callable(self._directory) else self._directory

    def configure_logging(self, logger):
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        path = os.path.join(self.directory, "omi_transcript_events.jsonl")
        separate_journal_tail(path)
        handler = OutputFileHandler(path,
                                    maxBytes=JOURNAL_MAX_BYTES, backupCount=JOURNAL_BACKUPS)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    def write(self, event, *, already_persisted=False):
        try:
            if not already_persisted:
                self.append(json.dumps(event, ensure_ascii=False))
        except Exception as error:
            self.on_error("jsonl", error)
            return False
        success = True
        try:
            self.on_commit(event)
        except Exception as error:
            self.on_error("delivery", error)
            success = False
        try:
            if self.notes is not None:
                if event["type"] == "gap":
                    self.notes.close_paragraph()
                if event.get("speaker_turns"):self.notes.close_paragraph()
                self.notes.add(event["timestamp"], display_text(event))
                if event.get("speaker_turns"):self.notes.close_paragraph()
                if event["type"] == "gap":
                    self.notes.close_paragraph()
        except Exception as error:
            self.on_error("markdown", error)
            success = False
            if self.notes is not None:
                self.notes.close_paragraph()
        if success:
            self.on_recovered()
        return True

    def flush(self):
        if self.notes is not None:
            self.notes.close_paragraph()

    def final(self, text, **timing):
        # Provider punctuation may be part of URLs, email addresses, or code.
        # Normalize whitespace without inventing sentence boundaries.
        text = " ".join(text.split())
        if not text:
            return False
        identity = (timing.get("session_id"), timing.get("utterance_id"))
        retryable = identity[1] is not None
        retry = self.failed_finals.get(identity) if retryable else None
        event = retry or self.create_event("transcript", text, **timing)
        persisted = False
        if retry is not None:
            # A write/flush can fail after the full line reached the file. Reuse the
            # same event ID and check the bounded rotated journal before appending.
            path = os.path.join(self.directory, "omi_transcript_events.jsonl")
            try:
                for candidate in [path] + [path+f".{index}" for index in range(1, JOURNAL_BACKUPS+1)]:
                    if not os.path.isfile(candidate):
                        continue
                    with open(candidate, encoding="utf-8", errors="replace") as journal:
                        for line in journal:
                            try:
                                persisted |= json.loads(line).get("event_id") == event["event_id"]
                            except (ValueError, AttributeError):
                                continue
                separate_journal_tail(path)
            except OSError as error:
                self.on_error("jsonl", error)
                return False
        if not self.write(event, already_persisted=persisted):
            if retryable:
                self.failed_finals[identity] = event
            return False
        self.failed_finals.pop(identity, None)
        return True


    def recent_events(self):
        """Bounded startup reads, including rotation backups; malformed lines are skipped."""
        events, seen = [], set()
        for suffix in ("", *(f".{i}" for i in range(1, JOURNAL_BACKUPS+1))):
            try:
                with open(os.path.join(self.directory, "omi_transcript_events.jsonl") + suffix, "rb") as stream:
                    size = stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, size - 1024 * 1024))
                    if stream.tell():
                        stream.readline()
                    lines = stream.read().splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(event, dict) or event.get("type", "transcript") not in ("transcript", "gap") or not isinstance(event.get("text"), str):
                    continue
                event.setdefault("type", "transcript")
                identity = event.get("event_id")
                if not isinstance(identity, str) or not identity:
                    event["event_id"] = identity = "legacy-" + hashlib.sha256(line).hexdigest()
                if identity not in seen:
                    events.append(event)
                    seen.add(identity)
                if len(events) == 200:
                    break
            if len(events) == 200:
                break
        return deque(reversed(events), maxlen=200)
