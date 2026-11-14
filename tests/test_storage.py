"""Persistence regressions: paragraph output and uncertain journal writes."""
import json
import logging
from pathlib import Path
import tempfile
import storage
import unittest
from unittest import mock

import bridge
from storage import DailyNotes, OutputFileHandler


class NotesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notes_class = DailyNotes

    def test_pause_word_limit_midnight_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            notes = self.notes_class(directory)
            entries = [("2026-10-24 23:59:40", "first"),
                       ("2026-10-24 23:59:48", "second"),
                       ("2026-10-24 23:59:57", "after pause"),
                       ("2026-10-25 00:00:02", "word " * 120),
                       ("2026-10-25 00:00:07", "next paragraph")]
            for stamp, text in entries:
                notes.add(stamp, text)
            first = Path(directory, "2026-10-24.md").read_text()
            second = Path(directory, "2026-10-25.md").read_text()
            self.assertIn("first second", first)
            self.assertEqual(first.count("**["), 2)
            self.assertEqual(second.count("**["), 2)
            restarted = self.notes_class(directory)
            restarted.add("2026-10-25 00:00:12", "continued after restart")
            updated = Path(directory, "2026-10-25.md").read_text()
            self.assertTrue(updated.startswith(second))
            self.assertIn("continued after restart", updated)


class LiteralTextTests(unittest.TestCase):
    def test_provider_punctuation_survives_journal_delivery_and_markdown(self):
        samples = (
            "Visit https://docs.Pipecat.ai/Foo.Bar?lang=enUS#API.",
            "Contact jane.Doe@Example.COM.",
            "Use namespace.Widget.run() and camelCase.Flag.",
            "U.S.A. uses v1.2.3-RC1.",
            "Literal hello.World?Yes!Please remains literal.",
        )
        with tempfile.TemporaryDirectory() as directory:
            appended, delivered, errors = [], [], []
            store = storage.TranscriptStore(directory=directory, append=appended.append,
                create_event=lambda kind, text, **timing: {"type": kind, "text": text,
                    "timestamp": "2026-11-14 12:00:00", **timing},
                on_commit=delivered.append, on_error=lambda *error: errors.append(error), on_recovered=lambda: None)
            store.notes = storage.DailyNotes(directory)
            for index, text in enumerate(samples):
                with self.subTest(index=index):
                    self.assertTrue(store.final(text, utterance_id=str(index)))
                    self.assertEqual(json.loads(appended[-1])["text"], text)
                    self.assertEqual(delivered[-1]["text"], text)
                    self.assertIn(text, Path(directory, "2026-11-14.md").read_text())
            self.assertFalse(errors)

    def test_whitespace_is_normalized_without_sentence_edits(self):
        delivered = []
        store = storage.TranscriptStore(directory="unused", append=lambda encoded: None,
            create_event=lambda kind, text, **timing: {"type": kind, "text": text},
            on_commit=delivered.append, on_error=lambda *args: None, on_recovered=lambda: None)
        self.assertTrue(store.final("  docs.Pipecat.ai\t\n stays\r\n literal.  "))
        self.assertEqual(delivered[-1]["text"], "docs.Pipecat.ai stays literal.")
        self.assertFalse(store.final(" \r\n\t "))
        self.assertEqual(len(delivered), 1)


class JournalRetryTests(unittest.TestCase):
    def test_flush_failure_after_complete_write_does_not_duplicate_the_journal(self):
        import bridge
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'omi_transcript_events.jsonl'
            logger=mock.Mock()
            def uncertain_write(encoded):
                path.write_text(encoded+'\n')
                raise OSError('flush failed after write')
            logger.info.side_effect=uncertain_write
            runtime=bridge.BridgeState()
            with mock.patch.object(bridge,'state',runtime), mock.patch.object(bridge,'LOG_DIR',directory), \
                 mock.patch.object(bridge,'STT_BACKEND','whisper'), mock.patch.object(bridge,'event_logger',logger):
                self.assertFalse(bridge.record_live_final('Recovered once.',utterance_id='u',session_id=1))
                self.assertFalse(runtime.recent_events)
                logger.info.side_effect=None
                self.assertTrue(bridge.record_live_final('Recovered once.',utterance_id='u',session_id=1))
                self.assertEqual(logger.info.call_count,1)
                self.assertEqual(len(path.read_text().splitlines()),1)
                self.assertEqual(len(runtime.recent_events),1)
                self.assertEqual(runtime.metrics['utterances_processed'],1)
                self.assertEqual(runtime.metrics['transcripts_emitted'],1)
                self.assertFalse(runtime.storage.failed_finals)

    def test_partial_failed_line_does_not_swallow_the_retried_final(self):
        import bridge
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'omi_transcript_events.jsonl'
            logger=mock.Mock()
            def partial_write(encoded):
                path.write_text(encoded[:25])
                raise OSError('partial write')
            logger.info.side_effect=partial_write
            with mock.patch.object(bridge,'state',bridge.BridgeState()), mock.patch.object(bridge,'LOG_DIR',directory), \
                 mock.patch.object(bridge,'STT_BACKEND','whisper'), mock.patch.object(bridge,'event_logger',logger):
                self.assertFalse(bridge.record_live_final('Recovered once.',utterance_id='u'))
                def append(encoded):
                    with path.open('a') as journal:journal.write(encoded+'\n')
                logger.info.side_effect=append
                self.assertTrue(bridge.record_live_final('Recovered once.',utterance_id='u'))
                lines=path.read_text().splitlines()
                self.assertEqual(len(lines),2)
                self.assertEqual(json.loads(lines[1])['text'],'Recovered once.')



class OutputHandlerTests(unittest.TestCase):
    def test_restart_separates_interrupted_tail_without_losing_the_next_final(self):
        previous = {"type": "transcript", "event_id": "previous", "text": "Previous words."}
        encoded = json.dumps(previous).encode()
        for tail in (b"", encoded, encoded + b"\n", encoded + b'\n{"text": "\xf0\x9f'):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as directory:
                path = Path(directory, "omi_transcript_events.jsonl")
                path.write_bytes(tail)
                logger = logging.getLogger("omi-journal-restart-test")
                runtime = bridge.BridgeState()
                with mock.patch.object(bridge, "state", runtime), \
                     mock.patch.object(bridge, "LOG_DIR", directory), \
                     mock.patch.object(bridge, "event_logger", logger):
                    try:
                        runtime.storage.configure_logging(logger)
                        self.assertTrue(bridge.record_live_final("After restart.", utterance_id="next"))
                        # Existing bytes, including the broken record, stay intact.
                        self.assertTrue(path.read_bytes().startswith(tail))
                        replay = list(runtime.storage.recent_events())
                        self.assertEqual(replay[-1]["text"], "After restart.")
                        self.assertEqual(sum(event["text"] == "After restart." for event in replay), 1)
                        self.assertEqual(sum(event["event_id"] == "previous" for event in replay), bool(tail))
                    finally:
                        for handler in logger.handlers[:]:
                            logger.removeHandler(handler)
                            handler.close()

    def test_saved_gaps_survive_restart_between_transcripts(self):
        with tempfile.TemporaryDirectory() as directory:
            events = [{"type": kind, "event_id": str(i), "text": kind}
                      for i, kind in enumerate(("transcript", "gap", "transcript"))]
            Path(directory, "omi_transcript_events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events))
            runtime = bridge.BridgeState()
            with mock.patch.object(bridge, "LOG_DIR", directory):
                self.assertEqual(list(runtime.storage.recent_events()), events)

    def test_disk_write_error_is_raised_instead_of_swallowed_by_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = OutputFileHandler(str(Path(directory, "test.log")), maxBytes=1024, backupCount=1)
            original = handler.stream
            handler.stream = mock.Mock()
            handler.stream.tell.return_value = 0
            handler.stream.write.side_effect = OSError("full")
            record = logging.LogRecord("test", logging.INFO, "", 1, "hello", (), None)
            try:
                with self.assertRaises(OSError):
                    handler.emit(record)
            finally:
                handler.stream = original
                handler.close()


class DailyNotesExternalTruncateTests(unittest.TestCase):
    def test_external_empty_or_partial_truncate_never_writes_nul_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            notes = storage.DailyNotes(directory, idle_flush=8)
            notes.add("2026-11-14 03:00:00", "First paragraph words.")
            notes.close_paragraph()
            notes.add("2026-11-14 03:00:02", "Second paragraph starts past header.")
            md_path = Path(directory, "2026-11-14.md")

            # Case 1: truncated to 0 bytes externally
            md_path.write_bytes(b"")
            notes.add("2026-11-14 03:00:03", "Recovered cleanly after zero truncate.")
            raw1 = md_path.read_bytes()
            self.assertNotIn(b"\x00", raw1)
            self.assertIn(b"Recovered cleanly after zero truncate.", raw1)

            # Case 2: truncated to 12 bytes (< notes.offset) externally
            notes.close_paragraph()
            notes.add("2026-11-14 03:00:04", "Another paragraph.")
            md_path.write_bytes(b"# Shortened\n")
            notes.add("2026-11-14 03:00:05", "Appended after shortened file.")
            raw2 = md_path.read_bytes()
            self.assertNotIn(b"\x00", raw2)
            self.assertIn(b"Appended after shortened file.", raw2)
