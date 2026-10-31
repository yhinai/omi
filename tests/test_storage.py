"""Persistence regressions: paragraph output and uncertain journal writes."""
import json
import logging
from pathlib import Path
import tempfile
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
