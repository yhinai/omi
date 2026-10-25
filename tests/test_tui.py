"""Viewer regressions; synthetic events and a local fake server, never live audio/API calls."""
import asyncio
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import unittest
from unittest import mock

from aiohttp import web
from aiohttp.test_utils import TestServer

import omi_tui as tui


def event(text, session=1, source=100, **kwargs):
    return {"text": text, "session_id": session, "audio_start_at": source,
            "timestamp": "2026-10-24 12:00:00", **kwargs}


class ReadingTests(unittest.TestCase):
    def test_related_chunks_flow_without_hiding_available_words(self):
        viewer = tui.Viewer()
        viewer.add(event("The sentence begins"))
        viewer.add(event("and continues naturally.", source=108))
        self.assertEqual(len(viewer.captions), 1)
        self.assertEqual(tui.rows_for(viewer, 76)[0].text,
                         "The sentence begins and continues naturally.")
        self.assertTrue(viewer.show_times)

    def test_session_gap_and_paragraph_length_create_reading_breaks(self):
        viewer = tui.Viewer()
        viewer.add(event("first thought"))
        viewer.add(event("another session", session=2, source=108))
        viewer.add(event("[GAP]", type="gap", duration_sec=2.4))
        viewer.add(event("words after the gap", session=2, source=116))
        self.assertEqual(len(viewer.captions), 4)
        self.assertEqual(viewer.captions[2].text, "2.4 seconds unavailable.")
        viewer.add(event("word " * 160, session=2, source=124))
        self.assertEqual(len(viewer.captions), 5)

    def test_bounded_history_deduplication_and_literal_text(self):
        viewer = tui.Viewer()
        viewer.add({"event_id": "same", "text": "literal [bold] text\x1b\x07"})
        viewer.add({"event_id": "same", "text": "duplicate"})
        self.assertEqual(len(viewer.captions), 1)
        self.assertEqual(viewer.captions[0].text, "literal [bold] text")
        for index in range(500):
            viewer.add(event(str(index), session=index, event_id=str(index)))
        self.assertEqual(len(viewer.captions), 200)
        self.assertEqual(len(viewer.seen), 400)
        self.assertEqual(viewer.captions[-1].words, ["499"])

    def test_unicode_wrap_fits_terminal_cells(self):
        for text in ("English words " * 20, "你好世界" * 10, "cafe\u0301 " * 20, "a" * 100):
            with self.subTest(text=text[:10]):
                lines = tui.wrap(text, 15)
                self.assertTrue(all(tui.cells(line) <= 15 for line in lines))
                self.assertEqual("".join("".join(lines).split()), "".join(text.split()))

    def test_scroll_anchor_stays_on_same_text_as_new_captions_arrive(self):
        viewer = tui.Viewer()
        for index in range(15):
            viewer.add(event(f"Paragraph {index}", session=index))
        rows = tui.rows_for(viewer, 50)
        tui.move(viewer, rows, 8, -5)
        anchor = viewer.anchor
        before = rows[tui.viewport(viewer, rows, 8)].text
        viewer.add(event("Fresh words", session=20))
        rows = tui.rows_for(viewer, 50)
        self.assertEqual(viewer.anchor, anchor)
        self.assertEqual(rows[tui.viewport(viewer, rows, 8)].text, before)
        self.assertEqual(viewer.unread, 1)
        tui.move(viewer, rows, 8, 1000)
        self.assertTrue(viewer.following)
        self.assertEqual(viewer.unread, 0)

    def test_resize_preserves_the_passage_instead_of_distance_from_bottom(self):
        viewer = tui.Viewer()
        for index in range(8):
            viewer.add(event(" ".join(f"word{j}" for j in range(30)), session=index))
        wide = tui.rows_for(viewer, 60)
        tui.move(viewer, wide, 6, -10)
        key, offset = viewer.anchor
        narrow = tui.rows_for(viewer, 30)
        start = tui.viewport(viewer, narrow, 6)
        self.assertEqual(narrow[start].key, key)
        self.assertLessEqual(narrow[start].offset, offset)
        self.assertGreater(narrow[start].offset + len(narrow[start].text) + 1, offset)

    def test_frozen_place_keeps_receiving_and_expired_history_is_bounded(self):
        viewer = tui.Viewer()
        viewer.add(event("old passage"))
        tui.move(viewer, tui.rows_for(viewer, 50), 6, 0)
        for index in range(210):
            viewer.add(event(f"Paragraph {index}", session=index + 2))
        rows = tui.rows_for(viewer, 50)
        self.assertFalse(viewer.following)
        self.assertEqual(tui.viewport(viewer, rows, 6), 0)
        self.assertEqual(viewer.unread, 210)
        self.assertEqual(len(viewer.captions), 200)

    def test_search_is_literal_and_navigates_matches_without_filtering_history(self):
        viewer = tui.Viewer()
        viewer.add(event("Find [a] here and [a] again"))
        viewer.add(event("Another paragraph", session=2))
        viewer.query = "[a]"
        tui.search(viewer)
        first = viewer.match
        tui.search(viewer, advance=True)
        self.assertNotEqual(viewer.match, first)
        tui.search(viewer, advance=True)
        self.assertEqual(viewer.match, first)
        self.assertEqual(len(viewer.captions), 2)
        self.assertFalse(viewer.following)
        viewer.query = "missing"
        tui.search(viewer)
        self.assertIsNone(viewer.match)

    def test_copy_uses_clean_paragraph_without_timestamps(self):
        viewer = tui.Viewer()
        viewer.add(event("first words"))
        viewer.add(event("continued", source=108))
        viewer.add(event("latest paragraph", session=2))
        rows = tui.rows_for(viewer, 50)
        self.assertEqual(tui.copy_text(viewer, rows, 2), "latest paragraph")
        viewer.query = "first"
        tui.search(viewer)
        self.assertEqual(tui.copy_text(viewer, rows, 2), "first words continued")

    def test_status_distinguishes_connection_audio_and_explicit_transcribing_state(self):
        viewer = tui.Viewer()
        self.assertEqual(viewer.status(100), "Connecting")
        viewer.connected = True
        self.assertEqual(viewer.status(100), "Ready")
        viewer.audio(bytes(320))
        now = tui.time.monotonic()
        self.assertEqual(viewer.status(now), "Listening")
        viewer.add({"type": "status", "stt_in_flight": True})
        self.assertEqual(viewer.status(now), "Transcribing")
        self.assertEqual(viewer.status(now + 10), "Ready")
        viewer.connected, viewer.connections = False, 1
        self.assertEqual(viewer.status(now), "Reconnecting")

    def test_partial_replaces_draft_but_does_not_modify_committed_text(self):
        viewer = tui.Viewer()
        viewer.add(event("Committed words"))
        viewer.add({"type": "partial", "text": "a draft"})
        viewer.add({"type": "partial", "text": "the revised draft"})
        self.assertEqual(viewer.captions[0].text, "Committed words")
        self.assertEqual(viewer.draft, "the revised draft")
        row = tui.rows_for(viewer, 76)[0]
        self.assertEqual(row.text, "Committed words the revised draft")
        self.assertEqual(row.soft_from, len("Committed words "))
        viewer.add(event("Final words", source=108))
        self.assertEqual(viewer.draft, "")

    def test_help_and_search_controls_are_on_demand(self):
        viewer = tui.Viewer()
        tui.handle_key(viewer, "?", [], 10)
        self.assertTrue(viewer.help_open)
        tui.handle_key(viewer, "\x1b", [], 10)
        self.assertFalse(viewer.help_open)
        tui.handle_key(viewer, "/", [], 10)
        for char in "hello":
            tui.handle_key(viewer, char, [], 10)
        self.assertEqual(viewer.query, "hello")
        tui.handle_key(viewer, "\x7f", [], 10)
        self.assertEqual(viewer.query, "hell")
        tui.handle_key(viewer, "\n", [], 10)
        self.assertFalse(viewer.searching)
        self.assertEqual(tui.handle_key(viewer, "q", [], 10), "quit")

    def test_invalid_events_and_pcm_do_not_damage_viewer(self):
        viewer = tui.Viewer()
        for value in (None, [], {"type": "unknown", "text": "ignore"}, {"text": ""}):
            viewer.add(value)
        viewer.audio(b"x")
        self.assertIsNone(viewer.last_audio)
        viewer.audio(bytes(320))
        self.assertIsNotNone(viewer.last_audio)
        self.assertFalse(viewer.captions)


class Screen:
    def __init__(self, height=26, width=100):
        self.height, self.width = height, width
        self.erase()

    def erase(self):
        self.lines = [[" "] * self.width for _ in range(self.height)]
        self.cleared = []

    def move(self, y, x):
        self.cursor = (y, x)

    def clearok(self, flag):
        pass

    def clrtoeol(self):
        y, x = self.cursor
        self.lines[y][x:] = [" "] * (self.width - x)
        self.cleared.append(y)

    def getmaxyx(self):
        return self.height, self.width

    def addstr(self, y, x, text, attr=0):
        for index, char in enumerate(text):
            self.lines[y][x + index] = char

    def noutrefresh(self):
        pass

    @property
    def text(self):
        return "\n".join("".join(line).rstrip() for line in self.lines)


class StreamOrderingTests(unittest.TestCase):
    def test_earlier_final_or_clear_cannot_erase_newer_draft(self):
        viewer = tui.Viewer()
        viewer.add({"type": "partial", "text": "Next thought", "stream_id": "s",
                    "session_id": 1, "utterance_id": "new", "revision": 1})
        viewer.add(event("Earlier final.", stream_id="s", utterance_id="old"))
        viewer.add({"type": "partial", "text": "", "stream_id": "s", "utterance_id": "old"})
        self.assertEqual(viewer.draft, "Next thought")
        viewer.add(event("Next thought finalized.", source=108, stream_id="s", utterance_id="new"))
        self.assertEqual(viewer.draft, "")
        viewer.add({"type": "partial", "text": "Late obsolete words", "stream_id": "s",
                    "utterance_id": "new", "revision": 99})
        self.assertEqual(viewer.draft, "")

    def test_partial_revisions_do_not_regress_and_new_session_stays_separate(self):
        viewer = tui.Viewer()
        viewer.add(event("Earlier conversation", stream_id="old"))
        draft = {"type": "partial", "stream_id": "new", "session_id": 1, "utterance_id": "u"}
        viewer.add({**draft, "text": "Current words", "revision": 2})
        viewer.add({**draft, "text": "Old words", "revision": 1})
        self.assertEqual(viewer.draft, "Current words")
        rows = tui.rows_for(viewer, 80)
        self.assertEqual(rows[0].text, "Earlier conversation")
        self.assertEqual(rows[-1].text, "Current words")
        self.assertEqual(viewer.captions[0].text, "Earlier conversation")

    def test_draft_starts_new_paragraph_after_sentence_at_paragraph_limit(self):
        viewer = tui.Viewer()
        viewer.add(event("word " * 89 + "end."))
        viewer.add({"type": "partial", "text": "New paragraph"})
        self.assertEqual(tui.rows_for(viewer, 100)[-1].text, "New paragraph")


class LayoutTests(unittest.TestCase):
    def test_history_reclaims_live_cushion_and_space_releases_then_restores_mouse(self):
        self.assertEqual(tui.geometry(30, 100, False)[3] - tui.geometry(30, 100, True)[3], 2)
        viewer = tui.Viewer(mouse_enabled=True)
        viewer.add(event("Words to select"))
        tui.handle_key(viewer, " ", tui.rows_for(viewer, 50), 20)
        self.assertTrue(viewer.holding)
        self.assertFalse(viewer.mouse_enabled)
        tui.handle_key(viewer, "g", tui.rows_for(viewer, 50), 20)
        self.assertFalse(viewer.holding)
        self.assertTrue(viewer.mouse_enabled)

    def test_copy_while_held_does_not_include_new_hidden_words(self):
        viewer = tui.Viewer()
        viewer.add(event("Visible words"))
        rows = tui.rows_for(viewer, 50)
        tui.handle_key(viewer, " ", rows, 20)
        viewer.add(event("arrived after freezing", source=108))
        self.assertEqual(tui.copy_text(viewer, tui.rows_for(viewer, 50), 20), "Visible words")
        tui.handle_key(viewer, "g", rows, 20)
        self.assertEqual(tui.copy_text(viewer, tui.rows_for(viewer, 50), 20),
                         "Visible words arrived after freezing")

    def test_repeated_frames_do_not_write_and_only_draft_row_changes(self):
        viewer = tui.Viewer(show_times=False)
        viewer.add(event("An earlier paragraph.", session=1))
        viewer.add(event("Current words", session=2))
        viewer.add({"type": "partial", "text": "first draft"})
        screen = Screen()
        with mock.patch.object(tui.curses, "doupdate") as update:
            tui.paint(screen, viewer, {"text": 0, "muted": 1, "accent": 2})
            screen.cleared.clear()
            tui.paint(screen, viewer, {"text": 0, "muted": 1, "accent": 2})
            self.assertEqual(update.call_count, 1)
            self.assertFalse(screen.cleared)
            viewer.add({"type": "partial", "text": "revised draft"})
            tui.paint(screen, viewer, {"text": 0, "muted": 1, "accent": 2})
            self.assertEqual(screen.cleared, [3])
            self.assertEqual(update.call_count, 2)
            self.assertIn("An earlier paragraph.", screen.text)
            self.assertIn("Current words revised draft", screen.text)
            self.assertNotIn("first draft", screen.text)

    def test_sync_output_resets_after_refresh_failure_and_has_plain_fallback(self):
        import io
        viewer = tui.Viewer(sync_output="on")
        screen = Screen()
        output = io.StringIO()
        with mock.patch.object(tui.sys, "stdout", output), \
             mock.patch.object(output, "isatty", return_value=True), \
             mock.patch.object(tui.curses, "doupdate", side_effect=tui.curses.error("refresh failed")):
            with self.assertRaises(tui.curses.error):
                tui.paint(screen, viewer, {"text": 0, "muted": 0, "accent": 0})
        self.assertEqual(output.getvalue(), "\x1b[?2026h\x1b[?2026l")
        self.assertIsNone(viewer.paint_cache)
        output = io.StringIO()
        viewer.sync_output = "off"
        with mock.patch.object(tui.sys, "stdout", output), \
             mock.patch.object(output, "isatty", return_value=True), \
             mock.patch.object(tui.curses, "doupdate"):
            tui.paint(screen, viewer, {"text": 0, "muted": 0, "accent": 0})
        self.assertEqual(output.getvalue(), "")
        self.assertIn("Your next words", screen.text)

    def test_sync_auto_uses_conservative_terminal_defaults(self):
        self.assertTrue(tui.synchronized_output("auto", {"TERM_PROGRAM": "ghostty"}))
        self.assertTrue(tui.synchronized_output("auto", {"TERM_PROGRAM": "WezTerm"}))
        self.assertFalse(tui.synchronized_output("auto", {"TERM_PROGRAM": "Apple_Terminal"}))
        self.assertFalse(tui.synchronized_output("auto", {"TERM_PROGRAM": "ghostty", "TMUX": "session"}))
        self.assertTrue(tui.synchronized_output("on", {"TMUX": "session"}))
    def test_timestamp_shares_first_text_row_and_does_not_repeat_on_wrapped_lines(self):
        viewer = tui.Viewer(show_times=True)
        viewer.add(event("Readable words " * 20, timestamp="2026-10-24 22:47:38"))
        rows = tui.rows_for(viewer, 30)
        self.assertEqual(rows[0].timestamp, "22:47:38")
        self.assertTrue(rows[0].text.startswith("Readable words"))
        self.assertTrue(all(not row.timestamp for row in rows[1:]))
        before = [(row.key, row.offset, row.text) for row in rows]
        viewer.show_times = False
        hidden = tui.rows_for(viewer, 30)
        self.assertEqual(before, [(row.key, row.offset, row.text) for row in hidden])
        self.assertTrue(all(not row.timestamp for row in hidden))

    def test_wide_layout_has_half_width_transcript_and_equal_margins(self):
        for width in (80, 100, 160, 240):
            column, left, top, available = tui.geometry(40, width)
            self.assertEqual(column, width // 2)
            self.assertLessEqual(abs(left - (width - left - column)), 1)
            self.assertEqual(top + available, 36)

    def test_native_selection_default_and_hold_keep_reception_and_reading_position(self):
        viewer = tui.Viewer()
        self.assertTrue(viewer.mouse_enabled)
        for i in range(30):
            viewer.add(event(f"Passage {i}", session=i))
        rows = tui.rows_for(viewer, 50)
        tui.handle_key(viewer, tui.curses.KEY_HOME, rows, 10)
        self.assertFalse(viewer.following)
        self.assertEqual(tui.viewport(viewer, rows, 10), 0)
        tui.handle_key(viewer, " ", rows, 10)
        self.assertTrue(viewer.holding)
        anchor = viewer.anchor
        viewer.add(event("Still arriving while selecting", session=99))
        self.assertEqual(viewer.anchor, anchor)
        self.assertEqual(viewer.captions[-1].text, "Still arriving while selecting")
        tui.handle_key(viewer, " ", tui.rows_for(viewer, 50), 10)
        self.assertTrue(viewer.following)
        self.assertFalse(viewer.holding)

    def test_page_shows_only_transcript_and_time_without_header_or_shortcut_bar(self):
        viewer = tui.Viewer(connected=True)
        viewer.add(event("A calm page of words, with enough room to read."))
        screen = Screen()
        with mock.patch.object(tui.curses, "doupdate"):
            tui.paint(screen, viewer, {"text": 0, "muted": 0, "accent": 0})
        self.assertNotIn("omi", screen.text)
        self.assertNotIn("Ready", screen.text)
        self.assertNotIn("●", screen.text)
        self.assertIn("A calm page of words", screen.text)
        self.assertNotIn("COMPLETED TRANSCRIPTS", screen.text)
        self.assertNotIn("q quit", screen.text)
        self.assertNotIn("▰", screen.text)
        self.assertNotIn("─", screen.text)
        self.assertEqual(screen.text.count("12:00:00"), 1)
        self.assertEqual(screen.lines[1].index("A"), 25)
        self.assertEqual("".join(screen.lines[1][15:23]), "12:00:00")
        self.assertNotIn("↑↓ scroll", screen.text)
        self.assertNotIn("y copy", screen.text)

    def test_narrow_screen_and_help_do_not_draw_out_of_bounds(self):
        for height, width in ((8, 20), (10, 32), (14, 40), (30, 120)):
            with self.subTest(size=(height, width)), mock.patch.object(tui.curses, "doupdate"):
                viewer = tui.Viewer(help_open=True)
                tui.paint(Screen(height, width), viewer, {"text": 0, "muted": 0, "accent": 0})


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_receives_existing_transcripts_without_sending_audio_or_api_calls(self):
        received = []
        async def endpoint(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_bytes(bytes(320))
            await ws.send_json({"type": "transcript", "event_id": "a", "text": "Blue lantern seven."})
            await ws.send_str("not-json")
            async for message in ws:
                received.append(message)
            return ws

        app = web.Application()
        app.router.add_get("/listen-ws", endpoint)
        server = TestServer(app)
        await server.start_server()
        viewer = tui.Viewer()
        task = asyncio.create_task(tui.receive(viewer, str(server.make_url("/listen-ws")).replace("http://", "ws://")))
        try:
            async with asyncio.timeout(2):
                while not viewer.captions:
                    await asyncio.sleep(.01)
            self.assertTrue(viewer.connected)
            self.assertEqual(viewer.captions[-1].text, "Blue lantern seven.")
            self.assertIsNotNone(viewer.last_audio)
            self.assertEqual(received, [])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await server.close()

    async def test_reconnect_marks_unreplayable_connection_gap(self):
        async def endpoint(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.close()
            return ws

        app = web.Application()
        app.router.add_get("/listen-ws", endpoint)
        server = TestServer(app)
        await server.start_server()
        viewer = tui.Viewer()
        task = asyncio.create_task(tui.receive(viewer, str(server.make_url("/listen-ws")).replace("http://", "ws://")))
        try:
            async with asyncio.timeout(3):
                while viewer.connections < 2:
                    await asyncio.sleep(.01)
            self.assertEqual(viewer.captions[-1].kind, "gap")
            self.assertIn("missing", viewer.captions[-1].text)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await server.close()

    async def test_copy_uses_pbcopy_only_when_invoked_and_reports_success(self):
        viewer = tui.Viewer()
        process = mock.Mock(returncode=0)
        process.communicate = mock.AsyncMock()
        with mock.patch.object(tui.asyncio, "create_subprocess_exec", new_callable=mock.AsyncMock, return_value=process) as create:
            await tui.copy_paragraph(viewer, "A clean paragraph.")
        self.assertEqual(create.call_args.args, ("/usr/bin/pbcopy",))
        process.communicate.assert_awaited_once_with(b"A clean paragraph.")
        self.assertEqual(viewer.toast, "Copied")


@unittest.skipUnless(os.name == "posix", "The viewer uses POSIX curses terminals")
class TerminalTests(unittest.TestCase):
    def test_help_search_resize_and_quit_restore_original_terminal_settings(self):
        import fcntl
        import pty
        import select
        import termios
        master, slave = pty.openpty()
        process = None
        try:
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 25, 100, 0, 0))
            before = termios.tcgetattr(slave)
            process = subprocess.Popen(
                [sys.executable, "-B", str(Path(tui.__file__).resolve()), "--url", "ws://127.0.0.1:1/listen-ws"],
                stdin=slave, stdout=slave, stderr=slave, env={**os.environ, "TERM": "xterm-256color"})
            output, stage = bytearray(), 0
            deadline = time.monotonic() + 6
            while process.poll() is None and time.monotonic() < deadline:
                if select.select([master], [], [], .05)[0]:
                    output.extend(os.read(master, 65536))
                if stage == 0 and b"Your next words" in output:
                    os.write(master, b"?")
                    stage = 1
                elif stage == 1 and b"Copy this paragraph" in output:
                    os.write(master, b"?/needle")
                    stage = 2
                elif stage == 2 and b"no match" in output:
                    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 18, 60, 0, 0))
                    import signal
                    process.send_signal(signal.SIGWINCH)
                    os.write(master, b"\rgq")
                    stage = 3
            self.assertEqual(stage, 3, "Terminal did not respond to its reading controls")
            self.assertEqual(process.poll(), 0, "Viewer did not exit cleanly")
            after = termios.tcgetattr(slave)
            after[3] &= ~getattr(termios, "PENDIN", 0)
            before[3] &= ~getattr(termios, "PENDIN", 0)
            self.assertEqual(after, before)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            os.close(master)
            os.close(slave)


class PageDetailTests(unittest.TestCase):
    def test_only_latest_chunk_softens_and_all_words_remain_available(self):
        viewer = tui.Viewer(show_times=False)
        with mock.patch.object(tui.time, "monotonic", return_value=100):
            viewer.add(event("Settled words"))
            viewer.add(event("newest words", source=108))
            row = tui.rows_for(viewer, 76)[0]
            self.assertEqual(row.text, "Settled words newest words")
            self.assertEqual(row.soft_from, len("Settled words "))
        with mock.patch.object(tui.time, "monotonic", return_value=102.1):
            self.assertIsNone(tui.rows_for(viewer, 76)[0].soft_from)

    def test_ellipsis_is_reported_work_not_an_animation_or_inferred_microphone(self):
        viewer = tui.Viewer(connected=True, show_times=False)
        with mock.patch.object(tui.time, "monotonic", return_value=100):
            viewer.add(event("1234567890"))
            self.assertFalse(viewer.pending(100))
            viewer.add({"type": "status", "buffer_sec": 1, "stt_in_flight": False})
            rows = tui.rows_for(viewer, 10)
            self.assertEqual(rows[0].text, "1234567890")
            self.assertEqual(rows[1].text, "…")
            viewer.add({"type": "status", "buffer_sec": 0, "stt_in_flight": False})
            self.assertNotIn("…", [r.text for r in tui.rows_for(viewer, 10)])
            viewer.add({"type": "status", "buffer_sec": 0, "stt_in_flight": True})
            self.assertTrue(viewer.pending(100))
        self.assertFalse(viewer.pending(106))

    def test_pending_work_in_another_session_does_not_animate_the_old_paragraph(self):
        viewer = tui.Viewer(connected=True)
        viewer.add(event("Earlier session", stream_id="old"))
        viewer.add({"type": "status", "buffer_sec": 5, "stream_id": "new", "session_id": 1})
        self.assertFalse(viewer.pending(tui.time.monotonic()))

    def test_replay_is_neutral_deduplicated_and_retains_resume_cursor(self):
        viewer = tui.Viewer()
        viewer.add({"type": "replay", "phase": "start", "reset": False})
        viewer.add(event("Saved text", event_id="one"))
        viewer.add(event("Saved text", event_id="one"))
        viewer.add({"type": "replay", "phase": "end"})
        self.assertTrue(viewer.replay_supported)
        self.assertFalse(viewer.replaying)
        self.assertEqual(len(viewer.captions), 1)
        self.assertEqual(viewer.last_event_id, "one")
        self.assertEqual(viewer.captions[0].soft_until, 0)

    def test_unavailable_replay_is_a_visible_note_not_silent_loss(self):
        viewer = tui.Viewer(last_event_id="expired")
        viewer.add({"type": "replay", "phase": "start", "reset": True})
        self.assertEqual(viewer.captions[0].kind, "gap")
        self.assertIn("outside", viewer.captions[0].text)
        self.assertIsNone(viewer.last_event_id)

    def test_sentence_breaks_are_preferred_but_unpunctuated_speech_still_has_a_hard_cap(self):
        viewer = tui.Viewer()
        viewer.add(event("word " * 89 + "done. " + "word " * 80))
        self.assertEqual([len(c.words) for c in viewer.captions], [90, 80])
        self.assertEqual(viewer.captions[0].words[-1], "done.")
        other = tui.Viewer()
        other.add(event("word " * 320))
        self.assertEqual([len(c.words) for c in other.captions], [150, 150, 20])

    def test_editorial_times_are_not_shown_on_every_processing_chunk_or_paragraph(self):
        viewer = tui.Viewer()
        viewer.add(event("word " * 150, source=100))
        viewer.add(event("another paragraph", source=108))
        self.assertEqual([c.show_stamp for c in viewer.captions], [True, False])
        viewer.add(event("Three minutes later", source=300))
        self.assertTrue(viewer.captions[-1].show_stamp)
        viewer.add(event("[gap]", type="gap"))
        viewer.add(event("After an interruption", source=308))
        self.assertTrue(viewer.captions[-1].show_stamp)

    def test_unread_counts_passages_instead_of_every_update_to_one_passage(self):
        viewer = tui.Viewer()
        viewer.add(event("Opening words"))
        tui.move(viewer, tui.rows_for(viewer, 50), 8, 0)
        viewer.add(event("another chunk", source=108))
        viewer.add(event("more of that thought", source=116))
        self.assertEqual(viewer.unread, 1)
        viewer.add(event("A new session", session=2, source=124))
        self.assertEqual(viewer.unread, 2)

    def test_focus_details_and_mouse_are_optional(self):
        viewer = tui.Viewer()
        viewer.add(event("Older words"))
        viewer.add(event("Latest words", session=2))
        tui.handle_key(viewer, "f", [], 10)
        rows = tui.rows_for(viewer, 76)
        self.assertEqual(next(r for r in rows if r.text == "Older words").style, "muted")
        self.assertEqual(next(r for r in rows if r.text == "Latest words").style, "text")
        tui.handle_key(viewer, "d", [], 10)
        self.assertTrue(viewer.diagnostics)
        tui.handle_key(viewer, "\x1b", [], 10)
        self.assertFalse(viewer.diagnostics)
        self.assertEqual(tui.handle_key(viewer, "m", [], 10), "mouse")
        self.assertFalse(viewer.mouse_enabled)
        tui.handle_key(viewer, "m", [], 10)
        self.assertTrue(viewer.mouse_enabled)

    def test_caption_url_preserves_other_parameters_and_encodes_opaque_cursor(self):
        url = tui.caption_url("wss://example.test/listen-ws?token=keep&replay=2", "a+b")
        from urllib.parse import parse_qs, urlsplit
        query = parse_qs(urlsplit(url).query)
        self.assertEqual(query, {"token": ["keep"], "mode": ["captions"], "replay": ["100"], "after": ["a+b"]})

    def test_transcript_uses_height_up_to_footer_without_empty_bottom_quarter(self):
        for height in (24, 30, 40):
            _, _, top, available = tui.geometry(height, 100)
            self.assertEqual(top + available, height - 4)

    def test_hiding_times_preserves_an_anchor_on_a_timestamp_row(self):
        viewer = tui.Viewer()
        for i in range(10):
            viewer.add(event(f"Passage {i}", session=i))
        caption = viewer.captions[5]
        viewer.following, viewer.anchor = False, (caption.key, -1)
        viewer.show_times = False
        rows = tui.rows_for(viewer, 76)
        self.assertEqual(rows[tui.viewport(viewer, rows, 4)].key, caption.key)


class ReplayConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_reconnection_resumes_by_event_id_without_a_fake_gap(self):
        requests = []
        connections = []
        async def endpoint(request):
            requests.append(dict(request.query))
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            connections.append(ws)
            number = len(connections)
            await ws.send_json({"type": "replay", "phase": "start", "reset": False})
            await ws.send_json(event(f"Caption {number}", event_id=str(number), source=number * 8))
            await ws.send_json({"type": "replay", "phase": "end"})
            await ws.send_json({"type": "status", "buffer_sec": 0, "stt_in_flight": False})
            async for _ in ws:
                pass
            return ws
        app = web.Application()
        app.router.add_get("/listen-ws", endpoint)
        server = TestServer(app)
        await server.start_server()
        viewer = tui.Viewer()
        task = asyncio.create_task(tui.receive(viewer, str(server.make_url("/listen-ws")).replace("http://", "ws://")))
        try:
            async with asyncio.timeout(3):
                while viewer.last_event_id != "1" or viewer.replaying:
                    await asyncio.sleep(.01)
                await connections[0].close()
                while viewer.last_event_id != "2" or viewer.replaying:
                    await asyncio.sleep(.01)
            self.assertEqual(requests[1]["after"], "1")
            self.assertEqual(requests[0]["mode"], "captions")
            self.assertFalse(any(c.kind == "gap" for c in viewer.captions))
            self.assertEqual(viewer.captions[-1].text, "Caption 1 Caption 2")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await server.close()


class CaptionPageIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_caption_endpoint_replays_then_drives_the_live_ellipsis(self):
        import omi_bridge as bridge
        runtime = bridge.BridgeState(stt_queue=asyncio.Queue(maxsize=24))
        runtime.recent_events.append(event("A saved passage.", event_id="saved", stream_id="old"))
        with mock.patch.object(bridge, "state", runtime), \
             mock.patch.object(bridge, "event_logger"), mock.patch.object(bridge, "transcript_logger"):
            server = TestServer(bridge.create_app(start_workers=False))
            await server.start_server()
            viewer = tui.Viewer()
            reader = asyncio.create_task(tui.receive(viewer, str(server.make_url("/listen-ws")).replace("http://", "ws://")))
            status_worker = asyncio.create_task(bridge.caption_status_worker())
            try:
                async with asyncio.timeout(4):
                    while viewer.last_event_id != "saved" or viewer.replaying or not viewer.last_status:
                        await asyncio.sleep(.01)
                    self.assertEqual(viewer.captions[0].soft_until, 0)
                    self.assertTrue(viewer.replay_supported)
                    bridge.ingest_raw_pcm(bytes(16000))
                    bridge.record_transcript_line("The next thought.", session_id=1, audio_start_at=time.time())
                    while viewer.last_event_id == "saved" or not viewer.pending(tui.time.monotonic()):
                        await asyncio.sleep(.01)
                    rows = tui.rows_for(viewer, 76)
                    self.assertTrue(any(row.suffix == " …" for row in rows))
                    self.assertEqual(viewer.captions[-1].text, "The next thought.")
                    runtime.current_session.pcm.clear()
                    while viewer.pending(tui.time.monotonic()):
                        await asyncio.sleep(.01)
                    self.assertFalse(any(row.suffix for row in tui.rows_for(viewer, 76)))
                    self.assertFalse(any(c.kind == "gap" for c in viewer.captions))
            finally:
                reader.cancel()
                status_worker.cancel()
                await asyncio.gather(reader, status_worker, return_exceptions=True)
                await server.close()


if __name__ == "__main__":
    unittest.main()
