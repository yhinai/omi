"""Additive viewer features: wrap cache, clipboard routing, search counts, export, diagnostics."""
import asyncio
from collections import deque
import curses
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import tui


def event(text, session=1, source=100, **kwargs):
    return {"text": text, "session_id": session, "audio_start_at": source,
            "timestamp": "2026-11-09 22:52:36", **kwargs}


class WrapCacheTests(unittest.TestCase):
    def test_second_pass_does_not_rewrap_retained_history(self):
        viewer = tui.Viewer()
        for index in range(5):
            viewer.add(event("word " * 40, session=index, event_id=f"e{index}"))
        tui.rows_for(viewer, 40)
        with mock.patch.object(tui, "wrap", side_effect=AssertionError("history was re-wrapped")):
            tui.rows_for(viewer, 40)

    def test_new_words_invalidate_only_their_own_caption(self):
        viewer = tui.Viewer()
        viewer.add(event("first passage", event_id="a"))
        viewer.add(event("second passage " * 5, session=2, source=200, event_id="b"))
        tui.rows_for(viewer, 40)
        viewer.add(event("second passage " * 5 + "extra", session=2, source=208, event_id="c"))
        with mock.patch.object(tui, "wrap", side_effect=AssertionError("stale cache reused")):
            with self.assertRaises(AssertionError):
                tui.rows_for(viewer, 40)

    def test_resize_invalidates_the_wrap(self):
        viewer = tui.Viewer()
        viewer.add(event("wide words " * 10))
        tui.rows_for(viewer, 60)
        with mock.patch.object(tui, "wrap", side_effect=AssertionError("stale width reused")):
            with self.assertRaises(AssertionError):
                tui.rows_for(viewer, 30)

    def test_rows_are_identical_to_uncached_wrapping(self):
        viewer = tui.Viewer()
        viewer.add(event("A sentence that is long enough to wrap across the column, twice over. " * 3))
        cached = tui.rows_for(viewer, 33)
        uncached = [(r.key, r.offset, r.text, r.style) for r in cached]
        for caption in viewer.captions:
            caption._cache = None
        self.assertEqual(uncached, [(r.key, r.offset, r.text, r.style) for r in tui.rows_for(viewer, 33)])

    def test_full_history_pass_is_fast(self):
        viewer = tui.Viewer()
        for index in range(200):
            viewer.add(event("word " * 149 + f"end{index}", session=index, event_id=f"e{index}"))
        tui.rows_for(viewer, 60)
        start = time.perf_counter()
        for _ in range(20):
            tui.rows_for(viewer, 60)
        per_pass = (time.perf_counter() - start) / 20
        self.assertLess(per_pass, 0.02, f"rows_for took {per_pass*1000:.1f} ms per pass")


class ClipboardTests(unittest.TestCase):
    def test_auto_prefers_osc52_over_ssh_and_tmux(self):
        self.assertEqual(tui.clipboard_mode("auto", {}), "pbcopy")
        self.assertEqual(tui.clipboard_mode("auto", {"SSH_CONNECTION": "a b c d"}), "osc52")
        self.assertEqual(tui.clipboard_mode("auto", {"SSH_TTY": "/dev/ttys001"}), "osc52")
        self.assertEqual(tui.clipboard_mode("auto", {"TMUX": "/tmp/tmux"}), "osc52")
        self.assertEqual(tui.clipboard_mode("pbcopy", {"SSH_CONNECTION": "a"}), "pbcopy")

    def test_osc52_writes_a_base64_escape(self):
        viewer = tui.Viewer()
        with mock.patch.object(tui.sys.stdout, "write") as write, \
             mock.patch.object(tui.sys.stdout, "flush"):
            asyncio.run(tui.copy_paragraph(viewer, "hello", mode="osc52"))
        self.assertEqual(write.call_args.args[0], "\x1b]52;c;aGVsbG8=\x07")
        self.assertEqual(viewer.toast, "Copied")

    def test_oversized_payload_falls_back_to_pbcopy(self):
        viewer = tui.Viewer()
        process = mock.Mock(returncode=0)
        process.communicate = mock.AsyncMock()
        with mock.patch.object(tui.asyncio, "create_subprocess_exec",
                               new_callable=mock.AsyncMock, return_value=process) as create:
            asyncio.run(tui.copy_paragraph(viewer, "x" * (tui.OSC52_LIMIT + 10), mode="osc52"))
        self.assertEqual(create.call_args.args, ("/usr/bin/pbcopy",))

    def test_empty_text_says_so(self):
        viewer = tui.Viewer()
        asyncio.run(tui.copy_paragraph(viewer, "", mode="osc52"))
        self.assertEqual(viewer.toast, "Nothing to copy")


class SearchCountTests(unittest.TestCase):
    def test_match_numbers_are_tracked(self):
        viewer = tui.Viewer()
        viewer.add(event("alpha one", event_id="a"))
        viewer.add(event("beta alpha", session=2, source=200, event_id="b"))
        viewer.query = "alpha"
        tui.search(viewer)
        self.assertEqual((viewer.match_number, viewer.match_total), (1, 2))
        tui.search(viewer, advance=True)
        self.assertEqual((viewer.match_number, viewer.match_total), (2, 2))
        viewer.query = "gamma"
        tui.search(viewer)
        self.assertEqual((viewer.match_number, viewer.match_total), (0, 0))

    def test_footer_reports_position_in_results(self):
        viewer = tui.Viewer(connected=True)
        viewer.add(event("alpha one", event_id="a"))
        viewer.add(event("beta alpha", session=2, source=200, event_id="b"))
        viewer.searching, viewer.query = True, "alpha"
        tui.search(viewer)
        screen = Screen()
        with mock.patch.object(tui.curses, "doupdate"):
            tui.paint(screen, viewer, {"text": 0, "muted": 1, "accent": 2})
        self.assertIn("/ alpha  · 1/2", screen.text)


class SessionTests(unittest.TestCase):
    def test_session_text_keeps_stamps_and_paragraph_breaks(self):
        viewer = tui.Viewer()
        viewer.add(event("first words", event_id="a"))
        viewer.add(event("latest", session=2, source=400, event_id="b"))
        text = tui.session_text(viewer)
        self.assertTrue(text.startswith("[22:52:36] first words"))
        self.assertIn("\n\n", text)

    def test_export_writes_markdown_and_reports_the_name(self):
        viewer = tui.Viewer()
        viewer.add(event("exported words", event_id="a"))
        with tempfile.TemporaryDirectory() as directory:
            message = tui.export_session(viewer, Path(directory))
            self.assertRegex(message, r"Exported omi-captions-\d{8}-\d{6}\.md")
            written = list(Path(directory).glob("omi-captions-*.md"))
            self.assertEqual(len(written), 1)
            self.assertIn("exported words", written[0].read_text())

    def test_empty_session_exports_nothing(self):
        self.assertEqual(tui.export_session(tui.Viewer(), Path("/tmp")), "Nothing to export")

    def test_copy_all_and_export_keys(self):
        viewer = tui.Viewer()
        self.assertEqual(tui.handle_key(viewer, "Y", [], 10), "copy_all")
        self.assertEqual(tui.handle_key(viewer, "e", [], 10), "export")


class DiagnosisTests(unittest.TestCase):
    def test_status_event_feeds_the_diagnostics_page(self):
        viewer = tui.Viewer(connected=True, diagnostics=True)
        viewer.add({"type": "status", "stt_in_flight": False, "last_packet_age_sec": 0.1,
                    "model": "whisper-large-v3-turbo-fp16", "latency_final_p50_sec": 2.478,
                    "latency_speech_start_to_first_partial_p50_sec": .456,
                    "latency_speech_end_to_final_p50_sec": .812,
                    "audio_duty": 0.951, "workers_alive": True, "last_error": None})
        screen = Screen()
        with mock.patch.object(tui.curses, "doupdate"):
            tui.paint(screen, viewer, {"text": 0, "muted": 1, "accent": 2})
        self.assertIn("whisper-large-v3-turbo-fp16", screen.text)
        self.assertIn("2.48s", screen.text)
        self.assertIn("Receipt → final p50: 2.48s", screen.text)
        self.assertIn("Speech → draft p50: 0.46s", screen.text)
        self.assertIn("Speech end → final p50: 0.81s", screen.text)
        self.assertIn("0.951", screen.text)
        self.assertIn("Workers: alive", screen.text)
        self.assertIn("Last error: none", screen.text)

    def test_unknown_or_invalid_latency_does_not_masquerade_as_a_measurement(self):
        viewer = tui.Viewer()
        viewer.add({"type": "status", "latency_final_p50_sec": 2.0,
                    "latency_speech_end_to_final_p50_sec": .8,
                    "latency_speech_start_to_first_partial_p50_sec": .4})
        viewer.add({"type": "status", "latency_final_p50_sec": None,
                    "latency_speech_end_to_final_p50_sec": None,
                    "latency_speech_start_to_first_partial_p50_sec": None})
        self.assertIsNone(viewer.p50)
        self.assertIsNone(viewer.speech_final_p50)
        self.assertIsNone(viewer.speech_draft_p50)
        for invalid in (True, -1, float("nan"), float("inf"), "0.2"):
            viewer.add({"type": "status", "latency_speech_end_to_final_p50_sec": invalid})
            self.assertIsNone(viewer.speech_final_p50)

    def test_degraded_bridge_notifies_once_then_goes_quiet(self):
        viewer = tui.Viewer(connected=True)
        viewer.add({"type": "status", "last_error": "TranscriptionOutputFailed"})
        self.assertEqual(viewer.toast, "Bridge degraded · d for details")
        viewer.toast_until = 0
        viewer.add({"type": "status", "last_error": "TranscriptionOutputFailed"})
        self.assertEqual(viewer.toast_until, 0)
        self.assertEqual(viewer.bridge_error, "TranscriptionOutputFailed")

    def test_a_healthy_bridge_arms_the_warning_again(self):
        viewer = tui.Viewer(connected=True)
        viewer.add({"type": "status", "last_error": "boom"})
        viewer.add({"type": "status", "last_error": None})
        viewer.toast_until = 0
        viewer.add({"type": "status", "last_error": "boom again"})
        self.assertEqual(viewer.toast, "Bridge degraded · d for details")

    def test_page_stays_chrome_free_when_the_bridge_is_healthy(self):
        viewer = tui.Viewer(connected=True)
        viewer.add({"type": "status", "stt_in_flight": False, "last_packet_age_sec": 0.1,
                    "model": "whisper-large-v3-turbo-fp16", "latency_final_p50_sec": 2.4,
                    "audio_duty": 0.9, "workers_alive": True})
        viewer.add(event("A calm page", event_id="a"))
        screen = Screen()
        with mock.patch.object(tui.curses, "doupdate"):
            tui.paint(screen, viewer, {"text": 0, "muted": 1, "accent": 2})
        self.assertNotIn("●", screen.text)
        self.assertNotIn("Live", screen.text)
        self.assertNotIn("p50", screen.text)
        self.assertIn("A calm page", screen.text)


class Screen:
    def __init__(self, height=26, width=100):
        self.height, self.width = height, width
        self.erase()

    def erase(self):
        self.lines = [[" "] * self.width for _ in range(self.height)]
        self.attrs = [[0] * self.width for _ in range(self.height)]
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
            if 0 <= y < self.height and 0 <= x + index < self.width:
                self.lines[y][x + index] = char
                self.attrs[y][x + index] = attr

    def noutrefresh(self):
        pass

    @property
    def text(self):
        return "\n".join("".join(line).rstrip() for line in self.lines)


class PauseBreakTests(unittest.TestCase):
    """A silence of two seconds or more opens a new paragraph."""

    def final(self, viewer, text, source, duration, session=1, uid="u", event_id=None):
        viewer.add({"type": "transcript", "text": text, "session_id": session,
                    "stream_id": "s", "utterance_id": uid, "event_id": event_id or uid,
                    "audio_start_at": source, "timestamp": "2026-11-09 22:52:36",
                    "sample_offset": 0, "sample_end_offset": int(duration * 16000)})

    def test_two_second_pause_starts_a_new_paragraph(self):
        viewer = tui.Viewer()
        self.final(viewer, "First thought.", 100, 3)            # audio ends at 103
        self.final(viewer, "After the pause.", 105.1, 2, uid="b", event_id="b")  # 2.1 s later
        self.assertEqual([c.text for c in viewer.captions], ["First thought.", "After the pause."])
        rows = tui.rows_for(viewer, 60)
        self.assertEqual([row.text for row in rows[:3]],
                         ["First thought.", "", "After the pause."])

    def test_exactly_two_seconds_is_still_a_break(self):
        viewer = tui.Viewer()
        self.final(viewer, "First.", 100, 3)
        self.final(viewer, "Second.", 105.0, 1, uid="b", event_id="b")
        self.assertEqual(len(viewer.captions), 2)

    def test_shorter_pause_keeps_flowing(self):
        viewer = tui.Viewer()
        self.final(viewer, "First thought.", 100, 3)
        self.final(viewer, "with more words", 104.9, 2, uid="b", event_id="b")  # 1.9 s
        self.assertEqual(len(viewer.captions), 1)
        self.assertEqual(viewer.captions[0].text, "First thought. with more words")

    def test_draft_after_a_pause_gets_its_own_paragraph(self):
        viewer = tui.Viewer()
        self.final(viewer, "Settled words.", 100, 3)   # ends at 103
        viewer.add({"type": "partial", "text": "A new thought", "stream_id": "s",
                    "session_id": 1, "utterance_id": "d", "audio_start_at": 105.8})
        rows = tui.rows_for(viewer, 76)
        self.assertEqual(rows[0].text, "Settled words.")
        self.assertTrue(any(row.text == "A new thought" for row in rows))
        self.assertFalse(any("Settled words. A new thought" == row.text for row in rows))

    def test_events_without_offsets_keep_the_legacy_window(self):
        viewer = tui.Viewer()
        viewer.add(event("No timing here", source=100, event_id="a"))
        viewer.add(event("still the same thought", source=108, event_id="b"))
        self.assertEqual(len(viewer.captions), 1)


class SelectionTests(unittest.TestCase):
    """Drag-selection inside the page, with the wheel still scrolling."""

    def setUp(self):
        self.viewer = tui.Viewer()
        self.viewer.add(event("alpha beta gamma", event_id="a"))
        self.viewer.add(event("delta epsilon", session=2, source=400, event_id="b"))
        self.rows = tui.rows_for(self.viewer, 40)
        self.first, self.second = self.viewer.captions[0].key, self.viewer.captions[1].key

    def test_selection_within_one_paragraph(self):
        self.viewer.selection = ((self.first, 0), (self.first, 11))
        self.assertEqual(tui.selection_text(self.viewer), "alpha beta")

    def test_selection_spans_paragraphs_with_a_blank_line(self):
        self.viewer.selection = ((self.first, 6), (self.second, 5))
        self.assertEqual(tui.selection_text(self.viewer), "beta gamma\n\ndelta")

    def test_selection_is_ordered_regardless_of_drag_direction(self):
        backwards = tui.Viewer()
        backwards.add(event("alpha beta gamma", event_id="a"))
        backwards.selection = ((backwards.captions[0].key, 11), (backwards.captions[0].key, 0))
        self.assertEqual(tui.selection_text(backwards), "alpha beta")

    def test_empty_selection_copies_nothing(self):
        self.viewer.selection = ((self.first, 5), (self.first, 5))
        self.assertEqual(tui.selection_text(self.viewer), "")

    def test_screen_cell_maps_to_a_text_position(self):
        position = tui.position_at(self.rows, 10, 1, 25, 1, 31, 0)
        self.assertEqual(position, (self.first, 6))

    def test_position_outside_the_body_is_ignored(self):
        self.assertIsNone(tui.position_at(self.rows, 10, 1, 25, 40, 30, 0))

    def test_selected_cells_are_reverse_video(self):
        self.viewer.selection = ((self.first, 6), (self.first, 16))
        screen = Screen()
        with mock.patch.object(tui.curses, "doupdate"):
            tui.paint(screen, self.viewer, {"text": 0, "muted": 1, "accent": 2})
        line = next(index for index, text in enumerate(screen.text.splitlines()) if "beta gamma" in text)
        start = screen.text.splitlines()[line].index("beta")
        self.assertEqual(screen.lines[line][start], "b")
        self.assertEqual(self.attrs(screen, line, start), tui.curses.A_REVERSE)
        self.assertNotEqual(self.attrs(screen, line, start - 1), tui.curses.A_REVERSE)

    def attrs(self, screen, y, x):
        return screen.attrs[y][x]

    def test_scrolling_keeps_the_selection(self):
        self.viewer.selection = ((self.first, 0), (self.first, 5))
        tui.move(self.viewer, self.rows, 5, -1)
        self.assertIsNotNone(self.viewer.selection)
        self.assertEqual(tui.selection_text(self.viewer), "alpha")

    def test_returning_live_clears_the_selection(self):
        self.viewer.selection = ((self.first, 0), (self.first, 5))
        self.viewer.live()
        self.assertIsNone(self.viewer.selection)


class ClockTests(unittest.TestCase):
    """One clock, top right of the reading column, seconds included."""

    def paint_into(self, screen, viewer, clock):
        with mock.patch.object(tui.curses, "doupdate"), \
             mock.patch.object(tui, "wall_clock", return_value=clock):
            tui.paint(screen, viewer, {"text": 0, "muted": 1, "accent": 2})

    def test_clock_sits_at_the_top_right_of_the_terminal(self):
        viewer = tui.Viewer(connected=True)
        viewer.add(event("A line of words", event_id="a"))
        screen = Screen(26, 100)
        self.paint_into(screen, viewer, "23:14:07")
        row = "".join(screen.lines[0])
        self.assertTrue(row.endswith("23:14:07"), row)
        self.assertEqual(len(row.rstrip()), 100)  # flush with the last column
        self.assertEqual(row.index("23:14:07"), 92)

    def test_clock_never_shifts_the_body(self):
        viewer = tui.Viewer(connected=True)
        viewer.add(event("A line of words", event_id="a"))
        screen = Screen(26, 100)
        self.paint_into(screen, viewer, "23:14:07")
        _, left, top, _ = tui.geometry(26, 100)
        self.assertEqual(top, 1)
        self.assertEqual("".join(screen.lines[1]).index("A"), left)

    def test_only_the_clock_row_repaints_when_the_second_changes(self):
        viewer = tui.Viewer(connected=True)
        viewer.add(event("A line of words", event_id="a"))
        screen = Screen(26, 100)
        self.paint_into(screen, viewer, "23:14:07")
        screen.cleared.clear()
        self.paint_into(screen, viewer, "23:14:08")
        self.assertEqual(screen.cleared, [0])
        self.assertIn("23:14:08", screen.text)

    def test_clock_stays_inside_a_narrow_terminal(self):
        viewer = tui.Viewer(connected=True)
        viewer.add(event("words", event_id="a"))
        screen = Screen(12, 40)
        self.paint_into(screen, viewer, "09:05:01")
        self.assertEqual("".join(screen.lines[0]).index("09:05:01"), 32)

    def test_the_view_wakes_at_least_once_a_second(self):
        viewer = tui.Viewer()
        delay = tui.next_transition(viewer, time.monotonic())
        self.assertTrue(delay is not None and 0 < delay <= 1.0, delay)


class FooterTests(unittest.TestCase):
    """No chrome unless it carries information: an unread count, a toast, a search line."""

    def paint_view(self, viewer, height=26):
        screen = Screen(height, 100)
        with mock.patch.object(tui.curses, "doupdate"), \
             mock.patch.object(tui, "wall_clock", return_value="23:14:07"):
            tui.paint(screen, viewer, {"text": 0, "muted": 1, "accent": 2})
        return screen

    def footer_of(self, screen):
        return "".join(screen.lines[-2]).strip()

    def test_following_shows_nothing_but_the_clock(self):
        viewer = tui.Viewer(connected=True)
        viewer.add(event("A line of words", event_id="a"))
        screen = self.paint_view(viewer)
        self.assertEqual(self.footer_of(screen), "")

    def test_reading_without_new_words_is_silent(self):
        viewer = tui.Viewer(connected=True)
        viewer.add(event("A line of words", event_id="a"))
        rows = tui.rows_for(viewer, 50)
        tui.move(viewer, rows, 10, -1)
        self.assertFalse(viewer.following)
        screen = self.paint_view(viewer)
        self.assertEqual(self.footer_of(screen), "")
        self.assertIn("A line of words", screen.text)

    def test_unread_is_a_count_not_a_sentence(self):
        viewer = tui.Viewer(connected=True)
        for index in range(6):
            viewer.add(event(f"paragraph {index} " + "word " * 12, session=index,
                             source=100 + index * 20, event_id=f"e{index}"))
        tui.move(viewer, tui.rows_for(viewer, 50), 4, -8)
        viewer.add(event("later words", session=99, source=5000, event_id="z"))
        footer = self.footer_of(self.paint_view(viewer, height=12))
        self.assertEqual(footer, "↓1")
        self.assertFalse(any(character.isalpha() for character in footer))

    def test_unread_is_hidden_when_everything_already_fits(self):
        viewer = tui.Viewer(connected=True)
        viewer.add(event("First", event_id="a"))
        tui.move(viewer, tui.rows_for(viewer, 50), 10, -1)
        viewer.add(event("Second", session=2, source=400, event_id="b"))
        self.assertEqual(viewer.unread, 1)
        self.assertEqual(self.footer_of(self.paint_view(viewer)), "")

    def test_holding_is_one_word(self):
        viewer = tui.Viewer(connected=True)
        viewer.add(event("A line of words", event_id="a"))
        viewer.holding = True
        self.assertEqual(self.footer_of(self.paint_view(viewer)), "Held")

    def test_search_line_still_reports_position(self):
        viewer = tui.Viewer(connected=True)
        viewer.add(event("alpha and alpha", event_id="a"))
        viewer.searching, viewer.query = True, "alpha"
        tui.search(viewer)
        self.assertIn("/ alpha  · 1/2", self.paint_view(viewer).text)


class WidthTests(unittest.TestCase):
    """Measure in cells over clusters, the way the terminal draws."""

    def test_flag_is_one_two_cell_glyph(self):
        self.assertEqual(len(tui.clusters("🇩🇪")), 1)
        self.assertEqual(tui.cells("🇩🇪"), 2)

    def test_zwj_family_is_one_cluster(self):
        family = "👨‍👩‍👧"
        self.assertEqual(len(tui.clusters(family)), 1)
        self.assertEqual(tui.cells(family), 2)

    def test_skin_tone_and_keycap_stay_attached(self):
        self.assertEqual(len(tui.clusters("👋🏽")), 1)
        self.assertEqual(tui.cells("👋🏽"), 2)
        self.assertEqual(tui.cells("1️⃣"), 1)

    def test_combining_marks_add_no_width(self):
        self.assertEqual(tui.cells("e\u0301"), 1)

    def test_cjk_stays_two_cells(self):
        self.assertEqual(tui.cells("日本"), 4)

    def test_fit_never_splits_a_cluster(self):
        self.assertEqual(tui.fit("👨‍👩‍👧x", 2), "👨‍👩‍👧")

    def test_wrap_stays_inside_the_measure_with_emoji(self):
        for line in tui.wrap("🇩🇪 " * 10, 9):
            self.assertLessEqual(tui.cells(line), 9)


class CaptionSafetyTests(unittest.TestCase):
    def test_final_ingestion_preserves_joined_emoji_and_persian_words(self):
        text = "A family 👨‍👩‍👧 says می\u200cروم."
        viewer = tui.Viewer()
        viewer.add(event(text, event_id="joined"))
        self.assertEqual(viewer.captions[0].text, text)
        self.assertEqual(tui.rows_for(viewer, 72)[0].text, text)
        self.assertEqual(tui.cells(viewer.captions[0].text), 23)

    def test_persian_non_joiner_has_no_cell_width(self):
        self.assertEqual(tui.cells("می\u200cروم"), 5)
        self.assertEqual(tui.clusters("ی\u200c"), ["ی\u200c"])
        self.assertEqual(tui.cells("\u200c"), 0)

    def test_draft_ingestion_preserves_joiners_without_terminal_controls(self):
        viewer = tui.Viewer()
        viewer.add({"type": "partial", "utterance_id": "draft",
                    "text": "👨‍👩‍👧\x1b\x07\x9b می\u200cروم\u202e\u2066"})
        self.assertEqual(viewer.draft, "👨‍👩‍👧 می\u200cروم")
        self.assertEqual(tui.rows_for(viewer, 72)[0].text, viewer.draft)

    def test_mouse_positions_never_split_a_joined_grapheme(self):
        family = "👨‍👩‍👧"
        text = family + "x"
        self.assertEqual(tui.cell_to_char(text, 1), 0)
        self.assertEqual(tui.cell_to_char(text, 2), len(family))
        self.assertEqual(tui.cell_to_char(text, 3), len(text))


class RawGraphemeSafetyTests(unittest.TestCase):
    def test_raw_overlay_preserves_joiners_without_emitting_remote_controls(self):
        capabilities = {name: ("<" + name + ">").encode()
                        for name in ("cup", "el", "sc", "rc", "sgr0")}
        writes = []
        def capture(fd, data):
            writes.append(bytes(data))
            return len(data)
        rows = [[(0, "👨‍👩‍👧 \x1b]52;c;untrusted\x07", 0)]]
        with mock.patch.object(tui.curses, "tigetstr", side_effect=capabilities.get), \
             mock.patch.object(tui.curses, "tparm", side_effect=lambda command, *args: command), \
             mock.patch.object(tui.curses, "pair_number", return_value=0), \
             mock.patch.object(tui.sys.stdout, "flush"), \
             mock.patch.object(tui.sys.stdout, "fileno", return_value=1), \
             mock.patch.object(tui.os, "write", side_effect=capture):
            tui.overlay_grapheme_rows(Screen(height=1), rows, {0})
        payload = b"".join(writes)
        self.assertIn("👨‍👩‍👧".encode(), payload)
        self.assertNotIn(b"\x1b", payload)
        self.assertNotIn(b"\x07", payload)
        self.assertTrue(payload.startswith(b"<sc>"))
        self.assertTrue(payload.endswith(b"<rc>"))


class CaptionTokenTests(unittest.TestCase):
    def test_local_viewer_uses_project_config_and_process_override(self):
        import config
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(config, "ROOT", Path(directory)):
            Path(directory, ".env").write_text("OMI_STREAM_TOKEN='project-token'\n")
            self.assertEqual(tui.caption_token(tui.DEFAULT_CAPTION_URL, environ={}), "project-token")
            self.assertEqual(tui.caption_token(tui.DEFAULT_CAPTION_URL,
                environ={"OMI_STREAM_TOKEN": "process-token"}), "process-token")
            Path(directory, ".env").write_text("OMI_STREAM_TOKEN=ingress\nOMI_CAPTION_TOKEN=viewer\n")
            self.assertEqual(tui.caption_token(tui.DEFAULT_CAPTION_URL, environ={}), "viewer")
            self.assertEqual(tui.caption_token(tui.DEFAULT_CAPTION_URL,
                environ={"OMI_CAPTION_TOKEN": ""}), "")

    def test_remote_viewer_never_loads_project_or_stream_credentials(self):
        import config
        with mock.patch.object(config, "load_env", side_effect=AssertionError("read local credentials")):
            for url in ("wss://example.com/listen", "ws://127.0.0.1:9999/listen",
                        "ws://localhost:8787/another-app", "wss://localhost:8787/listen"):
                with self.subTest(url=url):
                    self.assertEqual(tui.caption_token(url, environ={"OMI_STREAM_TOKEN": "private"}), "")
            self.assertEqual(tui.caption_token("wss://example.com/listen",
                environ={"OMI_CAPTION_TOKEN": "selected-remote-token"}), "selected-remote-token")

    def test_url_credentials_remain_supported_without_an_implicit_header(self):
        import config
        with mock.patch.object(config, "load_env", side_effect=AssertionError("read local credentials")):
            self.assertEqual(tui.caption_token(tui.DEFAULT_CAPTION_URL + "?token=url-token", environ={}), "")
            self.assertEqual(tui.caption_token(tui.DEFAULT_CAPTION_URL + "?key=url-key", environ={}), "")
            self.assertIn("token=url-token", tui.caption_url(tui.DEFAULT_CAPTION_URL + "?token=url-token"))

    def test_token_file_takes_precedence_and_is_not_put_in_the_url(self):
        with tempfile.TemporaryDirectory() as directory:
            token_file = Path(directory, "token")
            token_file.write_text("file-token\n")
            self.assertEqual(tui.caption_token("wss://example.com/listen", token_file,
                environ={"OMI_CAPTION_TOKEN": "environment-token"}), "file-token")
            self.assertNotIn("file-token", tui.caption_url("wss://example.com/listen"))

    def test_bad_credentials_fail_without_echoing_values(self):
        for token in ("private\nheader", "private\x1bcontrol", "x" * 4097):
            with self.subTest(length=len(token)):
                with self.assertRaises(ValueError) as raised:
                    tui.caption_token(tui.DEFAULT_CAPTION_URL, environ={"OMI_CAPTION_TOKEN": token})
                self.assertNotIn(token, str(raised.exception))
        with tempfile.TemporaryDirectory() as directory:
            token_file = Path(directory, "token")
            token_file.write_text("\n")
            with self.assertRaises(ValueError):
                tui.caption_token("wss://example.com/listen", token_file, environ={})

    def test_standalone_local_viewer_can_use_process_credentials(self):
        with mock.patch.dict(tui.sys.modules, {"bridge": None}):
            self.assertEqual(tui.caption_token(tui.DEFAULT_CAPTION_URL,
                environ={"OMI_STREAM_TOKEN": "standalone-token"}), "standalone-token")

    def test_startup_errors_do_not_echo_url_or_credential_paths(self):
        for argv in (["tui.py", "--url", "ws://[private-host"],
                     ["tui.py", "--token-file", "/nonexistent/private-token"]):
            stderr = io.StringIO()
            with self.subTest(option=argv[1]), mock.patch.object(tui.sys, "argv", argv), \
                 mock.patch.object(tui.sys, "stderr", stderr):
                with self.assertRaises(SystemExit) as raised:
                    tui.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertNotIn(argv[-1], stderr.getvalue())


class GapNoticeTests(unittest.TestCase):
    def footer(self, viewer):
        screen = Screen()
        with mock.patch.object(tui.curses, "doupdate"):
            tui.paint(screen, viewer, {"text": 0, "muted": 1, "accent": 2})
        return "".join(screen.lines[-2]).strip()

    def test_live_gap_is_visible_without_inserting_a_reading_row(self):
        viewer = tui.Viewer(connected=True)
        viewer.add(event("Settled words.", event_id="before"))
        viewer.add({"type": "gap", "text": "unavailable", "duration_sec": 2.4})
        self.assertEqual(self.footer(viewer), "2.4 seconds unavailable.")
        self.assertFalse(any("unavailable" in row.text for row in tui.rows_for(viewer, 72)))
        viewer.toast_until = 0
        self.assertEqual(self.footer(viewer), "")

    def test_expired_replay_cursor_shows_a_visible_notice(self):
        viewer = tui.Viewer(connected=True, last_event_id="expired")
        viewer.add({"type": "replay", "phase": "start", "reset": True})
        viewer.add(event("Retained words.", event_id="retained"))
        viewer.add({"type": "replay", "phase": "end"})
        self.assertIn("outside the available history", self.footer(viewer))

    def test_unreplayable_reconnect_notice_reaches_the_footer(self):
        viewer = tui.Viewer(connected=True)
        viewer.add({"type": "gap", "text": "Connection restored · some words may be missing"})
        self.assertEqual(self.footer(viewer), "Connection restored · some words may be missing")


class MeasureTests(unittest.TestCase):
    def test_ultrawide_windows_are_capped_to_a_readable_measure(self):
        column, _, _, _ = tui.geometry(40, 240)
        self.assertEqual(column, tui.READING_MEASURE)
        self.assertTrue(45 <= column <= 75)

    def test_measure_full_restores_half_width(self):
        self.assertEqual(tui.geometry(40, 240, measure=None)[0], 120)

    def test_parser_accepts_auto_full_and_numbers(self):
        self.assertEqual(tui.measure_value("auto"), tui.READING_MEASURE)
        self.assertIsNone(tui.measure_value("full"))
        self.assertEqual(tui.measure_value("90"), 90)
        self.assertEqual(tui.measure_value("5"), 30)
        with self.assertRaises(Exception):
            tui.measure_value("wide")


class TitleTests(unittest.TestCase):
    def test_title_is_osc_2(self):
        with mock.patch.object(tui.sys.stdout, "write") as write, \
             mock.patch.object(tui.sys.stdout, "flush"):
            tui.set_title("Omi · Listening")
        self.assertEqual(write.call_args.args[0], "\x1b]2;Omi · Listening\x07")


class ThemeTests(unittest.TestCase):
    """Theme detection, NO_COLOR, and palette roles bound to ANSI slots."""

    def test_colorfgbg_dark_and_light(self):
        self.assertEqual(tui.theme_from_env({"COLORFGBG": "15;0"}), "dark")
        self.assertEqual(tui.theme_from_env({"COLORFGBG": "0;15"}), "light")
        self.assertEqual(tui.theme_from_env({"COLORFGBG": "0;8"}), "light")
        self.assertEqual(tui.theme_from_env({}), "dark")
        self.assertEqual(tui.theme_from_env({"COLORFGBG": "nonsense"}), "dark")
        self.assertEqual(tui.theme_from_env({"COLORFGBG": "0;banana"}), "dark")

    def test_no_color_follows_the_standard(self):
        self.assertTrue(tui.monochrome_requested("auto", {"NO_COLOR": "1"}))
        self.assertFalse(tui.monochrome_requested("auto", {"NO_COLOR": ""}))
        self.assertTrue(tui.monochrome_requested("auto", {"TERM": "dumb"}))
        self.assertTrue(tui.monochrome_requested("never", {}))

    def test_palette_binds_roles_to_ansi_slots(self):
        self.assertEqual(tui.palette_spec("dark")["accent"][0], 6)
        self.assertEqual(tui.palette_spec("light")["accent"][0], 4)
        self.assertEqual(tui.palette_spec("dark")["muted"][0], 8)  # bright black, not A_DIM

    def test_mono_palette_uses_attributes_only(self):
        spec = tui.palette_spec("dark", mono=True)
        self.assertTrue(all(slot == -1 for slot, _ in spec.values()))
        self.assertNotEqual(spec["muted"][1], 0)
        self.assertNotEqual(spec["accent"][1], 0)


class SgrMouseTests(unittest.TestCase):
    """Rex and other SGR-1006 terminals: ncurses hands us ESC [ < b;x;y M|m as keys."""

    class Screen:
        def __init__(self, text):
            self.chars = list(text)

        def get_wch(self):
            if not self.chars:
                raise curses.error("no input")
            return self.chars.pop(0)

    def parse(self, sequence):
        pending = deque()
        result = tui.sgr_mouse(self.Screen(sequence), pending, timeout=0.05)
        return result, "".join(pending)

    def test_press_drag_and_release(self):
        self.assertEqual(self.parse("[<0;10;5M")[0], (9, 4, curses.BUTTON1_PRESSED))
        self.assertEqual(self.parse("[<32;20;5M")[0], (19, 4, curses.REPORT_MOUSE_POSITION))
        self.assertEqual(self.parse("[<0;20;5m")[0], (19, 4, curses.BUTTON1_RELEASED))

    def test_wheel_up_and_down(self):
        self.assertEqual(self.parse("[<64;5;5M")[0][2], curses.BUTTON4_PRESSED)
        self.assertEqual(self.parse("[<65;5;5M")[0][2], curses.REPORT_MOUSE_POSITION)

    def test_plain_escape_is_pushed_back_not_swallowed(self):
        result, pushed = self.parse("[A")
        self.assertIsNone(result)
        self.assertEqual(pushed, "[A")

    def test_modifier_bits_do_not_change_the_button(self):
        self.assertEqual(self.parse("[<4;10;5M")[0][2], curses.BUTTON1_PRESSED)


class ClockStateTests(unittest.TestCase):
    """The clock is the only chrome, so it carries the paused bit."""

    PALETTE = {"text": 0, "muted": 1, "accent": 2}

    def test_muted_while_following_live(self):
        self.assertEqual(tui.clock_attr(tui.Viewer(), self.PALETTE), 1)

    def test_accent_when_scrolled_back_or_held(self):
        scrolled = tui.Viewer()
        scrolled.following = False
        self.assertEqual(tui.clock_attr(scrolled, self.PALETTE), 2)
        held = tui.Viewer()
        held.holding = True
        self.assertEqual(tui.clock_attr(held, self.PALETTE), 2)


if __name__ == "__main__":
    unittest.main()
