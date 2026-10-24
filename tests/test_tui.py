"""Offline viewer checks; no wearable, Gemini calls, or running bridge required."""
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


class CaptionTests(unittest.TestCase):
    def test_word_reveal_has_a_fixed_maximum_duration(self):
        caption = tui.Caption("12:00:00", [f"word{i}" for i in range(200)], arrived=10)
        text, revealing = caption.visible(10)
        self.assertEqual(text, "word0")
        self.assertTrue(revealing)
        text, revealing = caption.visible(11.21)
        self.assertEqual(len(text.split()), 200)
        self.assertFalse(revealing)
        self.assertFalse(caption.visible(10, instant=True)[1])

    def test_new_chunks_finish_older_animation_without_queueing(self):
        viewer = tui.Viewer()
        viewer.add({"type": "transcript", "text": "old words here"})
        viewer.add({"type": "transcript", "text": "new words here"})
        with mock.patch.object(tui.time, "monotonic", return_value=viewer.captions[-1].arrived):
            rows = tui.rows_for(viewer, 80, instant=False)
        self.assertEqual(rows[0][1:], ("old words here", "text"))
        self.assertEqual(rows[2][1:], ("new", "new"))

    def test_bounded_history_deduplication_and_literal_text(self):
        viewer = tui.Viewer()
        viewer.add({"event_id": "same", "text": "literal [bold] text\x1b\x07"})
        viewer.add({"event_id": "same", "text": "duplicate"})
        self.assertEqual(len(viewer.captions), 1)
        self.assertEqual(" ".join(viewer.captions[0].words), "literal [bold] text")
        for index in range(500):
            viewer.add({"event_id": str(index), "text": str(index)})
        self.assertEqual(len(viewer.captions), 200)
        self.assertEqual(len(viewer.seen), 400)
        self.assertEqual(viewer.captions[-1].words, ["499"])

    def test_unicode_wrap_fits_terminal_cells(self):
        for text in ("English words " * 20, "你好世界" * 10, "cafe\u0301 " * 20, "a" * 100):
            with self.subTest(text=text[:10]):
                lines = tui.wrap(text, 15)
                self.assertTrue(all(tui.cells(line) <= 15 for line in lines))
                self.assertEqual("".join("".join(lines).split()), "".join(text.split()))

    def test_invalid_events_and_pcm_do_not_damage_viewer(self):
        viewer = tui.Viewer()
        for event in (None, [], {"type": "unknown", "text": "ignore"}, {"text": ""}):
            viewer.add(event)
        viewer.audio(b"x")
        self.assertIsNone(viewer.last_audio)
        viewer.audio(struct.pack("<h", 1000) * 16000)
        self.assertIsNotNone(viewer.last_audio)
        self.assertGreater(viewer.level, 0)
        self.assertEqual(len(viewer.captions), 0)

    def test_gap_is_shown_immediately(self):
        viewer = tui.Viewer()
        viewer.add({"type": "gap", "text": "[GAP: 8s audio untranscribed]"})
        rows = tui.rows_for(viewer, 80, instant=False)
        self.assertEqual(rows[0][1:], ("[GAP: 8s audio untranscribed]", "gap"))


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
        url = str(server.make_url("/listen-ws")).replace("http://", "ws://")
        task = asyncio.create_task(tui.receive(viewer, url))
        try:
            async with asyncio.timeout(2):
                while not viewer.captions:
                    await asyncio.sleep(0.01)
            self.assertTrue(viewer.connected)
            self.assertEqual(viewer.captions[-1].words, ["Blue", "lantern", "seven."])
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
        url = str(server.make_url("/listen-ws")).replace("http://", "ws://")
        task = asyncio.create_task(tui.receive(viewer, url))
        try:
            async with asyncio.timeout(3):
                while viewer.connections < 2:
                    await asyncio.sleep(0.01)
            self.assertEqual(viewer.captions[-1].kind, "gap")
            self.assertIn("missing", " ".join(viewer.captions[-1].words))
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await server.close()


@unittest.skipUnless(os.name == "posix", "The viewer uses POSIX curses terminals")
class TerminalTests(unittest.TestCase):
    def test_quit_restores_original_terminal_settings(self):
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
                [sys.executable, "-B", str(Path(tui.__file__).resolve()),
                 "--url", "ws://127.0.0.1:1/listen-ws"],
                stdin=slave, stdout=slave, stderr=slave,
                env={**os.environ, "TERM": "xterm-256color"})
            output, sent_quit = bytearray(), False
            deadline = time.monotonic() + 5
            while process.poll() is None and time.monotonic() < deadline:
                if select.select([master], [], [], 0.05)[0]:
                    output.extend(os.read(master, 65536))
                if not sent_quit and b"COMPLETED TRANSCRIPTS" in output:
                    os.write(master, b"q")
                    sent_quit = True
            self.assertTrue(sent_quit, "Terminal did not draw its header")
            self.assertEqual(process.poll(), 0, "Viewer did not exit cleanly")
            after = termios.tcgetattr(slave)
            # macOS sets PENDIN when restoring canonical mode. It is a transient
            # kernel input-reprocessing flag, not a changed interactive setting.
            after[3] &= ~getattr(termios, "PENDIN", 0)
            before[3] &= ~getattr(termios, "PENDIN", 0)
            self.assertEqual(after, before)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            os.close(master)
            os.close(slave)


if __name__ == "__main__":
    unittest.main()
