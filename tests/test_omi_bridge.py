"""Regression checks run entirely in a temporary directory, with no Gemini calls."""
import asyncio
from collections import deque
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch
import urllib.error
import wave

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        source = Path(os.environ.get("OMI_BRIDGE_PATH", Path(__file__).resolve().parents[1] / "omi_bridge.py"))
        target = Path(cls.temp.name) / "omi_bridge.py"
        shutil.copyfile(source, target)
        spec = importlib.util.spec_from_file_location("bridge_under_test", target)
        cls.bridge = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.bridge
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "OMI_RING_BUFFER_SEC": "300"}):
            spec.loader.exec_module(cls.bridge)
        cls.bridge.configure({"GEMINI_API_KEY": "test-key", "OMI_RING_BUFFER_SEC": "300"})
        for handler in cls.bridge.logger.handlers:
            handler.close()
        cls.bridge.logger.handlers = [logging.NullHandler()]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    async def asyncSetUp(self):
        b = self.bridge
        b.stt_queue = None
        b.audio_pcm_buffer = bytearray()
        b.audio_pre_tail = bytearray()
        b.audio_ring = deque()
        b.audio_ring_bytes = 0
        b.audio_sessions = deque()
        b.current_session = None
        b._next_session_id = 0
        b.stopping = False
        b.background_tasks.clear()
        b.webhook_tasks.clear()
        b.live_listeners.clear()
        b.daily_notes = None
        b.rolling_research_buffer.clear()
        for key in ("windows_processed", "windows_retried", "windows_dropped", "windows_skipped"):
            b.metrics[key] = 0
        for key in ("last_error", "last_packet_at", "last_stt_success_at"):
            b.metrics[key] = None
        b.metrics["stt_in_flight"] = False
        self.app = None
        self.calls = []
        self.lines = []
        def transcribe(*args):
            self.calls.append(args)
            return "sample transcript"
        self.api = patch.object(b, "_gemini_stt_request_sync", side_effect=transcribe).start()
        patch.object(b, "record_transcript_line", side_effect=self.lines.append).start()
        self.addCleanup(patch.stopall)
        self.pcm = struct.pack("<h", 500) * 16000

    async def asyncTearDown(self):
        if self.app is not None:
            await self.bridge.on_shutdown(self.app)
            await self.bridge.on_cleanup(self.app)

    async def start(self):
        self.app = {}
        await self.bridge.on_startup(self.app)
        await asyncio.sleep(.01)

    async def finish_window(self):
        await asyncio.wait_for(self.bridge.stt_queue.join(), 2)

    async def test_queue_is_created_on_server_loop(self):
        self.assertIsNone(self.bridge.stt_queue)
        await self.start()
        await self.bridge.enqueue_stt_window(self.pcm, 16000, 0, 1, 0)
        await self.finish_window()
        self.assertEqual(self.lines, ["sample transcript"])
        self.assertTrue(self.bridge.build_health_dict()["workers_alive"])

    async def test_retry_429_501_and_timeout_then_success(self):
        await self.start()
        self.api.side_effect = [urllib.error.HTTPError("test", 429, "rate limit", {}, None),
                                urllib.error.HTTPError("test", 501, "temporary", {}, None),
                                TimeoutError("slow"), "recovered"]
        with patch.object(self.bridge.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            await self.bridge.enqueue_stt_window(self.pcm, 16000, 0, 1, 0)
            await self.finish_window()
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2, 4])
        self.assertEqual(self.api.call_count, 4)
        self.assertEqual(self.lines, ["recovered"])
        self.assertEqual(self.bridge.metrics["windows_retried"], 3)
        self.assertIsNone(self.bridge.metrics["last_error"])

    async def test_permanent_error_is_not_retried(self):
        await self.start()
        self.api.side_effect = urllib.error.HTTPError("test", 401, "unauthorized", {}, None)
        await self.bridge.enqueue_stt_window(self.pcm, 16000, 0, 1, 0)
        await self.finish_window()
        self.assertEqual(self.api.call_count, 1)
        self.assertEqual(self.bridge.metrics["windows_dropped"], 1)
        self.assertEqual(self.bridge.build_health_dict()["status"], "degraded")

    async def test_exhausted_retries_are_counted(self):
        await self.start()
        self.api.side_effect = TimeoutError("slow")
        with patch.object(self.bridge.asyncio, "sleep", new_callable=AsyncMock):
            await self.bridge.enqueue_stt_window(self.pcm, 16000, 0, 1, 0)
            await self.finish_window()
        self.assertEqual(self.api.call_count, 4)
        self.assertEqual(self.bridge.metrics["windows_dropped"], 1)

    async def test_silence_is_retained_but_not_sent_to_gemini(self):
        await self.start()
        silent = bytes(16000)
        self.bridge.ingest_raw_pcm(silent)
        await self.bridge.flush_remaining_buffer()
        await self.finish_window()
        self.assertEqual(b"".join(chunk[2] for chunk in self.bridge.audio_ring), silent)
        self.api.assert_not_called()
        self.assertEqual(self.bridge.metrics["windows_skipped"], 1)

    async def test_short_tail_and_empty_boundary_reset_context(self):
        await self.start()
        self.bridge.ingest_raw_pcm(self.pcm[:2])
        await self.bridge.flush_remaining_buffer()
        await self.finish_window()
        await self.bridge.enqueue_stt_window(self.pcm, 16000, 0, 1, 0)
        await self.finish_window()
        self.assertEqual(len(self.calls[0][0]), 2)
        self.assertEqual(self.calls[1][-1], "")
        await self.bridge.flush_remaining_buffer()  # No samples, but still a session boundary.
        await self.bridge.enqueue_stt_window(self.pcm, 16000, 0, 1, 0)
        await self.finish_window()
        self.assertEqual(self.calls[2][-1], "")

    async def test_new_packet_closes_previous_idle_session(self):
        await self.start()
        self.bridge.ingest_raw_pcm(self.pcm[:2])
        self.bridge.current_session.last_packet_at -= 9
        self.bridge.ingest_raw_pcm(self.pcm[:4])
        await self.bridge.flush_remaining_buffer()
        await self.finish_window()
        self.assertEqual([len(c[0]) for c in self.calls], [2, 4])
        self.assertEqual([c[-1] for c in self.calls], ["", ""])

    async def test_real_slicer_flushes_idle_subsecond_tail(self):
        await self.start()
        self.bridge.ingest_raw_pcm(self.pcm[:160])
        self.bridge.current_session.last_packet_at -= 9
        await asyncio.sleep(.25)
        await self.finish_window()
        self.assertEqual(len(self.calls[0][0]), 160)
        self.assertFalse(self.bridge.audio_sessions)

    async def test_ring_exactly_trims_partial_oldest_packet(self):
        b = self.bridge
        chunk = self.pcm * 7
        for _ in range(43):
            b.audio_pcm_buffer.clear()
            b.ingest_raw_pcm(chunk)
        self.assertEqual(b.audio_ring_bytes, 9_600_000)
        response = await b.handle_audio_last(make_mocked_request("GET", "/audio/last?seconds=300"))
        with wave.open(io.BytesIO(response.body)) as wav:
            self.assertEqual(wav.getnframes(), 300 * 16000)
            self.assertEqual(wav.readframes(300 * 16000), (chunk * 43)[-9_600_000:])

    async def test_audio_export_and_invalid_parameters(self):
        self.bridge.ingest_raw_pcm(self.pcm)
        response = await self.bridge.handle_audio_last(make_mocked_request("GET", "/audio/last?seconds=1"))
        with wave.open(io.BytesIO(response.body)) as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 16000))
            self.assertEqual(wav.readframes(16000), self.pcm)
        for value in ("bad", "0", "301"):
            with self.assertRaises(web.HTTPBadRequest):
                await self.bridge.handle_audio_last(make_mocked_request("GET", f"/audio/last?seconds={value}"))

    async def test_invalid_rate_odd_samples_and_buffer_bound(self):
        b = self.bridge
        with self.assertRaises(web.HTTPBadRequest):
            b.ingest_raw_pcm(self.pcm, 8000)
        with self.assertRaises(web.HTTPBadRequest):
            b.ingest_raw_pcm(b"x")
        with self.assertRaises(web.HTTPServiceUnavailable):
            b.ingest_raw_pcm(bytes(b.MAX_BUFFER_BYTES + 2))
        self.assertEqual(len(b.audio_pcm_buffer), 0)

    async def test_full_queue_applies_backpressure_without_dropping(self):
        b = self.bridge
        b.stt_queue = asyncio.Queue(maxsize=24)
        for _ in range(24):
            await b.enqueue_stt_window(self.pcm, 16000, 0, 1, 0)
        waiting = asyncio.create_task(b.enqueue_stt_window(self.pcm, 16000, 0, 1, 0))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        self.assertEqual(b.stt_queue.qsize(), 24)
        self.assertEqual(b.metrics["windows_dropped"], 0)
        self.assertEqual(b.build_health_dict()["status"], "degraded")
        b.stt_queue.get_nowait()
        b.stt_queue.task_done()
        await asyncio.wait_for(waiting, 1)
        self.assertEqual(b.stt_queue.qsize(), 24)
        self.assertEqual(b.metrics["windows_dropped"], 0)

    async def test_dead_worker_is_not_healthy(self):
        await self.start()
        worker = self.app["slicer_task"]
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        response = await self.bridge.handle_health(None)
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.body)["status"], "degraded")
        self.assertFalse(json.loads(response.body)["workers_alive"])

    async def test_shutdown_drains_full_queue_and_short_tail(self):
        await self.start()
        b = self.bridge
        for _ in range(24):
            await b.enqueue_stt_window(self.pcm, 16000, 0, 1, 0)
        b.ingest_raw_pcm(self.pcm[:2])
        await b.on_shutdown(self.app)
        with self.assertRaises(web.HTTPServiceUnavailable):
            b.ingest_raw_pcm(self.pcm)
        await asyncio.wait_for(b.on_cleanup(self.app), 3)
        self.app = None
        self.assertEqual(self.api.call_count, 25)
        self.assertEqual(b.metrics["windows_dropped"], 0)
        self.assertTrue(b.stt_queue.empty())

    async def test_http_ingress_to_worker_and_playable_export(self):
        b = self.bridge
        await self.start()
        app = web.Application()
        app.router.add_post("/omi", b.handle_omi)
        app.router.add_get("/audio/last", b.handle_audio_last)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/omi?sample_rate=16000", data=self.pcm)
            self.assertEqual(response.status, 200)
            await b.flush_remaining_buffer()
            await self.finish_window()
            response = await client.get("/audio/last?seconds=1")
            with wave.open(io.BytesIO(await response.read())) as wav:
                self.assertEqual(wav.readframes(16000), self.pcm)
        self.assertEqual(self.lines, ["sample transcript"])


class NotesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        BridgeTests.setUpClass()
        cls.notes_class = BridgeTests.bridge.DailyNotes

    @classmethod
    def tearDownClass(cls):
        BridgeTests.tearDownClass()

    def test_pause_word_limit_midnight_and_rebuild(self):
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
            source = Path(directory, "source.log")
            source.write_text("".join(f"[{stamp}] {' '.join(text.split())}\n" for stamp, text in entries))
            rebuilt = self.notes_class(directory)
            rebuilt.rebuild(source)
            self.assertEqual(first, Path(directory, "2026-10-24.md").read_text())
            self.assertEqual(second, Path(directory, "2026-10-25.md").read_text())
            rebuilt.add("2026-10-25 00:00:12", "continued after restart")
            self.assertIn("next paragraph continued after restart", Path(directory, "2026-10-25.md").read_text())


if __name__ == "__main__":
    unittest.main()
