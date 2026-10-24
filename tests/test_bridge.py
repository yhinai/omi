"""Offline regressions for audio retention, configuration, and shutdown."""
import asyncio
from collections import namedtuple
import importlib.util
import io
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

if os.environ.get("OMI_BRIDGE_PATH"):
    spec = importlib.util.spec_from_file_location("omi_bridge", os.environ["OMI_BRIDGE_PATH"])
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    spec.loader.exec_module(bridge)
else:
    import omi_bridge as bridge

SPEECH = b"\x00\x10" * 8000  # Half a second above the silence gate.
VersionInfo = namedtuple("VersionInfo", "major minor micro releaselevel serial")


class RuntimeTests(unittest.TestCase):
    def runtime(self, version=(3, 15, 0, "final", 0), implementation="cpython", gil_disabled=0):
        runtime = SimpleNamespace(version_info=VersionInfo(*version),
                                  implementation=SimpleNamespace(name=implementation))
        config = SimpleNamespace(get_config_var=lambda name: gil_disabled)
        return mock.patch.multiple(bridge, sys=runtime, sysconfig=config)

    def test_standard_final_cpython315_patch_releases_are_supported(self):
        for micro in (0, 1, 99):
            for gil_disabled in (None, 0):
                with self.subTest(micro=micro, gil_disabled=gil_disabled), \
                     self.runtime(version=(3, 15, micro, "final", 0), gil_disabled=gil_disabled):
                    self.assertIsNone(bridge.require_python315())

    def test_other_python_versions_and_prereleases_are_rejected(self):
        versions = [(3, 14, 99, "final", 0), (3, 16, 0, "final", 0), (2, 7, 18, "final", 0)]
        versions += [(3, 15, 0, level, 1) for level in ("alpha", "beta", "candidate")]
        for version in versions:
            with self.subTest(version=version), self.runtime(version=version):
                with self.assertRaisesRegex(RuntimeError, "CPython 3\\.15"):
                    bridge.require_python315()

    def test_alternative_implementation_and_free_threaded_build_are_rejected(self):
        for implementation, gil_disabled in (("pypy", 0), ("cpython", 1)):
            with self.subTest(implementation=implementation, gil_disabled=gil_disabled), \
                 self.runtime(implementation=implementation, gil_disabled=gil_disabled):
                with self.assertRaisesRegex(RuntimeError, "CPython 3\\.15"):
                    bridge.require_python315()


class ImportTests(unittest.TestCase):
    def test_import_does_not_read_configuration_or_create_files(self):
        script = """
import aiohttp.web, importlib.util, os, sys
from unittest import mock
sys.dont_write_bytecode = True
before = dict(os.environ)
spec = importlib.util.spec_from_file_location('omi_bridge_import_probe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
with mock.patch('builtins.open', side_effect=AssertionError('file access during import')), \\
     mock.patch('os.makedirs', side_effect=AssertionError('mkdir during import')):
    spec.loader.exec_module(module)
assert dict(os.environ) == before, 'import changed environment'
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(Path(bridge.__file__).resolve())],
            capture_output=True, text=True, timeout=10,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_configuration_precedence_includes_gemini_key_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            local, fallback = Path(directory) / "local.env", Path(directory) / "fallback.env"
            local.write_text("GOOGLE_API_KEY=local-key\nOMI_CORE_WINDOW_SEC=6\n")
            fallback.write_text("GEMINI_API_KEY=fallback-key\nOMI_CORE_WINDOW_SEC=7\n")
            merged = bridge._load_dotenv(local_path=local, fallback_path=fallback, environ={})
            self.assertEqual(merged["GEMINI_API_KEY"], "local-key")
            self.assertEqual(merged["OMI_CORE_WINDOW_SEC"], "6")
            merged = bridge._load_dotenv(local_path=local, fallback_path=fallback,
                environ={"GOOGLE_API_KEY": "shell-key", "OMI_CORE_WINDOW_SEC": "5"})
            self.assertEqual(merged["GEMINI_API_KEY"], "shell-key")
            self.assertEqual(merged["OMI_CORE_WINDOW_SEC"], "5")


class AudioTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patches = [
            mock.patch.object(bridge, "stt_queue", asyncio.Queue(maxsize=24)),
            mock.patch.object(bridge, "GEMINI_API_KEY", "offline-test-key"),
            mock.patch.object(bridge, "record_transcript_line"),
            mock.patch.object(bridge, "flush_daily_paragraph"),
            mock.patch.object(bridge, "post_webhook_sync"),
        ]
        for patch in self.patches:
            patch.start()
        self.addCleanup(lambda: [patch.stop() for patch in reversed(self.patches)])
        bridge.audio_pcm_buffer.clear()
        bridge.audio_pre_tail.clear()
        bridge.audio_sessions.clear()
        bridge.current_session = None
        bridge._next_session_id = 0
        bridge.stopping = False
        bridge.background_tasks.clear()
        bridge.webhook_tasks.clear()
        bridge.live_listeners.clear()
        bridge.rolling_research_buffer.clear()
        bridge.audio_ring.clear()
        bridge.audio_ring_bytes = 0
        bridge.audio_sample_rate = 16000
        for name in ("windows_processed", "windows_retried", "windows_dropped"):
            bridge.metrics[name] = 0
        for name in ("last_packet_at", "last_stt_success_at", "last_error"):
            bridge.metrics[name] = None

    async def enqueue(self, reset=False):
        await bridge.enqueue_stt_window(SPEECH, 16000, 0, 0.5, 0, reset_context_after=reset)

    async def consume(self):
        await bridge.stt_queue.put(None)
        await asyncio.wait_for(bridge.stt_consumer_worker(), 2)
        await asyncio.wait_for(bridge.stt_queue.join(), 1)

    async def test_full_queue_waits_without_dropping_accepted_windows(self):
        bridge.stt_queue = asyncio.Queue(maxsize=1)
        await self.enqueue()
        waiting = asyncio.create_task(self.enqueue())
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        self.assertIsNotNone(bridge.stt_queue.get_nowait())
        bridge.stt_queue.task_done()
        await asyncio.wait_for(waiting, 1)
        self.assertEqual(bridge.stt_queue.qsize(), 1)
        self.assertEqual(bridge.metrics["windows_dropped"], 0)
        bridge.stt_queue.get_nowait()
        bridge.stt_queue.task_done()

    async def test_transient_failures_retry_three_times_with_backoff(self):
        errors = [urllib.error.HTTPError("https://example.invalid", code, "mock", {}, None)
                  for code in (429, 500, 501, 599)]
        errors += [TimeoutError("mock timeout"), socket.timeout("mock socket timeout"),
                   urllib.error.URLError("mock connection reset")]
        for error in errors:
            with self.subTest(error=repr(error)):
                await self.enqueue()
                with mock.patch.object(bridge, "_gemini_stt_request_sync",
                        side_effect=[error, error, error, "Recovered speech"]) as request, \
                     mock.patch.object(bridge.asyncio, "sleep", new_callable=mock.AsyncMock) as sleep:
                    await self.consume()
                self.assertEqual(request.call_count, 4)
                self.assertEqual([call.args[0] for call in sleep.await_args_list], [1, 2, 4])
                bridge.record_transcript_line.assert_called_with("Recovered speech")

    async def test_permanent_client_error_is_not_retried(self):
        await self.enqueue()
        error = urllib.error.HTTPError("https://example.invalid", 400, "mock", {}, None)
        with mock.patch.object(bridge, "_gemini_stt_request_sync", side_effect=error) as request:
            await self.consume()
        self.assertEqual(request.call_count, 1)
        self.assertEqual(bridge.metrics["windows_retried"], 0)
        bridge.record_transcript_line.assert_not_called()

    async def test_transient_http_response_is_closed_before_retry_backoff(self):
        await self.enqueue()
        body = io.BytesIO(b"mock rate limit response")
        self.addCleanup(body.close)
        error = urllib.error.HTTPError("https://example.invalid", 429, "mock", {}, body)

        async def check_backoff(delay):
            self.assertTrue(body.closed, "HTTP response remained open during retry backoff")

        with mock.patch.object(bridge, "_gemini_stt_request_sync",
                side_effect=[error, "Recovered speech"]) as request, \
             mock.patch.object(bridge.asyncio, "sleep", side_effect=check_backoff) as sleep:
            await self.consume()
        self.assertEqual(request.call_count, 2)
        sleep.assert_awaited_once_with(1.0)
        self.assertTrue(body.closed)
        bridge.record_transcript_line.assert_called_once_with("Recovered speech")

    async def test_permanent_http_response_is_closed_when_window_is_dropped(self):
        await self.enqueue()
        body = io.BytesIO(b"mock invalid request response")
        self.addCleanup(body.close)
        error = urllib.error.HTTPError("https://example.invalid", 400, "mock", {}, body)
        with mock.patch.object(bridge, "_gemini_stt_request_sync", side_effect=error) as request:
            await self.consume()
        self.assertEqual(request.call_count, 1)
        self.assertTrue(body.closed, "HTTP response remained open after a permanent error")
        bridge.record_transcript_line.assert_not_called()

    async def test_retry_exhaustion_finishes_queue_item_without_recording_text(self):
        await self.enqueue()
        with mock.patch.object(bridge, "_gemini_stt_request_sync", side_effect=TimeoutError("mock")) as request, \
             mock.patch.object(bridge.asyncio, "sleep", new_callable=mock.AsyncMock):
            await self.consume()
        self.assertEqual(request.call_count, 4)
        bridge.record_transcript_line.assert_not_called()

    async def test_context_is_sequential_and_resets_after_empty_session_flush(self):
        await self.enqueue()
        await self.enqueue()
        await bridge.flush_remaining_buffer(reset_context=True)
        await self.enqueue()
        with mock.patch.object(bridge, "_gemini_stt_request_sync",
                side_effect=["alpha beta", "alpha beta gamma", "alpha beta"]) as request:
            await self.consume()
        self.assertEqual([call.args[-1] for call in request.call_args_list], ["", "alpha beta", ""])
        self.assertEqual([call.args[0] for call in bridge.record_transcript_line.call_args_list],
                         ["alpha beta", "gamma", "alpha beta"])

    async def test_subsecond_tail_is_queued_and_context_does_not_cross_session(self):
        await self.enqueue()
        bridge.ingest_raw_pcm(SPEECH)
        await bridge.flush_remaining_buffer(reset_context=True)
        self.assertEqual(bridge.audio_pcm_buffer, bytearray())
        self.assertEqual(bridge.audio_pre_tail, bytearray())
        await self.enqueue()
        with mock.patch.object(bridge, "_gemini_stt_request_sync",
                side_effect=["first segment", "tail segment", "new session"]) as request:
            await self.consume()
        self.assertEqual(len(request.call_args_list[1].args[0]), len(SPEECH))
        self.assertEqual(request.call_args_list[1].args[3], 0.5)
        self.assertEqual(request.call_args_list[2].args[-1], "")

    async def test_silence_is_not_sent_to_gemini(self):
        await bridge.enqueue_stt_window(b"\0" * 16000, 16000, 0, 0.5, 0)
        with mock.patch.object(bridge, "_gemini_stt_request_sync") as request:
            await self.consume()
        request.assert_not_called()

    async def test_slicer_flushes_subsecond_audio_only_after_eight_seconds_idle(self):
        now = [100.0]
        clock = mock.Mock(wraps=bridge.time)
        clock.monotonic.side_effect = lambda: now[0]
        with mock.patch.object(bridge, "time", clock):
            bridge.ingest_raw_pcm(SPEECH)
            slicer = asyncio.create_task(bridge.window_slicer_worker())
            try:
                now[0] = 108.0
                await asyncio.sleep(0.25)
                self.assertTrue(bridge.stt_queue.empty())
                now[0] = 108.01
                for _ in range(300):
                    if not bridge.stt_queue.empty():
                        break
                    await asyncio.sleep(0.001)
                window = bridge.stt_queue.get_nowait()
                self.assertEqual(window.pcm, SPEECH)
                self.assertEqual(window.core_sec, 0.5)
                self.assertTrue(window.reset_context_after)
                bridge.stt_queue.task_done()
                self.assertEqual(bridge.buffered_audio_bytes(), 0)
            finally:
                slicer.cancel()
                await asyncio.gather(slicer, return_exceptions=True)

    async def test_new_session_does_not_join_audio_waiting_for_queue_admission(self):
        bridge.stt_queue = asyncio.Queue(maxsize=1)
        await self.enqueue()
        now = [100.0]
        clock = mock.Mock(wraps=bridge.time)
        clock.monotonic.side_effect = lambda: now[0]
        with mock.patch.object(bridge, "time", clock):
            bridge.ingest_raw_pcm(SPEECH)
            old = bridge.current_session
            now[0] = 109.0
            slicer = asyncio.create_task(bridge.window_slicer_worker())
            try:
                for _ in range(300):
                    if old.closed:
                        break
                    await asyncio.sleep(0.001)
                self.assertTrue(old.closed)
                self.assertEqual(bytes(old.pcm), SPEECH)
                new_audio = b"\x00\x20" * 8000
                bridge.ingest_raw_pcm(new_audio)
                self.assertEqual(len(bridge.audio_sessions), 2)
                self.assertIsNot(old, bridge.current_session)
                self.assertEqual(bytes(bridge.current_session.pcm), new_audio)
                bridge.stt_queue.get_nowait()
                bridge.stt_queue.task_done()
                window = await asyncio.wait_for(bridge.stt_queue.get(), 1)
                bridge.stt_queue.task_done()
                self.assertEqual(window.pcm, SPEECH)
                self.assertEqual(window.session_id, old.session_id)
                self.assertNotEqual(window.session_id, bridge.current_session.session_id)
                self.assertEqual(bytes(bridge.current_session.pcm), new_audio)
            finally:
                slicer.cancel()
                await asyncio.gather(slicer, return_exceptions=True)

    async def test_ingress_limit_counts_all_pending_sessions(self):
        now = [100.0]
        clock = mock.Mock(wraps=bridge.time)
        clock.monotonic.side_effect = lambda: now[0]
        with mock.patch.object(bridge, "time", clock), mock.patch.object(bridge, "MAX_BUFFER_BYTES", 8):
            bridge.ingest_raw_pcm(b"\x01\x01" * 2)
            now[0] = 109.0
            bridge.ingest_raw_pcm(b"\x02\x02" * 2)
            self.assertEqual(len(bridge.audio_sessions), 2)
            with self.assertRaises(web.HTTPServiceUnavailable):
                bridge.ingest_raw_pcm(b"\x03\x03")
            self.assertEqual(bridge.buffered_audio_bytes(), 8)
            self.assertEqual(bridge.audio_ring_bytes, 8)

    async def test_shutdown_drains_an_inflight_request_beyond_fifteen_seconds(self):
        entered, release = threading.Event(), threading.Event()

        def slow_request(*args):
            entered.set()
            release.wait(5)
            return "Accepted audio survives shutdown"

        async def background():
            await asyncio.Event().wait()

        await self.enqueue()
        app = {"slicer_task": asyncio.create_task(background()),
               "research_task": asyncio.create_task(background()),
               "stt_worker_task": asyncio.create_task(bridge.stt_consumer_worker())}
        cleanup = None
        with mock.patch.object(bridge, "_gemini_stt_request_sync", side_effect=slow_request):
            try:
                for _ in range(1000):
                    if entered.is_set():
                        break
                    await asyncio.sleep(0.001)
                self.assertTrue(entered.is_set())
                cleanup = asyncio.create_task(bridge.on_cleanup(app))
                for _ in range(4):
                    await asyncio.sleep(0)
                loop = asyncio.get_running_loop()
                original_time, debug = loop.time, loop.get_debug()
                loop.set_debug(False)
                await asyncio.sleep(0)
                try:
                    with mock.patch.object(loop, "time", lambda: original_time() + 20):
                        for _ in range(6):
                            await asyncio.sleep(0)
                        release.set()
                        await cleanup
                finally:
                    loop.set_debug(debug)
                bridge.record_transcript_line.assert_called_once_with("Accepted audio survives shutdown")
                self.assertFalse(app["stt_worker_task"].cancelled())
                await bridge.stt_queue.join()
            finally:
                release.set()
                tasks = [*app.values(), *([cleanup] if cleanup else [])]
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


class RouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patches = [mock.patch.object(bridge, "stt_queue", asyncio.Queue(maxsize=24)),
                        mock.patch.object(bridge, "GEMINI_API_KEY", ""),
                        mock.patch.object(bridge, "flush_daily_paragraph")]
        for patch in self.patches:
            patch.start()
        self.addCleanup(lambda: [patch.stop() for patch in reversed(self.patches)])
        bridge.audio_pcm_buffer.clear()
        bridge.audio_pre_tail.clear()
        bridge.audio_sessions.clear()
        bridge.current_session = None
        bridge._next_session_id = 0
        bridge.stopping = False

    async def client_for(self, app):
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        return client

    async def test_omi_setup_response_remains_compatible_without_gemini_key(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        response = await client.get("/omi")
        self.assertEqual(response.status, 200)
        data = await response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["is_setup_completed"])

    async def test_health_is_separate_and_existing_monitor_routes_remain_available(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        for path in ("/health", "/omi/unexpected"):
            with self.subTest(path=path):
                response = await client.get(path)
                self.assertEqual(response.status, 404)
        for path in ("/listen", "/audio/last"):
            response = await client.get(path)
            self.assertEqual(response.status, 200)
        async with client.ws_connect("/listen-ws"):
            pass

    async def test_local_health_app_exposes_telemetry_separately(self):
        client = await self.client_for(bridge.create_health_app())
        response = await client.get("/health")
        self.assertEqual(response.status, 200)
        data = await response.json()
        self.assertIn("queue_depth", data)
        self.assertFalse(data["gemini_key_configured"])
        response = await client.get("/omi")
        self.assertEqual(response.status, 404)

    async def test_pcm_ingress_validates_sample_rate_and_frame_alignment(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        for rate in ("0", "bad", "44100"):
            with self.subTest(rate=rate):
                response = await client.post("/omi?sample_rate=" + rate, data=SPEECH,
                                             headers={"Content-Type": "application/octet-stream"})
                self.assertEqual(response.status, 400)
        response = await client.post("/omi", data=b"\x01", headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(response.status, 400)
        response = await client.post("/omi", data=SPEECH, headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(response.status, 200)


if __name__ == "__main__":
    unittest.main()
