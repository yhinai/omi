"""Offline regressions for live ingress, configuration, replay, and persistence."""
import asyncio
from collections import namedtuple
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import bridge
import stt

bridge.logger.addHandler(logging.NullHandler())
bridge.logger.propagate = False

SPEECH = b"\x00\x10" * 8000  # Half a second above the silence gate.
VersionInfo = namedtuple("VersionInfo", "major minor micro releaselevel serial")


def live_state():
    runtime = bridge.BridgeState()
    runtime.live_stt = stt.LocalWhisperSTT(python="python", model="test", vad="vad",
        on_partial=bridge.publish_partial, on_final=bridge.record_live_final,
        on_gap=bridge.record_live_gap, on_latency=bridge.record_latency,
        on_connection=bridge.live_connection_changed)
    runtime.live_stt.flush_now = mock.AsyncMock()
    return runtime


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
spec = importlib.util.spec_from_file_location('bridge_import_probe', sys.argv[1])
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

    def test_process_configuration_overrides_project_file(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory)/"local.env"
            local.write_text("OMI_IDLE_FLUSH_SEC=6\n")
            self.assertEqual(bridge.load_env(local_path=local, environ={})["OMI_IDLE_FLUSH_SEC"], "6")
            self.assertEqual(bridge.load_env(local_path=local, environ={"OMI_IDLE_FLUSH_SEC":"5"})["OMI_IDLE_FLUSH_SEC"], "5")


class LiveConfigurationTests(unittest.TestCase):
    def configure(self, env):
        with mock.patch.object(bridge.os.path, "isfile", return_value=True), mock.patch.object(bridge.os, "access", return_value=True):
            bridge.configure(env)

    def tearDown(self):
        self.configure({})

    def test_local_defaults_and_audio_tuning(self):
        self.configure({"OMI_MAX_AUDIO_AGE_SEC":"12", "OMI_IDLE_FLUSH_SEC":"4"})
        self.assertEqual(bridge.MAX_AUDIO_AGE_SEC,12)
        self.assertEqual(bridge.IDLE_FLUSH_SEC,4)
        self.assertEqual(bridge.STT_BACKEND,"whisper")

    def test_missing_worker_and_models_are_rejected(self):
        with mock.patch.object(bridge.os.path,"isfile",return_value=False):
            with self.assertRaisesRegex(ValueError,"worker environment"):
                bridge.configure({})
        with mock.patch.object(bridge.os.path,"isfile",side_effect=[True,False]), mock.patch.object(bridge.os,"access",return_value=True):
            with self.assertRaisesRegex(ValueError,"model files"):
                bridge.configure({})


class RouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patches = [mock.patch.object(bridge, "state", live_state()),
                        mock.patch.object(bridge, "flush_daily_paragraph")]
        for patcher in self.patches:
            patcher.start()
        self.addCleanup(lambda: [patcher.stop() for patcher in reversed(self.patches)])

    async def client_for(self, app):
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        return client

    async def test_phone_overflow_closes_for_retry_without_losing_journal_or_viewer_finals(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        async with client.ws_connect("/listen?replay=0") as captions, client.ws_connect("/v1/listen") as phone:
            await captions.receive_json(timeout=1)  # replay start
            await captions.receive_json(timeout=1)  # replay end
            await captions.receive_json(timeout=1)  # status
            with mock.patch.object(bridge, "event_logger") as journal:
                for index in range(40):
                    self.assertTrue(bridge.record_transcript_line(f"Synthetic final {index}.", utterance_id=f"delivery-{index}"))
                self.assertEqual(journal.info.call_count, 40)
            async with asyncio.timeout(2):
                while True:
                    message = await phone.receive()
                    if message.type == bridge.aiohttp.WSMsgType.CLOSE:
                        self.assertEqual(message.data, 1013)
                        break
            finals = []
            while len(finals) < 40:
                event = await captions.receive_json(timeout=1)
                if event.get("type") == "transcript":
                    finals.append(event)
            self.assertEqual([event["text"] for event in finals], [f"Synthetic final {index}." for index in range(40)])
            self.assertEqual(len(bridge.state.recent_events), 40)
            self.assertFalse(captions.closed)
            self.assertEqual(bridge.build_health_dict()["phone_delivery_errors"], 1)
            self.assertEqual(bridge.state.metrics["last_output_error"], None)
        # Reconnection gets a clean delivery state rather than the old stalled queue.
        async with client.ws_connect("/v1/listen") as replacement:
            with mock.patch.object(bridge, "event_logger"):
                bridge.record_transcript_line("Delivery recovered.", utterance_id="recovered")
            self.assertEqual((await replacement.receive_json(timeout=1))["text"], "Delivery recovered.")
            self.assertEqual(bridge.state.metrics["phone_delivery_errors"], 1)

    async def test_phone_sender_failures_close_the_one_way_connection(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        for count, error in enumerate((TimeoutError(), OSError()), 1):
            with self.subTest(error=type(error).__name__):
                async with client.ws_connect("/v1/listen") as phone:
                    server = next(iter(bridge.state.stream_clients))
                    with mock.patch.object(server, "send_str", side_effect=error), \
                         mock.patch.object(bridge, "event_logger") as journal:
                        self.assertTrue(bridge.record_transcript_line("Synthetic delivery check.", utterance_id=f"send-{count}"))
                        message = await phone.receive(timeout=1)
                        self.assertEqual(message.type, bridge.aiohttp.WSMsgType.CLOSE)
                        self.assertEqual(message.data, 1013)
                        self.assertEqual(journal.info.call_count, 1)
                    self.assertEqual(bridge.state.metrics["phone_delivery_errors"], count)

    async def test_phone_delivery_failure_is_counted_once_and_degrades_active_health(self):
        phone = mock.Mock()
        phone.close = mock.AsyncMock(return_value=True)
        stream = bridge.PhoneStream(asyncio.Queue(maxsize=1), 0)
        bridge.state.stream_clients[phone] = stream
        worker = mock.Mock()
        worker.done.return_value = False
        bridge.state.background_tasks["test"] = worker
        bridge.state.live_stt.connected = True
        self.assertEqual(bridge.build_health_dict()["status"], "ok")
        bridge.fail_phone_delivery(phone, stream, "queue_full")
        bridge.fail_phone_delivery(phone, stream, "send_failed")
        self.assertEqual(bridge.build_health_dict()["status"], "degraded")
        self.assertEqual(bridge.state.metrics["phone_delivery_errors"], 1)
        async with asyncio.timeout(1):
            while not phone.close.await_count:
                await asyncio.sleep(0)
        phone.close.assert_awaited_once_with(code=1013, message=b"Caption delivery stalled; reconnect", drain=False)
        bridge.state.stream_clients.clear()

    async def test_retired_webhook_routes_return_404(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        for path in ("/omi", "/omi/ws"):
            response = await client.get(path)
            self.assertEqual(response.status, 404)
        response = await client.post("/omi", data=SPEECH)
        self.assertEqual(response.status, 404)
        self.assertEqual(bridge.state.live_stt.pending_bytes, 0)

    async def test_stopping_bridge_rejects_audio_connections(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        with mock.patch.object(bridge.state, "stopping", True):
            response = await client.get("/v1/listen")
            self.assertEqual(response.status, 503)

    async def test_disconnected_provider_is_degraded_even_when_workers_are_alive(self):
        worker = mock.Mock()
        worker.done.return_value = False
        with mock.patch.object(bridge.state, "background_tasks", {"worker": worker}):
            self.assertFalse(bridge.state.live_stt.connected)
            self.assertEqual(bridge.build_health_dict()["status"], "degraded")

    async def test_overloaded_stream_keeps_fresh_audio(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        raw = SPEECH * 24
        with mock.patch.object(bridge, "write_event", return_value=True):
            async with client.ws_connect("/v1/listen") as ws:
                await ws.send_bytes(raw)
                async with asyncio.timeout(1):
                    while bridge.state.live_stt.pending_bytes != bridge.BYTES_PER_SECOND * 10:
                        await asyncio.sleep(.01)
                self.assertEqual(bridge.state.live_stt.queue[0].pcm, raw[-bridge.BYTES_PER_SECOND * 10:])
                self.assertEqual(bridge.state.metrics["lost_audio_sec_queue_evicted"], 2)

    async def test_websocket_message_size_limit(self):
        with mock.patch.object(bridge, "MAX_INPUT_BYTES", bridge.BYTES_PER_SECOND):
            client = await self.client_for(bridge.create_app(start_workers=False))
            async with client.ws_connect("/v1/listen") as ws:
                await ws.send_bytes(bytes(bridge.BYTES_PER_SECOND))
                async with asyncio.timeout(1):
                    while not bridge.state.live_stt.pending_bytes:
                        await asyncio.sleep(.01)
                await ws.send_bytes(bytes(bridge.BYTES_PER_SECOND + 2))
                message = await ws.receive(timeout=1)
                self.assertEqual(message.data, 1009)

    async def test_health_is_separate_and_existing_monitor_routes_remain_available(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        for path in ("/health", "/omi/unexpected", "/listen-ws", "/listen-ndjson", "/live", "/tui", "/tui.py", "/audio/last", "/audio/last?seconds=60"):
            with self.subTest(path=path):
                response = await client.get(path)
                self.assertEqual(response.status, 404)
        async with client.ws_connect("/listen"):
            pass

    async def test_local_health_app_exposes_telemetry_separately(self):
        client = await self.client_for(bridge.create_health_app())
        response = await client.get("/health")
        self.assertEqual(response.status, 503)
        data = await response.json()
        self.assertIn("queue_depth", data)
        self.assertEqual(data["model"], "whisper-large-v3-turbo-fp16")
        self.assertEqual(data["stt_backend"], "whisper")
        response = await client.get("/omi")
        self.assertEqual(response.status, 404)

    async def test_health_returns_success_only_when_workers_are_healthy(self):
        client = await self.client_for(bridge.create_health_app())
        alive = mock.Mock()
        alive.done.return_value = False
        with mock.patch.object(bridge.state, "background_tasks", {"worker": alive}), \
             mock.patch.object(bridge.state.live_stt, "connected", True), \
             mock.patch.dict(bridge.state.metrics, {"last_error": None}):
            response = await client.get("/health")
            self.assertEqual(response.status, 200)
            self.assertEqual((await response.json())["status"], "ok")
            alive.done.return_value = True
            response = await client.get("/health")
            self.assertEqual(response.status, 503)
            self.assertEqual((await response.json())["status"], "degraded")

    async def test_pcm_stream_rejects_wav_headers_and_incomplete_samples(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        for raw in (b"RIFF" + bytes(40), b"\x01"):
            async with client.ws_connect("/v1/listen") as ws:
                await ws.send_bytes(raw)
                message = await ws.receive(timeout=1)
                self.assertEqual(message.data, 1003)
        self.assertEqual(bridge.state.live_stt.pending_bytes, 0)

    async def test_invalid_reconnect_preserves_working_stream(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        headers = {"Authorization": "Bearer secret-token", "X-Forwarded-For": "192.0.2.1"}
        with mock.patch.object(bridge, "STREAM_TOKEN", "secret-token"):
            async with client.ws_connect("/v1/listen", headers=headers) as phone:
                original = next(iter(bridge.state.stream_clients))
                for query in ("encoding=opus", "sample_rate=bad", "sample_rate=44100", "channels=2"):
                    response = await client.get("/v1/listen?" + query, headers=headers)
                    self.assertEqual(response.status, 400)
                    self.assertIn(original, bridge.state.stream_clients)
                    self.assertFalse(original.closed)
                # Valid parameters still require a successful WebSocket upgrade.
                response = await client.get("/v1/listen", headers=headers)
                self.assertEqual(response.status, 400)
                response = await client.get("/v1/listen", headers={**headers, "Authorization": "Bearer wrong"})
                self.assertEqual(response.status, 401)
                self.assertIn(original, bridge.state.stream_clients)
                self.assertFalse(original.closed)
                await phone.send_bytes(SPEECH)
                async with asyncio.timeout(1):
                    while not bridge.state.live_stt.pending_bytes:
                        await asyncio.sleep(.01)
                with mock.patch.object(bridge, "event_logger"):
                    bridge.record_transcript_line("Original stream still works")
                self.assertEqual((await phone.receive_json(timeout=1))["text"], "Original stream still works")

    async def test_valid_reconnect_replaces_previous_stream(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        headers = {"X-Forwarded-For": "192.0.2.1"}
        async with client.ws_connect("/v1/listen", headers=headers) as old_phone:
            old_server = next(iter(bridge.state.stream_clients))
            async with client.ws_connect("/v1/listen", headers=headers) as new_phone:
                await old_phone.receive(timeout=1)
                self.assertTrue(old_phone.closed)
                self.assertNotIn(old_server, bridge.state.stream_clients)
                self.assertEqual(len(bridge.state.stream_clients), 1)
                await new_phone.send_bytes(SPEECH)
                async with asyncio.timeout(1):
                    while not bridge.state.live_stt.pending_bytes:
                        await asyncio.sleep(.01)
                with mock.patch.object(bridge, "event_logger"):
                    bridge.record_transcript_line("Replacement stream works")
                self.assertEqual((await new_phone.receive_json(timeout=1))["text"], "Replacement stream works")

    async def test_phone_segments_use_connection_relative_audio_timing(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        bridge.state.live_stt.total_input_samples = 160000
        async with client.ws_connect("/v1/listen") as phone:
            with mock.patch.object(bridge, "event_logger") as events:
                for text, start, end in (("First", 176000, 200000), ("Second", 208000, 224000)):
                    bridge.record_transcript_line(text, sample_offset=start, sample_end_offset=end)
                    reply = await phone.receive_json(timeout=1)
                    segment = reply["segments"][0]
                    self.assertEqual(segment["start"], (start - 160000) / 16000)
                    self.assertEqual(segment["end"], (end - 160000) / 16000)
                    self.assertEqual(json.loads(events.info.call_args.args[0])["sample_end_offset"], end)
                bridge.state.live_stt.total_input_samples = 240000
                async with client.ws_connect("/v1/listen") as replacement:
                    # Old recognition must not be attached to the new recording.
                    bridge.record_transcript_line("Late old final", sample_offset=224000, sample_end_offset=240000)
                    bridge.record_transcript_line("New connection", sample_offset=240000, sample_end_offset=248000)
                    reply = await replacement.receive_json(timeout=1)
                    self.assertEqual(reply["text"], "New connection")
                    self.assertEqual(reply["segments"][0]["start"], 0)
                    self.assertEqual(reply["segments"][0]["end"], .5)

    async def test_direct_phone_websocket_adapter_ingests_pcm_and_returns_transcripts(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        for bad_url in ("/v1/listen?sample_rate=44100", "/v1/listen?sample_rate=bad",
                        "/v1/listen?encoding=opus", "/v1/listen?channels=2"):
            resp = await client.get(bad_url)
            self.assertEqual(resp.status, 400)
        with mock.patch.object(bridge, "STREAM_TOKEN", "secret-token"):
            unauth = await client.get("/v1/listen")
            self.assertEqual(unauth.status, 401)
            wrong = await client.get("/v1/listen", headers={"Authorization": "Token wrong"})
            self.assertEqual(wrong.status, 401)
            async with client.ws_connect("/v1/listen?encoding=linear16&sample_rate=16000&channels=1",
                                         headers={"Authorization": "Token secret-token",
                                                  "X-Forwarded-For": "192.168.68.71"}) as ws:
                conflict = await client.get("/v1/listen", headers={"Authorization": "Bearer secret-token"})
                self.assertEqual(conflict.status, 409)
                await ws.send_str(json.dumps({"type": "KeepAlive"}))
                await ws.send_bytes(SPEECH)
                async with asyncio.timeout(1):
                    while not bridge.state.live_stt.pending_bytes:
                        await asyncio.sleep(.01)
                self.assertEqual(bridge.state.metrics["audio_source"], "direct_ws")
                self.assertEqual(bridge.state.metrics["direct_ws_remote"], "192.168.68.71")
                with mock.patch.object(bridge, "event_logger"):
                    bridge.record_transcript_line("Direct phone transcript.Second sentence.")
                msg = await asyncio.wait_for(ws.receive_json(), timeout=1)
                self.assertEqual(msg["text"], "Direct phone transcript.Second sentence.")
                self.assertEqual(msg["segments"][0]["text"], "Direct phone transcript.Second sentence.")
                # Diagnostic text without source timing must not invent timestamps.
                self.assertNotIn("start", msg["segments"][0])
                self.assertNotIn("end", msg["segments"][0])
                self.assertEqual(set(msg), {"type", "is_final", "speech_final", "text", "segments"})
                self.assertEqual(msg["type"], "Results")
                self.assertTrue(msg["is_final"])
                self.assertTrue(msg["speech_final"])
                await ws.send_str(json.dumps({"type": "CloseStream"}))


class TimestampTests(unittest.TestCase):
    def test_audio_time_controls_midnight_notes_and_timing_metadata_is_retained(self):
        source = datetime(2026, 10, 24, 23, 59, 59).timestamp()
        processed = datetime(2026, 10, 25, 0, 2, 0).timestamp()
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(bridge.state.storage, "notes", bridge.DailyNotes(directory)), \
             mock.patch.object(bridge, "event_logger") as events, \
             mock.patch.object(bridge, "publish_caption") as publish:
            bridge.record_transcript_line("trigger spoken words", audio_start_at=source,
                received_at=source + .5, processed_at=processed, session_id=42, sample_offset=80000)
            note = Path(directory, "2026-10-24.md").read_text()
            self.assertIn("**[23:59:59]** trigger spoken words", note)
            self.assertFalse(Path(directory, "2026-10-25.md").exists())
            event = json.loads(events.info.call_args.args[0])
            self.assertEqual(event["audio_start_at"], source)
            self.assertEqual(event["received_at"], source + .5)
            self.assertEqual(event["processed_at"], processed)
            self.assertEqual(event["time_basis"], "estimated_from_receipt")
            self.assertEqual((event["session_id"], event["sample_offset"]), (42, 80000))
            self.assertEqual(publish.call_args.args[0]["audio_start_at"], source)




class ReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = live_state()
        patcher = mock.patch.object(bridge, "state", self.runtime)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_output_failure_does_not_drop_recognition_or_later_finals(self):
        notes = mock.Mock()
        notes.add.side_effect = OSError("disk full")
        self.runtime.storage.notes = notes
        with mock.patch.object(bridge, "event_logger") as events:
            self.assertTrue(bridge.record_live_final("First words", utterance_id="first"))
            self.assertTrue(bridge.record_live_final("Next words", utterance_id="next"))
        self.assertEqual(self.runtime.metrics["audio_gaps"], 0)
        self.assertEqual(self.runtime.metrics["output_errors"], 2)
        self.assertEqual(self.runtime.metrics["utterances_processed"], 2)
        self.assertEqual(self.runtime.metrics["transcripts_emitted"], 2)
        self.assertEqual(events.info.call_count, 2)

    async def test_jsonl_failure_prevents_projections_and_recovers_on_next_final(self):
        self.runtime.storage.notes = mock.Mock()
        with mock.patch.object(bridge, "event_logger") as events:
            events.info.side_effect = OSError("disk full")
            self.assertFalse(bridge.record_live_final("First words", utterance_id="first"))
            self.runtime.storage.notes.add.assert_not_called()
            self.assertFalse(self.runtime.recent_events)
            self.assertEqual(self.runtime.metrics["transcripts_emitted"], 0)
            self.assertIsNotNone(self.runtime.metrics["last_output_error"])
            events.info.side_effect = None
            self.assertTrue(bridge.record_live_final("Next words", utterance_id="next"))
        self.assertIsNone(self.runtime.metrics["last_output_error"])
        self.assertEqual(self.runtime.metrics["audio_gaps"], 0)
        self.assertEqual(self.runtime.metrics["utterances_processed"], 2)
        self.assertEqual(self.runtime.metrics["transcripts_emitted"], 1)

    async def test_jsonl_is_written_before_markdown_and_markdown_failure_is_reported(self):
        order = []
        notes = mock.Mock()
        def failed_markdown(*args):
            order.append("markdown")
            raise OSError("full")
        notes.add.side_effect = failed_markdown
        self.runtime.storage.notes = notes
        with mock.patch.object(bridge, "event_logger") as events:
            events.info.side_effect = lambda *args: order.append("jsonl")
            self.assertTrue(bridge.record_transcript_line("hello"))
        self.assertEqual(order, ["jsonl", "markdown"])
        self.assertEqual(self.runtime.metrics["output_errors"], 1)
        self.assertEqual(self.runtime.metrics["audio_gaps"], 0)
        notes.close_paragraph.assert_called_once()

    async def test_gap_is_persisted_in_jsonl_and_daily_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            self.runtime.storage.notes = bridge.DailyNotes(directory)
            with mock.patch.object(bridge, "event_logger") as events:
                bridge.record_gap("stale", .125, session_id=4, sample_offset=8000)
            event = json.loads(events.info.call_args.args[0])
            self.assertEqual(event["type"], "gap")
            self.assertEqual(event["duration_sec"], .125)
            self.assertTrue(event["event_id"])
            note = next(Path(directory).glob("*.md")).read_text()
            self.assertIn("[GAP: 0.125s", note)

    async def test_bridge_state_instances_do_not_share_mutable_buffers(self):
        other = bridge.BridgeState()
        self.runtime.metrics["audio_gaps"] = 3
        self.runtime.recent_events.append({"text": "caption"})
        self.assertEqual(other.metrics["audio_gaps"], 0)
        self.assertFalse(other.recent_events)


class OutputHandlerTests(unittest.TestCase):

    def test_invalid_live_audio_settings_fail_startup(self):
        with mock.patch.object(bridge.os.path, "isfile", return_value=True), \
             mock.patch.object(bridge.os, "access", return_value=True):
            for name in ("OMI_MAX_AUDIO_AGE_SEC", "OMI_IDLE_FLUSH_SEC"):
                for value in ("0", "-1", "nan", "inf"):
                    with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                        bridge.configure({name: value})
            bridge.configure({})


class LatencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = live_state()
        self.now = 100.0
        clock = mock.Mock(wraps=bridge.time)
        clock.monotonic.side_effect = lambda: self.now
        clock.time.side_effect = lambda: 10000 - self.now  # Wall clock deliberately moves backwards.
        for patcher in (mock.patch.object(bridge, "state", self.state),
                        mock.patch.object(bridge, "time", clock),
                        mock.patch("stt.time", clock)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_empty_health_reports_no_latency_observations_or_pending_audio(self):
        health = bridge.build_health_dict()
        self.assertIsNone(health["oldest_pending_audio_age_sec"])
        for summary in health["latency"].values():
            self.assertEqual(summary, {"samples": 0, "last": None, "p50": None, "p95": None})

    def test_caption_status_exposes_provider_word_and_packet_timing(self):
        for metric, value in (("provider_word_end_to_partial_sec", .2),
                              ("provider_word_end_to_final_sec", .6),
                              ("final_word_packet_to_result_sec", .4)):
            bridge.record_latency(metric, value)
        status = bridge.caption_status()
        self.assertEqual(status["latency_provider_word_end_to_partial_p50_sec"], .2)
        self.assertEqual(status["latency_provider_word_end_to_final_p50_sec"], .6)
        self.assertEqual(status["latency_final_word_packet_to_result_p50_sec"], .4)

    def test_live_queue_age_uses_receipt_clock(self):
        bridge.ingest_raw_pcm(SPEECH)
        self.now += 5
        self.assertEqual(bridge.build_health_dict()["oldest_pending_audio_age_sec"], 5)
        self.assertEqual(bridge.caption_status()["buffer_sec"], .5)
        self.assertEqual(bridge.caption_status()["queue_depth"], 1)
        self.state.live_stt.queue.clear()
        self.state.live_stt.pending_bytes = 0
        self.assertIsNone(bridge.build_health_dict()["oldest_pending_audio_age_sec"])

    def test_percentiles_are_nearest_rank_and_history_is_bounded(self):
        for i in range(200):
            bridge.record_latency("receipt_to_final_sec", i)
        summary = bridge.build_health_dict()["latency"]["receipt_to_final_sec"]
        self.assertEqual(summary, {"samples": 128, "last": 199, "p50": 135, "p95": 193})


class CaptionFeedTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = live_state()
        for patcher in (mock.patch.object(bridge, "state", self.runtime),
                        mock.patch.object(bridge, "event_logger")):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(TestServer(bridge.create_app(start_workers=False)))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    def seed(self, count):
        for i in range(count):
            self.runtime.recent_events.append({"type": "transcript", "event_id": f"e{i}",
                "text": f"Caption {i}", "timestamp": "2026-10-24 12:00:00"})

    async def replay(self, ws):
        start = await ws.receive_json(timeout=2)
        events = [await ws.receive_json(timeout=2) for _ in range(start["count"])]
        end = await ws.receive_json(timeout=2)
        self.assertEqual((start["type"], start["phase"], end["phase"]), ("replay", "start", "end"))
        return start, events

    async def test_default_listen_replays_captions_and_never_streams_raw_audio(self):
        self.seed(130)
        async with self.client.ws_connect("/listen") as ws:
            start, events = await self.replay(ws)
            self.assertFalse(start["reset"])
            self.assertEqual([e["event_id"] for e in events], [f"e{i}" for i in range(30, 130)])
            status = await ws.receive_json(timeout=2)
            self.assertEqual(status["type"], "status")
            bridge.ingest_raw_pcm(b"\x01\x01" * 160)
            bridge.record_transcript_line("Fresh caption")
            fresh = await ws.receive_json(timeout=2)
            self.assertEqual(fresh["text"], "Fresh caption")
            self.assertEqual(fresh["stream_id"], self.runtime.stream_id)
            self.assertEqual(len(self.runtime.recent_events), 131)

    async def test_resume_cursor_recovers_all_retained_events_after_it(self):
        self.seed(10)
        async with self.client.ws_connect("/listen?replay=1&after=e6") as ws:
            start, events = await self.replay(ws)
            self.assertFalse(start["reset"])
            self.assertEqual([e["event_id"] for e in events], ["e7", "e8", "e9"])

    async def test_expired_cursor_explicitly_resets_and_bounds_replay(self):
        self.seed(10)
        async with self.client.ws_connect("/listen?replay=3&after=expired") as ws:
            start, events = await self.replay(ws)
            self.assertTrue(start["reset"])
            self.assertEqual([e["event_id"] for e in events], ["e7", "e8", "e9"])

    async def test_new_live_event_follows_replay_without_a_subscription_hole(self):
        self.seed(20)
        async with self.client.ws_connect("/listen?replay=20") as ws:
            bridge.record_transcript_line("Arrived during replay")
            _, events = await self.replay(ws)
            self.assertEqual(len(events), 20)
            messages = [await ws.receive_json(timeout=2), await ws.receive_json(timeout=2)]
            self.assertEqual(sum(e.get("text") == "Arrived during replay" for e in messages), 1)

    async def test_caption_backpressure_closes_for_replay_instead_of_losing_old_captions_silently(self):
        listener = bridge.CaptionListener()
        self.runtime.caption_listeners["test"] = listener
        for i in range(64):
            bridge.publish_caption({"type": "transcript", "event_id": str(i)})
        bridge.publish_caption({"type": "transcript", "event_id": "overflow"})
        self.assertTrue(listener.closed)
        self.assertEqual(listener.queue.qsize(), 64)
        self.assertEqual(listener.queue.get_nowait()["event_id"], "0")
        self.assertTrue(listener.ready.is_set())
        self.runtime.caption_listeners.clear()

    async def test_status_does_not_consume_reliable_caption_queue_capacity(self):
        listener = bridge.CaptionListener()
        self.runtime.caption_listeners["test"] = listener
        for _ in range(64):
            listener.queue.put_nowait({"type": "transcript"})
        sleeping = asyncio.Event()
        async def pause(*args):
            sleeping.set()
            await asyncio.Event().wait()
        with mock.patch.object(bridge.asyncio, "sleep", side_effect=pause):
            worker = asyncio.create_task(bridge.caption_status_worker())
            try:
                await asyncio.wait_for(sleeping.wait(), 1)
                self.assertEqual(listener.status["type"], "status")
                self.assertEqual(listener.queue.qsize(), 64)
                self.assertFalse(listener.closed)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
        self.runtime.caption_listeners.clear()

    async def test_invalid_caption_parameters_fail_before_websocket_upgrade(self):
        for query in ("mode=audio", "mode=unknown", "replay=-1", "replay=201", "replay=bad"):
            response = await self.client.get("/listen?" + query)
            self.assertEqual(response.status, 400)

    async def test_jsonl_failure_does_not_publish_or_cache_an_uncommitted_event(self):
        bridge.event_logger.info.side_effect = OSError("full")
        listener = bridge.CaptionListener()
        self.runtime.caption_listeners["test"] = listener
        self.assertFalse(bridge.record_transcript_line("not saved"))
        self.assertFalse(self.runtime.recent_events)
        self.assertTrue(listener.queue.empty())
        self.runtime.caption_listeners.clear()

    async def test_seed_reads_rotation_in_order_skips_malformed_lines_and_preserves_ids(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(bridge, "LOG_DIR", directory):
            path = Path(directory, "omi_transcript_events.jsonl")
            older = [{"type": "transcript", "event_id": f"e{i}", "text": str(i)} for i in range(50)]
            newer = [{"type": "transcript", "event_id": f"e{i}", "text": str(i)} for i in range(50, 230)]
            path.with_name(path.name + ".1").write_text("\n".join(json.dumps(e) for e in older) + "\n")
            path.write_text("\n".join(json.dumps(e) for e in newer) + '\n{"broken"\n')
            bridge.seed_recent_events()
            self.assertEqual([e["event_id"] for e in self.runtime.recent_events], [f"e{i}" for i in range(30, 230)])
            path.write_text(json.dumps({"text": "Legacy transcript"}) + "\n")
            bridge.seed_recent_events()
            identity = self.runtime.recent_events[-1]["event_id"]
            bridge.seed_recent_events()
            self.assertEqual(self.runtime.recent_events[-1]["event_id"], identity)
            self.assertTrue(identity.startswith("legacy-"))


if __name__ == "__main__":
    unittest.main()


class PartialAndFinalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = bridge.BridgeState()
        patcher = mock.patch.object(bridge, "state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)


    def test_partial_updates_are_coalesced_and_never_written_or_replayed(self):
        listener = bridge.CaptionListener()
        self.state.caption_listeners["test"] = listener
        listener.queue.put_nowait({"type": "transcript", "text": "Already final"})
        with mock.patch.object(bridge, "record_transcript_line") as record, \
             mock.patch.object(bridge, "event_logger") as events:
            for revision in range(100):
                bridge.publish_partial({"type": "partial", "utterance_id": "u1", "revision": revision, "text": str(revision)})
        self.assertEqual(listener.partial["text"], "99")
        self.assertEqual(listener.queue.qsize(), 1)
        self.assertFalse(self.state.recent_events)
        record.assert_not_called()
        events.info.assert_not_called()


    def test_old_partial_clear_does_not_erase_a_new_utterance(self):
        bridge.publish_partial({"type": "partial", "utterance_id": "new", "text": "Current words"})
        bridge.publish_partial({"type": "partial", "utterance_id": "old", "text": ""})
        self.assertEqual(self.state.partial_caption["utterance_id"], "new")


    def test_live_ingress_forwards_audio(self):
        stream = mock.Mock()
        self.state.live_stt = stream
        bridge.ingest_raw_pcm(SPEECH)
        stream.feed.assert_called_once()
        self.assertEqual(stream.feed.call_args.args[0], SPEECH)


    def test_final_events_are_marked_and_persisted_once(self):
        with mock.patch.object(bridge, "publish_saved", return_value=True) as output:
            bridge.record_live_final("Final speech", utterance_id="u1", session_id=1)
        event = output.call_args.args[0]
        self.assertEqual((event["type"],event["is_final"],event["utterance_id"]), ("transcript",True,"u1"))
        self.assertEqual(self.state.metrics["transcripts_emitted"],1)
        self.assertEqual(self.state.metrics["utterances_processed"],1)
