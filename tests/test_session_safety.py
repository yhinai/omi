"""Connection ownership under delayed WebSocket close and shutdown."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest import mock

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import bridge
from test_bridge import live_state


class PhoneSocket:
    """Control close timing so buffered old frames can race a replacement."""
    def __init__(self, *, delay_close=False):
        self.frames = asyncio.Queue()
        self.prepared = asyncio.Event()
        self.close_allowed = asyncio.Event()
        self.handler_closing = asyncio.Event()
        self.close_count = 0
        if not delay_close:
            self.close_allowed.set()

    async def prepare(self, request):
        self.prepared.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        frame = await self.frames.get()
        if frame is None:
            raise StopAsyncIteration
        return frame

    async def close(self, **kwargs):
        self.close_count += 1
        if self.close_count >= 2:
            self.handler_closing.set()
        await self.close_allowed.wait()
        self.frames.put_nowait(None)


class SessionOwnershipTests(unittest.IsolatedAsyncioTestCase):
    def test_health_exposes_aggregate_signal_metrics_without_raw_audio(self):
        with mock.patch.object(bridge, "state", live_state()):
            bridge.ingest_raw_pcm(b"\xff\x7f\x00\x80")
            quality = bridge.build_health_dict()["audio_quality"]
            self.assertEqual(quality["sampled_samples"], 2)
            self.assertEqual(quality["clipped_sample_pct"], 100)
            self.assertEqual(quality["coverage"], "complete")
            self.assertNotIn("pcm", quality)
            self.assertNotIn("samples", quality)

    async def test_replaced_socket_cannot_feed_or_flush_or_end_the_new_session(self):
        for kind, data in ((aiohttp.WSMsgType.BINARY, b"\x00\x10" * 800),
                           (aiohttp.WSMsgType.TEXT, '{"type":"CloseStream"}')):
            with self.subTest(kind=kind):
                runtime = live_state()
                old, new = PhoneSocket(delay_close=True), PhoneSocket()
                request = SimpleNamespace(query={}, headers={}, remote="192.0.2.1")
                tasks = []
                with mock.patch.object(bridge, "state", runtime), \
                     mock.patch.object(bridge, "STREAM_TOKEN", ""), \
                     mock.patch.object(bridge.web, "WebSocketResponse", side_effect=[old, new]):
                    try:
                        async with asyncio.timeout(2):
                            tasks.append(asyncio.create_task(bridge.handle_omi_ws(request)))
                            await old.prepared.wait()
                            tasks.append(asyncio.create_task(bridge.handle_omi_ws(request)))
                            await new.prepared.wait()
                            old.frames.put_nowait(SimpleNamespace(type=kind, data=data))
                            await old.handler_closing.wait()
                            self.assertEqual(runtime.live_stt.total_input_samples, 0)
                            self.assertEqual(runtime.metrics["stream_audio_bytes"], 0)
                            self.assertIsNone(runtime.metrics["session_ended_at"])
                            runtime.live_stt.flush_now.assert_not_awaited()
                            new.frames.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.BINARY,
                                                                 data=b"\x00\x10" * 1600))
                            new.frames.put_nowait(None)
                            await tasks[1]
                            self.assertEqual(runtime.live_stt.total_input_samples, 1600)
                            self.assertEqual(runtime.metrics["stream_audio_bytes"], 3200)
                            self.assertIsNotNone(runtime.metrics["session_ended_at"])
                            # The unexpected active disconnect flushes its source boundary.
                            self.assertEqual(runtime.live_stt.controls[-1][1], 1600)
                    finally:
                        old.close_allowed.set()
                        old.frames.put_nowait(None)
                        new.frames.put_nowait(None)
                        await asyncio.gather(*tasks, return_exceptions=True)

    async def test_shutdown_while_waiting_for_admission_rejects_upgrade(self):
        runtime = live_state()
        request = SimpleNamespace(query={}, headers={}, remote="192.0.2.1")
        with mock.patch.object(bridge, "state", runtime), mock.patch.object(bridge, "STREAM_TOKEN", ""):
            await runtime.stream_admission_lock.acquire()
            task = asyncio.create_task(bridge.handle_omi_ws(request))
            await asyncio.sleep(0)
            runtime.stopping = True
            runtime.stream_admission_lock.release()
            with self.assertRaises(web.HTTPServiceUnavailable):
                await task
            self.assertFalse(runtime.stream_clients)

    async def test_handshake_finishing_after_shutdown_does_not_admit_a_client(self):
        for handler in (bridge.handle_omi_ws, bridge.handle_listen):
            with self.subTest(handler=handler.__name__):
                runtime, socket = live_state(), PhoneSocket()
                finish_handshake = asyncio.Event()
                async def prepare(request):
                    socket.prepared.set()
                    await finish_handshake.wait()
                request = SimpleNamespace(query={}, headers={}, remote="192.0.2.1")
                with mock.patch.object(bridge, "state", runtime), \
                     mock.patch.object(bridge, "STREAM_TOKEN", ""), \
                     mock.patch.object(bridge, "CAPTION_TOKEN", ""), \
                     mock.patch.object(bridge.web, "WebSocketResponse", return_value=socket), \
                     mock.patch.object(socket, "prepare", side_effect=prepare):
                    task = asyncio.create_task(handler(request))
                    await asyncio.wait_for(socket.prepared.wait(), 1)
                    await bridge.on_shutdown({})
                    finish_handshake.set()
                    self.assertIs(await asyncio.wait_for(task, 1), socket)
                    self.assertFalse(runtime.stream_clients)
                    self.assertFalse(runtime.caption_listeners)
                    self.assertIsNone(runtime.metrics["session_started_at"])


class CaptionAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_caption_capacity_is_enforced_and_recovers_after_disconnect(self):
        with mock.patch.object(bridge, "state", live_state()), \
             mock.patch.object(bridge, "CAPTION_TOKEN", ""), \
             mock.patch.object(bridge, "MAX_CAPTION_CLIENTS", 1):
            async with TestClient(TestServer(bridge.create_app(start_workers=False))) as client:
                async with client.ws_connect("/listen?replay=0") as ws:
                    await ws.receive_json(timeout=1)
                    response = await client.get("/listen")
                    self.assertEqual(response.status, 503)
                    self.assertEqual(bridge.build_health_dict()["caption_clients"], 1)
                    self.assertEqual(bridge.build_health_dict()["caption_client_limit"], 1)
                async with asyncio.timeout(1):
                    while bridge.state.caption_listeners:
                        await asyncio.sleep(0)
                async with client.ws_connect("/listen?replay=0") as ws:
                    self.assertEqual((await ws.receive_json(timeout=1))["phase"], "start")

    async def test_committed_gap_reaches_viewer_and_replays_after_reconnect(self):
        import tui
        runtime = live_state()
        viewer = tui.Viewer()
        with mock.patch.object(bridge, "state", runtime), \
             mock.patch.object(bridge, "CAPTION_TOKEN", ""), \
             mock.patch.object(bridge, "event_logger"):
            async with TestClient(TestServer(bridge.create_app(start_workers=False))) as client:
                async with client.ws_connect("/listen?replay=0") as ws:
                    for _ in range(3):  # Replay start/end and initial status.
                        await ws.receive_json(timeout=1)
                    bridge.record_gap("stale", .75, session_id=1)
                    event = await ws.receive_json(timeout=1)
                    self.assertEqual(event["type"], "gap")
                    viewer.add(event)
                    self.assertIn("0.8 seconds unavailable", viewer.toast)
                    self.assertFalse(tui.rows_for(viewer, 72))
                async with client.ws_connect("/listen?replay=1") as ws:
                    self.assertEqual((await ws.receive_json(timeout=1))["count"], 1)
                    self.assertEqual((await ws.receive_json(timeout=1))["event_id"], event["event_id"])

    async def test_caption_auth_precedes_upgrade_and_does_not_trust_proxy_headers(self):
        with mock.patch.object(bridge, "state", live_state()), \
             mock.patch.object(bridge, "CAPTION_TOKEN", "caption-secret"):
            async with TestClient(TestServer(bridge.create_app(start_workers=False))) as client:
                for headers in ({}, {"Authorization": "Bearer wrong"},
                                {"CF-Connecting-IP": "127.0.0.1"}, {"X-Omi-Token": "é"}):
                    response = await client.get("/listen", headers=headers)
                    self.assertEqual(response.status, 401)
                self.assertFalse(bridge.state.caption_listeners)
                async with client.ws_connect("/listen?replay=0", headers={"Authorization": "Bearer caption-secret"}) as ws:
                    self.assertEqual((await ws.receive_json(timeout=1))["phase"], "start")

    def test_caption_token_defaults_to_stream_token_with_explicit_opt_out(self):
        with mock.patch.object(bridge.os.path, "isfile", return_value=True), \
             mock.patch.object(bridge.os, "access", return_value=True), \
             mock.patch.object(bridge, "STREAM_TOKEN"), mock.patch.object(bridge, "CAPTION_TOKEN"), \
             mock.patch.object(bridge, "WHISPER_ENV"):
            bridge.configure({"OMI_STREAM_TOKEN": "phone-secret", "OMI_LANGUAGE": "auto",
                              "OMI_LANGUAGES": "en,ar", "OMI_HOTWORDS": "", "UNRELATED_SECRET": "not-forwarded"})
            self.assertEqual(bridge.CAPTION_TOKEN, "phone-secret")
            self.assertEqual(bridge.WHISPER_ENV, {"OMI_LANGUAGE": "auto", "OMI_LANGUAGES": "en,ar", "OMI_HOTWORDS": ""})
            bridge.configure({"OMI_STREAM_TOKEN": "phone-secret", "OMI_CAPTION_TOKEN": "viewer-secret"})
            self.assertEqual(bridge.CAPTION_TOKEN, "viewer-secret")
            bridge.configure({"OMI_STREAM_TOKEN": "phone-secret", "OMI_CAPTION_TOKEN": ""})
            self.assertEqual(bridge.CAPTION_TOKEN, "")

    def test_invalid_language_allowlist_fails_before_worker_startup(self):
        with mock.patch.object(bridge.os.path, "isfile", return_value=True), \
             mock.patch.object(bridge.os, "access", return_value=True):
            for env in ({"OMI_LANGUAGE": "ar"}, {"OMI_LANGUAGES": ""},
                        {"OMI_LANGUAGE": "auto", "OMI_LANGUAGES": " , "}):
                with self.subTest(env=env), self.assertRaisesRegex(ValueError, "allowlist"):
                    bridge.configure(env)
            bridge.configure({"OMI_LANGUAGE": " AR ", "OMI_LANGUAGES": " en, Ar "})
            bridge.configure({})
