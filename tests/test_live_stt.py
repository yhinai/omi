"""Live STT transport, partial/final separation, receipt timing and bounded-loss checks."""
import asyncio
from collections import deque
import json
from types import SimpleNamespace
import unittest
from unittest import mock

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
import live_stt as live
import omi_bridge as bridge

SPEECH = b"\x00\x10" * 1600


class LiveTests(unittest.IsolatedAsyncioTestCase):
    def make(self, **kwargs):
        options = dict(session=mock.Mock(), api_key="test-key", model="gemini-3.5-transcribe-live",
            on_partial=mock.Mock(), on_final=mock.Mock(return_value=True), on_gap=mock.Mock(),
            on_latency=mock.Mock(), on_connection=mock.Mock())
        options.update(kwargs)
        return live.GeminiLiveSTT(**options)

    def seed(self, stream):
        stream.spans.append(live.Span(0, 16000, 1000, 100, 999, 0, 1, True))
        stream.sent_samples = 16000
        stream.handle_message({"voiceActivity": {"type": "ACTIVITY_START", "audioOffset": "0.1s"}})

    def test_binary_json_from_gemini_is_supported(self):
        message = SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=b'{"setupComplete":{}}')
        self.assertEqual(live.decode_message(message), {"setupComplete": {}})
        with self.assertRaises(ConnectionError):
            live.decode_message(SimpleNamespace(type=aiohttp.WSMsgType.CLOSE))

    def test_partials_replace_hypotheses_and_first_partial_is_timed_only_once(self):
        stream = self.make()
        self.seed(stream)
        with mock.patch.object(live.time, "monotonic", return_value=101):
            stream.handle_message({"serverContent": {"interimInputTranscription": {"text": "The bridge"}}})
        with mock.patch.object(live.time, "monotonic", return_value=102):
            stream.handle_message({"serverContent": {"interimInputTranscription": {"text": "The bridge works."}}})
        self.assertEqual(stream.on_partial.call_count, 2)
        first, second = [call.args[0] for call in stream.on_partial.call_args_list]
        self.assertEqual(first["utterance_id"], second["utterance_id"])
        self.assertEqual((first["revision"], second["revision"]), (1, 2))
        stream.on_final.assert_not_called()
        stream.on_latency.assert_called_once_with("receipt_to_first_partial_sec", 1)

    def test_vad_end_before_final_preserves_receipt_timing_and_commits_once(self):
        stream = self.make()
        self.seed(stream)
        stream.handle_message({"voiceActivity": {"type": "ACTIVITY_END", "audioOffset": "0.9s"}})
        with mock.patch.object(live.time, "monotonic", return_value=103):
            stream.handle_message({"serverContent": {"inputTranscription": {"text": "Final words."}}})
        stream.on_final.assert_called_once()
        self.assertEqual(stream.on_final.call_args.kwargs["received_at"], 1000)
        stream.on_latency.assert_any_call("receipt_to_final_sec", 3)
        stream.on_latency.assert_any_call("receipt_to_transcript_sec", 3)
        self.assertIsNone(stream.turn)

    def test_incomplete_partial_is_discarded_with_gap_instead_of_saved_as_final(self):
        stream = self.make()
        self.seed(stream)
        stream.handle_message({"serverContent": {"interimInputTranscription": {"text": "Unfinished words"}}})
        stream.handle_message({"voiceActivity": {"type": "ACTIVITY_END", "audioOffset": "0.9s"}})
        stream.handle_message({"serverContent": {"generationComplete": True}})
        stream.on_final.assert_not_called()
        self.assertEqual(stream.on_gap.call_args.args[0], "failed")
        self.assertEqual(stream.on_partial.call_args.args[0]["text"], "")
        self.assertFalse(stream.speech_pending)

    def test_same_words_in_distinct_utterances_are_not_deduplicated(self):
        stream = self.make()
        for _ in range(2):
            self.seed(stream)
            stream.handle_message({"serverContent": {"inputTranscription": {"text": "Thank you."}}})
            stream.handle_message({"voiceActivity": {"type": "ACTIVITY_END", "audioOffset": "1s"}})
        self.assertEqual(stream.on_final.call_count, 2)
        ids = [call.kwargs["utterance_id"] for call in stream.on_final.call_args_list]
        self.assertNotEqual(*ids)

    def test_queue_caps_audio_and_preserves_the_fresh_suffix(self):
        stream = self.make()
        raw = SPEECH * 120  # Twelve seconds.
        stream.feed(raw, 1000, 100)
        self.assertEqual(stream.pending_bytes, 320000)
        self.assertEqual(stream.queue[0].sample_offset, 32000)
        self.assertEqual(stream.queue[0].pcm, raw[-320000:])
        self.assertEqual(stream.on_gap.call_args.args[:2], ("queue_evicted", 2))

    async def test_audio_is_forwarded_in_small_frames_without_waiting_for_a_window(self):
        stream = self.make()
        stream.feed(SPEECH * 3, 1000, 100)
        stream.stopping = True
        ws = SimpleNamespace(send_json=mock.AsyncMock())
        await stream._send(ws)
        messages = [call.args[0]["realtimeInput"] for call in ws.send_json.call_args_list]
        self.assertEqual(len([item for item in messages if "audio" in item]), 3)
        self.assertTrue(messages[-1]["audioStreamEnd"])
        self.assertEqual(stream.sent_samples, 4800)

    async def test_send_failure_retries_only_unsent_suffix_and_reports_uncertain_audio(self):
        stream = self.make()
        stream.feed(SPEECH * 3, 1000, 100)
        stream.stopping = True
        ws = SimpleNamespace(send_json=mock.AsyncMock(side_effect=[None, ConnectionError()]))
        with self.assertRaises(ConnectionError):
            await stream._send(ws)
        self.assertEqual(stream.pending_bytes, len(SPEECH))
        self.assertEqual(stream.queue[0].sample_offset, 3200)
        stream._connection_gap()
        self.assertAlmostEqual(stream.on_gap.call_args.args[1], .2)
        self.assertEqual(stream.pending_bytes, len(SPEECH))

    async def test_stale_unsent_packets_are_skipped_before_network_send(self):
        stream = self.make()
        stream.feed(SPEECH, 1000, 1)
        ws = SimpleNamespace(send_json=mock.AsyncMock())
        task = asyncio.create_task(stream._send(ws))
        try:
            await asyncio.sleep(.01)
            self.assertEqual(stream.on_gap.call_args.args[:2], ("stale", .1))
            self.assertTrue(all("audio" not in call.args[0]["realtimeInput"] for call in ws.send_json.call_args_list))
        finally:
            stream.stopping = True
            stream.wake.set()
            await asyncio.wait_for(task, 1)

    async def test_silence_does_not_use_up_the_continuous_speech_finalization_budget(self):
        stream = self.make()
        stream.feed(bytes(len(SPEECH)*4) + SPEECH, 1000, 100)
        stream.stopping = True
        ws = SimpleNamespace(send_json=mock.AsyncMock())
        # Shorten only the byte budget to exercise a long silence without a slow test.
        with mock.patch.object(live, "BPS", 320):
            await stream._send(ws)
        ends = [call for call in ws.send_json.call_args_list if call.args[0]["realtimeInput"].get("audioStreamEnd")]
        self.assertEqual(len(ends), 1)  # Only shutdown, never a cut at the first voiced frame.

    def test_vad_boundary_inside_a_frame_does_not_leave_finalized_audio_pending(self):
        stream = self.make()
        stream.spans.append(live.Span(0, 1600, 1000, 100, 999.9, 0, 1, True))
        stream.sent_samples = 1600
        stream.handle_message({"voiceActivity": {"type": "ACTIVITY_START", "audioOffset": "0s"}})
        stream.handle_message({"serverContent": {"inputTranscription": {"text": "Words"}}})
        stream.handle_message({"voiceActivity": {"type": "ACTIVITY_END", "audioOffset": "0.09s"}})
        self.assertIsNone(stream.pending_age)
        self.assertFalse(stream.speech_pending)

    async def test_connection_loss_reconnects_with_fresh_audio_and_records_unfinalized_gap(self):
        ready, first, final, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
        connections = []
        async def socket(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            number = len(connections)
            connections.append(ws)
            async for message in ws:
                data = json.loads(message.data)
                if "setup" in data:
                    await ws.send_json({"setupComplete": {}})
                elif "audio" in data.get("realtimeInput", {}):
                    await ws.send_json({"voiceActivity": {"type": "ACTIVITY_START", "audioOffset": "0s"}})
                    if number == 0:
                        await ws.send_json({"serverContent": {"interimInputTranscription": {"text": "Unfinished"}}})
                        await ws.close(code=1011)
                        break
                    await ws.send_json({"serverContent": {"inputTranscription": {"text": "Fresh final"}}})
                    await ws.send_json({"voiceActivity": {"type": "ACTIVITY_END", "audioOffset": "0.1s"}})
                elif data.get("realtimeInput", {}).get("audioStreamEnd"):
                    await ws.send_json({"serverContent": {"generationComplete": True}})
            return ws
        app = web.Application()
        app.router.add_get("/live", socket)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        async with aiohttp.ClientSession() as session:
            stream = self.make(session=session,
                on_connection=lambda connected,error: ready.set() if connected else disconnected.set(),
                on_partial=lambda event: first.set() if event.get("text") else None,
                on_final=lambda text,**timing: final.set())
            with mock.patch.object(live, "LIVE_URL", str(server.make_url("/live"))):
                task = asyncio.create_task(stream.run())
                try:
                    await asyncio.wait_for(ready.wait(), 1)
                    stream.feed(SPEECH, bridge.time.time(), bridge.time.monotonic())
                    await asyncio.wait_for(first.wait(), 1)
                    await asyncio.wait_for(disconnected.wait(), 1)
                    stream.feed(SPEECH, bridge.time.time(), bridge.time.monotonic())
                    await asyncio.wait_for(final.wait(), 4)
                    self.assertEqual(len(connections), 2)
                    self.assertGreaterEqual(stream.reconnects, 1)
                    self.assertEqual(stream.on_gap.call_args.args[0], "failed")
                    self.assertAlmostEqual(stream.on_gap.call_args.args[1], .1)
                finally:
                    await stream.close()
                    await task

    async def test_real_local_websocket_setup_partials_and_shutdown_finalization(self):
        ready, partial = asyncio.Event(), asyncio.Event()
        async def socket(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            samples = 0
            async for message in ws:
                data = json.loads(message.data)
                if "setup" in data:
                    self.assertEqual(data["setup"]["generationConfig"]["responseModalities"], ["TEXT"])
                    await ws.send_bytes(b'{"setupComplete":{}}')
                elif "audio" in data.get("realtimeInput", {}):
                    samples += 1600
                    await ws.send_json({"voiceActivity": {"type": "ACTIVITY_START", "audioOffset": "0s"}})
                    await ws.send_json({"serverContent": {"interimInputTranscription": {"text": "Live"}}})
                elif data.get("realtimeInput", {}).get("audioStreamEnd"):
                    await ws.send_json({"serverContent": {"inputTranscription": {"text": "Live final."}}})
                    await ws.send_json({"voiceActivity": {"type": "ACTIVITY_END", "audioOffset": str(samples/16000)+"s"}})
                    await ws.send_json({"serverContent": {"generationComplete": True}})
            return ws
        app = web.Application()
        app.router.add_get("/live", socket)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        async with aiohttp.ClientSession() as session:
            stream = self.make(session=session,
                on_connection=lambda connected,error: ready.set() if connected else None,
                on_partial=lambda event: partial.set() if event.get("text") else None)
            with mock.patch.object(live, "LIVE_URL", str(server.make_url("/live"))):
                task = asyncio.create_task(stream.run())
                try:
                    await asyncio.wait_for(ready.wait(), 1)
                    stream.feed(SPEECH, bridge.time.time(), bridge.time.monotonic())
                    await asyncio.wait_for(partial.wait(), 1)
                    await stream.close()
                    await task
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
            stream.on_final.assert_called_once()
            self.assertEqual(stream.on_final.call_args.args[0], "Live final.")
            self.assertFalse(stream.connected)
            self.assertFalse(stream.queue)
            stream.on_gap.assert_not_called()


class BridgeLiveTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = bridge.BridgeState(stt_queue=asyncio.Queue(maxsize=24))
        patcher = mock.patch.object(bridge, "state", self.state)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_partial_updates_are_coalesced_and_never_written_or_replayed(self):
        listener = bridge.CaptionListener()
        self.state.caption_listeners["test"] = listener
        listener.queue.put_nowait({"type": "transcript", "text": "Already final"})
        with mock.patch.object(bridge, "record_transcript_line") as record, \
             mock.patch.object(bridge, "schedule_webhook") as webhook, \
             mock.patch.object(bridge, "event_logger") as events:
            for revision in range(100):
                bridge.publish_partial({"type": "partial", "utterance_id": "u1", "revision": revision, "text": str(revision)})
        self.assertEqual(listener.partial["text"], "99")
        self.assertEqual(listener.queue.qsize(), 1)
        self.assertFalse(self.state.recent_events)
        self.assertFalse(self.state.rolling_research_buffer)
        record.assert_not_called()
        webhook.assert_not_called()
        events.info.assert_not_called()

    def test_old_partial_clear_does_not_erase_a_new_utterance(self):
        bridge.publish_partial({"type": "partial", "utterance_id": "new", "text": "Current words"})
        bridge.publish_partial({"type": "partial", "utterance_id": "old", "text": ""})
        self.assertEqual(self.state.partial_caption["utterance_id"], "new")

    def test_live_ingress_retains_ring_audio_and_bypasses_batch_buffer(self):
        stream = mock.Mock()
        self.state.live_stt = stream
        bridge.ingest_raw_pcm(SPEECH)
        stream.feed.assert_called_once()
        self.assertEqual(self.state.audio_ring[-1][2], SPEECH)
        self.assertFalse(self.state.audio_sessions)
        self.assertTrue(self.state.stt_queue.empty())

    def test_final_events_are_marked_and_persisted_once(self):
        with mock.patch.object(bridge, "write_event", return_value=True) as output:
            bridge.record_live_final("Final speech", utterance_id="u1", session_id=1)
        event = output.call_args.args[0]
        self.assertEqual((event["type"],event["is_final"],event["utterance_id"]), ("transcript",True,"u1"))
        self.assertEqual(self.state.metrics["transcripts_emitted"],1)
        self.assertEqual(self.state.metrics["windows_processed"],1)

    async def test_live_startup_uses_no_batch_worker_and_stops_before_http_session_closes(self):
        from live_stt import GeminiLiveSTT
        started = asyncio.Event()
        stopped = asyncio.Event()
        async def run(stream):
            stream.task = asyncio.current_task()
            stream.connected = True
            started.set()
            await stopped.wait()
        async def close(stream):
            self.assertFalse(self.state.http_session.closed)
            stopped.set()
        app = {}
        with mock.patch.object(bridge, "GEMINI_API_KEY", "test"), \
             mock.patch.object(bridge, "STT_MODE", "live"), \
             mock.patch.object(bridge, "seed_recent_events"), \
             mock.patch.object(GeminiLiveSTT, "run", run), \
             mock.patch.object(GeminiLiveSTT, "close", close):
            await bridge.on_startup(app)
            try:
                await asyncio.wait_for(started.wait(), 1)
                self.assertNotIn("slicer_task", app)
                self.assertNotIn("stt_worker_task", app)
                self.assertEqual(bridge.build_health_dict()["stt_mode"], "live")
                self.assertEqual(bridge.build_health_dict()["core_window_sec"], 0)
                self.assertTrue(bridge.build_health_dict()["stream_connected"])
            finally:
                await bridge.on_cleanup(app)
        self.assertIsNone(self.state.http_session)
