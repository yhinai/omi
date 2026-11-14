"""Offline Deepgram protocol, timing, bounded replay and persistence regressions."""
import asyncio
import json
import math
from types import SimpleNamespace
import time
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit

import aiohttp

import deepgram_stt as deepgram
from common import RATE, BPS, QUEUE_MAX_BYTES, QUEUE_MAX_PACKETS


class Socket:
    def __init__(self, auto_metadata=True):
        self.incoming = asyncio.Queue()
        self.audio, self.controls, self.send_times = [], [], []
        self.auto_metadata = auto_metadata
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def event(self, event):
        self.incoming.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(event)))

    def disconnect(self):
        self.incoming.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))

    async def send_bytes(self, raw):
        self.audio.append(raw)
        self.send_times.append(time.monotonic())

    async def send_json(self, event):
        self.controls.append(event)
        if event["type"] == "CloseStream" and self.auto_metadata:
            self.event({"type": "Metadata", "duration": sum(map(len, self.audio))/BPS})

    async def receive(self):
        return await self.incoming.get()


class Session:
    def __init__(self, sockets=()):
        self.sockets = list(sockets)
        self.calls = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def ws_connect(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.sockets:
            self.sockets.append(Socket())
        return self.sockets.pop(0)


def result(text="Hello.", start=0, duration=.1, final=True, speech_final=False, words=None):
    alternative = {"transcript": text}
    if words is not None:
        alternative["words"] = words
    return {"type": "Results", "start": start, "duration": duration,
            "is_final": final, "speech_final": speech_final,
            "channel": {"alternatives": [alternative]},
            "metadata": {"request_id": "synthetic-request", "model_info": {
                "name": "general-nova-3", "version": "synthetic-model-version"}}}


async def until(predicate, timeout=2):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.001)


class DeepgramTests(unittest.IsolatedAsyncioTestCase):
    def make(self, sockets=(), **settings):
        self.session = Session(sockets)
        self.partial, self.final, self.gap = mock.Mock(), mock.Mock(return_value=True), mock.Mock()
        self.latency, self.connection = mock.Mock(), mock.Mock()
        self.stream = deepgram.DeepgramSTT(api_key="synthetic-private-key", on_partial=self.partial,
            on_final=self.final, on_gap=self.gap, on_latency=self.latency,
            on_connection=self.connection, session_factory=lambda **kwargs: self.session, **settings)
        self.task = asyncio.create_task(self.stream.run())
        self.addAsyncCleanup(self.stop)
        return self.stream

    async def stop(self):
        if not self.task.done():
            await self.stream.close()
        await asyncio.gather(self.task, return_exceptions=True)

    def feed(self, raw=b"\x00\x10"*1600, received_at=None, received_mono=None):
        self.stream.feed(raw, time.time() if received_at is None else received_at,
                         time.monotonic() if received_mono is None else received_mono)

    async def test_lazy_connection_fixed_options_and_private_header(self):
        stream = self.make(diarize=True)
        await asyncio.sleep(0)
        self.assertTrue(stream.ready)
        self.assertFalse(stream.connected)
        self.assertEqual(stream.connection_state, "idle")
        self.assertFalse(self.session.calls)
        self.assertTrue(await stream.flush_now())
        self.assertFalse(self.session.calls)  # Empty drains need no provider acknowledgment.
        self.feed()
        await until(lambda: bool(self.session.calls))
        url, settings = self.session.calls[0]
        self.assertNotIn("synthetic-private-key", url)
        self.assertEqual(settings["headers"], {"Authorization": "Token synthetic-private-key"})
        options = {key: values[0] for key, values in parse_qs(urlsplit(url).query).items()}
        self.assertEqual(options, {"model": "nova-3", "language": "en", "encoding": "linear16",
            "sample_rate": "16000", "channels": "1", "interim_results": "true", "endpointing": "300",
            "punctuate": "true", "smart_format": "true", "no_delay": "true", "diarize": "true",
            "vad_events": "true", "mip_opt_out": "true", "redact_usage": "true"})

    async def test_interims_replace_then_is_final_commits_without_speech_final(self):
        socket = Socket()
        stream = self.make([socket])
        self.feed()
        await until(lambda: bool(socket.audio))
        socket.event(result("Hel", final=False))
        socket.event(result("Hello", final=False))
        socket.event(result(words=[{"word": "hello", "punctuated_word": "Hello.",
            "start": 0, "end": .1, "confidence": .98, "speaker": 1}]))
        await until(lambda: self.final.call_count == 1)
        self.assertTrue(stream.connected)
        self.assertEqual(stream.connection_state, "connected")
        self.assertFalse(any(event["type"] == "CloseStream" for event in socket.controls))
        first, second = [call.args[0] for call in self.partial.call_args_list if call.args[0]["text"]]
        self.assertEqual(first["utterance_id"], second["utterance_id"])
        self.assertEqual([first["revision"], second["revision"]], [1, 2])
        fields = self.final.call_args.kwargs
        self.assertEqual((fields["sample_offset"], fields["sample_end_offset"]), (0, 1600))
        self.assertEqual(fields["recognition"]["resolved_model"], "general-nova-3")
        self.assertEqual(fields["recognition"]["words"][0]["speaker"], 1)
        self.assertFalse(stream.retained)
        # Duplicate immutable responses are source-deduplicated.
        socket.event(result())
        await asyncio.sleep(.01)
        self.assertEqual(self.final.call_count, 1)

    async def test_overlapping_final_word_timings_remove_only_committed_prefix(self):
        socket = Socket()
        stream = self.make([socket])
        self.feed()
        await until(lambda: bool(socket.audio))
        socket.event(result("Hello.", duration=.05))
        await until(lambda: self.final.call_count == 1)
        socket.event(result("Hello world.", words=[
            {"word": "hello", "start": 0, "end": .05},
            {"word": "world", "punctuated_word": "World.", "start": .05, "end": .1}]))
        await until(lambda: self.final.call_count == 2)
        self.assertEqual(self.final.call_args.args[0], "World.")
        self.assertEqual(self.final.call_args.kwargs["sample_offset"], 800)
        self.assertEqual(len(self.final.call_args.kwargs["recognition"]["words"]), 1)

    async def test_repeated_speech_final_closes_only_its_own_speech_state(self):
        socket = Socket()
        stream = self.make([socket])
        self.feed()
        await until(lambda: bool(socket.audio))
        socket.event(result("Hello", final=False))
        socket.event(result("Hello."))
        await until(lambda: self.final.call_count == 1)
        self.assertTrue(stream.speech_pending)
        socket.event(result("Hello.", speech_final=True))
        await until(lambda: not stream.speech_pending)
        self.feed()
        await until(lambda: len(socket.audio) == 2)
        socket.event(result("Next", start=.1, duration=.1, final=False))
        await until(lambda: stream.last_partial is not None)
        socket.event(result("Hello.", speech_final=True))
        await asyncio.sleep(.01)
        self.assertTrue(stream.speech_pending)
        self.assertEqual(stream.last_partial["text"], "Next")
        self.assertEqual(self.final.call_count, 1)

    async def test_new_speech_started_is_protected_before_its_first_interim(self):
        socket = Socket()
        stream = self.make([socket])
        self.feed()
        await until(lambda: bool(socket.audio))
        socket.event(result("Previous.", speech_final=True))
        await until(lambda: self.final.call_count == 1)
        self.feed()
        await until(lambda: len(socket.audio) == 2)
        socket.event({"type": "SpeechStarted", "timestamp": .15})
        await until(lambda: stream.speech_pending)
        self.assertIsNone(stream.last_partial)
        socket.event(result("Previous.", speech_final=True))
        await asyncio.sleep(.01)
        self.assertTrue(stream.speech_pending)
        socket.event(result("Next.", start=.1, duration=.1, speech_final=True))
        await until(lambda: self.final.call_count == 2)
        self.assertFalse(stream.speech_pending)

    async def test_arrival_connection_and_persistence_latencies_are_distinct(self):
        clock = mock.Mock(wraps=time)
        clock.monotonic.return_value = 100
        class TimedSocket(Socket):
            async def __aenter__(self):
                clock.monotonic.return_value = 100.25
                return self
        socket = TimedSocket()
        with mock.patch.object(deepgram, "time", clock):
            stream = self.make([socket])
            def slow_failed_save(text, **fields):
                self.assertIn(mock.call("receipt_to_final_sec", 5.5), self.latency.call_args_list)
                clock.monotonic.return_value = 101
                return False
            self.final.side_effect = slow_failed_save
            self.feed(received_mono=95)
            await until(lambda: bool(socket.audio))
            clock.monotonic.return_value = 100.5
            socket.event(result())
            await until(lambda: bool(stream.pending_finals))
            self.assertFalse(any(call.args[0] == "final_to_persist_sec" for call in self.latency.call_args_list))
            self.final.side_effect = None
            clock.monotonic.return_value = 103
            stream.retry_saves()
            metrics = {call.args[0]: call.args[1] for call in self.latency.call_args_list}
            self.assertEqual(metrics["provider_connect_sec"], .25)
            self.assertEqual(metrics["receipt_to_final_sec"], 5.5)
            self.assertEqual(metrics["receipt_to_transcript_sec"], 8)
            self.assertEqual(metrics["final_to_persist_sec"], 2.5)
            self.assertNotIn("_final_received_mono", self.final.call_args.kwargs)
            self.assertTrue(all(math.isfinite(value) and value >= 0 for value in metrics.values()))

    async def test_nonfinite_or_negative_latency_observations_cannot_pollute_health(self):
        stream = self.make()
        stream._latency("invalid", float("inf"))
        stream._latency("invalid", float("nan"))
        stream._latency("clock_skew", -1)
        self.latency.assert_called_once_with("clock_skew", 0)

    async def test_word_timing_slop_cannot_create_negative_ranges_or_lose_transcript(self):
        socket = Socket()
        stream = self.make([socket])
        self.feed()
        await until(lambda: bool(socket.audio))
        socket.event(result("Hello there.", words=[
            {"word": "hello", "start": 0, "end": .105},
            {"word": "there", "start": .101, "end": .109}]))
        await until(lambda: self.final.called)
        self.assertEqual(self.final.call_args.args[0], "Hello there.")
        words = self.final.call_args.kwargs["recognition"]["words"]
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0]["sample_end_offset"], 1600)
        self.assertTrue(all(word["sample_offset"] < word["sample_end_offset"] for word in words))
        self.assertFalse(any(call.args[0].startswith("provider_word_end_to_") for call in self.latency.call_args_list))

    def test_word_end_latency_preserves_original_packet_clock_across_splits_and_epoch_offsets(self):
        for offset in (0, RATE*7):
            for committed_prefix in (0, 1600):
                with self.subTest(offset=offset, committed_prefix=committed_prefix):
                    latency, final = mock.Mock(), mock.Mock(return_value=True)
                    stream = deepgram.DeepgramSTT(api_key="synthetic", on_partial=mock.Mock(),
                        on_final=final, on_gap=mock.Mock(), on_latency=latency, on_connection=mock.Mock())
                    stream.total_input_samples = offset
                    stream.feed(bytes(6400), received_at=200, received_mono=100)
                    original = stream.queue.popleft()
                    stream.pending_bytes = 0
                    stream.retained.extend((original.slice(offset, offset+1600), original.slice(offset+1600)))
                    if committed_prefix:
                        stream._release(offset+committed_prefix)
                    epoch = deepgram._Epoch(start=offset, end=offset+3200, session=stream.session_id)
                    payload = result("Word.", start=.1, duration=.1,
                        words=[{"word": "word", "start": .15, "end": .18}])
                    clock = mock.Mock(wraps=time)
                    clock.monotonic.return_value = 100.5
                    with mock.patch.object(deepgram, "time", clock):
                        stream._handle_result({**payload, "is_final": False}, epoch)
                        stream._handle_result(payload, epoch)
                        # A duplicate final must not add another word-lag sample.
                        stream._handle_result(payload, epoch)
                    metrics = {call.args[0]: call.args[1] for call in latency.call_args_list}
                    self.assertAlmostEqual(metrics["provider_word_end_to_partial_sec"], .52)
                    self.assertAlmostEqual(metrics["provider_word_end_to_final_sec"], .52)
                    self.assertAlmostEqual(metrics["final_word_packet_to_result_sec"], .5)
                    self.assertEqual(sum(call.args[0] == "provider_word_end_to_final_sec"
                                         for call in latency.call_args_list), 1)
                    self.assertEqual(final.call_args.kwargs["sample_offset"], offset+1600)
                    self.assertFalse(stream.retained)

    def test_results_without_complete_word_timing_do_not_invent_word_lag_metrics(self):
        for text, words in (("Word.", None), ("", []),
                            ("Word.", [{"word": "word", "start": 0, "end": .05}]*513)):
            with self.subTest(text=bool(text), word_count=None if words is None else len(words)):
                latency = mock.Mock()
                stream = deepgram.DeepgramSTT(api_key="synthetic", on_partial=mock.Mock(),
                    on_final=mock.Mock(return_value=True), on_gap=mock.Mock(),
                    on_latency=latency, on_connection=mock.Mock())
                stream.feed(bytes(3200), time.time(), time.monotonic())
                stream.retained.append(stream.queue.popleft())
                stream.pending_bytes = 0
                epoch = deepgram._Epoch(start=0, end=1600, session=stream.session_id)
                stream._handle_result(result(text, words=words), epoch)
                self.assertFalse(any(call.args[0].startswith("provider_word_end_to_")
                                     or call.args[0] == "final_word_packet_to_result_sec"
                                     for call in latency.call_args_list))

    async def test_failed_output_callback_cannot_mutate_retry_provenance(self):
        socket = Socket()
        stream = self.make([socket])
        def failed(text, **fields):
            fields["recognition"]["requested_model"] = "mutated by callback"
            return False
        self.final.side_effect = failed
        self.feed()
        await until(lambda: bool(socket.audio))
        socket.event(result())
        await until(lambda: bool(stream.pending_finals))
        self.assertEqual(stream.pending_finals[0]["recognition"]["requested_model"], "nova-3")
        self.final.side_effect = None
        stream.retry_saves()
        self.assertEqual(self.final.call_args.kwargs["recognition"]["requested_model"], "nova-3")

    async def test_empty_final_and_metadata_are_real_flush_barriers(self):
        socket = Socket(auto_metadata=False)
        stream = self.make([socket])
        self.feed()
        await until(lambda: bool(socket.audio))
        socket.event(result("", final=True))
        await until(lambda: stream.committed_end == 1600)
        drain = asyncio.create_task(stream.flush_now(timeout=1))
        await until(lambda: bool(socket.controls))
        self.assertFalse(drain.done())
        socket.event({"type": "Metadata", "duration": .1})
        self.assertTrue(await drain)
        self.assertFalse(stream.pending_bytes)
        self.assertFalse(self.final.called)
        self.assertFalse(self.gap.called)

    async def test_expected_rotation_keeps_source_sessions_separate(self):
        first, second = Socket(), Socket()
        stream = self.make([first, second])
        stream.new_session()  # Initial empty boundary must not open an empty socket.
        self.feed()
        await until(lambda: bool(first.audio))
        old_session = stream.session_id
        stream.new_session()
        self.feed()
        await until(lambda: bool(second.audio))
        self.assertEqual(len(first.audio), 1)
        self.assertTrue(first.closed)
        second.event(result("New session."))
        await until(lambda: self.final.called)
        self.assertEqual(self.final.call_args.kwargs["sample_offset"], 1600)
        self.assertGreater(self.final.call_args.kwargs["session_id"], old_session)
        self.assertEqual(stream.reconnects, 0)
        self.assertFalse(any(call.args[0] is False for call in self.connection.call_args_list))

    async def test_connection_loss_replays_only_unresolved_audio_with_original_time(self):
        first, second = Socket(), Socket()
        with mock.patch.object(deepgram, "RECONNECT_INITIAL_SECONDS", .001):
            stream = self.make([first, second])
            self.feed(received_at=100)
            await until(lambda: bool(first.audio))
            first.event(result("First.", duration=.05))
            first.disconnect()
            await until(lambda: bool(second.audio))
            self.assertEqual(sum(map(len, second.audio)), 1600)
            second.event(result("Second.", duration=.05))
            await until(lambda: self.final.call_count == 2)
            calls = self.final.call_args_list
            self.assertEqual([call.kwargs["sample_offset"] for call in calls], [0, 800])
            self.assertEqual([call.kwargs["sample_end_offset"] for call in calls], [800, 1600])
            self.assertAlmostEqual(calls[0].kwargs["audio_start_at"], 99.9)
            self.assertAlmostEqual(calls[1].kwargs["audio_start_at"], 99.95)
            self.assertNotEqual(calls[0].kwargs["recognition"]["connection_epoch"],
                                calls[1].kwargs["recognition"]["connection_epoch"])
            self.assertFalse(self.gap.called)

    async def test_save_failure_protects_audio_and_provenance_across_connection_loss(self):
        first = Socket()
        with mock.patch.object(deepgram, "RECONNECT_INITIAL_SECONDS", .001):
            stream = self.make([first])
            self.final.return_value = False
            self.feed()
            await until(lambda: bool(first.audio))
            first.event(result())
            first.disconnect()
            await until(lambda: bool(stream.protected_pcm))
            self.assertFalse(stream.save_ready.is_set())
            self.assertFalse(stream.ready)
            self.assertEqual(len(stream.pending_finals), 1)
            self.assertFalse(stream.queue)
            before = self.final.call_args.kwargs.copy()
            self.final.return_value = True
            stream.retry_saves()
            self.assertEqual(self.final.call_args.kwargs, before)
            self.assertFalse(stream.pending_finals)
            self.assertFalse(stream.protected_pcm)
            self.assertTrue(stream.save_ready.is_set())
            self.assertFalse(stream.ready)  # Saving cannot turn a failed provider healthy.
            self.assertEqual(stream.committed_end, 1600)
            self.assertEqual(len(self.session.calls), 1)

    async def test_pacing_caps_buffered_audio_and_large_packet_source_time_is_preserved(self):
        socket = Socket()
        stream = self.make([socket])
        self.feed(b"\x00\x10"*3200, received_at=100)
        await until(lambda: len(socket.audio) == 2)
        self.assertGreaterEqual(socket.send_times[1]-socket.send_times[0], .075)
        socket.event(result("Second chunk.", start=.1, duration=.1))
        await until(lambda: self.final.called)
        self.assertAlmostEqual(self.final.call_args.kwargs["audio_start_at"], 99.9)
        self.assertEqual(self.final.call_args.kwargs["received_at"], 100)

    async def test_idle_time_cannot_be_banked_to_burst_later_audio(self):
        socket = Socket()
        self.make([socket])
        self.feed()
        await until(lambda: bool(socket.audio))
        await asyncio.sleep(.25)
        self.feed(b"\x00\x10"*4800)
        await until(lambda: len(socket.audio) == 4)
        self.assertGreaterEqual(socket.send_times[2]-socket.send_times[1], .075)
        self.assertGreaterEqual(socket.send_times[3]-socket.send_times[2], .075)

    async def test_unexpected_short_or_missing_metadata_never_releases_unconfirmed_audio(self):
        for metadata, requested_close in (({"type": "Metadata", "duration": .1}, False),
                                           ({"type": "Metadata", "duration": .05}, True),
                                           ({"type": "Metadata"}, True)):
            with self.subTest(metadata=metadata), mock.patch.object(deepgram, "RECONNECT_INITIAL_SECONDS", .001):
                first, second = Socket(auto_metadata=False), Socket(auto_metadata=False)
                stream = self.make([first, second])
                self.feed()
                await until(lambda: bool(first.audio))
                drain = asyncio.create_task(stream.flush_now(timeout=1)) if requested_close else None
                if drain is not None:
                    await until(lambda: bool(first.controls))
                first.event(metadata)
                await until(lambda: bool(second.audio))
                self.assertEqual(stream.committed_end, 0)
                self.assertEqual(sum(map(len, second.audio)), 3200)
                self.assertFalse(self.final.called)
                self.assertFalse(self.gap.called)
                if drain is None:
                    drain = asyncio.create_task(stream.flush_now(timeout=1))
                await until(lambda: any(event["type"] == "CloseStream" for event in second.controls))
                second.event({"type": "Metadata", "duration": .1})
                self.assertTrue(await drain)
                await self.stop()

    async def test_gap_in_source_audio_drains_before_next_timeline(self):
        first, second = Socket(), Socket()
        stream = self.make([first, second], idle_flush=1000)
        self.feed()
        await until(lambda: bool(first.audio))
        self.feed(received_mono=time.monotonic()-46)
        self.feed()
        await until(lambda: bool(second.audio))
        self.assertEqual(sum(map(len, first.audio)), 3200)
        self.assertEqual(sum(map(len, second.audio)), 3200)
        self.assertEqual(self.gap.call_args.args[0], "stale")
        second.event(result("After the gap."))
        await until(lambda: self.final.called)
        self.assertEqual(self.final.call_args.kwargs["sample_offset"], 3200)

    async def test_queue_eviction_and_stale_audio_report_source_gaps(self):
        stream = self.make()
        self.feed(bytes(BPS*12), received_at=100)
        self.assertEqual(stream.pending_bytes, QUEUE_MAX_BYTES)
        self.assertEqual(stream.queue[0].sample_offset, RATE*2)
        self.assertAlmostEqual(stream.queue[0].audio_start_at, 90)
        self.assertEqual(self.gap.call_args.args[:2], ("queue_evicted", 2))
        # A separate stale fixture avoids sending this large synthetic burst.
        stream.queue.clear()
        stream.pending_bytes = 0
        self.feed(received_mono=time.monotonic()-46)
        await until(lambda: self.gap.call_count == 2)
        self.assertEqual(self.gap.call_args.args[0], "stale")
        self.assertFalse(stream.pending_bytes)

    async def test_packet_count_and_flush_control_history_are_bounded(self):
        stream = self.make()
        for _ in range(QUEUE_MAX_PACKETS+10):
            self.feed(bytes(2))
            stream.request_flush()
        self.assertLessEqual(len(stream.queue), QUEUE_MAX_PACKETS)
        self.assertLessEqual(len(stream.controls), QUEUE_MAX_PACKETS)

    async def test_keepalive_is_a_text_control_message(self):
        socket = Socket()
        with mock.patch.object(deepgram, "KEEPALIVE_SECONDS", .01):
            self.make([socket])
            self.feed()
            await until(lambda: {"type": "KeepAlive"} in socket.controls)

    async def test_flush_timeout_does_not_release_unresolved_audio(self):
        socket = Socket(auto_metadata=False)
        stream = self.make([socket])
        self.feed()
        await until(lambda: bool(socket.audio))
        with self.assertLogs("bridge", level="WARNING"):
            self.assertFalse(await stream.flush_now(timeout=.01))
        self.assertEqual(len(stream.retained[0].pcm), 3200)
        self.assertFalse(stream.last_flush_completed)
        socket.event({"type": "Metadata", "duration": .1})
        await until(lambda: not stream.retained)

    async def test_concurrent_flushes_can_share_a_boundary_without_losing_waiters(self):
        socket = Socket(auto_metadata=False)
        stream = self.make([socket])
        self.feed()
        await until(lambda: bool(socket.audio))
        with self.assertLogs("bridge", level="WARNING"):
            short = asyncio.create_task(stream.flush_now(timeout=.01))
            long = asyncio.create_task(stream.flush_now(timeout=1))
            self.assertFalse(await short)
        self.assertFalse(long.done())
        socket.event({"type": "Metadata", "duration": .1})
        self.assertTrue(await long)

    async def test_error_payload_and_exception_text_are_not_logged(self):
        socket = Socket()
        with mock.patch.object(deepgram, "RECONNECT_INITIAL_SECONDS", .001):
            stream = self.make([socket])
            self.feed()
            await until(lambda: bool(socket.audio))
            with self.assertLogs("bridge", level="WARNING") as logs:
                socket.event({"type": "Error", "code": "NET-0001", "description": "synthetic-private-key private spoken words"})
                await until(lambda: stream.reconnects == 1)
            text = " ".join(logs.output)
            self.assertIn("ConnectionError", text)
            self.assertIn("NET-0001", text)
            self.assertNotIn("synthetic-private-key", text)
            self.assertNotIn("private spoken words", text)

    async def test_auth_failure_reports_only_safe_http_status(self):
        class RejectedSocket(Socket):
            async def __aenter__(self):
                raise aiohttp.WSServerHandshakeError(None, (), status=401,
                    message="synthetic-private-key and a private URL")
        stream = self.make([RejectedSocket()])
        with self.assertLogs("bridge", level="WARNING") as logs:
            self.feed()
            await until(lambda: stream.reconnects == 1)
        self.assertIn("http_status=401", " ".join(logs.output))
        self.assertNotIn("synthetic-private-key", " ".join(logs.output))
        self.connection.assert_called_with(False, "WSServerHandshakeError: HTTP 401")
        self.assertEqual(stream.connection_state, "reconnecting")

    async def test_result_failure_logs_only_allowlisted_reason(self):
        for message, expected in (("Deepgram word outside source timeline", "word_outside_timeline"),
                                  ("private speech and synthetic-private-key", "None")):
            with self.subTest(message=expected):
                stream = self.make([Socket()])
                with mock.patch.object(stream, "_epoch", side_effect=ValueError(message)):
                    with self.assertLogs("bridge", level="WARNING") as logs:
                        self.feed()
                        await until(lambda: stream.reconnects == 1)
                output = " ".join(logs.output)
                self.assertIn("reason=" + expected, output)
                self.assertNotIn("private speech", output)
                self.assertNotIn("synthetic-private-key", output)
                await self.stop()

    async def test_successful_recognition_resets_reconnect_wait(self):
        for productive, expected in ((False, [1, 2, 4]), (True, [1, 1, 1])):
            with self.subTest(productive=productive):
                stream = self.make()
                waits = []
                attempts = 0
                original_wait_for = asyncio.wait_for

                async def fail_epoch(session):
                    nonlocal attempts
                    attempts += 1
                    if attempts == 4:
                        stream.stopping = True
                    if productive:
                        stream.partial_updates += 1
                    raise ConnectionError("synthetic interruption")

                async def fast_wait(awaitable, timeout):
                    if (getattr(awaitable, "cr_frame", None) is not None
                            and awaitable.cr_frame.f_locals.get("self") is stream.stop_event):
                        waits.append(timeout)
                        awaitable.close()
                        raise TimeoutError
                    return await original_wait_for(awaitable, timeout)

                with mock.patch.object(stream, "_epoch", side_effect=fail_epoch), \
                        mock.patch.object(asyncio, "wait_for", side_effect=fast_wait):
                    self.feed()
                    with self.assertLogs("bridge", level="WARNING"):
                        await self.task
                self.assertEqual(waits, expected)

    async def test_constructor_rejects_unsupported_model_language_and_invalid_policy(self):
        for settings in ({"language": "ar"}, {"model": "nova-2"}, {"endpointing_ms": 0},
                         {"endpointing_ms": True}, {"diarize": "true"}, {"max_age": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                deepgram.DeepgramSTT(api_key="synthetic", on_partial=mock.Mock(), on_final=mock.Mock(),
                    on_gap=mock.Mock(), on_latency=mock.Mock(), on_connection=mock.Mock(), **settings)


if __name__ == "__main__":
    unittest.main()
