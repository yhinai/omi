"""Local segmentation, nonblocking finals, source ownership and recovery regressions."""
import asyncio
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import whisper
from common import BPS
import unittest
from unittest import mock
from types import SimpleNamespace

from stt import LocalWhisperSTT
from whisper import FRAME, Stream
from common import Packet, WhisperPolicy, RATE, WHISPER_POLICY


class Recognizer:
    def reset_vad(self):
        pass

    def probability(self, pcm):
        return .9 if pcm[:2] == b"\x00\x10" else 0

    def decode(self, pcm):
        return "Recognized words.", .001, ""


def packet(pcm, offset=0, session=1):
    return {"pcm_base64": base64.b64encode(pcm).decode(), "sample_offset": offset,
        "session_id": session, "received_at": time.time(), "received_mono": time.monotonic(),
        "audio_start_at": time.time()-len(pcm)/32000}


def feed(stream, pcm, offset=0, session=1):
    for start in range(0, len(pcm), 3200):
        stream.feed(packet(pcm[start:start+3200], offset+start//2, session))


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.stream = Stream(Recognizer(), output=self.events.append)

    def test_short_speech_finalizes_after_half_second_of_received_silence(self):
        speech = b"\x00\x10" * (FRAME*4)
        feed(self.stream, speech)
        self.assertIsNotNone(self.stream.active)
        feed(self.stream, bytes(FRAME*2*15), len(speech)//2)
        self.assertFalse(self.stream.closed)
        feed(self.stream, bytes(FRAME*2), len(speech)//2+FRAME*15)
        self.assertIsNone(self.stream.active)
        self.assertEqual(len(self.stream.closed), 1)
        self.assertEqual(self.stream.closed[0].pcm, speech+bytes(FRAME*2*16))

    def test_preroll_and_unaligned_tail_preserve_the_source_samples(self):
        opening = bytes(FRAME*2*5)
        speech = b"\x00\x10" * (FRAME*2+17)
        feed(self.stream, opening+speech)
        self.stream.flush("close", (len(opening)+len(speech))//2)
        turn = self.stream.closed[0]
        self.assertEqual(bytes(turn.pcm), opening+speech)
        self.assertEqual((turn.first.sample_offset, turn.end), (0, len(turn.pcm)//2))

    def test_session_change_closes_old_source_before_accepting_new_source(self):
        speech = b"\x00\x10"*(FRAME*4)
        feed(self.stream, speech)
        feed(self.stream, speech, len(speech)//2, session=2)
        self.assertEqual(self.stream.closed[0].first.session_id, 1)
        self.assertEqual(self.stream.active.first.session_id, 2)
        self.assertEqual(self.stream.closed[0].end, self.stream.active.first.sample_offset)

    def test_long_continuous_speech_is_bounded_without_dropping_pcm(self):
        speech = b"\x00\x10" * (int(RATE*WHISPER_POLICY.max_turn_seconds)+FRAME*5)
        feed(self.stream, speech)
        self.stream.flush("end", len(speech)//2)
        turns = list(self.stream.closed)
        self.assertEqual(len(turns), 2)
        self.assertEqual(b"".join(bytes(t.pcm) for t in turns), speech)
        self.assertEqual(turns[0].end, turns[1].first.sample_offset)

    def test_silence_does_not_create_an_utterance(self):
        feed(self.stream, bytes(32000*3))
        self.stream.flush("end", 48000)
        self.assertIsNone(self.stream.active)
        self.assertFalse(self.stream.closed)

    def test_draft_refresh_requires_a_meaningful_amount_of_new_audio(self):
        feed(self.stream, b"\x00\x10"*8000)
        turn = self.stream.active
        self.assertTrue(self.stream.draft_due(turn, 1))
        turn.decoded_samples = len(turn.pcm)//2
        self.stream.last_decode = 1
        feed(self.stream, b"\x00\x10"*512, 8000)
        self.assertFalse(self.stream.draft_due(turn, 2))
        feed(self.stream, b"\x00\x10"*3200, 8512)
        self.assertTrue(self.stream.draft_due(turn, 2))
        self.assertFalse(self.stream.draft_due(turn, 1.1))

    def test_policy_rejects_invalid_context_lengths_and_probabilities(self):
        for settings in ({"max_turn_seconds": 31}, {"update_seconds": float('nan')},
                         {"pause_seconds": 0}, {"continue_probability": .9}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                WhisperPolicy(**settings)

    def test_sparse_arrivals_avoid_repeated_nearly_identical_decodes(self):
        feed(self.stream,b"\x00\x10"*8000)
        turn=self.stream.active
        decodes=1
        turn.decoded_samples=len(turn.pcm)//2
        self.stream.last_decode=1
        for index in range(20):
            feed(self.stream,b"\x00\x10"*FRAME,8000+index*FRAME)
            now=1+(index+1)*.25
            if self.stream.draft_due(turn,now):
                decodes+=1
                turn.decoded_samples=len(turn.pcm)//2
                self.stream.last_decode=now
        # The previous any-new-sample rule schedules 21 decodes in this trace.
        # This is a scheduler-only fixture, not an ASR speed/accuracy benchmark.
        self.assertEqual(decodes,3)

    def test_trailing_silence_skips_drafts_and_resumed_speech_enables_them(self):
        samples = FRAME * 16
        feed(self.stream, b"\x00\x10" * samples)
        turn = self.stream.active
        turn.decoded_samples = len(turn.pcm) // 2
        turn.decoded_voice_end = turn.voice_end
        self.stream.last_decode = 1

        feed(self.stream, bytes(FRAME * 2 * 8), samples)
        self.assertIs(self.stream.active, turn)
        self.assertFalse(self.stream.draft_due(turn, 2))

        feed(self.stream, b"\x00\x10" * (FRAME * 2), samples + FRAME * 8)
        self.assertTrue(self.stream.draft_due(turn, 2))


class DecodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_configuration_overrides_inherited_environment_without_mutating_it(self):
        worker = '''import json,os,sys
with open(sys.argv[1], 'w') as out:
 json.dump({key:os.environ.get(key) for key in ('OMI_LANGUAGE','OMI_LANGUAGES','OMI_HOTWORDS','OMI_TEST_INHERITED')}, out)
print(json.dumps({'type':'ready','load_warm_sec':0}),flush=True)
for line in sys.stdin:
 request=json.loads(line)
 if request['op']=='shutdown':break
'''
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {
                "OMI_LANGUAGE": "en", "OMI_TEST_INHERITED": "kept"}):
            probe = Path(directory) / "environment.json"
            ready = asyncio.Event()
            settings = {"OMI_LANGUAGE": "auto", "OMI_LANGUAGES": "en,ar", "OMI_HOTWORDS": ""}
            stream = LocalWhisperSTT(python=sys.executable, model="model", vad="vad",
                on_partial=mock.Mock(), on_final=mock.Mock(), on_gap=mock.Mock(),
                on_latency=mock.Mock(), on_connection=lambda connected, _: ready.set() if connected else None,
                worker_env=settings)
            settings["OMI_LANGUAGE"] = "changed after construction"
            stream.command = [sys.executable, "-u", "-c", worker, str(probe)]
            task = asyncio.create_task(stream.run())
            try:
                await asyncio.wait_for(ready.wait(), 3)
                self.assertEqual(json.loads(probe.read_text()), {
                    "OMI_LANGUAGE": "auto", "OMI_LANGUAGES": "en,ar",
                    "OMI_HOTWORDS": "", "OMI_TEST_INHERITED": "kept"})
                self.assertEqual(os.environ["OMI_LANGUAGE"], "en")
            finally:
                await stream.close()
                await task

    async def test_first_speech_latency_excludes_retained_silence(self):
        events = []
        stream = None
        def output(event):
            events.append(event)
            if event["type"] == "partial" and event.get("text"):
                stream.flush("done", stream.expected_offset)
                stream.stopping = True
        stream = Stream(Recognizer(), output=output)
        for offset in range(0, RATE * 2, 3200):
            stream.feed({**packet(bytes(3200), offset // 2),
                         "received_mono": 100 + offset / 32000})
        for offset in range(0, 9600, 3200):
            stream.feed({**packet(b"\x00\x10" * 1600, RATE + offset // 2),
                         "received_mono": 101 + offset / 32000})
        clock = mock.Mock(wraps=time)
        clock.monotonic.return_value = 101.5
        with mock.patch("whisper.time", clock):
            await asyncio.wait_for(stream.decode_loop(), 2)
        metrics = {event["name"]: event["seconds"] for event in events if event["type"] == "latency"}
        self.assertAlmostEqual(metrics["speech_start_to_first_partial_sec"], .5)
        self.assertGreater(metrics["receipt_to_first_partial_sec"], 1.3)
        self.assertEqual(len([event for event in events if event["type"] == "final"]), 1)

    async def test_worker_exit_during_save_failure_keeps_recognized_audio_out_of_replay(self):
        worker = '''import sys,json,os
print(json.dumps({'type':'ready','load_warm_sec':0}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r['op']=='packet':
  import base64
  end=r['sample_offset']+len(base64.b64decode(r['pcm_base64']))//2
  print(json.dumps({'type':'final','text':'Recovered.','utterance_id':'u','session_id':r['session_id'],'sample_offset':r['sample_offset'],'sample_end_offset':end,'received_at':r['received_at'],'audio_start_at':r['audio_start_at']}),flush=True)
  print(json.dumps({'type':'covered','sample_end_offset':end}),flush=True)
  os._exit(2)
 if r['op']=='shutdown':break
'''
        writable=False
        saved=[]
        def save(text,**fields):
            if not writable:return False
            saved.append(text)
            return True
        stream=LocalWhisperSTT(python=sys.executable,model='model',vad='vad',on_partial=mock.Mock(),
            on_final=save,on_gap=mock.Mock(),on_latency=mock.Mock(),on_connection=mock.Mock())
        stream.command=[sys.executable,'-u','-c',worker]
        task=asyncio.create_task(stream.run())
        try:
            stream.feed(b'\x00\x10'*1600,time.time(),time.monotonic())
            async with asyncio.timeout(3):
                while not stream.protected_pcm:await asyncio.sleep(.01)
            self.assertFalse(stream.queue)
            self.assertEqual(len(stream.pending_finals),1)
            self.assertEqual(len(stream.protected_pcm[0].pcm),3200)
            writable=True
            await asyncio.wait_for(stream.save_ready.wait(),2)
            self.assertFalse(stream.protected_pcm)
            self.assertEqual(saved,['Recovered.'])
        finally:
            await stream.close()
            await task

    async def test_local_startup_does_not_allocate_a_cloud_http_client(self):
        import bridge
        stopped=asyncio.Event()
        fake=mock.Mock()
        fake.run=stopped.wait
        async def close():
            stopped.set()
        fake.close=close
        runtime=bridge.BridgeState()
        app={}
        with mock.patch.object(bridge,'state',runtime), mock.patch.object(bridge,'STT_BACKEND','whisper'), \
             mock.patch.object(bridge,'seed_recent_events'), mock.patch.object(bridge,'DailyNotes'), \
             mock.patch('stt.LocalWhisperSTT',return_value=fake), \
             mock.patch.object(bridge.aiohttp,'ClientSession',side_effect=AssertionError('Cloud HTTP allocated')):
            await bridge.on_startup(app)
            self.assertFalse(hasattr(runtime, "http_session"))
            await bridge.on_cleanup(app)
            self.assertFalse(runtime.background_tasks)

    async def test_closed_turn_is_decoded_even_when_draft_refresh_is_throttled(self):
        events=[]
        stream=Stream(Recognizer(),output=events.append)
        feed(stream,b"\x00\x10"*8000)
        stream.active.decoded_samples=len(stream.active.pcm)//2
        stream.active.decoded_voice_end=stream.active.voice_end
        stream.last_decode=time.monotonic()+60
        stream.flush('end',8000)
        stream.stopping=True
        await asyncio.wait_for(stream.decode_loop(),2)
        self.assertEqual([e['text'] for e in events if e['type']=='final'],['Recognized words.'])

    async def test_failed_save_is_retried_during_idle_without_losing_source(self):
        final = mock.Mock(side_effect=[False, True])
        stream = LocalWhisperSTT(python=sys.executable, model="model", vad="vad",
            on_partial=mock.Mock(), on_final=final, on_gap=mock.Mock(),
            on_latency=mock.Mock(), on_connection=mock.Mock())
        stream.retained.append(Packet(bytes(16000), time.time(), time.monotonic(), 0, 1))
        event = {"type":"final", "text":"Saved after retry", "utterance_id":"u", "session_id":1,
            "sample_offset":0,"sample_end_offset":8000,"received_at":time.time(),"audio_start_at":time.time()-.5}
        stream.handle_message(event)
        stream.handle_message({"type":"covered","sample_end_offset":8000})
        self.assertFalse(stream.save_ready.is_set())
        self.assertEqual(len(stream.retained),1)
        self.assertEqual(stream.committed_end,0)
        task = asyncio.create_task(stream._save_loop())
        try:
            await asyncio.wait_for(stream.save_ready.wait(), 2)
            self.assertFalse(stream.retained)
            self.assertFalse(stream.pending_finals)
            self.assertEqual(stream.final_count,1)
            self.assertEqual(stream.committed_end,8000)
            stream.handle_message(event)
            self.assertEqual(final.call_count,2)
        finally:
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)

    async def test_next_speech_is_buffered_while_previous_final_is_decoded(self):
        entered, release = threading.Event(), threading.Event()
        class Blocking(Recognizer):
            def decode(self, pcm):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("Test decoder was not released")
                return "Recognized words.", .001, ""
        events = []
        stream = Stream(Blocking(), output=events.append)
        speech = b"\x00\x10"*(FRAME*8)
        feed(stream, speech)
        stream.flush("first", len(speech)//2)
        task = asyncio.create_task(stream.decode_loop())
        try:
            async with asyncio.timeout(2):
                while not entered.is_set():
                    await asyncio.sleep(.005)
            feed(stream, speech, len(speech)//2, session=2)
            self.assertIsNotNone(stream.active)
            self.assertEqual(stream.active.first.session_id, 2)
            self.assertFalse(any(e["type"] == "final" for e in events))
            stream.flush("second", len(speech))
            stream.stopping = True
            release.set()
            await asyncio.wait_for(task, 3)
            finals = [e for e in events if e["type"] == "final"]
            self.assertEqual([e["session_id"] for e in finals], [1, 2])
            self.assertEqual(finals[0]["sample_end_offset"], finals[1]["sample_offset"])
            self.assertEqual([e["request_id"] for e in events if e["type"] == "flushed"], ["first", "second"])
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_worker_crash_replays_unresolved_pcm_and_commits_once(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory)/"crashed"
            worker = '''import sys,json,os
marker=sys.argv[1]
print(json.dumps({'type':'ready','load_warm_sec':0}),flush=True)
end=0;start=None;first=None
for line in sys.stdin:
 r=json.loads(line)
 if r['op']=='packet':
  import base64
  if not os.path.exists(marker):
   open(marker,'w').close();os._exit(2)
  if first is None:first=r;start=r['sample_offset']
  end=r['sample_offset']+len(base64.b64decode(r['pcm_base64']))//2
 elif r['op'] in ('flush','shutdown'):
  if first is not None:
   print(json.dumps({'type':'final','text':'Recovered speech.','utterance_id':'recovered','session_id':first['session_id'],'sample_offset':start,'sample_end_offset':end,'received_at':first['received_at'],'audio_start_at':first['audio_start_at']}),flush=True)
   first=None
  print(json.dumps({'type':'covered','sample_end_offset':end}),flush=True)
  print(json.dumps({'type':'flushed','request_id':r.get('request_id')}),flush=True)
  if r['op']=='shutdown':break
'''
            finals, gaps = [], []
            stream = LocalWhisperSTT(python=sys.executable, model="model", vad="vad",
                on_partial=lambda e: None, on_final=lambda text, **kw: finals.append((text, kw)),
                on_gap=lambda *args, **kw: gaps.append(args), on_latency=lambda *args: None,
                on_connection=lambda *args: None)
            stream.command = [sys.executable, "-u", "-c", worker, str(marker)]
            task = asyncio.create_task(stream.run())
            try:
                stream.feed(b"\x00\x10"*3200, time.time(), time.monotonic())
                async with asyncio.timeout(5):
                    while stream.reconnects < 1 or not stream.connected:
                        await asyncio.sleep(.01)
                await stream.flush_now()
                self.assertEqual(len(finals), 1)
                self.assertEqual((finals[0][1]['sample_offset'], finals[0][1]['sample_end_offset']), (0, 3200))
                self.assertFalse(gaps)
                self.assertFalse(stream.retained)
            finally:
                await stream.close()
                await task


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.final, self.gap = mock.Mock(), mock.Mock()
        self.stream = LocalWhisperSTT(python="python", model="model", vad="vad",
            on_partial=mock.Mock(), on_final=self.final, on_gap=self.gap,
            on_latency=mock.Mock(), on_connection=mock.Mock())

    def test_partials_are_ephemeral_and_duplicate_source_final_is_not_saved_twice(self):
        self.stream.handle_message({"type": "partial", "utterance_id": "a", "text": "Draft"})
        self.final.assert_not_called()
        final = {"type": "final", "utterance_id": "a", "text": "Final", "session_id": 1,
            "sample_offset": 0, "sample_end_offset": 8000, "received_at": time.time(), "audio_start_at": time.time()-.5}
        self.stream.handle_message(final)
        self.stream.handle_message(final)
        self.final.assert_called_once()

    def test_failed_finals_retry_in_order_and_covered_events_cannot_release_them(self):
        self.stream.retained.append(Packet(bytes(32000),time.time(),time.monotonic(),0,1))
        def final(end):
            return {"type":"final","text":str(end),"utterance_id":str(end),"session_id":1,
                "sample_offset":end-8000,"sample_end_offset":end,"received_at":time.time(),"audio_start_at":time.time()}
        self.final.return_value=False
        self.stream.handle_message(final(8000))
        self.stream.handle_message(final(16000))
        self.stream.handle_message({"type":"covered","sample_end_offset":16000})
        self.assertEqual(len(self.stream.retained[0].pcm),32000)
        self.assertEqual(self.stream.final_count,0)
        self.final.return_value=True
        self.stream.retry_saves()
        self.assertEqual([call.args[0] for call in self.final.call_args_list[-2:]],['8000','16000'])
        self.assertFalse(self.stream.retained)
        self.assertEqual(self.stream.final_count,2)

    def test_exception_while_saving_keeps_the_final_retryable(self):
        self.final.side_effect=OSError('disk unavailable')
        self.stream.handle_message({"type":"final","text":"retry","utterance_id":"u","session_id":1,
            "sample_offset":0,"sample_end_offset":8000,"received_at":time.time(),"audio_start_at":time.time()})
        self.assertEqual(self.stream.save_error,'OSError')
        self.assertEqual(len(self.stream.pending_finals),1)
        self.assertEqual(self.stream.committed_end,0)

    def test_unsent_overload_keeps_newest_audio_and_reports_eviction(self):
        self.stream.feed(b"\x00\x10"*(16000*12), time.time(), time.monotonic())
        self.assertEqual(self.stream.pending_bytes, 32000*10)
        self.assertEqual(self.stream.queue[0].sample_offset, 32000)
        self.assertEqual(self.gap.call_args.args[:2], ("queue_evicted", 2))

    def test_source_release_trims_only_covered_prefix(self):
        self.stream.feed(bytes(3200), time.time(), time.monotonic())
        self.stream.retained.append(self.stream.queue.popleft())
        self.stream._release(512)
        self.assertEqual(self.stream.retained[0].sample_offset, 512)
        self.assertEqual(len(self.stream.retained[0].pcm), (1600-512)*2)




class FlushAndDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def make(self):
        return LocalWhisperSTT(python=sys.executable, model='model', vad='vad',
            on_partial=mock.Mock(), on_final=mock.Mock(), on_gap=mock.Mock(),
            on_latency=mock.Mock(), on_connection=mock.Mock())

    async def test_flush_reports_completion_after_worker_ack(self):
        stream = self.make()
        task = asyncio.create_task(stream.flush_now(timeout=1))
        await asyncio.sleep(0)
        identity, _ = stream.controls[0]
        stream.handle_message({'type':'flushed','request_id':identity})
        self.assertTrue(await task)
        self.assertTrue(stream.last_flush_completed)
        self.assertFalse(stream.flush_waiters)

    async def test_flush_reports_save_timeout_without_releasing_source(self):
        stream = self.make()
        stream.retained.append(Packet(bytes(3200),time.time(),time.monotonic(),0,1))
        stream.save_ready.clear()
        with self.assertLogs('bridge', level='WARNING'):
            task = asyncio.create_task(stream.flush_now(timeout=.02))
            await asyncio.sleep(0)
            stream.handle_message({'type':'flushed','request_id':stream.controls[0][0]})
            self.assertFalse(await task)
        self.assertFalse(stream.last_flush_completed)
        self.assertEqual(len(stream.retained[0].pcm),3200)
        self.assertFalse(stream.flush_waiters)

    async def test_shutdown_timeout_is_reported(self):
        stream = self.make()
        stream.task = asyncio.create_task(asyncio.sleep(60))
        with mock.patch('stt.SHUTDOWN_TIMEOUT',.02), self.assertLogs('bridge',level='ERROR'):
            await stream.close()
        self.assertFalse(stream.last_flush_completed)
        self.assertTrue(stream.task.cancelled())

    async def test_stderr_is_bounded_and_logs_only_runtime_metadata(self):
        from common import WORKER_DIAGNOSTIC_BYTES
        stream = self.make()
        reader = asyncio.StreamReader()
        stream.process = SimpleNamespace(stderr=reader)
        reader.feed_data(b'x'*20000+b'\nRuntimeError: out of memory private spoken phrase SecretError\n')
        reader.feed_eof()
        await stream._read_diagnostics()
        self.assertLessEqual(len(stream.worker_diagnostics),WORKER_DIAGNOSTIC_BYTES)
        with self.assertLogs('bridge',level='ERROR') as output:
            stream._report_diagnostics(ConnectionError())
        text=' '.join(output.output)
        self.assertIn('RuntimeError',text)
        self.assertIn('out of memory',text)
        self.assertNotIn('private spoken phrase',text)
        self.assertNotIn('SecretError',text)
        self.assertFalse(stream.worker_diagnostics)


def _pkt(pcm: bytes, offset: int = 0, session: int = 1, mono: float = 100.0, wall: float = 1700000000.0):
    return {
        "pcm_base64": base64.b64encode(pcm).decode("ascii"),
        "sample_offset": offset,
        "session_id": session,
        "received_at": wall + len(pcm) / BPS,
        "received_mono": mono + len(pcm) / BPS,
        "audio_start_at": wall,
    }



def _feed_frames(stream: whisper.Stream, pcm: bytes, offset: int = 0, session: int = 1,
                 mono_start: float = 100.0, wall_start: float = 1700000000.0):
    step = whisper.FRAME * 2
    for pos in range(0, len(pcm), step):
        chunk = pcm[pos : pos + step]
        sec = pos / BPS
        stream.feed(_pkt(chunk, offset + pos // 2, session, mono_start + sec, wall_start + sec))



class QueueCadenceAndTimestampTests(unittest.TestCase):
    def make_adapter(self, gaps):
        return LocalWhisperSTT(
            python="python", model="m", vad="v",
            on_partial=lambda e: None,
            on_final=lambda *a, **kw: True,
            on_gap=lambda reason, sec, **kw: gaps.append((reason, sec, kw)),
            on_latency=lambda *a: None,
            on_connection=lambda *a: None,
        )

    def test_real_phone_10ms_cadence_buffers_up_to_10_seconds_without_eviction(self):
        gaps = []
        adapter = self.make_adapter(gaps)
        frame_10ms = b"\x00\x10" * 160
        for idx in range(1000):  # 1,000 x 10ms = 10.0s = 320,000 bytes
            adapter.feed(frame_10ms, 1700000000.0 + idx * 0.01, 100.0 + idx * 0.01)
        self.assertEqual(adapter.pending_bytes, 320000)
        self.assertEqual(len(adapter.queue), 1000)
        self.assertEqual(gaps, [])

    def test_real_phone_10ms_overload_evicts_oldest_and_preserves_audio_start_at(self):
        gaps = []
        adapter = self.make_adapter(gaps)
        frame_10ms = b"\x00\x10" * 160
        for idx in range(1200):  # 12.0s -> 2.0s evicted, newest 10.0s retained
            adapter.feed(frame_10ms, 1700000000.01 + idx * 0.01, 100.01 + idx * 0.01)
        self.assertEqual(adapter.pending_bytes, 320000)
        self.assertEqual(adapter.queue[0].sample_offset, 32000)
        self.assertAlmostEqual(sum(sec for reason, sec, _ in gaps if reason == "queue_evicted"), 2.0, places=4)
        self.assertAlmostEqual(adapter.queue[0].audio_start_at, 1700000002.0, places=4)

    def test_partial_packet_trim_and_release_advance_audio_start_at_exactly(self):
        gaps = []
        adapter = self.make_adapter(gaps)
        # Feed 11.5s as one large packet -> front 1.5s trimmed
        adapter.feed(bytes(int(11.5 * BPS)), 1700000011.5, 111.5)
        pkt = adapter.queue[0]
        self.assertEqual(pkt.sample_offset, 24000)
        self.assertAlmostEqual(pkt.audio_start_at, 1700000001.5, places=5)
        adapter.retained.append(adapter.queue.popleft())
        adapter._release(32000)  # release another 0.5s (8,000 samples)
        self.assertEqual(adapter.retained[0].sample_offset, 32000)
        self.assertAlmostEqual(adapter.retained[0].audio_start_at, 1700000002.0, places=5)



class SegmentationAndPrefinalTests(unittest.IsolatedAsyncioTestCase):
    def test_whisper_policy_validates_confirm_frames_and_prefinal_window(self):
        for bad in (
            {"confirm_frames": 0},
            {"confirm_frames": True},
            {"prefinal_quiet_seconds": -0.1},
            {"prefinal_quiet_seconds": 0.8, "pause_seconds": 0.5},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                WhisperPolicy(**bad)

    async def test_single_frame_transient_is_rejected_without_gap_while_two_frames_confirm(self):
        class FakeRec:
            def reset_vad(self):
                pass

            def probability(self, pcm):
                return 0.85 if pcm[:2] == b"\x00\x10" else 0.01

            def decode(self, pcm):
                return "Yes.", 0.01, ""

        # 1 frame (32 ms) transient + 16 frames silence -> unconfirmed, zero finals, zero gaps
        events_1 = []
        stream_1 = whisper.Stream(FakeRec(), output=events_1.append)
        _feed_frames(stream_1, b"\x00\x10" * whisper.FRAME + bytes(whisper.FRAME * 2 * 16))
        stream_1.stopping = True
        await asyncio.wait_for(stream_1.decode_loop(), 2)
        self.assertEqual([e for e in events_1 if e["type"] == "final"], [])
        self.assertEqual([e for e in events_1 if e["type"] == "gap"], [])
        self.assertEqual([e["reason"] for e in events_1 if e["type"] == "no_text"], ["unconfirmed"])

        # 2 frames (64 ms) voiced + 16 frames silence -> confirmed and finalized
        events_2 = []
        stream_2 = whisper.Stream(FakeRec(), output=events_2.append)
        _feed_frames(stream_2, b"\x00\x10" * (whisper.FRAME * 2) + bytes(whisper.FRAME * 2 * 16))
        stream_2.stopping = True
        await asyncio.wait_for(stream_2.decode_loop(), 2)
        self.assertEqual([e["text"] for e in events_2 if e["type"] == "final"], ["Yes."])

    def test_draft_due_waits_for_post_onset_speech_not_preroll_silence(self):
        class FakeRec:
            def reset_vad(self):
                pass

            def probability(self, pcm):
                return 0.9 if pcm[:2] == b"\x00\x10" else 0.0

            def decode(self, pcm):
                return "Draft", 0.01, ""

        stream = whisper.Stream(FakeRec(), output=lambda e: None)
        # 31 frames (992 ms) pre-roll silence + 2 frames (64 ms) speech -> total pcm > 1.0s, speech = 64 ms < 250 ms
        _feed_frames(stream, bytes(whisper.FRAME * 2 * 31) + b"\x00\x10" * (whisper.FRAME * 2))
        self.assertIsNotNone(stream.active)
        self.assertFalse(stream.draft_due(stream.active, 10.0))
        # Feed 6 more speech frames -> total speech = 8 frames = 256 ms >= 250 ms -> draft_due True!
        _feed_frames(stream, b"\x00\x10" * (whisper.FRAME * 6), offset=whisper.FRAME * 33)
        self.assertTrue(stream.draft_due(stream.active, 10.0))

    async def test_prefinal_settle_decode_reused_on_clean_pause_without_second_gpu_call(self):
        decode_calls = []

        class CountingRec:
            def reset_vad(self):
                pass

            def probability(self, pcm):
                return 0.9 if pcm[:2] == b"\x00\x10" else 0.02

            def decode(self, pcm):
                decode_calls.append(len(pcm) // 2)
                return "Settled cleanly.", 0.05, ""

        events = []
        stream = whisper.Stream(CountingRec(), output=events.append)
        # 10 speech frames + 8 quiet frames (256 ms >= prefinal_quiet_seconds, < 512 ms pause)
        _feed_frames(stream, b"\x00\x10" * (whisper.FRAME * 10) + bytes(whisper.FRAME * 2 * 8))
        # Mark draft as already decoded at end of speech so prefinal is what triggers
        stream.active.decoded_samples = whisper.FRAME * 10
        stream.active.decoded_voice_end = stream.active.voice_end
        self.assertTrue(stream.prefinal_due(stream.active))

        # Run one prefinal pass by closing after it settles
        task = asyncio.create_task(stream.decode_loop())
        try:
            async with asyncio.timeout(2):
                while stream.active.settled_result is None:
                    await asyncio.sleep(0.01)
            self.assertEqual(len(decode_calls), 1)
            # Feed remaining 8 quiet frames -> closes by pause at 512 ms and reuses settled_result!
            _feed_frames(stream, bytes(whisper.FRAME * 2 * 8), offset=whisper.FRAME * 18, mono_start=101.0)
            stream.stopping = True
            await asyncio.wait_for(task, 2)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        self.assertEqual(len(decode_calls), 1)
        finals = [e for e in events if e["type"] == "final"]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["text"], "Settled cleanly.")
        lat = {e["name"]: e["seconds"] for e in events if e["type"] == "latency"}
        self.assertIn("turn_boundary_wait_sec", lat)
        self.assertLess(lat["turn_close_to_final_sec"], 0.05)

    async def test_prefinal_invalidated_when_subthreshold_voice_activity_appears(self):
        decode_calls = []

        class SubthresholdRec:
            def reset_vad(self):
                pass

            def probability(self, pcm):
                tag = pcm[:2]
                if tag == b"\x00\x10":
                    return 0.9
                if tag == b"\x00\x05":
                    return 0.22  # < continue_probability (0.30), >= 0.15 -> invalidates settled reuse!
                return 0.01

            def decode(self, pcm):
                decode_calls.append(len(pcm) // 2)
                return f"Decoded-{len(decode_calls)}", 0.01, ""

        events = []
        stream = whisper.Stream(SubthresholdRec(), output=events.append)
        _feed_frames(stream, b"\x00\x10" * (whisper.FRAME * 10) + bytes(whisper.FRAME * 2 * 8))
        stream.active.decoded_samples = whisper.FRAME * 10
        stream.active.decoded_voice_end = stream.active.voice_end
        task = asyncio.create_task(stream.decode_loop())
        try:
            async with asyncio.timeout(2):
                while stream.active.settled_result is None:
                    await asyncio.sleep(0.01)
            # 1 frame with prob 0.22 + 7 quiet frames -> closes by pause, must re-decode!
            _feed_frames(
                stream,
                b"\x00\x05" * whisper.FRAME + bytes(whisper.FRAME * 2 * 7),
                offset=whisper.FRAME * 18,
            )
            stream.stopping = True
            await asyncio.wait_for(task, 2)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        self.assertEqual(len(decode_calls), 2)
        self.assertEqual([e["text"] for e in events if e["type"] == "final"], ["Decoded-2"])

    async def test_language_and_no_speech_drops_skip_failure_gaps_while_degenerate_reports_voiced_span(self):
        reasons = iter([("", 0.01, "language"), ("", 0.01, "no_speech"), ("", 0.01, "degenerate")])

        class ReasonRec:
            def reset_vad(self):
                pass

            def probability(self, pcm):
                return 0.9 if pcm[:2] == b"\x00\x10" else 0.0

            def decode(self, pcm):
                return next(reasons)

        events = []
        stream = whisper.Stream(ReasonRec(), output=events.append, policy=WhisperPolicy(prefinal_quiet_seconds=0.0))
        # Turn 1 (language), Turn 2 (no_speech), Turn 3 (degenerate with 31-frame pre-roll + 4 voiced frames + 16 quiet frames)
        for idx in range(3):
            clip = bytes(whisper.FRAME * 2 * 20) + b"\x00\x10" * (whisper.FRAME * 4) + bytes(whisper.FRAME * 2 * 16)
            _feed_frames(stream, clip, offset=idx * whisper.FRAME * 40, session=idx + 1)
        stream.stopping = True
        await asyncio.wait_for(stream.decode_loop(), 2)
        gaps = [e for e in events if e["type"] == "gap"]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["session_id"], 3)
        # Voiced span is 4 * 512 / 16000 = 0.128s, NOT 40 * 512 / 16000 = 1.28s!
        self.assertAlmostEqual(gaps[0]["seconds"], (whisper.FRAME * 4) / RATE, places=4)
