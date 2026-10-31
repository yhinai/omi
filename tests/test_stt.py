"""Local segmentation, nonblocking finals, source ownership and recovery regressions."""
import asyncio
import base64
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from stt import LocalWhisperSTT
from whisper import FRAME, MAX_TURN, Stream
from common import Packet, WhisperPolicy


class Recognizer:
    def reset_vad(self):
        pass

    def probability(self, pcm):
        return .9 if pcm[:2] == b"\x00\x10" else 0

    def decode(self, pcm):
        return "Recognized words.", .001


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
        speech = b"\x00\x10" * (MAX_TURN+FRAME*5)
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


class DecodeTests(unittest.IsolatedAsyncioTestCase):
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
             mock.patch.object(bridge,'LocalWhisperSTT',return_value=fake), \
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
                return "Recognized words.", .001
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


class JournalRetryTests(unittest.TestCase):
    def test_flush_failure_after_complete_write_does_not_duplicate_the_journal(self):
        import bridge
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'omi_transcript_events.jsonl'
            logger=mock.Mock()
            def uncertain_write(encoded):
                path.write_text(encoded+'\n')
                raise OSError('flush failed after write')
            logger.info.side_effect=uncertain_write
            runtime=bridge.BridgeState()
            with mock.patch.object(bridge,'state',runtime), mock.patch.object(bridge,'LOG_DIR',directory), \
                 mock.patch.object(bridge,'STT_BACKEND','whisper'), mock.patch.object(bridge,'event_logger',logger):
                self.assertFalse(bridge.record_live_final('Recovered once.',utterance_id='u',session_id=1))
                self.assertFalse(runtime.recent_events)
                logger.info.side_effect=None
                self.assertTrue(bridge.record_live_final('Recovered once.',utterance_id='u',session_id=1))
                self.assertEqual(logger.info.call_count,1)
                self.assertEqual(len(path.read_text().splitlines()),1)
                self.assertEqual(len(runtime.recent_events),1)
                self.assertEqual(runtime.metrics['utterances_processed'],1)
                self.assertEqual(runtime.metrics['transcripts_emitted'],1)
                self.assertFalse(runtime.failed_finals)

    def test_partial_failed_line_does_not_swallow_the_retried_final(self):
        import bridge
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'omi_transcript_events.jsonl'
            logger=mock.Mock()
            def partial_write(encoded):
                path.write_text(encoded[:25])
                raise OSError('partial write')
            logger.info.side_effect=partial_write
            with mock.patch.object(bridge,'state',bridge.BridgeState()), mock.patch.object(bridge,'LOG_DIR',directory), \
                 mock.patch.object(bridge,'STT_BACKEND','whisper'), mock.patch.object(bridge,'event_logger',logger):
                self.assertFalse(bridge.record_live_final('Recovered once.',utterance_id='u'))
                def append(encoded):
                    with path.open('a') as journal:journal.write(encoded+'\n')
                logger.info.side_effect=append
                self.assertTrue(bridge.record_live_final('Recovered once.',utterance_id='u'))
                lines=path.read_text().splitlines()
                self.assertEqual(len(lines),2)
                self.assertEqual(json.loads(lines[1])['text'],'Recovered once.')
