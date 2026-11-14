"""Private-process cleanup, protocol backpressure and Omi's Pipecat provenance contract."""
import asyncio
import json
import sys
import time
import unittest
from unittest import mock

import aiohttp
from pipecat_stt import PipelineSocket, PipecatDeepgramSTT
from test_deepgram_stt import Socket, result, until

READY = {"kind":"ready","protocol":1,"pipecat":"1.16.0","deepgram_sdk":"7.19.0"}


class PipelineTransportTests(unittest.IsolatedAsyncioTestCase):
    def socket(self, body):
        self.children=[]
        script="import sys,json\nprint("+repr(json.dumps(READY))+",flush=True)\n"+body
        async def launch(*args, **kwargs):
            child=await asyncio.create_subprocess_exec(sys.executable,'-B','-c',script,**kwargs)
            self.children.append(child)
            return child
        return PipelineSocket(sys.executable,'synthetic-secret',{},process_factory=launch)

    async def test_acknowledged_audio_and_events_use_the_private_pipe(self):
        socket=self.socket('''for line in sys.stdin:
 m=json.loads(line)
 if m['op']=='stop':break
 print(json.dumps({'kind':'ack','id':m['id']}),flush=True)
 print(json.dumps({'kind':'event','event':{'type':'Metadata','duration':0.1}}),flush=True)
''')
        async with socket:
            await socket.send_bytes(bytes(3200))
            message=await socket.receive()
            self.assertEqual(message.type,aiohttp.WSMsgType.TEXT)
            self.assertEqual(json.loads(message.data)['duration'],.1)
        self.assertIsNotNone(self.children[0].returncode)

    async def test_unexpected_worker_exit_is_reported_without_raw_diagnostics(self):
        socket=self.socket("sys.stderr.write('private transcript and credentials\\n')\n")
        async with socket:
            message=await asyncio.wait_for(socket.receive(),2)
            self.assertEqual(message.type,aiohttp.WSMsgType.CLOSED)
            self.assertNotIn('private',str(socket.failure))
        self.assertIsNotNone(self.children[0].returncode)

    async def test_cancelled_send_removes_ack_waiter_and_owned_process_is_stopped(self):
        socket=self.socket("for line in sys.stdin:\n if json.loads(line)['op']=='stop':break\n")
        async with socket:
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(socket.send_bytes(bytes(320)),.03)
            self.assertFalse(socket.waiters)
        self.assertIsNotNone(self.children[0].returncode)

    async def test_wrong_runtime_version_never_marks_pipeline_ready(self):
        async def launch(*args,**kwargs):
            return await asyncio.create_subprocess_exec(sys.executable,'-B','-c',
                "print('{\"kind\":\"ready\",\"protocol\":0}',flush=True)",**kwargs)
        socket=PipelineSocket(sys.executable,'synthetic',{},process_factory=launch)
        with self.assertRaises(ConnectionError):await socket.__aenter__()
        self.assertIsNotNone(socket.process.returncode)

    async def test_cancelled_launch_still_reaps_the_owned_child(self):
        children = []
        async def launch(*args, **kwargs):
            await asyncio.sleep(.03)
            child = await asyncio.create_subprocess_exec(sys.executable, '-B', '-c',
                "import sys,json\nfor line in sys.stdin:\n if json.loads(line)['op']=='stop':break\n", **kwargs)
            children.append(child)
            return child
        socket = PipelineSocket(sys.executable, 'synthetic', {}, process_factory=launch)
        opening = asyncio.create_task(socket.__aenter__())
        await asyncio.sleep(.005)
        opening.cancel()
        with self.assertRaises(asyncio.CancelledError):await opening
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].returncode)


class PipecatAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_save_retries_immutable_pipecat_provenance_and_flush_waits(self):
        peer=Socket()
        save=mock.Mock(return_value=False)
        stream=PipecatDeepgramSTT(python=sys.executable,api_key='synthetic',socket_factory=lambda *args:peer,
            on_partial=mock.Mock(),on_final=save,on_gap=mock.Mock(),on_latency=mock.Mock(),on_connection=mock.Mock())
        task=asyncio.create_task(stream.run())
        try:
            stream.feed(bytes(3200),time.time(),time.monotonic())
            await until(lambda:peer.audio)
            peer.event(result())
            await until(lambda:save.called)
            self.assertEqual(stream.pending_final_count,1)
            fields=save.call_args.kwargs
            self.assertEqual(fields['recognition']['orchestration'],'pipecat')
            self.assertEqual(fields['recognition']['pipecat_version'],'1.16.0')
            flush=asyncio.create_task(stream.flush_now(1))
            await asyncio.sleep(.01)
            self.assertFalse(flush.done())
            save.return_value=True
            stream.retry_saves()
            self.assertTrue(await flush)
            self.assertEqual(stream.final_count,1)
            self.assertEqual(save.call_args.kwargs['recognition'],fields['recognition'])
        finally:
            await stream.close()
            await task

    def test_selection_keeps_general_deepgram_configuration_and_whisper_separate(self):
        from backend_settings import DeepgramConfig
        direct=DeepgramConfig.from_env({'DEEPGRAM_API_KEY':'synthetic'})
        self.assertEqual(direct.pipeline,'direct')
        selected=DeepgramConfig.from_env({'DEEPGRAM_API_KEY':'synthetic','OMI_DEEPGRAM_PIPELINE':'pipecat'})
        self.assertEqual(selected.pipeline,'pipecat')
        with self.assertRaises(ValueError):
            DeepgramConfig.from_env({'DEEPGRAM_API_KEY':'synthetic','OMI_DEEPGRAM_PIPELINE':'typo'})

    def test_pipeline_start_does_not_inflate_provider_connection_metric(self):
        latency=mock.Mock()
        stream=PipecatDeepgramSTT(python=sys.executable,api_key='synthetic',on_final=mock.Mock(),
            on_partial=mock.Mock(),on_gap=mock.Mock(),on_latency=latency,on_connection=mock.Mock())
        stream._pipeline_session.socket=mock.Mock(provider_connect_sec=.1)
        stream._latency('provider_connect_sec',.6)
        self.assertEqual(latency.call_args_list,[mock.call('pipeline_start_sec',.6),mock.call('provider_connect_sec',.1)])
