"""Exercise the real Pipecat pipeline with an in-memory SDK peer, without cloud traffic."""
import asyncio
import base64
from contextlib import asynccontextmanager
import importlib.util
import json
from types import SimpleNamespace
import unittest

AVAILABLE = importlib.util.find_spec("pipecat") is not None


@unittest.skipUnless(AVAILABLE, "Optional Pipecat environment required")
class PipecatRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_transcription_frames_and_terminal_metadata_are_ordered(self):
        from loguru import logger
        logger.remove()
        from deepgram.core.events import EventType
        from deepgram.listen.v1.types import ListenV1Results
        import pipecat_worker
        reader = asyncio.StreamReader()
        output = []
        options = {}
        callbacks = {}
        closed = asyncio.Event()

        class Peer:
            def on(self, event, callback):callbacks[event] = callback
            async def start_listening(self):await closed.wait()
            async def send_media(self, raw):
                payload = {"type": "Results", "channel_index": [0, 1], "start": 0, "duration": len(raw)/32000,
                    "is_final": True, "speech_final": True,
                    "channel": {"alternatives": [{"transcript": "Hello.", "confidence": .9, "words": []}]},
                    "metadata": {"request_id": "00000000-0000-0000-0000-000000000001", "model_info": {
                        "name": "general-nova-3", "version": "test", "arch": "nova-3"},
                        "model_uuid": "00000000-0000-0000-0000-000000000002"}}
                await callbacks[EventType.MESSAGE](ListenV1Results.model_validate(payload))
            async def send_close_stream(self, request):
                await callbacks[EventType.MESSAGE](SimpleNamespace(model_dump=lambda **kw:{"type":"Metadata","duration":.1}))
                closed.set()
            async def send_keep_alive(self, request):pass

        @asynccontextmanager
        async def connect(**kwargs):
            options.update(kwargs)
            yield Peer()
        def emit(message):
            output.append(message)
            if message.get("kind") == "ready":
                reader.feed_data((json.dumps({"op":"audio","id":1,"pcm":base64.b64encode(bytes(3200)).decode()})+'\n').encode())
                reader.feed_data(b'{"op":"CloseStream","id":2}\n')
            if message.get("event",{}).get("type") == "Metadata":
                reader.feed_data(b'{"op":"stop"}\n')
        counts = await asyncio.wait_for(pipecat_worker.serve(reader, {}, emit,
            SimpleNamespace(listen=SimpleNamespace(v1=SimpleNamespace(connect=connect)))), 15)
        self.assertEqual(counts["audio"], 1)
        self.assertEqual(counts["final"], 1)
        events = [m["event"]["type"] for m in output if m.get("kind")=="event"]
        self.assertEqual(events,["Results","Metadata"])
        self.assertEqual(options["mip_opt_out"], "true")
        self.assertEqual(options["redact_usage"], "true")
        self.assertEqual(options["sample_rate"], "16000")
        from deepgram import AsyncDeepgramClient
        import inspect
        inspect.signature(AsyncDeepgramClient(api_key="synthetic").listen.v1.connect).bind(**options)
        self.assertEqual(options["request_options"]["additional_query_parameters"]["no_delay"], "true")

    async def test_sdk_send_failure_never_acknowledges_audio_or_reconnects_under_same_clock(self):
        from loguru import logger
        logger.remove()
        import pipecat_worker
        reader=asyncio.StreamReader()
        connections=[]
        output=[]
        class Peer:
            def on(self,*args):pass
            async def start_listening(self):await asyncio.Event().wait()
            async def send_media(self,raw):raise OSError('synthetic provider failure with private data')
            async def send_close_stream(self,request):pass
        @asynccontextmanager
        async def connect(**kwargs):
            connections.append(True)
            yield Peer()
        def emit(message):
            output.append(message)
            if message.get('kind')=='ready':
                reader.feed_data((json.dumps({'op':'audio','id':1,'pcm':base64.b64encode(bytes(320)).decode()})+'\n').encode())
        with self.assertRaises(ConnectionError):
            await asyncio.wait_for(pipecat_worker.serve(reader,{},emit,
                SimpleNamespace(listen=SimpleNamespace(v1=SimpleNamespace(connect=connect)))),10)
        self.assertEqual(len(connections),1)
        self.assertFalse(any(m.get('kind')=='ack' for m in output))
        self.assertNotIn('private',str(output))
