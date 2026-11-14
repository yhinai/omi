"""Exercise Omi's shim through the shared capture engine without a radio."""
import asyncio
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import vexo_source
from omi_device import OmiBand
from config import ROOT
from vexo_open.audio import ogg_packets

class OmiCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self,codec):
        stop=asyncio.Event();loop=asyncio.get_running_loop()
        packets=ogg_packets(ROOT/'vendor/vexo-open-kit/dashboard/speech/sample-speech.opus')[:10] if codec==21 else [bytes(320)]*10
        class Device:
            info={'model':'Synthetic Omi','codec':codec}
            loss_counters={'device_lost_notifications':0,'device_old_notifications':0}
            def __init__(self):self.handlers={};self.off=False;self.closed=False;self.codec=codec
            def on(self,event,fn):self.handlers[event]=fn
            def configure(self,*,audio):
                if audio:
                    def produce():
                        for seq,packet in enumerate(packets):
                            self.handlers['audio'](SimpleNamespace(seq=seq,opus=packet))
                            time.sleep(.02 if codec==21 else .01)
                        loop.call_soon_threadsafe(stop.set)
                    self.thread=threading.Thread(target=produce);self.thread.start()
                else:self.off=True
            def disconnect(self):self.thread.join();self.closed=True
        class Socket:
            close_code=1000
            def __init__(self):self.sent=[];self.controls=[];self.closed=asyncio.Event()
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def send_bytes(self,data):self.sent.append(data)
            async def send_json(self,data):
                self.controls.append(data)
                if data.get('type')=='CloseStream':self.closed.set()
            def __aiter__(self):return self
            async def __anext__(self):await self.closed.wait();raise StopAsyncIteration
        device=Device();ws=Socket();calls=[]
        def connect(*args,**kwargs):calls.append(kwargs);return ws
        stats=dict(received_packets=0,sent_samples=0,lost_packets=0,old_packets=0,queue_overflows=0,discarded_packets=0,corrupt_packets=0)
        address='00000000-0000-0000-0000-000000000001'
        with mock.patch.object(OmiBand,'connect',return_value=device) as open_device, mock.patch.object(vexo_source,'find_port',side_effect=AssertionError('No serial access allowed')):
            await vexo_source.capture(SimpleNamespace(ws_connect=connect),{'OMI_STREAM_TOKEN':'synthetic'},stop,43,stats,'ble',address)
        open_device.assert_called_once_with(address)
        self.assertEqual(calls[0]['headers']['X-Audio-Source'],'omi_ble')
        self.assertEqual(calls[0]['headers']['X-Omi-Device'],address)
        self.assertTrue(device.off and device.closed)
        self.assertEqual(stats['received_packets'],10)
        self.assertEqual(stats['sent_samples'],3200 if codec==21 else 1600)
        self.assertEqual(stats['discarded_packets'],0)
        self.assertEqual(ws.controls[-1],{'type':'CloseStream'})
        self.assertFalse(ws.controls[-2]['discontinuity'])

    async def test_opus_20ms_is_decoded_and_drained(self):await self.exercise(21)
    async def test_native_pcm_10ms_is_forwarded_without_opus_decode(self):await self.exercise(0)
