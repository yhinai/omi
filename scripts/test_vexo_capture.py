"""Run with .venv-vexo/bin/python -m unittest scripts.test_vexo_capture -q."""
import unittest
from unittest import mock
from types import SimpleNamespace
import vexo_source as source
from vexo_open.audio import ogg_packets, PcmDecoder
from config import ROOT

class CaptureTests(unittest.TestCase):
    def test_rollover_duplicates_and_old_packets(self):
        seq=source.Sequence()
        for n in [65534,65535,0,1]: self.assertTrue(seq.accept(n))
        self.assertFalse(seq.accept(1))
        self.assertFalse(seq.accept(0))
        self.assertEqual(seq.old,2)
        self.assertEqual(seq.lost,0)

    def test_loss_breaks_session(self):
        seq=source.Sequence(); seq.accept(10)
        with self.assertRaises(source.Discontinuity): seq.accept(13)
        self.assertEqual(seq.lost,2)

    def test_sample_pcm_format_and_duration(self):
        packets=ogg_packets(ROOT/'vendor/vexo-open-kit/dashboard/speech/sample-speech.opus')
        decoder=PcmDecoder(conceal=False)
        pcm=b''.join(decoder.feed(p) for p in packets)+decoder.flush()
        self.assertEqual(len(pcm), len(packets)*640)
        self.assertEqual(decoder.lost,0)

    def test_port_filter_never_selects_samsung(self):
        phone=SimpleNamespace(vid=1256,pid=26720,product='SAMSUNG',manufacturer='Samsung',device='/dev/phone')
        with mock.patch.object(source,'comports',return_value=[phone]):
            with self.assertRaises(source.Discontinuity):source.find_port()

    def test_busy_port_is_not_opened(self):
        band=SimpleNamespace(vid=0x2fe3,pid=4,product='Vexo Open band',manufacturer='Vexo Open',device='/dev/vexo')
        with mock.patch.object(source,'comports',return_value=[band]),mock.patch.object(source.subprocess,'run',return_value=SimpleNamespace(returncode=0)):
            with self.assertRaisesRegex(source.Discontinuity,'port_in_use'):source.find_port()

class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, *, loss=False, flood=False, transport="serial"):
        import asyncio
        import time
        import threading
        packets=ogg_packets(ROOT/'vendor/vexo-open-kit/dashboard/speech/sample-speech.opus')[:10]
        if flood:packets=packets*500
        stop=asyncio.Event()
        loop=asyncio.get_running_loop()
        class FakeBand:
            number=43
            handlers={}
            configured=[]
            def on(self,event,handler):self.handlers[event]=handler
            def configure(self,**options):
                self.configured.append(options)
                if options.get('audio'):
                    def produce():
                        for n,p in enumerate(packets):
                            self.handlers['audio'](SimpleNamespace(seq=n+(1 if loss and n>=5 else 0),opus=p))
                            if not flood:time.sleep(.02)
                        loop.call_soon_threadsafe(stop.set)
                    self.thread=threading.Thread(target=produce);self.thread.start()
            def disconnect(self):
                self.thread.join()
                self.handlers['disconnect'](None)
        class Socket:
            close_code=1000
            sent=[]
            controls=[]
            closed=asyncio.Event()
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def send_bytes(self,data):self.sent.append(data)
            async def send_json(self,data):
                self.controls.append(data)
                if data.get('type') == 'CloseStream':self.closed.set()
            def __aiter__(self):return self
            async def __anext__(self):
                await self.closed.wait()
                raise StopAsyncIteration
        band=FakeBand(); ws=Socket()
        session=SimpleNamespace(ws_connect=lambda *a,**kw:ws)
        stats=dict(received_packets=0,sent_samples=0,lost_packets=0,old_packets=0,queue_overflows=0,discarded_packets=0,corrupt_packets=0)
        with mock.patch.object(source,'find_port',return_value='/dev/fake') as find, mock.patch.object(source.Band,'connect',return_value=band) as connect, mock.patch.object(source,'MotionTelemetry',return_value=SimpleNamespace(on_motion=lambda e:None,on_event=lambda e:None,on_battery=lambda e:None,tick=mock.AsyncMock(),close=mock.Mock())):
            if loss or flood:
                with self.assertRaises(source.Discontinuity):
                    await source.capture(session,{'OMI_STREAM_TOKEN':'synthetic-test-token'},stop,43,stats,transport)
            else:
                await source.capture(session,{'OMI_STREAM_TOKEN':'synthetic-test-token'},stop,43,stats,transport)
        if transport == 'ble':
            find.assert_not_called()
            connect.assert_called_once_with(43,transport='ble')
        self.assertEqual(band.configured[-1],{'audio':False})
        if loss or flood:
            self.assertTrue(ws.controls[-2]['discontinuity'])
            return stats
        self.assertEqual(stats['received_packets'],10)
        self.assertEqual(stats['discarded_packets'],0)
        self.assertEqual(stats['sent_samples'],3200)
        self.assertEqual(ws.controls[-1],{'type':'CloseStream'})
        self.assertFalse(ws.controls[-2]['discontinuity'])

    async def test_normal_stop_turns_mic_off_drains_pcm_and_closes_stream(self):
        await self.exercise()

    async def test_loss_ends_session_without_inserting_pcm(self):
        stats=await self.exercise(loss=True)
        self.assertEqual(stats['lost_packets'],1)
        self.assertLess(stats['sent_samples'],3200)

    async def test_overload_is_bounded_and_explicit(self):
        stats=await self.exercise(flood=True)
        self.assertGreater(stats['queue_overflows'],0)
        self.assertGreater(stats['discarded_packets'],0)

    async def test_bluetooth_uses_same_pcm_drain_without_serial_discovery(self):
        await self.exercise(transport="ble")
