"""Vexo Bluetooth or USB capture only. Recognition and persistence belong to the bridge."""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import struct
import av

import aiohttp
from config import ROOT, load_env, VEXO_BAND
from motion import MotionTelemetry
sys.path.insert(0, str(ROOT / 'vendor/vexo-open-kit/python'))
from vexo_open.sync import Band
from vexo_open.audio import PcmDecoder
from serial.tools.list_ports import comports


class Discontinuity(Exception):
    pass


class Sequence:
    def __init__(self):
        self.expected = None
        self.lost = self.old = 0

    def accept(self, seq):
        gap = 0 if self.expected is None else (seq - self.expected) & 65535
        if gap > 32768:
            self.old += 1
            return False
        self.lost += gap
        self.expected = (seq + 1) & 65535
        if gap:
            raise Discontinuity('packet_loss')
        return True


def find_port():
    ports = [p.device for p in comports() if p.vid == 0x2fe3 and p.pid == 4
             and 'Vexo' in ((p.product or '') + (p.manufacturer or ''))]
    if len(ports) != 1:
        raise Discontinuity('expected_one_vexo_usb_device')
    # Never displace a browser/dashboard or another serial owner.
    check = subprocess.run(['lsof', ports[0]], capture_output=True)
    if check.returncode == 0:
        raise Discontinuity('usb_port_in_use_close_dashboard')
    return ports[0]


async def capture(session, env, stop, band_number, stats, transport="serial"):
    # Decoder import and initialization complete BEFORE enabling microphone.
    decoder = PcmDecoder(conceal=False)
    validator = av.CodecContext.create('opus', 'r')
    validator.extradata = b'OpusHead' + struct.pack('<BBHIhB', 1, 1, 0, 16000, 0, 0)
    sequence = Sequence()
    opus, pcm = queue.Queue(50), queue.Queue(50)
    failed, decode_stop = threading.Event(), threading.Event()
    reason = ['']
    closing = threading.Event()
    def fail(why):
        reason[0] = why
        failed.set()
    def on_audio(a):
        if failed.is_set():
            return
        try:
            opus.put_nowait((a.seq, bytes(a.opus), time.monotonic()))
        except queue.Full:
            stats['queue_overflows'] += 1
            stats['discarded_packets'] += 1
            fail('capture_queue_overflow')
    def decode():
        try:
            while not decode_stop.is_set() or not opus.empty():
                try:
                    seq, packet, received = opus.get(timeout=.1)
                except queue.Empty:
                    continue
                if not sequence.accept(seq):
                    stats['old_packets'] += 1
                    continue
                # Feed without seq: Sequence ends sessions on ANY loss. No PLC/gap fill.
                try:
                    validator.decode(av.Packet(packet))
                except Exception:
                    stats['corrupt_packets'] += 1
                    raise Discontinuity('corrupt_opus') from None
                raw = decoder.feed(packet)
                if not 0 < len(raw) <= 640 or len(raw) % 2:
                    raise Discontinuity('unexpected_pcm_duration')
                pcm.put((raw, received), timeout=1)
                stats['received_packets'] += 1
        except Exception:
            stats['discarded_packets'] += 1
            stats['lost_packets'] += sequence.lost
            fail('decode_or_sequence_discontinuity')
    options = {"port": find_port()} if transport == "serial" else {}
    token = env.get('OMI_STREAM_TOKEN', '')
    if not token:
        raise ValueError('OMI_STREAM_TOKEN required')
    async with session.ws_connect('http://127.0.0.1:8787/v1/listen?encoding=linear16&sample_rate=16000&channels=1',
            headers={'Authorization': 'Bearer ' + token, 'X-Audio-Source': 'vexo', 'X-Vexo-Band': str(band_number), 'X-Vexo-Transport': transport},
            heartbeat=20) as ws:
        async def receive_results():
            async for _ in ws:
                pass  # Captions/persistence remain the bridge's responsibility.
            if not closing.is_set():
                fail('bridge_disconnected')
        receiver = asyncio.create_task(receive_results())
        telemetry = MotionTelemetry()
        band = None
        worker = None
        try:
            band = await asyncio.to_thread(Band.connect, band_number, transport=transport, **options)
            if band.number != band_number:
                raise Discontinuity('wrong_band')
            band.on('audio', on_audio)
            band.on('motion', telemetry.on_motion)
            band.on('event', telemetry.on_event)
            band.on('disconnect', lambda _: None if closing.is_set() else fail('device_disconnected'))
            band.on('stream-taken', lambda _: fail('source_taken'))
            worker = threading.Thread(target=decode)
            worker.start()
            print(f'MICROPHONE ON: Vexo-HACK-{band_number:02}; audio goes to the selected bridge recognizer.', flush=True)
            await asyncio.to_thread(band.configure, audio=True)
            deadline = time.monotonic()
            last = deadline
            report_at = deadline
            while not stop.is_set() and not failed.is_set():
                await telemetry.tick(band)
                try:
                    raw, received = pcm.get_nowait()
                except queue.Empty:
                    if time.monotonic() - last > 3:
                        raise Discontinuity('audio_stalled')
                    await asyncio.sleep(.005)
                    continue
                last = time.monotonic()
                if last - received > 2:
                    raise Discontinuity('audio_backlog')
                await asyncio.sleep(max(0, deadline-time.monotonic()))
                await asyncio.wait_for(ws.send_bytes(raw), 2)
                stats['sent_samples'] += len(raw)//2
                if time.monotonic() - report_at >= 1:
                    await ws.send_json({'type': 'SourceStatus', 'counters': stats})
                    report_at = time.monotonic()
                deadline = max(deadline + len(raw)/32000, time.monotonic())
            if failed.is_set():
                raise Discontinuity(reason[0])
        finally:
            closing.set()
            telemetry.close()
            if band is not None:
                try:
                    await asyncio.to_thread(band.configure, audio=False)
                except Exception:
                    pass
                try:
                    await asyncio.to_thread(band.disconnect)
                except Exception:
                    fail('device_disconnect_failed')
            decode_stop.set()
            if worker:
                await asyncio.to_thread(worker.join, 3)
            # Drain accepted queued audio on normal stop only. Broken sessions never
            # compress an unknown hole into uninterrupted speech.
            if not failed.is_set() and stop.is_set():
                while not pcm.empty():
                    raw, _ = pcm.get_nowait()
                    await asyncio.sleep(len(raw)/32000)
                    await asyncio.wait_for(ws.send_bytes(raw), 2)
                    stats['sent_samples'] += len(raw)//2
            if stop.is_set() and not failed.is_set():
                tail = decoder.flush()
                if tail:
                    await ws.send_bytes(tail)
                    stats['sent_samples'] += len(tail)//2
            stats['discarded_packets'] += opus.qsize() + pcm.qsize()
            await ws.send_json({'type': 'SourceStatus', 'counters': stats, 'discontinuity': not stop.is_set() or failed.is_set()})
            await ws.send_json({'type': 'CloseStream'})
            try:
                await asyncio.wait_for(receiver, 20)
            finally:
                if not receiver.done():
                    receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
            if ws.close_code != 1000:
                raise Discontinuity('bridge_drain_unconfirmed')
            print('MICROPHONE OFF; bridge drain completed.', flush=True)


async def run(number, transport="serial"):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    stats = dict(received_packets=0, sent_samples=0, lost_packets=0, old_packets=0, queue_overflows=0, discarded_packets=0, corrupt_packets=0, sessions=0)
    env = load_env()
    async with aiohttp.ClientSession() as session:
        while not stop.is_set():
            stats['sessions'] += 1
            try:
                await capture(session, env, stop, number, stats, transport)
            except Exception as error:
                # No provider bodies, tokens or microphone content in diagnostics.
                print('Source session ended:', str(error) if isinstance(error, Discontinuity) else type(error).__name__, flush=True)
            print(json.dumps(stats), flush=True)
            try:
                await asyncio.wait_for(stop.wait(), 2)
            except TimeoutError:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--band', type=int, choices=(VEXO_BAND,), required=True)
    parser.add_argument("--transport", choices=("serial", "ble"), default="serial")
    args = parser.parse_args()
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/vexo-source.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(run(args.band, args.transport))
