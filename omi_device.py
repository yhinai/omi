"""Direct Omi BLE input, following BasedHardware/omi's published device protocol.
No STT client, phone app, firmware writes, or audio files.
"""
import asyncio
from collections import defaultdict
import struct
import threading
from types import SimpleNamespace

SERVICE = '19b10000-e8f2-537e-4f6c-d104768a1214'
AUDIO = '19b10001-e8f2-537e-4f6c-d104768a1214'
CODEC = '19b10002-e8f2-537e-4f6c-d104768a1214'


class PacketError(Exception):
    pass


class Reassembler:
    """Notification counter advances per fragment; index 0 begins a new codec frame."""
    def __init__(self):
        self.expected = None
        self.index = -1
        self.pending = bytearray()
        self.old = self.lost = 0

    def feed(self, data):
        if len(data) <= 3:
            raise PacketError('short_notification')
        seq,index = struct.unpack_from('<HB', data)
        gap = 0 if self.expected is None else (seq-self.expected)&65535
        if gap > 32768:
            self.old += 1
            return None
        self.lost += gap
        if gap:
            raise PacketError('notification_loss')
        self.expected = (seq+1)&65535
        result = None
        if index == 0:
            result = bytes(self.pending) if self.pending else None
            self.pending.clear()
        elif not self.pending or index != self.index+1:
            raise PacketError('fragment_order')
        self.index = index
        self.pending.extend(data[3:])
        if len(self.pending)>4096:
            raise PacketError('frame_too_large')
        return result

    def flush(self):
        result=bytes(self.pending)
        self.pending.clear()
        return result


class OmiBand:
    """Small synchronous device interface for the existing capture/queue engine."""
    @classmethod
    def connect(cls,address):
        obj=cls(address)
        try:
            obj.call(obj.open())
            return obj
        except BaseException:
            obj.disconnect()
            raise

    def __init__(self,address):
        self.address=address
        self.handlers=defaultdict(list)
        self.loop=asyncio.new_event_loop()
        self.thread=threading.Thread(target=self.loop.run_forever,daemon=True)
        self.thread.start()
        self.client=None
        self.closing=False
        self.streaming=False
        self.reassembler=Reassembler()
        self.frame=0
        self.codec=None
        self.info={}

    def call(self,coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine,self.loop).result(timeout=30)

    async def open(self):
        from bleak import BleakClient, BleakScanner
        d=await BleakScanner.find_device_by_address(self.address,timeout=10,service_uuids=[SERVICE])
        if d is None:raise PacketError('configured_omi_not_advertising')
        self.client=BleakClient(d,disconnected_callback=lambda _:self.emit('disconnect',None) if not self.closing else None)
        await self.client.connect()
        if not any(s.uuid.lower()==SERVICE for s in self.client.services):raise PacketError('not_an_omi_service')
        raw=await self.client.read_gatt_char(CODEC)
        if len(raw)!=1 or raw[0] not in {0,20,21}:raise PacketError('unsupported_omi_codec')
        self.codec=raw[0]
        self.name=d.name or 'Omi'
        for key,short in [('model','2a24'),('firmware','2a26'),('hardware','2a27'),('manufacturer','2a29')]:
            uuid=f'0000{short}-0000-1000-8000-00805f9b34fb'
            if self.client.services.get_characteristic(uuid):
                value=await self.client.read_gatt_char(uuid)
                self.info[key]=bytes(value).decode('utf-8',errors='replace')[:120]
        battery='00002a19-0000-1000-8000-00805f9b34fb'
        if self.client.services.get_characteristic(battery):
            self.info['battery_percent']=(await self.client.read_gatt_char(battery))[0]
        self.info['codec']=self.codec

    def on(self,event,handler):
        self.handlers[event].append(handler)

    def emit(self,event,value):
        for fn in self.handlers[event]:fn(value)

    def packet(self,payload):
        if payload:
            self.emit('audio',SimpleNamespace(seq=self.frame,opus=payload))
            self.frame=(self.frame+1)&65535

    def receive(self,_sender,data):
        try:
            self.packet(self.reassembler.feed(bytes(data)))
        except PacketError:
            self.emit('disconnect',None)

    async def set_audio(self,enabled):
        if enabled and not self.streaming:
            await self.client.start_notify(AUDIO,self.receive)
            self.streaming=True
        elif not enabled and self.streaming:
            await self.client.stop_notify(AUDIO)
            self.streaming=False
            self.packet(self.reassembler.flush())

    def configure(self,*,audio):
        self.call(self.set_audio(audio))

    async def close(self):
        self.closing=True
        if self.client and self.client.is_connected:
            await self.set_audio(False)
            await self.client.disconnect()

    def disconnect(self):
        if not self.loop.is_running():return
        try:self.call(self.close())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=3)
            self.loop.close()

    @property
    def loss_counters(self):
        return {'device_lost_notifications':self.reassembler.lost,'device_old_notifications':self.reassembler.old}
