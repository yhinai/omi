"""Local motion telemetry and conservative, calibrated control proposals."""
from collections import deque
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from config import ROOT, VEXO_BAND

LEASE = ROOT / 'logs/vexo-motion.lease'
STATE = ROOT / 'logs/vexo-motion.json'


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w') as output:
        json.dump(value, output, allow_nan=False)
    os.replace(temporary, path)


class MotionTelemetry:
    """Callbacks only replace small in-memory values. Capture loop writes at 10 Hz."""
    def __init__(self):
        self.lock = threading.Lock()
        self.generation = uuid.uuid4().hex
        self.battery = None
        self.sample = None
        self.recent_samples = deque(maxlen=50)
        self.events = deque(maxlen=30)
        self.count = 0
        self.enabled = False
        self.rate = 0
        self.controlling = False
        self.error = None
        self.last_poll = self.last_publish = 0

    def on_battery(self, event):
        with self.lock:
            self.battery = {k:event.get(k) for k in ('percent','millivolts','state','charging')}
            self.battery['received_at'] = time.time()

    def on_motion(self, event):
        value = {k: event.get(k) for k in ('t','n','ax','ay','az','gx','gy','gz')}
        value['received_at'] = time.time()
        with self.lock:
            self.sample = value
            self.recent_samples.append(value)
            self.count += 1

    def on_event(self, event):
        value = {k:event.get(k) for k in ('t','kind','arg','face')}
        value.update(received_at=time.time(), event_id=uuid.uuid4().hex)
        with self.lock:
            self.events.append(value)

    async def tick(self, band):
        import asyncio
        now = time.monotonic()
        if now - self.last_poll >= .5:
            self.last_poll = now
            try:
                wanted = 0 <= time.time() - LEASE.stat().st_mtime < 3
                rate = 25 if wanted else 0
                controlling = wanted and LEASE.read_text() == 'control'
            except (OSError, UnicodeError):
                wanted = False
                rate = 0
                controlling = False
            self.controlling = controlling
            if wanted != self.enabled or rate != self.rate:
                try:
                    await asyncio.to_thread(band.configure, motion=rate if wanted else False, events=wanted)
                    self.enabled = wanted
                    self.rate = rate
                    self.error = None
                except Exception as error:
                    self.error = type(error).__name__
        if now - self.last_publish >= (.04 if self.enabled and self.controlling else .1 if self.enabled else 1):
            self.last_publish = now
            await asyncio.to_thread(self.publish)

    def publish(self, connected=True):
        with self.lock:
            value = dict(band=VEXO_BAND, generation=self.generation, connected=connected,
                         enabled=self.enabled and connected, battery=self.battery, sample=self.sample if connected else None,
                         samples=self.count, recent_samples=list(self.recent_samples) if connected else [], events=list(self.events), published_at=time.time(), error=self.error)
        try:
            atomic_json(STATE, value)
        except OSError as error:
            self.error = type(error).__name__

    def close(self):
        self.enabled = False
        self.publish(connected=False)


def attitude(sample):
    x,y,z = (float(sample[k]) for k in ('ax','ay','az'))
    return math.degrees(math.atan2(y,z)), math.degrees(math.atan2(-x,math.hypot(y,z)))


class Controller:
    """Tilt is cursor velocity, not absolute pointing. Every session starts disarmed."""
    def __init__(self):
        self.generation = None
        self.neutral = None
        self.armed_until = 0
        self.wave = deque()
        self.last_dismiss = 0
        self.last_sample = None
        self.gain = 1.0
        self.dead_zone = 6.0

    def disarm(self):
        self.armed_until = 0
        self.wave.clear()

    def calibrate(self, sample):
        self.neutral = attitude(sample)
        self.wave.clear()

    def update(self, state, now=None):
        now = time.time() if now is None else now
        if state.get('generation') != self.generation:
            self.generation = state.get('generation')
            self.neutral = None
            self.disarm()
        sample = state.get('sample')
        if (not state.get('connected') or not state.get('enabled') or not sample
                or not 0 <= now-sample.get('received_at',0) < .3):
            self.disarm()
            return {'fresh':False,'armed':False,'dx':0,'dy':0,'dismiss':False}
        roll,pitch = attitude(sample)
        if self.neutral is None:self.calibrate(sample)
        def velocity(angle):
            if abs(angle)<self.dead_zone:return 0
            return max(-12,min(12,math.copysign((abs(angle)-self.dead_zone)*.35*self.gain,angle)))
        result=dict(fresh=True,armed=now<self.armed_until,roll=roll,pitch=pitch,
                    dx=velocity((roll-self.neutral[0]+180)%360-180),dy=velocity(pitch-self.neutral[1]),dismiss=False)
        key=sample.get('received_at')
        if key != self.last_sample:
            self.last_sample=key
            speed=float(sample.get('gz') or 0)
            direction=1 if speed>120 else -1 if speed< -120 else 0
            if direction and (not self.wave or self.wave[-1][1]!=direction):self.wave.append((now,direction))
            while self.wave and now-self.wave[0][0]>.8:self.wave.popleft()
            if len(self.wave)>=3 and now-self.last_dismiss>2:
                result['dismiss']=True
                self.last_dismiss=now
                self.wave.clear()
        return result
