"""Persistent Gemini Live STT. Partials are ephemeral; only final callbacks persist text."""
import asyncio
import base64
from collections import deque
from dataclasses import dataclass
import json
import math
import struct
import time
import uuid

import aiohttp

RATE, BPS, FRAME_BYTES = 16000, 32000, 3200
LIVE_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"


@dataclass
class Packet:
    pcm: bytes
    received_at: float
    received_mono: float
    sample_offset: int
    session_id: int


@dataclass
class Span:
    start: int
    end: int
    received_at: float
    received_mono: float
    audio_start_at: float
    sample_offset: int
    session_id: int
    speech: bool


@dataclass
class Turn:
    identity: str
    span: Span
    first_partial: bool = False
    revision: int = 0
    finalized: bool = False
    end_sample: int | None = None


def decode_message(message):
    if message.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
        raise ConnectionError("Gemini Live connection closed")
    value = json.loads(message.data)
    if not isinstance(value, dict) or "error" in value:
        raise ConnectionError("Gemini Live protocol error")
    return value


class GeminiLiveSTT:
    def __init__(self, *, session, api_key, model, on_partial, on_final, on_gap, on_latency,
                 on_connection, max_age=45, idle_flush=8, rms_threshold=80):
        self.session, self.api_key, self.model = session, api_key, model
        self.on_partial, self.on_final, self.on_gap = on_partial, on_final, on_gap
        self.on_latency, self.on_connection = on_latency, on_connection
        self.max_age, self.idle_flush, self.rms_threshold = max_age, idle_flush, rms_threshold
        self.queue, self.spans = deque(), deque()
        self.wake, self.flushed = asyncio.Event(), asyncio.Event()
        self.stop_event = asyncio.Event()
        self.pending_bytes = 0
        self.total_input_samples = self.session_id = self.sent_samples = self.covered_samples = 0
        self.last_packet_mono = None
        self.connected = self.stopping = self.rotating = self.end_sent = False
        self.ws = self.task = self.turn = None
        self.reconnects = self.partial_updates = self.final_count = 0
        self.has_vad = False
        self.flush_sample = 0

    @property
    def pending_age(self):
        starts = [self.queue[0].received_mono] if self.queue else []
        starts += [span.received_mono for span in self.spans
                   if span.end > self.covered_samples and span.speech]
        if self.turn is not None and not self.turn.finalized:
            starts.append(self.turn.span.received_mono)
        return max(0, time.monotonic() - min(starts)) if starts else None

    @property
    def speech_pending(self):
        return self.turn is not None or any(s.speech and s.end > self.covered_samples for s in self.spans)

    def gap(self, reason, seconds, item):
        if seconds > 0:
            self.on_gap(reason, seconds, session_id=item.session_id, sample_offset=item.sample_offset,
                        audio_start_at=(item.received_at - len(item.pcm) / BPS if isinstance(item, Packet)
                                        else item.audio_start_at), received_at=item.received_at)

    def feed(self, raw, received_at, received_mono):
        if self.stopping:
            raise RuntimeError("Live STT is stopping")
        if not raw:
            return
        if self.last_packet_mono is None or received_mono - self.last_packet_mono > self.idle_flush:
            self.session_id += 1
        self.last_packet_mono = received_mono
        packet = Packet(raw, received_at, received_mono, self.total_input_samples, self.session_id)
        self.total_input_samples += len(raw) // 2
        self.queue.append(packet)
        self.pending_bytes += len(raw)
        self._bound_queue()
        self.wake.set()

    def _bound_queue(self):
        # Bound both bytes (10 seconds) and metadata count; favor the newest audio.
        while self.pending_bytes > BPS * 10 or len(self.queue) > 256:
            old = self.queue[0]
            remove = min(len(old.pcm), max(self.pending_bytes - BPS * 10, 0))
            if len(self.queue) > 256:
                remove = len(old.pcm)
            self.gap("queue_evicted", remove / BPS, old)
            self.pending_bytes -= remove
            if remove == len(old.pcm):
                self.queue.popleft()
            else:
                old.pcm = old.pcm[remove:]
                old.sample_offset += remove // 2

    def _span_at(self, offset=None):
        if offset is not None:
            for span in self.spans:
                if span.start <= offset < span.end:
                    return span
        return next((s for s in self.spans if s.end > self.covered_samples and s.speech),
                    self.spans[-1] if self.spans else None)

    def _turn(self):
        if self.turn is None:
            span = self._span_at()
            if span is not None:
                self.turn = Turn(uuid.uuid4().hex, span)
        return self.turn

    def handle_message(self, data):
        if "error" in data:
            raise ConnectionError("Gemini Live server error")
        if "goAway" in data:
            self.rotating = True
            self.wake.set()
        activity = data.get("voiceActivity")
        if isinstance(activity, dict):
            self.has_vad = True
            offset = activity.get("audioOffset", "")
            try:
                seconds = float(str(offset).removesuffix("s"))
                offset = round(seconds * RATE) if math.isfinite(seconds) and seconds >= 0 else None
            except (ValueError, OverflowError):
                offset = None
            if activity.get("type") == "ACTIVITY_START":
                span = self._span_at(offset)
                if span is not None:
                    self.turn = Turn(uuid.uuid4().hex, span)
            elif activity.get("type") == "ACTIVITY_END":
                end = min(offset, self.sent_samples) if offset is not None else self.sent_samples
                # Coverage bookkeeping uses whole sent frames; a VAD boundary can fall
                # inside their trailing silence. Do not leave that frame pending forever.
                end = next((span.end for span in self.spans if span.start < end <= span.end), end)
                if self.turn is not None:
                    self.turn.end_sample = end
                    # Some servers send VAD end before the finalized text. Preserve its receipt baseline.
                    if self.turn.finalized:
                        self.covered_samples = max(self.covered_samples, end)
                        self.turn = None
                else:
                    self.covered_samples = max(self.covered_samples, end)
        content = data.get("serverContent", {})
        if not isinstance(content, dict):
            raise ValueError("Malformed Gemini Live content")
        interim = content.get("interimInputTranscription")
        if isinstance(interim, dict) and isinstance(interim.get("text"), str) and interim["text"].strip():
            turn = self._turn()
            if turn is not None and not turn.finalized:
                if not turn.first_partial:
                    self.on_latency("receipt_to_first_partial_sec", time.monotonic() - turn.span.received_mono)
                    turn.first_partial = True
                turn.revision += 1
                self.partial_updates += 1
                self.on_partial({"type": "partial", "utterance_id": turn.identity, "revision": turn.revision,
                                 "session_id": turn.span.session_id, "text": interim["text"][:16000]})
        final = content.get("inputTranscription")
        if isinstance(final, dict) and isinstance(final.get("text"), str) and final["text"].strip():
            turn = self._turn()
            if turn is not None and not turn.finalized:
                if len(final["text"]) > 32000:
                    raise ValueError("Gemini Live final transcript exceeds limit")
                saved = self.on_final(final["text"], utterance_id=turn.identity,
                    session_id=turn.span.session_id, sample_offset=turn.span.sample_offset,
                    received_at=turn.span.received_at, audio_start_at=turn.span.audio_start_at)
                self.on_latency("receipt_to_final_sec", time.monotonic() - turn.span.received_mono)
                if saved is not False:
                    self.on_latency("receipt_to_transcript_sec", time.monotonic() - turn.span.received_mono)
                turn.finalized = True
                self.final_count += 1
                self.on_partial({"type": "partial", "utterance_id": turn.identity, "text": ""})
                if turn.end_sample is not None or not self.has_vad:
                    self.covered_samples = max(self.covered_samples,
                        turn.end_sample if turn.end_sample is not None else self.sent_samples)
                    self.turn = None
        if content.get("generationComplete") and self.turn is not None and not self.turn.finalized and self.turn.end_sample is not None:
            # A completed segment can legitimately contain no intelligible speech.
            if self.turn.first_partial:
                duration = max(0, self.turn.end_sample - max(self.covered_samples, self.turn.span.start)) / RATE
                self.gap("failed", duration, self.turn.span)
            self.covered_samples = max(self.covered_samples, self.turn.end_sample)
            self.on_partial({"type": "partial", "utterance_id": self.turn.identity, "text": ""})
            self.turn = None
        if content.get("generationComplete") and self.end_sent:
            if self.turn is not None and not self.turn.finalized:
                # Never promote an unfinished hypothesis just because an audio segment ended.
                if self.turn.first_partial:
                    spans = [s for s in self.spans if s.end > self.covered_samples and s.speech]
                    if spans:
                        self.gap("failed", sum(s.end-max(s.start,self.covered_samples) for s in spans)/RATE, spans[0])
                self.on_partial({"type": "partial", "utterance_id": self.turn.identity, "text": ""})
            self.turn = None
            self.covered_samples = max(self.covered_samples, self.flush_sample)
            self.flushed.set()
        # Keep bounded receipt metadata for late VAD/final offsets, even after coverage is finalized.

    def _connection_gap(self):
        candidates = [s for s in self.spans if s.end > self.covered_samples and s.speech]
        if candidates and (self.turn is None or not self.turn.finalized):
            self.gap("failed", sum(s.end - max(s.start, self.covered_samples) for s in candidates) / RATE,
                     candidates[0])
        if self.turn is not None:
            self.on_partial({"type": "partial", "utterance_id": self.turn.identity, "text": ""})
        self.spans.clear()
        self.turn = None
        self.sent_samples = self.covered_samples = 0
        self.has_vad = False
        self.flush_sample = 0

    async def _end_audio(self, ws):
        self.flushed.clear()
        self.end_sent = True
        self.flush_sample = self.sent_samples
        await asyncio.wait_for(ws.send_json({"realtimeInput": {"audioStreamEnd": True}}), 5)

    async def _send(self, ws):
        idle_ended, since_end, prior_session = False, 0, None
        while True:
            if self.queue:
                packet = self.queue.popleft()
                self.pending_bytes -= len(packet.pcm)
                if not self.stopping and time.monotonic() - packet.received_mono > self.max_age:
                    self.gap("stale", len(packet.pcm) / BPS, packet)
                    continue
                if prior_session is not None and packet.session_id != prior_session:
                    await self._end_audio(ws)
                prior_session = packet.session_id
                idle_ended = False
                for pos in range(0, len(packet.pcm), FRAME_BYTES):
                    frame = packet.pcm[pos:pos+FRAME_BYTES]
                    n = len(frame) // 2
                    rms = math.sqrt(sum(sample[0] ** 2 for sample in struct.iter_unpack("<h", frame)) / n)
                    span = Span(self.sent_samples, self.sent_samples + n, packet.received_at,
                        packet.received_mono, packet.received_at - (len(packet.pcm) - pos) / BPS,
                        packet.sample_offset + pos // 2, packet.session_id, rms >= self.rms_threshold)
                    self.spans.append(span)
                    self.sent_samples += n
                    try:
                        self.end_sent = False
                        await asyncio.wait_for(ws.send_json({"realtimeInput": {"audio": {
                            "data": base64.b64encode(frame).decode("ascii"), "mimeType": "audio/pcm;rate=16000"}}}), 5)
                    except BaseException:
                        # The attempted frame is uncertain; the unsent suffix can safely retry.
                        remaining = packet.pcm[pos+len(frame):]
                        if remaining:
                            self.queue.appendleft(Packet(remaining, packet.received_at, packet.received_mono,
                                                         packet.sample_offset + (pos+len(frame)) // 2, packet.session_id))
                            self.pending_bytes += len(remaining)
                            self._bound_queue()
                        raise
                    since_end = since_end + len(frame) if self.speech_pending else 0
                    if len(self.spans) > 2048:
                        old = self.spans.popleft()
                        if old.speech and old.end > self.covered_samples:
                            self.gap("failed", (old.end - max(old.start, self.covered_samples)) / RATE, old)
                    # Finalize a long uninterrupted utterance without waiting to batch its audio.
                    if since_end >= BPS * 30 and self.speech_pending:
                        await self._end_audio(ws)
                        since_end = 0
                continue
            if self.stopping or self.rotating:
                await self._end_audio(ws)
                return
            if (not idle_ended and self.last_packet_mono is not None
                    and time.monotonic() - self.last_packet_mono > self.idle_flush):
                await self._end_audio(ws)
                idle_ended, since_end = True, 0
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), .2)
            except TimeoutError:
                pass

    async def _receive(self, ws):
        async for message in ws:
            self.handle_message(decode_message(message))
        raise ConnectionError("Gemini Live connection closed")

    async def run(self):
        self.task = asyncio.current_task()
        delay = 1
        try:
            while not self.stopping or self.queue:
                send = receive = None
                try:
                    async with self.session.ws_connect(LIVE_URL, headers={"x-goog-api-key": self.api_key},
                            timeout=aiohttp.ClientWSTimeout(ws_close=5), max_msg_size=2*1024*1024,
                            heartbeat=30) as ws:
                        self.ws = ws
                        setup = {"setup": {"model": "models/" + self.model,
                            "generationConfig": {"responseModalities": ["TEXT"]},
                            "inputAudioTranscription": {"mode": "VERBATIM"},
                            "realtimeInputConfig": {"automaticActivityDetection": {"silenceDurationMs": 400}}}}
                        await asyncio.wait_for(ws.send_json(setup), 10)
                        first = decode_message(await asyncio.wait_for(ws.receive(), 15))
                        if "setupComplete" not in first:
                            raise ConnectionError("Gemini Live setup failed")
                        self.connected, self.rotating = True, False
                        delay = 1
                        self.on_connection(True, None)
                        send, receive = asyncio.create_task(self._send(ws)), asyncio.create_task(self._receive(ws))
                        done, _ = await asyncio.wait((send, receive), return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            await task
                        # Sender ends only on shutdown/rotation; give final recognition a bounded drain.
                        try:
                            await asyncio.wait_for(self.flushed.wait(), 5 if self.speech_pending else .3)
                        except TimeoutError:
                            pass
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    label = f"HTTP {error.status}" if isinstance(error, aiohttp.ClientResponseError) else type(error).__name__
                    self.on_connection(False, label)
                finally:
                    self.connected = False
                    tasks = [t for t in (send, receive) if t is not None]
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    self.ws = None
                    self._connection_gap()
                if self.stopping:
                    break
                self.reconnects += 1
                try:
                    # New uploads must not bypass rate-limit/network reconnect backoff.
                    await asyncio.wait_for(self.stop_event.wait(), delay)
                except TimeoutError:
                    pass
                delay = min(15, delay * 2)
        finally:
            self.connected = False

    async def close(self):
        self.stopping = True
        self.wake.set()
        self.stop_event.set()
        if self.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.task), 20)
            except TimeoutError:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
        while self.queue:
            packet = self.queue.popleft()
            self.gap("failed", len(packet.pcm) / BPS, packet)
        self.pending_bytes = 0
