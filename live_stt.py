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
    last_activity_mono: float = 0.0
    last_text: str = ""
    closed_sample: int | None = None


def _sanitize_error_detail(value):
    text = " ".join(str(value or "").split())
    return text[:120] if text else ""


def decode_message(message, ws=None):
    if message.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
        code = getattr(ws, "close_code", None) or (getattr(message, "data", None) if message.type == aiohttp.WSMsgType.CLOSE else None)
        extra = _sanitize_error_detail(getattr(message, "extra", None))
        detail = f" (code={code}{', ' + extra if extra else ''})" if code is not None else ""
        raise ConnectionError(f"Gemini Live connection closed{detail}")
    value = json.loads(message.data)
    if not isinstance(value, dict) or "error" in value:
        err = value.get("error") if isinstance(value, dict) else None
        if isinstance(err, dict):
            code = err.get("code")
            status = _sanitize_error_detail(err.get("status"))
            raise ConnectionError(f"Gemini Live protocol error (code={code}, status={status})")
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
        self.ws = self.task = self.turn = self.closing_turn = None
        self.reconnects = self.partial_updates = self.final_count = 0
        self.has_vad = False
        self.flush_sample = 0
        self.injected_samples = 0
        self.flush_requested = False

    def request_flush(self):
        self.flush_requested = True
        self.wake.set()

    async def flush_now(self, timeout=2.5):
        if not (self.queue or self.speech_pending):
            return
        self.flushed.clear()
        self.request_flush()
        try:
            await asyncio.wait_for(self.flushed.wait(), timeout)
        except TimeoutError:
            pass

    @property
    def pending_age(self):
        starts = [self.queue[0].received_mono] if self.queue else []
        starts += [span.received_mono for span in self.spans
                   if span.end > self.covered_samples and span.speech]
        for active in (self.closing_turn, self.turn):
            if active is not None and not active.finalized:
                starts.append(active.span.received_mono)
        return max(0, time.monotonic() - min(starts)) if starts else None

    @property
    def speech_pending(self):
        return (self.turn is not None or self.closing_turn is not None
                or any(s.speech and s.end > self.covered_samples for s in self.spans))

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
                self.turn = Turn(uuid.uuid4().hex, span, last_activity_mono=time.monotonic())
        return self.turn

    def _expire_closing_turn(self):
        if self.closing_turn is None:
            return
        ct = self.closing_turn
        end_bound = ct.end_sample or ct.closed_sample or self.flush_sample or self.sent_samples
        if not ct.finalized and ct.first_partial:
            duration = max(0, end_bound - max(self.covered_samples, ct.span.start)) / RATE
            self.gap("failed", duration, ct.span)
            self.on_partial({"type": "partial", "utterance_id": ct.identity, "text": ""})
        self.covered_samples = max(self.covered_samples, end_bound)
        self.closing_turn = None

    def _retire_ambient_spans(self, now):
        """Retire expired closing turns with a gap; when server VAD is active and no turn is open, retire 3s-old pre-VAD noise."""
        if self.closing_turn is not None and self.closing_turn.last_activity_mono and now - self.closing_turn.last_activity_mono > 5.0:
            self._expire_closing_turn()
        if not self.has_vad or self.turn is not None or self.closing_turn is not None:
            return
        cutoff_sample = self.sent_samples - RATE * 3
        for span in self.spans:
            if span.end <= cutoff_sample and now - span.received_mono >= 3.0:
                self.covered_samples = max(self.covered_samples, span.end)
            else:
                break

    def handle_message(self, data):
        now = time.monotonic()
        if "error" in data:
            err = data.get("error")
            if isinstance(err, dict):
                raise ConnectionError(f"Gemini Live server error (code={err.get('code')}, status={_sanitize_error_detail(err.get('status'))})")
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
                raw_offset = round(seconds * RATE) if math.isfinite(seconds) and seconds >= 0 else None
                offset = max(0, raw_offset - self.injected_samples) if raw_offset is not None else None
            except (ValueError, OverflowError):
                offset = None
            if activity.get("type") == "ACTIVITY_START":
                span = self._span_at(offset)
                if span is not None:
                    if self.turn is not None and not self.turn.finalized and self.turn.closed_sample is None:
                        if span.start >= self.turn.span.start:
                            self.covered_samples = max(self.covered_samples, self.turn.span.start)
                            self.turn.last_activity_mono = now
                    else:
                        # Frames before ACTIVITY_START contained no speech according to server VAD.
                        self.covered_samples = max(self.covered_samples, span.start)
                        self.turn = Turn(uuid.uuid4().hex, span, last_activity_mono=now)
            elif activity.get("type") == "ACTIVITY_END":
                end = min(offset, self.sent_samples) if offset is not None else self.sent_samples
                # Coverage bookkeeping uses whole sent frames; a VAD boundary can fall
                # inside their trailing silence. Do not leave that frame pending forever.
                end = next((span.end for span in self.spans if span.start < end <= span.end), end)
                if self.closing_turn is not None and (self.turn is None or end <= (self.closing_turn.closed_sample or end)):
                    self.closing_turn.end_sample = end
                    self.closing_turn.last_activity_mono = now
                    if self.closing_turn.finalized:
                        self.covered_samples = max(self.covered_samples, end)
                        self.closing_turn = None
                elif self.turn is not None:
                    if self.turn.closed_sample is None and end <= self.turn.span.start:
                        self.covered_samples = max(self.covered_samples, end)
                    else:
                        self.turn.end_sample = end
                        self.turn.last_activity_mono = now
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
            turn = (self.closing_turn if (self.turn is None and self.closing_turn is not None and not self.closing_turn.finalized)
                    else self._turn())
            if turn is not None and not turn.finalized:
                turn.last_activity_mono = now
                if not turn.first_partial:
                    self.on_latency("receipt_to_first_partial_sec", now - turn.span.received_mono)
                    turn.first_partial = True
                turn.revision += 1
                turn.last_text = interim["text"].strip()
                if turn.last_text.endswith((".", "?", "!")):
                    self.wake.set()
                self.partial_updates += 1
                self.on_partial({"type": "partial", "utterance_id": turn.identity, "revision": turn.revision,
                                 "session_id": turn.span.session_id, "text": interim["text"][:16000]})
        final = content.get("inputTranscription")
        if isinstance(final, dict) and isinstance(final.get("text"), str) and final["text"].strip():
            turn = self.closing_turn if (self.closing_turn is not None and not self.closing_turn.finalized) else self._turn()
            if turn is not None and not turn.finalized:
                turn.last_activity_mono = now
                if len(final["text"]) > 32000:
                    raise ValueError("Gemini Live final transcript exceeds limit")
                saved = self.on_final(final["text"], utterance_id=turn.identity,
                    session_id=turn.span.session_id, sample_offset=turn.span.sample_offset,
                    received_at=turn.span.received_at, audio_start_at=turn.span.audio_start_at)
                self.on_latency("receipt_to_final_sec", now - turn.span.received_mono)
                if saved is not False:
                    self.on_latency("receipt_to_transcript_sec", now - turn.span.received_mono)
                turn.finalized = True
                self.final_count += 1
                self.on_partial({"type": "partial", "utterance_id": turn.identity, "text": ""})
                if turn is self.closing_turn:
                    if turn.end_sample is not None or not self.has_vad:
                        self.covered_samples = max(self.covered_samples,
                            turn.end_sample if turn.end_sample is not None else (turn.closed_sample or self.sent_samples))
                        self.closing_turn = None
                    self.flushed.set()
                elif turn.end_sample is not None or not self.has_vad:
                    self.covered_samples = max(self.covered_samples,
                        turn.end_sample if turn.end_sample is not None else self.sent_samples)
                    self.turn = None
                if self.end_sent:
                    self.flushed.set()
        if content.get("generationComplete"):
            if self.closing_turn is not None:
                self._expire_closing_turn()
                self.flushed.set()
            else:
                if self.turn is not None and self.turn.finalized:
                    self.covered_samples = max(self.covered_samples,
                        self.turn.end_sample if self.turn.end_sample is not None else self.sent_samples)
                    self.turn = None
                if self.turn is not None and not self.turn.finalized and self.turn.end_sample is not None:
                    # A completed segment can legitimately contain no intelligible speech.
                    if self.turn.first_partial:
                        duration = max(0, self.turn.end_sample - max(self.covered_samples, self.turn.span.start)) / RATE
                        self.gap("failed", duration, self.turn.span)
                    self.covered_samples = max(self.covered_samples, self.turn.end_sample)
                    self.on_partial({"type": "partial", "utterance_id": self.turn.identity, "text": ""})
                    self.turn = None
                if self.end_sent:
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
        self._retire_ambient_spans(time.monotonic())
        candidates = [s for s in self.spans if s.end > self.covered_samples and s.speech]
        unfinalized = (self.closing_turn is not None and not self.closing_turn.finalized) or (self.turn is None or not self.turn.finalized)
        if candidates and unfinalized:
            self.gap("failed", sum(s.end - max(s.start, self.covered_samples) for s in candidates) / RATE,
                     candidates[0])
        for active in (self.closing_turn, self.turn):
            if active is not None:
                self.on_partial({"type": "partial", "utterance_id": active.identity, "text": ""})
        self.spans.clear()
        self.turn = self.closing_turn = None
        self.sent_samples = self.covered_samples = 0
        self.has_vad = False
        self.flush_sample = 0
        self.injected_samples = 0
        self.flush_requested = False

    async def _end_audio(self, ws, *, wait=False, active=False, quiet_samples=0):
        if self.connected and not active and self.end_sent:
            return
        self.flushed.clear()
        self.end_sent = True
        self.flush_sample = self.sent_samples
        if self.turn is not None:
            self.turn.closed_sample = self.sent_samples
        if active and self.connected:
            pad_samples = max(0, int(RATE * 0.3) - quiet_samples)
            if pad_samples:
                silence_b64 = base64.b64encode(b"\x00" * (pad_samples * 2)).decode("ascii")
                await asyncio.wait_for(ws.send_json({"realtimeInput": {"audio": {
                    "data": silence_b64, "mimeType": "audio/pcm;rate=16000"}}}), 5)
                self.injected_samples += pad_samples
            await asyncio.wait_for(ws.send_json({"realtimeInput": {"activityEnd": {}}}), 5)
        await asyncio.wait_for(ws.send_json({"realtimeInput": {"audioStreamEnd": True}}), 5)
        if wait and self.connected:
            has_words = self.turn is not None and self.turn.first_partial and not self.turn.finalized
            try:
                await asyncio.wait_for(self.flushed.wait(), 2.0 if has_words else 0.5)
            except TimeoutError:
                pass
            if self.turn is not None and self.turn.closed_sample is not None:
                if not self.turn.finalized:
                    self._expire_closing_turn()
                    self.closing_turn = self.turn
                else:
                    self.covered_samples = max(self.covered_samples,
                        self.turn.end_sample or self.turn.closed_sample or self.flush_sample)
                self.turn = None

    async def _send(self, ws):
        idle_ended, since_end, prior_session, activity_open, quiet_samples = False, 0, None, False, 0
        while True:
            self._retire_ambient_spans(time.monotonic())
            if self.turn is not None and not self.turn.finalized and self.turn.last_activity_mono and time.monotonic() - self.turn.last_activity_mono > 15:
                await self._end_audio(ws, wait=True, active=activity_open, quiet_samples=quiet_samples)
                activity_open, since_end, quiet_samples = False, 0, 0
                if (self.turn is not None and not self.turn.finalized) or (self.closing_turn is not None and not self.closing_turn.finalized):
                    raise TimeoutError("Gemini Live turn stalled after 15s")
            if self.queue:
                packet = self.queue.popleft()
                self.pending_bytes -= len(packet.pcm)
                if not self.stopping and time.monotonic() - packet.received_mono > self.max_age:
                    self.gap("stale", len(packet.pcm) / BPS, packet)
                    continue
                if prior_session is not None and packet.session_id != prior_session:
                    await self._end_audio(ws, wait=True, active=activity_open, quiet_samples=quiet_samples)
                    activity_open, since_end, quiet_samples = False, 0, 0
                prior_session = packet.session_id
                idle_ended = False
                for pos in range(0, len(packet.pcm), FRAME_BYTES):
                    frame = packet.pcm[pos:pos+FRAME_BYTES]
                    n = len(frame) // 2
                    rms = math.sqrt(sum(sample[0] ** 2 for sample in struct.iter_unpack("<h", frame)) / n)
                    speech_floor = max(self.rms_threshold, 220) if self.connected else self.rms_threshold
                    span = Span(self.sent_samples, self.sent_samples + n, packet.received_at,
                        packet.received_mono, packet.received_at - (len(packet.pcm) - pos) / BPS,
                        packet.sample_offset + pos // 2, packet.session_id, rms >= speech_floor)
                    frame_sent = False
                    try:
                        self.end_sent = False
                        if self.connected and not activity_open and span.speech and rms >= speech_floor:
                            self.covered_samples = max(self.covered_samples, span.start)
                            if self.turn is None:
                                self.turn = Turn(uuid.uuid4().hex, span, last_activity_mono=time.monotonic())
                            await asyncio.wait_for(ws.send_json({"realtimeInput": {"activityStart": {}}}), 5)
                            activity_open = True
                        self.spans.append(span)
                        self.sent_samples += n
                        frame_sent = True
                        await asyncio.wait_for(ws.send_json({"realtimeInput": {"audio": {
                            "data": base64.b64encode(frame).decode("ascii"), "mimeType": "audio/pcm;rate=16000"}}}), 5)
                    except BaseException:
                        # If failed before audio send, retry this frame too; otherwise retry the unsent suffix.
                        retry_pos = (pos + len(frame)) if frame_sent else pos
                        remaining = packet.pcm[retry_pos:]
                        if remaining:
                            self.queue.appendleft(Packet(remaining, packet.received_at, packet.received_mono,
                                                         packet.sample_offset + retry_pos // 2, packet.session_id))
                            self.pending_bytes += len(remaining)
                            self._bound_queue()
                        raise
                    self._retire_ambient_spans(time.monotonic())
                    if activity_open or self.speech_pending:
                        since_end += len(frame)
                        quiet_samples = quiet_samples + n if rms < 320 else 0
                    else:
                        since_end, quiet_samples = 0, 0
                    if len(self.spans) > 2048:
                        old = self.spans.popleft()
                        if old.speech and old.end > self.covered_samples:
                            self.gap("failed", (old.end - max(old.start, self.covered_samples)) / RATE, old)
                    # Finalize at a sentence/pause boundary (or 8s max when live, 12s in unit mocks) so finals commit fast without clipping words.
                    has_words = self.turn is not None and self.turn.first_partial
                    sentence_done = (has_words and self.turn.last_text.endswith((".", "?", "!"))
                                     and since_end >= BPS * 3.5)
                    pause_done = (self.connected and has_words
                                  and since_end >= BPS * 3.5 and quiet_samples >= int(RATE * 0.4))
                    empty_done = (self.connected and not has_words
                                  and since_end >= BPS * 2.5 and quiet_samples >= int(RATE * 1.0))
                    max_cut = BPS * 8 if self.connected else BPS * 12
                    if (sentence_done or pause_done or empty_done or since_end >= max_cut) and (activity_open or self.speech_pending):
                        await self._end_audio(ws, wait=True, active=activity_open, quiet_samples=quiet_samples)
                        activity_open, since_end, quiet_samples = False, 0, 0
                continue
            if self.flush_requested:
                self.flush_requested = False
                if activity_open or self.speech_pending:
                    await self._end_audio(ws, wait=True, active=activity_open, quiet_samples=quiet_samples)
                    idle_ended, activity_open, since_end, quiet_samples = True, False, 0, 0
            if self.stopping or self.rotating:
                await self._end_audio(ws, active=activity_open, quiet_samples=quiet_samples)
                return
            if (activity_open and since_end >= BPS * 3.0 and self.turn is not None
                    and self.turn.last_text.endswith((".", "?", "!"))):
                await self._end_audio(ws, wait=True, active=activity_open, quiet_samples=quiet_samples)
                activity_open, since_end, quiet_samples = False, 0, 0
            if (not idle_ended and self.last_packet_mono is not None
                    and time.monotonic() - self.last_packet_mono > min(self.idle_flush, 1.8 if activity_open else self.idle_flush)):
                await self._end_audio(ws, wait=True, active=activity_open, quiet_samples=quiet_samples)
                idle_ended, activity_open, since_end, quiet_samples = True, False, 0, 0
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), .04)
            except TimeoutError:
                pass

    async def _receive(self, ws):
        async for message in ws:
            self.handle_message(decode_message(message, ws))
        code = getattr(ws, "close_code", None)
        raise ConnectionError(f"Gemini Live connection closed{f' (code={code})' if code is not None else ''}")

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
                            "realtimeInputConfig": {"automaticActivityDetection": {"disabled": True}}}}
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
                    if isinstance(error, aiohttp.ClientResponseError):
                        label = f"HTTP {error.status}"
                    elif isinstance(error, (ConnectionError, TimeoutError, ValueError)) and str(error).startswith("Gemini Live"):
                        label = f"{type(error).__name__}: {_sanitize_error_detail(error)}"
                    else:
                        label = type(error).__name__
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
