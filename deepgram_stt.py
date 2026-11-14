"""Optional English Deepgram streaming adapter; no credentials or audio are logged.

Each connection carries one contiguous source timeline. CloseStream + final
Metadata is the completion barrier; is_final segments are persisted immediately.
Unresolved PCM remains bounded in memory and is replayed after connection loss.
"""
import asyncio
from collections import deque
import copy
from dataclasses import dataclass, field
import json
import logging
import math
import time
from urllib.parse import urlencode
import uuid

import aiohttp

from common import (RATE, BPS, PACKET_MAX_BYTES, QUEUE_MAX_BYTES,
                    QUEUE_MAX_PACKETS, REPLAY_MAX_BYTES, SHUTDOWN_TIMEOUT)

logger = logging.getLogger("bridge")
MAX_SEND_RATE = 1.2
KEEPALIVE_SECONDS = 3.0
MAX_UNACKNOWLEDGED_BYTES = BPS * 30
MAX_RETAINED_PACKETS = 4096
MAX_PENDING_FINALS = 128
DRAIN_TIMEOUT = 15.0
RECONNECT_INITIAL_SECONDS = 1.0
PROVIDER_CODES = {"DATA-0000", "NET-0000", "NET-0001", "NET-0002"}
RESULT_FAILURES = {
    "Invalid Deepgram result timing": "invalid_result_timing",
    "Deepgram result outside source timeline": "result_outside_timeline",
    "Deepgram word outside source timeline": "word_outside_timeline",
    "Deepgram result did not match retained source audio": "missing_source_audio",
    "Invalid Deepgram transcript": "invalid_transcript",
    "Invalid Deepgram word": "invalid_word",
    "Overlapping Deepgram text requires word timing": "overlap_without_word_timing",
}


@dataclass
class _Packet:
    pcm: bytes
    received_at: float
    received_mono: float
    sample_offset: int
    session_id: int
    audio_start_at: float

    @property
    def end(self):
        return self.sample_offset + len(self.pcm) // 2

    def slice(self, start, end=None):
        end = self.end if end is None else end
        return _Packet(self.pcm[(start-self.sample_offset)*2:(end-self.sample_offset)*2],
            self.received_at, self.received_mono, start, self.session_id,
            self.audio_start_at + (start-self.sample_offset)/RATE)


@dataclass
class _Epoch:
    identity: str = field(default_factory=lambda: uuid.uuid4().hex)
    start: int | None = None
    end: int = 0
    session: int | None = None
    next_send_at: float = 0.0
    closing: bool = False
    complete: bool = False
    revision: int = 0


class DeepgramSTT:
    def __init__(self, *, api_key, on_partial, on_final, on_gap, on_latency,
                 on_connection, model="nova-3", language="en", endpointing_ms=300,
                 diarize=False, max_age=45, idle_flush=8, session_factory=None, enrich_speaker=None, keyterms=()):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("A Deepgram API key is required")
        if model != "nova-3" or language.lower() not in ("en", "en-us"):
            raise ValueError("Deepgram supports Nova-3 English only")
        if (isinstance(endpointing_ms, bool) or not isinstance(endpointing_ms, int)
                or not 1 <= endpointing_ms <= 5000 or not isinstance(diarize, bool)):
            raise ValueError("Invalid Deepgram endpointing or diarization setting")
        if not all(math.isfinite(value) and value > 0 for value in (max_age, idle_flush)):
            raise ValueError("Deepgram audio age and idle timeout must be positive")
        self._api_key = api_key.strip()
        self.model, self.language, self.endpointing_ms = model, language.lower(), endpointing_ms
        self.diarize = diarize
        self.enrich_speaker = enrich_speaker
        self.keyterms = tuple(keyterms)
        self.on_partial, self.on_final, self.on_gap = on_partial, on_final, on_gap
        self.on_latency, self.on_connection = on_latency, on_connection
        self.max_age, self.idle_flush = max_age, idle_flush
        self._session_factory = session_factory or aiohttp.ClientSession
        self.queue, self.retained, self.protected_pcm, self.controls = deque(), deque(), deque(), deque()
        self.pending_finals = deque()
        self.pending_bytes = self.total_input_samples = self.session_id = 0
        self.committed_end = self.release_end = 0
        self.connected = self.stopping = self._speech_pending = False
        self._speech_started_sample = None
        self._connecting = self._retrying = False
        self.reconnects = self.partial_updates = self.final_count = self.turns_without_text = 0
        self.no_text_causes, self.gap_causes = {}, {}
        self.save_error = self.last_flush_completed = self.last_partial = None
        self.last_packet_mono = None
        self.resolved_model = None
        self.task = None
        self.wake, self.stop_event, self.save_ready = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.save_ready.set()
        self.flush_waiters = {}
        self._flush_users = {}

    @property
    def ready(self):
        """Healthy intake runtime; an idle socket differs from a provider failure."""
        return (self.task is not None and not self.task.done()
                and not self.stopping and not self._retrying)

    @property
    def pending_age(self):
        times = [items[0].received_mono for items in (self.queue, self.retained, self.protected_pcm) if items]
        return max(0, time.monotonic()-min(times)) if times else None

    @property
    def connection_state(self):
        if self.task is not None and self.task.done() or self.stopping and not self.connected:
            return "stopped"
        if self.connected:
            return "connected"
        if self._retrying and (self.queue or self.retained or self.protected_pcm):
            return "reconnecting"
        return "connecting" if self._connecting else "idle"

    @property
    def speech_pending(self):
        return self._speech_pending or bool(self.queue) or bool(self.pending_finals)

    @property
    def pending_final_count(self):
        return len(self.pending_finals)

    def connection_url(self):
        options = {
            "model": self.model, "language": self.language, "encoding": "linear16",
            "sample_rate": RATE, "channels": 1, "interim_results": "true",
            "endpointing": self.endpointing_ms, "punctuate": "true", "smart_format": "true",
            "no_delay": "true", "diarize": str(self.diarize).lower(), "vad_events": "true",
            "mip_opt_out": "true", "redact_usage": "true"}
        return "wss://api.deepgram.com/v1/listen?" + urlencode([*options.items(), *(("keyterm", term) for term in self.keyterms)])

    def new_session(self):
        self.request_flush()
        self.session_id += 1

    def feed(self, raw, received_at, received_mono):
        if self.stopping:
            raise RuntimeError("Deepgram transcription is stopping")
        if len(raw) % 2:
            raise ValueError("Expected complete PCM16 samples")
        if not raw:
            return
        if self.last_packet_mono is None or received_mono-self.last_packet_mono > self.idle_flush:
            self.session_id += 1
        self.last_packet_mono = received_mono
        self.queue.append(_Packet(raw, received_at, received_mono, self.total_input_samples,
                                  self.session_id, received_at-len(raw)/BPS))
        self.total_input_samples += len(raw)//2
        self.pending_bytes += len(raw)
        self._bound_queue()
        self.wake.set()

    def _gap(self, reason, packet, detail):
        if not packet.pcm:
            return
        self.gap_causes[detail] = self.gap_causes.get(detail, 0) + 1
        self.on_gap(reason, len(packet.pcm)/BPS, session_id=packet.session_id,
            sample_offset=packet.sample_offset, received_at=packet.received_at,
            audio_start_at=packet.audio_start_at, detail=detail)

    def _latency(self, name, seconds):
        if math.isfinite(seconds):
            self.on_latency(name, max(0, seconds))

    def _bound_queue(self):
        while self.pending_bytes > QUEUE_MAX_BYTES or len(self.queue) > QUEUE_MAX_PACKETS:
            packet = self.queue[0]
            count = (len(packet.pcm) if len(self.queue) > QUEUE_MAX_PACKETS else
                     min(len(packet.pcm), self.pending_bytes-QUEUE_MAX_BYTES))
            end = packet.sample_offset + count//2
            self._gap("queue_evicted", packet.slice(packet.sample_offset, end), "deepgram_unsent_queue_evicted")
            self.pending_bytes -= count
            self.queue.popleft()
            if end < packet.end:
                self.queue.appendleft(packet.slice(end))

    def request_flush(self):
        end = self.total_input_samples
        if self.controls and (self.controls[-1][1] == end or len(self.controls) >= QUEUE_MAX_PACKETS):
            identity, _ = self.controls.pop()
        else:
            identity = uuid.uuid4().hex
        self.controls.append((identity, end))
        self.wake.set()
        return identity

    async def flush_now(self, timeout=5):
        identity = self.request_flush()
        waiter = self.flush_waiters.setdefault(identity, asyncio.Event())
        self._flush_users[identity] = self._flush_users.get(identity, 0) + 1
        try:
            async with asyncio.timeout(timeout):
                await waiter.wait()
                await self.save_ready.wait()
            self.last_flush_completed = True
            return True
        except TimeoutError:
            self.last_flush_completed = False
            logger.warning("Deepgram transcription flush timed out; accepted audio remains pending")
            return False
        finally:
            self._flush_users[identity] -= 1
            if not self._flush_users[identity]:
                self._flush_users.pop(identity, None)
                self.flush_waiters.pop(identity, None)

    def _complete_controls(self, end):
        while self.controls and self.controls[0][1] <= end:
            identity, _ = self.controls.popleft()
            if identity in self.flush_waiters:
                self.flush_waiters[identity].set()

    def _release(self, end):
        self.release_end = max(self.release_end, end)
        end = min(self.release_end, self.pending_finals[0]["sample_offset"]) if self.pending_finals else self.release_end
        self.committed_end = max(self.committed_end, end)
        for items in (self.queue, self.retained, self.protected_pcm):
            while items and items[0].end <= end:
                packet = items.popleft()
                if items is self.queue:
                    self.pending_bytes -= len(packet.pcm)
            if items and items[0].sample_offset < end:
                packet = items.popleft()
                items.appendleft(packet.slice(end))
                if items is self.queue:
                    self.pending_bytes -= (end-packet.sample_offset)*2

    def retry_saves(self):
        while self.pending_finals:
            event = self.pending_finals[0]
            fields = {key: value for key, value in event.items()
                      if key not in ("text", "_received_mono", "_final_received_mono")}
            fields["recognition"] = copy.deepcopy(event["recognition"])
            try:
                saved = self.on_final(event["text"], **fields)
            except Exception as error:
                saved = False
                self.save_error = type(error).__name__
            if saved is False:
                self.save_error = self.save_error or "Transcript save failed"
                self.save_ready.clear()
                return
            self.pending_finals.popleft()
            self.final_count += 1
            self._release(event["sample_end_offset"])
            saved_at = time.monotonic()
            self._latency("receipt_to_transcript_sec", saved_at-event["_received_mono"])
            self._latency("final_to_persist_sec", saved_at-event["_final_received_mono"])
        self.save_error = None
        self.save_ready.set()
        self.wake.set()

    async def _save_loop(self):
        while True:
            await asyncio.sleep(.5)
            if self.pending_finals:
                self.retry_saves()

    def _source(self, start):
        for packet in self.retained:
            if packet.sample_offset <= start < packet.end:
                return packet.slice(start)
        raise ValueError("Deepgram result did not match retained source audio")

    def _word_end_latency(self, end, received_mono, *, final):
        """Provider-word timing mapped to bridge receipt; not acoustic hardware lag.

        The interpolated estimate assumes a packet's last sample corresponds to
        its receipt. The separate packet metric uses the observed receipt only,
        and both include any provider, orchestration and return-transport delay.
        """
        if end is None:
            return
        try:
            packet = self._source(end-1)
        except ValueError:
            return  # Missing timing authority is unknown, never a fabricated zero.
        word_end_mono = (packet.received_mono + packet.audio_start_at-packet.received_at
                         + (end-packet.sample_offset)/RATE)
        name = "provider_word_end_to_final_sec" if final else "provider_word_end_to_partial_sec"
        self._latency(name, received_mono-word_end_mono)
        if final:
            self._latency("final_word_packet_to_result_sec", received_mono-packet.received_mono)

    @staticmethod
    def _seconds(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid Deepgram result timing")
        return value

    def _clear_partial(self, identity=None):
        if self.last_partial is not None and (identity is None or identity == self.last_partial["utterance_id"]):
            self.on_partial({**self.last_partial, "text": ""})
            self.last_partial = None

    def _speech_final(self, event, end):
        # Endpoint markers can repeat an already committed segment. They still
        # close its speech state, but cannot close a newer active hypothesis.
        if event.get("speech_final") is True and (self.last_partial is None
                or self.last_partial["sample_end_offset"] <= end) and (
                self._speech_started_sample is None or self._speech_started_sample <= end):
            self._speech_pending = False
            self._speech_started_sample = None

    @staticmethod
    def _provider_failure(code=None):
        error = ConnectionError("Deepgram provider stream failed")
        error.provider_code = code if code in PROVIDER_CODES else None
        return error

    def _handle_result(self, event, epoch):
        received_mono = event.get("_omi_received_mono", time.monotonic())
        if epoch.start is None:
            return
        start = epoch.start + round(self._seconds(event.get("start"))*RATE)
        end = min(epoch.end, start + round(self._seconds(event.get("duration"))*RATE))
        if start < epoch.start or start > epoch.end or end < start:
            raise ValueError("Deepgram result outside source timeline")
        alternatives = event.get("channel", {}).get("alternatives", [])
        alternative = alternatives[0] if alternatives else {}
        text = alternative.get("transcript", "")
        if not isinstance(text, str) or len(text) > 32000:
            raise ValueError("Invalid Deepgram transcript")
        text = " ".join(text.split())
        watermark = self.pending_finals[-1]["sample_end_offset"] if self.pending_finals else self.committed_end
        if end <= watermark:
            self._speech_final(event, end)
            return
        words = []
        provider_words = alternative.get("words", [])
        for word in provider_words[:512]:
            word_start = epoch.start + round(self._seconds(word.get("start"))*RATE)
            word_end = epoch.start + round(self._seconds(word.get("end"))*RATE)
            if not epoch.start <= word_start <= word_end <= epoch.end + RATE//100:
                raise ValueError("Deepgram word outside source timeline")
            if word_end <= watermark:
                continue
            label = word.get("punctuated_word", word.get("word", ""))
            if not isinstance(label, str):
                raise ValueError("Invalid Deepgram word")
            mapped_start = min(epoch.end, max(word_start, watermark))
            mapped_end = min(word_end, epoch.end)
            if mapped_end <= mapped_start:
                continue  # Timing slop may put a word wholly beyond sent audio.
            item = {"word": label[:512], "sample_offset": mapped_start,
                    "sample_end_offset": mapped_end}
            confidence = word.get("confidence")
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and math.isfinite(confidence):
                item["confidence"] = max(0, min(1, confidence))
            if isinstance(word.get("speaker"), int) and not isinstance(word["speaker"], bool) and word["speaker"] >= 0:
                item["speaker"] = word["speaker"]
            if self.enrich_speaker is not None and isinstance(word.get("_voice_identity"), dict):
                item["voice_identity"] = dict(word["_voice_identity"])
            words.append(item)
        last_word_end = None
        if provider_words and len(provider_words) <= 512:
            reported_end = epoch.start + round(self._seconds(provider_words[-1].get("end"))*RATE)
            if watermark < reported_end <= epoch.end:
                last_word_end = reported_end
        if start < watermark:
            if text and not alternative.get("words"):
                raise ValueError("Overlapping Deepgram text requires word timing")
            start = watermark
            text = " ".join(word["word"] for word in words)
        identity = f"deepgram-{epoch.identity}-{start}"
        metadata = event.get("metadata") or {}
        model_info = metadata.get("model_info") or {}
        recognition = {"provider": "deepgram", "requested_model": self.model,
                       "connection_epoch": epoch.identity, "language": self.language, "words": words}
        for key, value in (("resolved_model", model_info.get("name")),
                           ("model_version", model_info.get("version")), ("request_id", metadata.get("request_id"))):
            if isinstance(value, str):
                recognition[key] = value[:256]
        self.resolved_model = recognition.get("resolved_model", self.resolved_model)
        source = self._source(start) if start < end else None
        if event.get("is_final") is True:
            if text and source is not None:
                if len(self.pending_finals) >= MAX_PENDING_FINALS:
                    raise BufferError("Deepgram pending final capacity reached")
                self.pending_finals.append({"text": text, "utterance_id": identity,
                    "session_id": epoch.session, "sample_offset": start, "sample_end_offset": end,
                    "received_at": source.received_at, "audio_start_at": source.audio_start_at,
                    "_received_mono": source.received_mono, "_final_received_mono": received_mono,
                    "recognition": recognition})
                self._word_end_latency(last_word_end, received_mono, final=True)
                self._release(end)
                self._latency("receipt_to_final_sec", received_mono-source.received_mono)
                self.retry_saves()
            elif not text:
                self._release(end)
            self._clear_partial(identity)
            self._speech_final(event, end)
        elif text and source is not None:
            epoch.revision += 1
            first = self.last_partial is None or identity != self.last_partial["utterance_id"]
            partial = {"type": "partial", "utterance_id": identity, "session_id": epoch.session,
                "revision": epoch.revision, "text": text, "audio_start_at": source.audio_start_at,
                "sample_offset": start, "sample_end_offset": end}
            self.last_partial = partial
            self.partial_updates += 1
            self._speech_pending = True
            self._word_end_latency(last_word_end, received_mono, final=False)
            if first:
                self._latency("receipt_to_first_partial_sec", received_mono-source.received_mono)
            self.on_partial(partial)

    async def _send(self, ws, epoch):
        while True:
            self.wake.clear()
            await self.save_ready.wait()
            boundary = self.controls[0][1] if self.controls else None
            next_packet = self.queue[0] if self.queue else None
            close = (boundary is not None and (next_packet is None or next_packet.sample_offset >= boundary))
            close |= next_packet is None and epoch.start is None
            if epoch.start is not None:
                close |= bool(next_packet and (next_packet.sample_offset != epoch.end or next_packet.session_id != epoch.session))
                close |= (sum(len(packet.pcm) for packet in self.retained) >= MAX_UNACKNOWLEDGED_BYTES
                          or len(self.retained) >= MAX_RETAINED_PACKETS)
                close |= bool(not next_packet and (self.stopping or (
                    self.last_packet_mono is not None and time.monotonic()-self.last_packet_mono >= self.idle_flush)))
            if close:
                epoch.closing = True
                await asyncio.wait_for(ws.send_json({"type": "CloseStream"}), 5)
                return
            if next_packet is not None:
                if time.monotonic()-next_packet.received_mono > self.max_age:
                    self.queue.popleft()
                    self.pending_bytes -= len(next_packet.pcm)
                    self._gap("stale", next_packet, "deepgram_unsent_audio_stale")
                    continue
                if epoch.start is not None:
                    wait = epoch.next_send_at-time.monotonic()
                    if wait > 0:
                        await asyncio.sleep(wait)
                        continue  # Re-read intake after pacing: eviction can replace its head.
                count = min(PACKET_MAX_BYTES, len(next_packet.pcm))
                if boundary is not None:
                    count = min(count, (boundary-next_packet.sample_offset)*2)
                chunk_end = next_packet.sample_offset+count//2
                chunk = next_packet.slice(next_packet.sample_offset, chunk_end)
                self.queue.popleft()
                if chunk_end < next_packet.end:
                    self.queue.appendleft(next_packet.slice(chunk_end))
                self.pending_bytes -= count
                if epoch.start is None:
                    epoch.start = chunk.sample_offset
                    epoch.session = chunk.session_id
                # Idle time never earns credit to burst future audio faster.
                epoch.next_send_at = max(time.monotonic(), epoch.next_send_at) + count/BPS/MAX_SEND_RATE
                epoch.end = chunk.end
                self.retained.append(chunk)  # An uncertain send remains replayable.
                await asyncio.wait_for(ws.send_bytes(chunk.pcm), 5)
                self._latency("receipt_to_send_sec", time.monotonic()-chunk.received_mono)
                if sum(len(packet.pcm) for packet in self.retained) > REPLAY_MAX_BYTES:
                    raise TimeoutError("Deepgram did not release source audio")
                continue
            try:
                await asyncio.wait_for(self.wake.wait(), KEEPALIVE_SECONDS)
            except TimeoutError:
                await asyncio.wait_for(ws.send_json({"type": "KeepAlive"}), 5)

    async def _annotate_speakers(self, event, epoch):
        if self.enrich_speaker is None or not self.diarize or event.get("is_final") is not True or epoch.start is None:
            return
        alternatives=(event.get("channel") or {}).get("alternatives") or []
        words=alternatives[0].get("words", []) if alternatives else []
        if len(words)>512:return
        groups=[]
        for word in words:
            speaker=word.get("speaker")
            if not groups or groups[-1][0].get("speaker")!=speaker:groups.append([])
            groups[-1].append(word)
        for word in words:word["_voice_identity"]={"status":"uncertain","reason":"insufficient_or_missing_audio"}
        try:
            async with asyncio.timeout(4):
                for group in groups[:8]:
                    speaker=group[0].get("speaker")
                    if not isinstance(speaker,int) or isinstance(speaker,bool):continue
                    start=epoch.start+round(self._seconds(group[0].get("start"))*RATE)
                    end=min(epoch.start+round(self._seconds(group[-1].get("end"))*RATE),start+8*RATE)
                    if not epoch.start<=start<end<=epoch.end or end-start<RATE*1.5:continue
                    cursor=start; pieces=[]
                    for packet in self.retained:
                        if packet.session_id!=epoch.session or packet.end<=cursor:continue
                        if packet.sample_offset>cursor:break
                        finish=min(end,packet.end)
                        pieces.append(packet.pcm[(cursor-packet.sample_offset)*2:(finish-packet.sample_offset)*2])
                        cursor=finish
                        if cursor==end:break
                    if cursor!=end:continue
                    identity=await self.enrich_speaker(b"".join(pieces))
                    for word in group:word["_voice_identity"]=identity
        except Exception:
            pass # Verification failure stays uncertain and never drops recognized words.

    async def _receive(self, ws, epoch):
        while True:
            try:
                message = await asyncio.wait_for(ws.receive(), 5)
            except TimeoutError:
                if self.pending_age is not None and self.pending_age > self.max_age and not self.pending_finals:
                    raise TimeoutError("Deepgram source audio stalled")
                continue
            if message.type == aiohttp.WSMsgType.TEXT:
                event = json.loads(message.data)
                kind = event.get("type")
                if kind == "Results":
                    event["_omi_received_mono"] = time.monotonic()
                    await self._annotate_speakers(event, epoch)
                    self._handle_result(event, epoch)
                elif kind == "SpeechStarted":
                    self._speech_pending = True
                    timestamp = event.get("timestamp")
                    if (epoch.start is not None and isinstance(timestamp, (int, float))
                            and not isinstance(timestamp, bool) and math.isfinite(timestamp) and timestamp >= 0):
                        self._speech_started_sample = epoch.start + round(timestamp*RATE)
                elif kind == "Metadata":
                    duration = self._seconds(event.get("duration"))
                    sent = epoch.end-epoch.start if epoch.start is not None else 0
                    if not epoch.closing or abs(round(duration*RATE)-sent) > 1:
                        raise ConnectionError("Deepgram terminal metadata did not confirm the sent source")
                    epoch.complete = True
                    self._release(epoch.end)
                    return
                elif kind == "Error":
                    raise self._provider_failure(event.get("code"))
            elif message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                reason = getattr(message, "extra", "")
                code = next((code for code in PROVIDER_CODES if isinstance(reason, str) and reason.startswith(code)), None)
                raise self._provider_failure(code)

    def _requeue_retained(self):
        recognized_end = self.pending_finals[-1]["sample_end_offset"] if self.pending_finals else self.committed_end
        replay = deque()
        for packet in self.retained:
            split = min(packet.end, max(packet.sample_offset, recognized_end))
            if split > packet.sample_offset:
                self.protected_pcm.append(packet.slice(packet.sample_offset, split))
            if split < packet.end:
                replay.append(packet.slice(split))
        self.retained.clear()
        self.queue.extendleft(reversed(replay))
        self.pending_bytes = sum(len(packet.pcm) for packet in self.queue)
        self._bound_queue()

    async def _epoch(self, session):
        epoch = _Epoch()
        connecting_at = time.monotonic()
        async with session.ws_connect(self.connection_url(),
                headers={"Authorization": "Token " + self._api_key}, heartbeat=20,
                max_msg_size=1024*1024, timeout=aiohttp.ClientWSTimeout(ws_close=5)) as ws:
            self.connected = True
            self._retrying = False
            self._latency("provider_connect_sec", time.monotonic()-connecting_at)
            self.on_connection(True, None)
            sender, receiver = asyncio.create_task(self._send(ws, epoch)), asyncio.create_task(self._receive(ws, epoch))
            try:
                done, _ = await asyncio.wait((sender, receiver), return_when=asyncio.FIRST_COMPLETED)
                if sender in done:
                    await sender
                    await asyncio.wait_for(receiver, DRAIN_TIMEOUT)
                else:
                    await receiver
                if not epoch.complete:
                    raise ConnectionError("Deepgram completion was not confirmed")
                self._complete_controls(epoch.end)
            finally:
                for task in (sender, receiver):
                    task.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)
                self.connected = False
                self._speech_pending = False
                self._speech_started_sample = None
                self._clear_partial()

    async def run(self):
        self.task = asyncio.current_task()
        saver = asyncio.create_task(self._save_loop())
        delay = RECONNECT_INITIAL_SECONDS
        try:
            async with self._session_factory(timeout=aiohttp.ClientTimeout(total=None, connect=10)) as session:
                while True:
                    self.wake.clear()
                    await self.save_ready.wait()
                    self._complete_controls(self.queue[0].sample_offset if self.queue else self.total_input_samples)
                    if not self.queue:
                        if self.stopping:
                            break
                        await self.wake.wait()
                        continue
                    try:
                        self._connecting = True
                        progress = (self.partial_updates, self.final_count)
                        await self._epoch(session)
                        delay = RECONNECT_INITIAL_SECONDS
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        self._connecting = False
                        self._retrying = True
                        if progress != (self.partial_updates, self.final_count):
                            # A productive connection breaks the sequence of failed
                            # attempts. Occasional interruptions must not inherit a
                            # 15-second retry from earlier, recovered failures.
                            delay = RECONNECT_INITIAL_SECONDS
                        # Exception messages and provider payloads may contain secrets/text.
                        status = error.status if isinstance(error, aiohttp.WSServerHandshakeError) and error.status in {
                            400, 401, 402, 403, 408, 413, 429, 500, 502, 503, 504} else None
                        code = getattr(error, "provider_code", None)
                        code = code if isinstance(code, str) and code in PROVIDER_CODES else None
                        reason = RESULT_FAILURES.get(str(error)) if isinstance(error, ValueError) else None
                        logger.warning("Deepgram stream failure: %s; http_status=%s; provider_code=%s; reason=%s",
                                       type(error).__name__, status, code, reason)
                        self.on_connection(False, type(error).__name__ + (f": HTTP {status}" if status else "")
                                           + (f": {code}" if code else ""))
                        self.reconnects += 1
                        if self.stopping:
                            break
                        try:
                            await asyncio.wait_for(self.stop_event.wait(), delay)
                        except TimeoutError:
                            pass
                        delay = min(delay*2, 15)
                    finally:
                        self._connecting = False
                        self.connected = False
                        self._requeue_retained()
        finally:
            self.connected = False
            self._connecting = False
            saver.cancel()
            await asyncio.gather(saver, return_exceptions=True)

    async def close(self):
        self.stopping = True
        self.wake.set()
        self.stop_event.set()
        completed = True
        if self.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.task), SHUTDOWN_TIMEOUT)
            except TimeoutError:
                completed = False
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
                logger.error("Deepgram shutdown drain timed out")
        self.retry_saves()
        if self.pending_finals:
            self.last_flush_completed = False
            raise OSError("Shutdown left unsaved Deepgram finals")
        for items in (self.queue, self.retained, self.protected_pcm):
            while items:
                completed = False
                self._gap("failed", items.popleft(), "deepgram_shutdown_unsent")
        self.pending_bytes = 0
        self.last_flush_completed = completed
        return completed
