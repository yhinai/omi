"""Async bridge adapter for the isolated Python 3.14 Whisper worker.

Keep source PCM until the worker releases it. A worker restart replays unresolved
audio, retaining original sample positions and phone-session ownership.
"""
import asyncio
import base64
from collections import deque
import json
import logging
import os
import re
from pathlib import Path
import time
import uuid

from common import (BPS, Packet, QUEUE_MAX_BYTES, QUEUE_MAX_PACKETS,
                    REPLAY_MAX_BYTES, PACKET_MAX_BYTES, SHUTDOWN_TIMEOUT,
                    WORKER_DIAGNOSTIC_BYTES)

logger = logging.getLogger("bridge")


class LocalWhisperSTT:
    def __init__(self, *, python, model, vad, on_partial, on_final, on_gap,
                 on_latency, on_connection, max_age=45, idle_flush=8, worker_env=None):
        self.command = [str(python), str(Path(__file__).with_name("whisper.py")),
                        "--model", str(model), "--vad", str(vad), "--idle-flush", str(idle_flush)]
        self.on_partial, self.on_final, self.on_gap = on_partial, on_final, on_gap
        self.on_latency, self.on_connection = on_latency, on_connection
        self.max_age, self.idle_flush = max_age, idle_flush
        self.worker_env = dict(worker_env or {})
        self.queue, self.retained, self.controls = deque(), deque(), deque()
        self.pending_bytes = self.total_input_samples = self.session_id = 0
        self.connected = self.stopping = self._speech_pending = False
        self.last_packet_mono = None
        self.reconnects = self.partial_updates = self.final_count = self.turns_without_text = 0
        self.no_text_causes = {}
        self.gap_causes = {}
        self.pending_final_count = 0
        self.wake, self.stop_event = asyncio.Event(), asyncio.Event()
        self.task = self.process = None
        self.flush_waiters = {}
        self.last_partial = None
        self.committed_end = 0
        self.load_warm_sec = None
        self.pending_finals = deque()
        self.save_ready = asyncio.Event()
        self.save_ready.set()
        self.release_end = 0
        self.protected_pcm = deque()
        self.save_error = None
        self.last_flush_completed = None
        self.worker_diagnostics = bytearray()

    @property
    def pending_age(self):
        starts = [q[0].received_mono for q in (self.protected_pcm, self.retained, self.queue) if q]
        return max(0, time.monotonic()-min(starts)) if starts else None

    @property
    def speech_pending(self):
        return self._speech_pending or bool(self.queue) or bool(self.pending_finals)

    def new_session(self):
        self.request_flush()
        self.session_id += 1

    def gap(self, reason, seconds, packet, detail):
        if seconds <= 0:
            return
        self.gap_causes[detail] = self.gap_causes.get(detail, 0) + 1
        self.on_gap(reason, seconds, session_id=packet.session_id, sample_offset=packet.sample_offset,
            received_at=packet.received_at, audio_start_at=packet.audio_start_at, detail=detail)

    def feed(self, raw, received_at, received_mono):
        if self.stopping:
            raise RuntimeError("Local STT is stopping")
        if not raw:
            return
        if self.last_packet_mono is None or received_mono-self.last_packet_mono > self.idle_flush:
            self.session_id += 1
        self.last_packet_mono = received_mono
        self.queue.append(Packet(raw, received_at, received_mono, self.total_input_samples, self.session_id))
        self.total_input_samples += len(raw)//2
        self.pending_bytes += len(raw)
        self._bound_queue()
        self.wake.set()

    def _bound_queue(self):
        while self.pending_bytes > QUEUE_MAX_BYTES or len(self.queue) > QUEUE_MAX_PACKETS:
            packet = self.queue[0]
            count = len(packet.pcm) if len(self.queue)>QUEUE_MAX_PACKETS else min(len(packet.pcm), self.pending_bytes-QUEUE_MAX_BYTES)
            self.gap("queue_evicted", count/BPS, packet, "local_unsent_queue_evicted")
            self.pending_bytes -= count
            if count == len(packet.pcm):
                self.queue.popleft()
            else:
                packet.pcm = packet.pcm[count:]
                packet.sample_offset += count//2
                packet.audio_start_at += count/BPS

    def request_flush(self):
        identity = uuid.uuid4().hex
        self.controls.append((identity, self.total_input_samples))
        self.wake.set()
        return identity

    async def flush_now(self, timeout=5):
        identity = self.request_flush()
        waiter = self.flush_waiters[identity] = asyncio.Event()
        try:
            async with asyncio.timeout(timeout):
                await waiter.wait()
                await self.save_ready.wait()
            self.last_flush_completed = True
            return True
        except TimeoutError:
            self.last_flush_completed = False
            logger.warning("Local transcription flush timed out; accepted audio remains pending")
            return False
        finally:
            self.flush_waiters.pop(identity, None)

    def _release(self, end):
        self.release_end = max(self.release_end, end)
        if self.pending_finals:
            end = min(end, self.pending_finals[0]["sample_offset"])
        for queue in (self.protected_pcm, self.retained):
            while queue and queue[0].sample_offset+len(queue[0].pcm)//2 <= end:
                queue.popleft()
            if queue and queue[0].sample_offset < end:
                packet = queue[0]
                trimmed = (end-packet.sample_offset)*2
                packet.pcm = packet.pcm[trimmed:]
                packet.sample_offset = end
                packet.audio_start_at += trimmed/BPS

    def retry_saves(self):
        """Commit in source order; failed saves keep both the final and its PCM."""
        while self.pending_finals:
            event = self.pending_finals[0]
            fields = {key: event[key] for key in ("utterance_id", "session_id", "sample_offset",
                "sample_end_offset", "received_at", "audio_start_at",
                "speech_start_offset", "voice_end_offset") if key in event}
            try:
                saved = self.on_final(event["text"], **fields)
            except OSError as error:
                saved = False
                self.save_error = type(error).__name__
            if saved is False:
                self.save_error = self.save_error or "Transcript save failed"
                self.save_ready.clear()
                return
            self.pending_finals.popleft()
            self.final_count += 1
            self.committed_end = event["sample_end_offset"]
            self._release(max(self.release_end, self.committed_end))
            self.on_latency("receipt_to_transcript_sec", time.monotonic()-self._first_receipt(event))
        self.save_error = None
        self.save_ready.set()
        self.wake.set()

    async def _save_loop(self):
        while True:
            await asyncio.sleep(.5)
            if self.pending_finals:
                self.retry_saves()

    def handle_message(self, event):
        kind = event["type"]
        if kind == "partial":
            self.last_partial = event if event.get("text") else None
            self.partial_updates += bool(event.get("text"))
            self.on_partial(event)
        elif kind == "final":
            end = event["sample_end_offset"]
            if end > self.committed_end and not any(e["sample_end_offset"] == end for e in self.pending_finals):
                self.pending_finals.append(dict(event))
                self.retry_saves()
        elif kind == "gap":
            fields = {key: value for key, value in event.items() if key not in ("type", "reason", "seconds")}
            self.gap_causes[event["detail"]] = self.gap_causes.get(event["detail"], 0)+1
            self.on_gap(event["reason"], event["seconds"], **fields)
        elif kind == "covered":
            self._release(event["sample_end_offset"])
        elif kind == "latency":
            self.on_latency(event["name"], event["seconds"])
        elif kind == "flushed":
            waiter = self.flush_waiters.get(event.get("request_id"))
            if waiter is not None:
                waiter.set()
        elif kind == "no_text":
            self.turns_without_text += 1
            reason = event.get("reason") or "unknown"
            self.no_text_causes[reason] = self.no_text_causes.get(reason, 0) + 1
        elif kind == "status":
            self.pending_final_count = event["pending_finals"]
            self._speech_pending = event["speech_pending"]

    @staticmethod
    def _first_receipt(event):
        # Wall/monotonic conversion only for this persistence diagnostic. The
        # worker's recognition metrics use the original monotonic receipt.
        return time.monotonic() - max(0, time.time()-event["received_at"])

    async def _write(self, message):
        self.process.stdin.write((json.dumps(message, separators=(",", ":"))+"\n").encode())
        await asyncio.wait_for(self.process.stdin.drain(), 5)

    async def _send(self):
        shutdown_sent = False
        while True:
            self.wake.clear()
            await self.save_ready.wait()  # Backpressure during output failure; intake stays bounded.
            if self.controls and (not self.queue or self.queue[0].sample_offset >= self.controls[0][1]):
                identity, end = self.controls.popleft()
                await self._write({"op": "flush", "request_id": identity, "sample_end_offset": end})
                continue
            if self.queue:
                packet = self.queue[0]
                if time.monotonic()-packet.received_mono > self.max_age:
                    self.gap("stale", len(packet.pcm)/BPS, packet, "local_unsent_audio_stale")
                    self.queue.popleft()
                    self.pending_bytes -= len(packet.pcm)
                    continue
                size = min(PACKET_MAX_BYTES, len(packet.pcm))
                if self.controls:
                    size = min(size, max(0, self.controls[0][1]-packet.sample_offset)*2)
                chunk = Packet(packet.pcm[:size], packet.received_at, packet.received_mono,
                               packet.sample_offset, packet.session_id, packet.audio_start_at)
                # Retain before writing: a crash between write and acknowledgment
                # is replayable; committed source offsets suppress duplicate finals.
                self.retained.append(chunk)
                packet.pcm = packet.pcm[size:]
                packet.sample_offset += size//2
                packet.audio_start_at += size/BPS
                self.pending_bytes -= size
                if not packet.pcm:
                    self.queue.popleft()
                await self._write({"op": "packet", "pcm_base64": base64.b64encode(chunk.pcm).decode("ascii"),
                    "sample_offset": chunk.sample_offset, "session_id": chunk.session_id,
                    "received_at": chunk.received_at, "received_mono": chunk.received_mono,
                    "audio_start_at": chunk.audio_start_at})
                self.on_latency("receipt_to_send_sec", time.monotonic()-chunk.received_mono)
                # A stalled worker must not build an unbounded replay copy.
                if sum(len(item.pcm) for item in self.retained) > REPLAY_MAX_BYTES:
                    raise TimeoutError("Local worker did not release source audio")
                continue
            if self.stopping and not shutdown_sent:
                await self._write({"op": "shutdown", "sample_end_offset": self.total_input_samples})
                shutdown_sent = True
            await self.wake.wait()

    async def _receive(self):
        while True:
            try:
                line = await asyncio.wait_for(self.process.stdout.readline(), 5)
            except TimeoutError:
                if not self.pending_finals and self.pending_age is not None and self.pending_age > self.max_age:
                    raise TimeoutError("Local source audio stalled")
                continue
            if not line:
                break
            self.handle_message(json.loads(line))
        if not self.stopping:
            raise ConnectionError("Local worker exited")

    async def run(self):
        self.task = asyncio.current_task()
        saver = asyncio.create_task(self._save_loop())
        delay = 1
        try:
            while True:
                sender = receiver = diagnostics = None
                try:
                    self.process = await asyncio.create_subprocess_exec(*self.command,
                        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE, limit=131072,
                        env={**os.environ, **self.worker_env})
                    self.worker_diagnostics.clear()
                    diagnostics = asyncio.create_task(self._read_diagnostics())
                    ready = json.loads(await asyncio.wait_for(self.process.stdout.readline(), 60))
                    if ready.get("type") != "ready":
                        raise ConnectionError("Invalid local worker startup")
                    self.connected, delay = True, 1
                    self.load_warm_sec = ready["load_warm_sec"]
                    self.on_connection(True, None)
                    sender = asyncio.create_task(self._send())
                    receiver = asyncio.create_task(self._receive())
                    done, _ = await asyncio.wait((sender, receiver), return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        await task
                    if not self.stopping:
                        raise ConnectionError("Local worker stopped")
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    # Model error strings can include recognized text. Log only
                    # exception names and allowlisted runtime failure signals.
                    await asyncio.sleep(0)
                    self._report_diagnostics(error)
                    self.on_connection(False, type(error).__name__)
                finally:
                    self.connected = False
                    for task in (sender, receiver):
                        if task is not None:
                            task.cancel()
                    await asyncio.gather(*(task for task in (sender, receiver) if task is not None), return_exceptions=True)
                    if self.process is not None and self.process.returncode is None:
                        self.process.terminate()
                        try:
                            await asyncio.wait_for(self.process.wait(), 5)
                        except TimeoutError:
                            self.process.kill()
                            await self.process.wait()
                    if diagnostics is not None:
                        try:
                            await asyncio.wait_for(asyncio.shield(diagnostics), 1)
                        except TimeoutError:
                            diagnostics.cancel()
                        await asyncio.gather(diagnostics, return_exceptions=True)
                    if self.last_partial:
                        self.on_partial({**self.last_partial, "text": ""})
                        self.last_partial = None
                    # Recognized-but-unsaved audio is protected, not re-recognized
                    # after a worker crash. Its immutable final remains retryable.
                    recognized_end = self.pending_finals[-1]["sample_end_offset"] if self.pending_finals else self.committed_end
                    replay = deque()
                    for packet in self.retained:
                        count = min(len(packet.pcm), max(0, recognized_end-packet.sample_offset)*2)
                        if count:
                            self.protected_pcm.append(Packet(packet.pcm[:count], packet.received_at,
                                packet.received_mono, packet.sample_offset, packet.session_id, packet.audio_start_at))
                        if count < len(packet.pcm):
                            replay.append(Packet(packet.pcm[count:], packet.received_at, packet.received_mono,
                                packet.sample_offset+count//2, packet.session_id, packet.audio_start_at+count/BPS))
                    self.queue.extendleft(reversed(replay))
                    self.retained.clear()
                    self.pending_bytes = sum(len(packet.pcm) for packet in self.queue)
                    self._bound_queue()
                    self.pending_final_count = 0
                    self._speech_pending = False
                if self.stopping:
                    break
                self.reconnects += 1
                try:
                    await asyncio.wait_for(self.stop_event.wait(), delay)
                except TimeoutError:
                    pass
                delay = min(delay*2, 15)
        finally:
            self.connected = False
            saver.cancel()
            await asyncio.gather(saver, return_exceptions=True)

    async def close(self):
        completed = True
        self.stopping = True
        self.wake.set()
        self.stop_event.set()
        if self.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.task), SHUTDOWN_TIMEOUT)
            except TimeoutError:
                completed = False
                self.last_flush_completed = False
                logger.error("Local transcription shutdown drain timed out")
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
        if self.queue:
            completed = False
        for packet in self.queue:
            self.gap("failed", len(packet.pcm)/BPS, packet, "local_shutdown_unsent")
        self.queue.clear()
        self.pending_bytes = 0
        self.retry_saves()
        if self.pending_finals:
            self.last_flush_completed = False
            raise OSError("Shutdown left unsaved finals; memory-only recovery is lost when the bridge exits")
        self.last_flush_completed = completed
        return completed

    async def _read_diagnostics(self):
        while chunk := await self.process.stderr.read(1024):
            self.worker_diagnostics.extend(chunk)
            del self.worker_diagnostics[:-WORKER_DIAGNOSTIC_BYTES]

    def _report_diagnostics(self, error):
        text = self.worker_diagnostics.decode("utf-8", errors="replace")
        known = {"RuntimeError", "MemoryError", "ValueError", "TypeError", "ImportError",
                 "ModuleNotFoundError", "FileNotFoundError", "OSError", "InvalidArgument"}
        types = sorted(known.intersection(re.findall(r"\b([A-Za-z]+(?:Error|Exception))\b", text)))
        allowed = ("out of memory", "Metal GPU is unavailable", "No module named", "No such file or directory")
        signals = [signal for signal in allowed if signal.casefold() in text.casefold()]
        logger.error("Whisper worker failure: %s; runtime exceptions=%s; signals=%s; stderr_bytes=%d",
                     type(error).__name__, ",".join(types) or "none", ",".join(signals) or "none",
                     len(self.worker_diagnostics))
        self.worker_diagnostics.clear()
