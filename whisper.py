"""Private PCM worker: CPU Silero segmentation and one resident MLX GPU model.

The bridge owns transport and persistence. This process owns speech boundaries;
only an immutable, closed audio snapshot can produce a final event.
"""
import argparse
import asyncio
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import stat
import time
import types
import uuid
from common import RATE, WHISPER_POLICY

FRAME = 512
PRE_ROLL = 31  # 992 ms retained before the voice trigger; no extra waiting.
MAX_TURN = int(WHISPER_POLICY.max_turn_seconds * RATE)


def emit(event):
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


class Recognizer:
    def __init__(self, model_path, vad_path):
        import mlx.core as mx
        import numpy as np
        import onnxruntime as ort
        from mlx_whisper.transcribe import ModelHolder

        self.mx, self.np, self.path = mx, np, str(model_path)
        mx.set_default_device(mx.gpu)
        if not mx.metal.is_available():
            raise RuntimeError("Metal GPU is unavailable")
        model = ModelHolder.get_model(self.path, mx.float16)

        def fused(attention, q, k, v, mask=None):
            batch, length, state = q.shape
            heads = attention.n_head
            q = q.reshape(batch, length, heads, -1).transpose(0, 2, 1, 3)
            k = k.reshape(batch, k.shape[1], heads, -1).transpose(0, 2, 1, 3)
            v = v.reshape(batch, v.shape[1], heads, -1).transpose(0, 2, 1, 3)
            mask = mask[:length, :length] if mask is not None else None
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=(state // heads)**-.5, mask=mask)
            return out.transpose(0, 2, 1, 3).reshape(batch, length, state), None

        for block in model.encoder.blocks:
            block.attn.qkv_attention = types.MethodType(fused, block.attn)
        for block in model.decoder.blocks:
            for attention in (block.attn, block.cross_attn):
                attention.qkv_attention = types.MethodType(fused, attention)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.vad = ort.InferenceSession(str(vad_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.reset_vad()
        self.decode(bytes(RATE * 2))  # Warmup output is deliberately discarded.

    def reset_vad(self):
        self.vad_state = self.np.zeros((2, 1, 128), self.np.float32)
        self.context = self.np.zeros((1, 64), self.np.float32)

    def probability(self, pcm):
        np = self.np
        frame = np.frombuffer(pcm, dtype="<i2").astype(np.float32).reshape(1, FRAME) / 32768
        value = np.concatenate([self.context, frame], axis=1)
        result, self.vad_state = self.vad.run(None, {
            "input": value, "state": self.vad_state, "sr": np.array(RATE, np.int64)})
        self.context = value[:, -64:]
        return float(result[0][0])

    def decode(self, pcm):
        import mlx_whisper
        self.mx.set_default_device(self.mx.gpu)
        audio = self.np.frombuffer(pcm, dtype="<i2").astype(self.np.float32) / 32768
        start = time.monotonic()
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=self.path, language="en",
            task="transcribe", temperature=0.0, condition_on_previous_text=False,
            fp16=True, word_timestamps=False, verbose=None)
        self.mx.synchronize()
        return result["text"].strip(), time.monotonic() - start


@dataclass
class AudioFrame:
    pcm: bytes
    sample_offset: int
    session_id: int
    received_at: float
    received_mono: float
    audio_start_at: float

    @property
    def end(self):
        return self.sample_offset + len(self.pcm) // 2


@dataclass
class Utterance:
    first: AudioFrame
    pcm: bytearray = field(default_factory=bytearray)
    identity: str = field(default_factory=lambda: uuid.uuid4().hex)
    confirmed: bool = False
    quiet_samples: int = 0
    last_voice_mono: float = 0.0
    end: int = 0
    closed_mono: float = 0.0
    decoded_samples: int = 0
    revision: int = 0
    text: str = ""
    first_partial: bool = False


class Stream:
    def __init__(self, recognizer, *, idle_flush=8, output=emit, policy=WHISPER_POLICY):
        self.recognizer, self.idle_flush, self.output = recognizer, idle_flush, output
        self.policy = policy
        self.active = self.inference_turn = None
        self.inference_final = False
        self.closed, self.pre_roll = deque(), deque(maxlen=PRE_ROLL)
        self.carry = bytearray()
        self.carry_meta = None
        self.expected_offset = self.session_id = None
        self.covered = 0
        self.last_packet = time.monotonic()
        self.idle_flushed = False
        self.last_decode = 0.0
        self.barriers = deque()
        self.changed = asyncio.Event()
        self.stopping = False
        self.last_status = None

    def event(self, kind, **fields):
        self.output({"type": kind, **fields})

    def partial(self, turn, text):
        turn.revision += 1
        self.event("partial", utterance_id=turn.identity, session_id=turn.first.session_id,
                   revision=turn.revision, text=text)

    def release(self):
        candidates = [t.first.sample_offset for t in self.closed]
        candidates += [t.first.sample_offset for t in (self.active, self.inference_turn) if t is not None]
        if self.pre_roll:
            candidates.append(self.pre_roll[0].sample_offset)
        if self.carry_meta is not None:
            candidates.append(self.carry_meta.sample_offset)
        end = min(candidates) if candidates else (self.expected_offset or self.covered)
        if end > self.covered:
            self.covered = end
            self.event("covered", sample_end_offset=end)
        status = (len(self.closed)+int(self.inference_final), bool(self.active is not None or self.closed or self.inference_turn is not None))
        if status != self.last_status:
            self.event("status", pending_finals=status[0], speech_pending=status[1])
            self.last_status = status

    def close_turn(self):
        if self.active is not None:
            self.active.closed_mono = time.monotonic()
            self.closed.append(self.active)
            self.active = None
            self.changed.set()

    def tail(self):
        # A non-512-aligned last packet remains real source audio, not padding.
        if self.carry and self.active is not None:
            self.active.pcm.extend(self.carry)
            self.active.end = self.carry_meta.end
        self.carry.clear()
        self.carry_meta = None

    def feed(self, packet):
        raw = base64.b64decode(packet["pcm_base64"], validate=True)
        offset, session = packet["sample_offset"], packet["session_id"]
        if not raw or len(raw) % 2 or len(raw) > 3200:
            raise ValueError("Expected at most 100 ms of PCM16")
        if session != self.session_id or offset != self.expected_offset:
            self.tail()
            self.close_turn()
            self.pre_roll.clear()
            self.recognizer.reset_vad()
        self.session_id = session
        self.expected_offset = offset + len(raw) // 2
        self.last_packet = time.monotonic()
        self.idle_flushed = False
        if self.carry_meta is None:
            self.carry_meta = AudioFrame(b"", offset, session, packet["received_at"],
                packet["received_mono"], packet["audio_start_at"])
        self.carry.extend(raw)
        while len(self.carry) >= FRAME * 2:
            meta = self.carry_meta
            frame = AudioFrame(bytes(self.carry[:FRAME * 2]), meta.sample_offset, session,
                               meta.received_at, meta.received_mono, meta.audio_start_at)
            del self.carry[:FRAME * 2]
            self.carry_meta = AudioFrame(b"", frame.end, session, packet["received_at"],
                packet["received_mono"], frame.audio_start_at + FRAME / RATE) if self.carry else None
            self.consume(frame, self.recognizer.probability(frame.pcm))
        # Include a short tail's actual length when closing a stream.
        if self.carry_meta is not None:
            self.carry_meta.pcm = bytes(self.carry)
        self.release()

    def consume(self, frame, probability):
        if self.active is None:
            self.pre_roll.append(frame)
            if probability < self.policy.start_probability:
                return
            self.active = Utterance(self.pre_roll[0])
            for previous in self.pre_roll:
                self.active.pcm.extend(previous.pcm)
            self.pre_roll.clear()
        else:
            self.active.pcm.extend(frame.pcm)
        turn = self.active
        turn.end = frame.end
        turn.confirmed |= probability >= self.policy.confirm_probability
        if probability >= self.policy.continue_probability:
            turn.quiet_samples = 0
            turn.last_voice_mono = frame.received_mono
        else:
            turn.quiet_samples += len(frame.pcm) // 2
        if turn.quiet_samples >= RATE*self.policy.pause_seconds or len(turn.pcm) >= RATE*self.policy.max_turn_seconds*2:
            self.close_turn()
        self.changed.set()

    def flush(self, identity, end):
        self.tail()
        self.close_turn()
        self.pre_roll.clear()
        self.recognizer.reset_vad()
        self.barriers.append((identity, end))
        self.changed.set()

    def acknowledge_flushes(self):
        while self.barriers:
            identity, end = self.barriers[0]
            if any(t.first.sample_offset < end for t in self.closed) or (
                    self.inference_turn is not None and self.inference_turn.first.sample_offset < end):
                break
            self.barriers.popleft()
            self.event("flushed", request_id=identity)

    async def decode_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            self.changed.clear()
            if not self.idle_flushed and time.monotonic() - self.last_packet >= self.idle_flush:
                self.tail()
                self.close_turn()
                self.pre_roll.clear()
                self.recognizer.reset_vad()
                self.idle_flushed = True
            turn = self.closed[0] if self.closed else self.active
            final = bool(self.closed)
            eligible = turn is not None and (final or self.draft_due(turn, time.monotonic()))
            if eligible:
                if final:
                    self.closed.popleft()
                self.inference_turn = turn
                self.inference_final = final
                self.release()
                self.last_decode = time.monotonic()
                pcm = bytes(turn.pcm)  # The decoder never reads a mutable buffer.
                turn.decoded_samples = len(pcm) // 2
                if final and not turn.confirmed:
                    text, elapsed = "", 0.0
                else:
                    text, elapsed = await loop.run_in_executor(None, self.recognizer.decode, pcm)
                self.event("latency", name="local_decode_sec", seconds=elapsed)
                if final:
                    if text:
                        self.event("final", text=text, utterance_id=turn.identity,
                            session_id=turn.first.session_id, sample_offset=turn.first.sample_offset,
                            sample_end_offset=turn.end, received_at=turn.first.received_at,
                            audio_start_at=turn.first.audio_start_at)
                        self.event("latency", name="receipt_to_final_sec", seconds=time.monotonic()-turn.first.received_mono)
                        self.event("latency", name="turn_close_to_final_sec", seconds=time.monotonic()-turn.closed_mono)
                        if turn.last_voice_mono:
                            self.event("latency", name="speech_end_to_final_sec", seconds=time.monotonic()-turn.last_voice_mono)
                    elif turn.confirmed:
                        self.event("gap", reason="failed", seconds=len(pcm)/(RATE*2),
                            session_id=turn.first.session_id, sample_offset=turn.first.sample_offset,
                            received_at=turn.first.received_at, audio_start_at=turn.first.audio_start_at,
                            detail="local_confirmed_speech_without_text")
                    else:
                        self.event("no_text")
                    self.partial(turn, "")
                elif text and text != turn.text:
                    if not turn.first_partial:
                        self.event("latency", name="receipt_to_first_partial_sec", seconds=time.monotonic()-turn.first.received_mono)
                        turn.first_partial = True
                    turn.text = text
                    self.partial(turn, text)
                self.inference_turn = None
                self.inference_final = False
                self.release()
                self.acknowledge_flushes()
                continue
            self.release()
            self.acknowledge_flushes()
            if self.stopping and not self.closed and self.active is None:
                return
            try:
                await asyncio.wait_for(self.changed.wait(), .04)
            except TimeoutError:
                pass

    def draft_due(self, turn, now):
        samples = len(turn.pcm)//2
        if samples < RATE*self.policy.initial_seconds:
            return False
        # A timer tick plus a few jittered samples does not justify decoding the
        # whole utterance again. Finals always bypass draft throttling.
        new_audio = samples-turn.decoded_samples
        return ((turn.decoded_samples == 0 or new_audio >= RATE*self.policy.update_seconds)
                and now-self.last_decode >= self.policy.update_seconds)


async def serve(args):
    loop = asyncio.get_running_loop()
    # A single inference thread avoids concurrent GPU jobs; the main loop keeps
    # consuming PCM and running the tiny CPU VAD while a final is decoded.
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper"))
    started = time.monotonic()
    recognizer = await loop.run_in_executor(None, Recognizer, args.model, args.vad)
    stream = Stream(recognizer, idle_flush=args.idle_flush)
    transport_reader = asyncio.StreamReader(limit=65536)
    protocol = asyncio.StreamReaderProtocol(transport_reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    emit({"type": "ready", "load_warm_sec": time.monotonic()-started})
    decoder = asyncio.create_task(stream.decode_loop())
    try:
        while True:
            line_task = asyncio.create_task(transport_reader.readline())
            done, _ = await asyncio.wait((line_task, decoder), return_when=asyncio.FIRST_COMPLETED)
            if decoder in done:
                line_task.cancel()
                await asyncio.gather(line_task, return_exceptions=True)
                await decoder
                raise RuntimeError("Decoder stopped unexpectedly")
            line = line_task.result()
            if not line:
                break
            request = json.loads(line)
            if request["op"] == "packet":
                stream.feed(request)
            elif request["op"] in ("flush", "shutdown"):
                stream.flush(request.get("request_id"), request["sample_end_offset"])
                if request["op"] == "shutdown":
                    break
            else:
                raise ValueError("Unknown worker operation")
    finally:
        stream.flush(None, stream.expected_offset or 0)
        stream.stopping = True
        stream.changed.set()
        await decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vad", type=Path, required=True)
    parser.add_argument("--idle-flush", type=float, default=8)
    args = parser.parse_args()
    if not stat.S_ISFIFO(os.fstat(sys.stdin.fileno()).st_mode):
        parser.error("Worker stdin must be a private pipe from the bridge")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
