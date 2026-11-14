"""Private PCM worker: CPU Silero segmentation and one resident MLX GPU model.

The bridge owns transport and persistence. This process owns speech boundaries;
only an immutable, closed audio snapshot can produce a final event.
"""
import argparse
import asyncio
import base64
from collections import Counter, deque
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
import zlib
from common import RATE, WHISPER_POLICY, PACKET_MAX_BYTES

FRAME = 512
PRE_ROLL = 31  # 992 ms retained before the voice trigger; no extra waiting.

DEFAULT_HOTWORDS = (
    "A talk about AI infrastructure by Yahya Alhinai in Oman, mentioning Omi, MLX, "
    "vLLM, SGLang, KV cache, Triton, Helion, Mojo, CUDA and GPU kernels."
)
HOTWORDS_MAX_CHARS = 800
MAX_WORDS_PER_SECOND = 8.0
DEFAULT = object()  # sentinel: "use the configured language"


def degenerate(text, prompt="", seconds=None):
    """True when a transcription cannot be trusted.

    Whisper prepends initial_prompt to its decoder context, so on awkward audio it
    can copy or loop the prompt instead of transcribing. Four independent tests,
    each cheap and each sufficient on its own:

      * physically impossible output (more words than could be spoken in the clip)
      * a single 3-gram covering 40% of the words
      * compression ratio >= 4.0 (normal speech sits near 2-2.6), including scripts without spaces
      * most words coming straight from the prompt
    """
    if len(text) >= 48:
        compressed = len(zlib.compress(text.encode("utf-8"), 6)) or 1
        if len(text) / compressed >= 4.0:
            return True
    words = [word.strip(".,!?;:\"'").lower() for word in text.split()]
    count = len(words)
    if not count:
        return False
    if seconds is not None and count > max(8.0, seconds * MAX_WORDS_PER_SECOND):
        return True
    if count >= 8:
        grams = Counter(tuple(words[index:index + 3]) for index in range(count - 2))
        if grams and grams.most_common(1)[0][1] * 3 >= count * 0.4:
            return True
    if prompt and count >= 8:
        prompt_words = {word.strip(".,!?;:\"'").lower() for word in prompt.split()}
        if sum(1 for word in words if word in prompt_words) / count >= 0.6:
            return True
    return False


def language_setting(raw=None):
    """Target language, or None for per-utterance detection (OMI_LANGUAGE=auto)."""
    value = (os.environ.get("OMI_LANGUAGE", "en") if raw is None else raw).strip().lower()
    return None if value in ("", "auto") else value


def language_allowlist(raw=None):
    """Languages whose transcripts may be published (OMI_LANGUAGES, comma separated)."""
    value = os.environ.get("OMI_LANGUAGES", "en") if raw is None else raw
    return {part.strip().lower() for part in str(value).split(",") if part.strip()}


def hotword_prompt(raw=None):
    """Return the decoder prompt that biases spelling toward names and jargon.

    Whisper keeps only the last 224 tokens of an initial_prompt, so a long prompt
    silently drops its own beginning; the cap keeps the useful terms. Empty or
    whitespace-only returns "" and disables biasing.
    """
    text = os.environ.get("OMI_HOTWORDS", DEFAULT_HOTWORDS) if raw is None else raw
    return " ".join(str(text).split())[:HOTWORDS_MAX_CHARS].strip()


def emit(event):
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


class Recognizer:
    def __init__(self, model_path, vad_path):
        import mlx.core as mx
        import numpy as np
        import onnxruntime as ort
        from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import DecodingOptions
        from mlx_whisper.tokenizer import get_tokenizer
        from mlx_whisper.transcribe import ModelHolder

        self.mx, self.np, self.path = mx, np, str(model_path)
        self._log_mel = log_mel_spectrogram
        self._pad_or_trim = pad_or_trim
        self._n_frames = N_FRAMES
        self._decoding_options = DecodingOptions
        self.hotwords = hotword_prompt()
        self.language = language_setting()
        self.languages = language_allowlist()
        self._cached_features = (None, None)
        self._prompt_token_cache = {}
        self._last_no_speech = False
        mx.set_default_device(mx.gpu)
        if not mx.metal.is_available():
            raise RuntimeError("Metal GPU is unavailable")
        model = ModelHolder.get_model(self.path, mx.float16)
        self.model = model
        self._tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=self.language or "en",
            task="transcribe",
        )
        if self.hotwords:
            self._prompt_token_cache[self.hotwords] = list(self._tokenizer.encode(" " + self.hotwords.strip()))

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
        """Return (text, seconds, reason); reason names a silent drop when text is empty."""
        seconds = len(pcm) / 2 / RATE
        self._last_no_speech = False
        text, language, elapsed = self._transcribe(pcm, self.hotwords, 0.0)
        if language and language not in self.languages:
            return "", elapsed, "language"
        if text and degenerate(text, self.hotwords, seconds):
            # Suspicious output: ask again without the prompt, slightly warmer, and
            # with language detection on. That both breaks prompt-echo loops and
            # tells us when the audio was never the target language to begin with.
            self._last_no_speech = False
            retry, retry_language, retry_elapsed = self._transcribe(pcm, "", 0.2, language=None)
            elapsed += retry_elapsed
            if retry_language and retry_language not in self.languages:
                return "", elapsed, "language"
            if degenerate(retry, "", seconds):
                return "", elapsed, "degenerate"
            if not retry and getattr(self, "_last_no_speech", False):
                return "", elapsed, "no_speech"
            return retry, elapsed, ""
        if not text and getattr(self, "_last_no_speech", False):
            return "", elapsed, "no_speech"
        return text, elapsed, ""

    def _transcribe(self, pcm, prompt, temperature, language=DEFAULT):
        mx = self.mx
        mx.set_default_device(mx.gpu)
        target_language = self.language if language is DEFAULT else language
        start = time.monotonic()
        cached_pcm, features = self._cached_features
        if cached_pcm is not pcm or features is None:
            audio = self.np.frombuffer(pcm, dtype="<i2").astype(self.np.float32) / 32768
            mel = self._pad_or_trim(
                self._log_mel(audio, n_mels=self.model.dims.n_mels),
                self._n_frames,
                axis=-2,
            ).astype(mx.float16)
            features = self.model.encoder(mel[None])[0]
            self._cached_features = (pcm, features)
        prompt_tokens = None
        if prompt:
            prompt_tokens = self._prompt_token_cache.get(prompt)
            if prompt_tokens is None:
                prompt_tokens = list(self._tokenizer.encode(" " + prompt.strip()))
                if len(self._prompt_token_cache) < 8:
                    self._prompt_token_cache[prompt] = prompt_tokens
        seconds = max(0.25, len(pcm) / 2 / RATE)
        max_tokens = min(224, max(56, int(seconds * 20) + 16))
        options = self._decoding_options(
            task="transcribe",
            language=target_language,
            temperature=temperature,
            sample_len=max_tokens,
            prompt=prompt_tokens,
            fp16=True,
            without_timestamps=False,
        )
        result = self.model.decode(features, options)
        mx.synchronize()
        elapsed = time.monotonic() - start
        detected = result.language or target_language
        if result.no_speech_prob > 0.6 and result.avg_logprob <= -1.0:
            self._last_no_speech = True
            return "", detected, elapsed
        return result.text.strip(), detected, elapsed


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
    speech_start_offset: int = 0
    speech_start_at: float = 0.0
    speech_received_at: float = 0.0
    confirm_count: int = 0
    confirmed: bool = False
    quiet_samples: int = 0
    last_voice_mono: float = 0.0
    speech_started_mono: float | None = None
    voice_end: int = 0
    decoded_voice_end: int = 0
    end: int = 0
    closed_mono: float = 0.0
    closed_by_pause: bool = False
    decoded_samples: int = 0
    settled_result: tuple | None = None
    settled_voice_end: int = 0
    settled_max_prob_after: float = 0.0
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
                   revision=turn.revision, text=text, audio_start_at=turn.speech_start_at,
                   sample_offset=turn.first.sample_offset, speech_start_offset=turn.speech_start_offset)

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

    def close_turn(self, *, by_pause=False):
        if self.active is not None:
            self.active.closed_mono = time.monotonic()
            self.active.closed_by_pause = by_pause
            self.closed.append(self.active)
            self.active = None
            self.changed.set()

    def tail(self):
        # A non-512-aligned last packet remains real source audio, not padding.
        if self.carry and self.active is not None:
            self.active.pcm.extend(self.carry)
            self.active.end = self.carry_meta.end
            self.active.settled_result = None
        self.carry.clear()
        self.carry_meta = None

    def feed(self, packet):
        raw = base64.b64decode(packet["pcm_base64"], validate=True)
        offset, session = packet["sample_offset"], packet["session_id"]
        if not raw or len(raw) % 2 or len(raw) > PACKET_MAX_BYTES:
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
            self.active = Utterance(
                self.pre_roll[0],
                speech_start_offset=frame.sample_offset,
                speech_start_at=frame.audio_start_at,
                speech_received_at=frame.received_at,
                speech_started_mono=frame.received_mono,
                voice_end=frame.end,
            )
            for previous in self.pre_roll:
                self.active.pcm.extend(previous.pcm)
            self.pre_roll.clear()
        else:
            self.active.pcm.extend(frame.pcm)
        turn = self.active
        turn.end = frame.end
        if probability >= self.policy.confirm_probability:
            turn.confirm_count += 1
            if turn.confirm_count >= self.policy.confirm_frames:
                turn.confirmed = True
        if probability >= self.policy.continue_probability:
            turn.quiet_samples = 0
            turn.last_voice_mono = frame.received_mono
            turn.voice_end = frame.end
            turn.settled_result = None
            turn.settled_max_prob_after = 0.0
        else:
            turn.quiet_samples += len(frame.pcm) // 2
            if turn.settled_voice_end:
                turn.settled_max_prob_after = max(turn.settled_max_prob_after, probability)
        paused = turn.quiet_samples >= RATE * self.policy.pause_seconds
        if paused or len(turn.pcm) >= RATE * self.policy.max_turn_seconds * 2:
            self.close_turn(by_pause=paused)
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

    def prefinal_due(self, turn):
        if not self.policy.prefinal_quiet_seconds or not turn.confirmed:
            return False
        if turn.quiet_samples < RATE * self.policy.prefinal_quiet_seconds:
            return False
        return turn.settled_result is None or turn.settled_voice_end < turn.voice_end

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
            prefinal = bool(not final and turn is not None and self.prefinal_due(turn))
            eligible = turn is not None and (final or prefinal or self.draft_due(turn, time.monotonic()))
            if eligible:
                if final:
                    self.closed.popleft()
                self.inference_turn = turn
                self.inference_final = final
                self.release()
                pcm = bytes(turn.pcm)  # The decoder never reads a mutable buffer.
                voice_end_snap = turn.voice_end
                turn.decoded_samples = len(pcm) // 2
                turn.decoded_voice_end = voice_end_snap
                if prefinal:
                    turn.settled_voice_end = voice_end_snap
                    turn.settled_max_prob_after = 0.0
                if final and not turn.confirmed:
                    text, elapsed, reason = "", 0.0, "unconfirmed"
                elif (final and turn.settled_result is not None
                      and turn.closed_by_pause
                      and turn.settled_voice_end == turn.voice_end
                      and turn.settled_max_prob_after < 0.15):
                    text, elapsed, reason = turn.settled_result
                else:
                    self.last_decode = time.monotonic()
                    text, elapsed, reason = await loop.run_in_executor(
                        None, self.recognizer.decode, pcm)
                    self.event("latency", name="local_decode_sec", seconds=elapsed)
                if prefinal and turn.voice_end == voice_end_snap:
                    turn.settled_result = (text, elapsed, reason)
                if final:
                    if text:
                        self.event("final", text=text, utterance_id=turn.identity,
                            session_id=turn.first.session_id, sample_offset=turn.first.sample_offset,
                            sample_end_offset=turn.end, speech_start_offset=turn.speech_start_offset,
                            voice_end_offset=turn.voice_end, received_at=turn.first.received_at,
                            audio_start_at=turn.speech_start_at)
                        now_mono = time.monotonic()
                        self.event("latency", name="receipt_to_final_sec", seconds=now_mono-turn.first.received_mono)
                        self.event("latency", name="turn_close_to_final_sec", seconds=max(0.0, now_mono-turn.closed_mono))
                        if turn.last_voice_mono:
                            self.event("latency", name="turn_boundary_wait_sec",
                                seconds=max(0.0, turn.closed_mono-turn.last_voice_mono))
                            self.event("latency", name="speech_end_to_final_sec", seconds=max(0.0, now_mono-turn.last_voice_mono))
                    else:
                        self.event("no_text", reason=reason or "empty")
                        if turn.confirmed and reason not in ("language", "no_speech"):
                            voiced_sec = max(FRAME, turn.voice_end - turn.speech_start_offset) / RATE
                            self.event("gap", reason="failed", seconds=voiced_sec,
                                session_id=turn.first.session_id, sample_offset=turn.speech_start_offset,
                                received_at=turn.speech_received_at, audio_start_at=turn.speech_start_at,
                                detail="local_confirmed_speech_without_text")
                    self.partial(turn, "")
                elif turn is self.active and text and text != turn.text:
                    if not turn.first_partial:
                        self.event("latency", name="receipt_to_first_partial_sec", seconds=time.monotonic()-turn.first.received_mono)
                        if turn.speech_started_mono is not None:
                            self.event("latency", name="speech_start_to_first_partial_sec",
                                seconds=time.monotonic()-turn.speech_started_mono)
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
        speech_samples = turn.end - turn.speech_start_offset if turn.end else samples
        if speech_samples < RATE*self.policy.initial_seconds:
            return False
        # Appending only trailing silence does not add words. Save that GPU work
        # for the settled pre-final or the next speech frame.
        if turn.decoded_samples and turn.voice_end <= turn.decoded_voice_end:
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
