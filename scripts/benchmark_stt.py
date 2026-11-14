#!/usr/bin/env python3.15
"""Benchmark direct transcription callbacks using synthetic English audio only.

Default: local Whisper. Explicit --backends whisper,deepgram uploads the generated
clips to the paid cloud provider. No wearable audio, live sockets or journals are used.
"""
import argparse
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge
from config import load_env
from common import BPS, RATE

PHRASES = (
    ("conversation", "Please remind me to bring the charger to the meeting tomorrow."),
    ("hackathon", "Omi hackathon check. Blue lantern seven."),
    ("technical_numbers", "The API returned status two hundred after three requests in one point five seconds."),
)
LATENCY_NAMES = {
    "receipt_to_send_sec", "receipt_to_first_partial_sec", "receipt_to_transcript_sec",
    "receipt_to_final_sec", "turn_close_to_final_sec", "speech_end_to_final_sec",
    "local_decode_sec", "turn_boundary_wait_sec", "speech_onset_to_first_partial_sec",
    "speech_onset_to_final_sec",
    "provider_word_end_to_final_sec", "provider_word_end_to_partial_sec",
    "final_word_packet_to_result_sec", "pipeline_start_sec", "provider_connect_sec",
}
SMALL_NUMBERS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                 "fifteen sixteen seventeen eighteen nineteen").split()
TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()


def integer_words(number):
    if number < 20:
        return [SMALL_NUMBERS[number]]
    if number < 100:
        return [TENS[number // 10]] + (integer_words(number % 10) if number % 10 else [])
    if number < 1000:
        return integer_words(number // 100) + ["hundred"] + (integer_words(number % 100) if number % 100 else [])
    if number < 10000:
        return integer_words(number // 1000) + ["thousand"] + (integer_words(number % 1000) if number % 1000 else [])
    return [SMALL_NUMBERS[int(digit)] for digit in str(number)]


def normalize(text):
    """Ignore case/punctuation; normalize small numbers and API/HTTP spelling.

    This is a formatting-tolerant lexical WER, not a semantic accuracy measure.
    Contractions, dates, homophones and other acronyms can still count as errors.
    """
    words = []
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text.casefold())
    for token in re.findall(r"[a-z]+|\d+(?:\.\d+)?", text):
        if token in ("api", "http"):
            words.extend(token)
        elif token[0].isdigit():
            whole, _, fraction = token.partition(".")
            words.extend(integer_words(int(whole)) if len(whole) <= 7 else [SMALL_NUMBERS[int(n)] for n in whole])
            if fraction:
                words.append("point")
                words.extend(SMALL_NUMBERS[int(n)] for n in fraction)
        else:
            words.append(token)
    return words


def word_error_rate(reference, hypothesis):
    expected, actual = normalize(reference), normalize(hypothesis)
    previous = list(range(len(actual) + 1))
    for row, left in enumerate(expected, 1):
        current = [row]
        for column, right in enumerate(actual, 1):
            current.append(min(previous[column] + 1, current[-1] + 1,
                               previous[column - 1] + (left != right)))
        previous = current
    errors = previous[-1]
    return {"wer": errors / len(expected) if expected else None,
            "word_errors": errors, "reference_words": len(expected), "recognized_words": len(actual)}


@dataclass(frozen=True)
class Clip:
    name: str
    expected: str
    pcm: bytes
    clip_bytes: int  # Original synthesized WAV, excluding the appended silence.


def synthetic_clips(directory):
    """Generate a fixed corpus locally; deliberately accepts no user audio path."""
    clips = []
    for name, phrase in PHRASES:
        aiff, wav = Path(directory, name + ".aiff"), Path(directory, name + ".wav")
        subprocess.run(["/usr/bin/say", "-r", "160", "-o", str(aiff), phrase],
                       check=True, capture_output=True, timeout=20)
        subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
                       check=True, capture_output=True, timeout=20)
        with wave.open(str(wav), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, RATE, "NONE"):
                raise ValueError("Invalid synthetic audio format")
            count = audio.getnframes()
            if not 0 < count <= RATE * 30:
                raise ValueError("Invalid synthetic audio duration")
            pcm = audio.readframes(count)
            if len(pcm) != count * 2:
                raise ValueError("Incomplete synthetic audio")
        clips.append(Clip(name, phrase, pcm + bytes(BPS * 2), len(pcm)))
    clips.append(Clip("silence_control", "", bytes(BPS * 3), BPS * 3))
    return clips


def summary(values):
    if not values:
        return {"samples": 0, "p50": None, "p95": None}
    ordered = sorted(values)
    return {"samples": len(values), "p50": statistics.median(ordered),
            "p95": ordered[math.ceil(len(ordered) * .95) - 1]}


async def trial(backend_name, clip, factory, *, startup_timeout=60, sleep=asyncio.sleep,
                monotonic=time.monotonic, wall_time=time.time, chunk_ms=100):
    """One isolated backend instance; times end at callbacks, before delivery/UI."""
    if isinstance(chunk_ms, bool) or not isinstance(chunk_ms, int) or not 10 <= chunk_ms <= 100:
        raise ValueError("Use a feed chunk from 10 to 100 milliseconds")
    chunk_bytes = BPS * chunk_ms // 1000
    started = None
    partial_times, final_times, finals = [], [], []
    first_prefix = None
    prefix = normalize(clip.expected)[:3]
    latencies = defaultdict(lambda: deque(maxlen=256))
    gaps = []
    connection_failures = 0
    any_text = False
    backend = task = None
    sent_bytes = 0
    failure = None
    flushed = closed = False

    def prefix_observed(text):
        nonlocal first_prefix
        if started is not None and prefix and first_prefix is None and normalize(text)[:len(prefix)] == prefix:
            first_prefix = monotonic() - started

    def partial(event):
        nonlocal any_text
        text = event.get("text", "")
        if isinstance(text, str) and text.strip():
            any_text = True
            if started is not None:
                partial_times.append(monotonic() - started)
            prefix_observed(" ".join([*finals, text]))

    def final(text, **timing):
        nonlocal any_text
        if isinstance(text, str) and text.strip():
            any_text = True
            finals.append(text)
            if started is not None:
                final_times.append(monotonic() - started)
            prefix_observed(" ".join(finals))
        return True  # Benchmark acceptance only; no journal or bridge publication.

    def gap(reason, seconds, **timing):
        if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0:
            gaps.append(float(seconds))

    def latency(name, seconds):
        if name in LATENCY_NAMES and isinstance(seconds, (int, float)) and math.isfinite(seconds):
            latencies[name].append(float(seconds))

    def connection(connected, error):
        nonlocal connection_failures
        connection_failures += bool(not connected and error)

    def check_task():
        if task.done():
            # Retrieve but never print an exception: provider text can contain URLs/keys.
            if not task.cancelled():
                task.exception()
            raise RuntimeError("Backend exited before the trial drained")

    try:
        backend = factory(dict(on_partial=partial, on_final=final, on_gap=gap,
                               on_latency=latency, on_connection=connection))
        task = asyncio.create_task(backend.run())
        async with asyncio.timeout(startup_timeout):
            while True:
                check_task()
                ready = getattr(backend, "ready", False)
                if isinstance(ready, asyncio.Event):
                    ready = ready.is_set()
                if ready is True or backend.connected:
                    break
                await sleep(.01)
        started = monotonic()
        for offset in range(0, len(clip.pcm), chunk_bytes):
            check_task()
            raw = clip.pcm[offset:offset + chunk_bytes]
            backend.feed(raw, wall_time(), monotonic())
            sent_bytes += len(raw)
            await sleep(max(0, started + sent_bytes / BPS - monotonic()))
        check_task()
        async with asyncio.timeout(31):
            flushed = await backend.flush_now(timeout=30) is not False
        if not flushed:
            failure = "FlushIncomplete"
    except asyncio.CancelledError:
        raise
    except Exception as error:
        failure = "TimeoutError" if isinstance(error, TimeoutError) else "BackendFailure"
    finally:
        if backend is not None:
            try:
                async with asyncio.timeout(25):
                    closed = await backend.close() is not False
                    if task is not None:
                        await task
                if not closed:
                    failure = failure or "CloseIncomplete"
            except asyncio.CancelledError:
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                raise
            except Exception:
                failure = failure or "CleanupFailure"
            finally:
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
    text = " ".join(finals)
    if not clip.expected and any_text:
        failure = failure or "SilenceProducedText"
    if clip.expected and not finals:
        failure = failure or "NoFinalText"
    if gaps:
        failure = failure or "UntranscribedAudioGap"
    clip_seconds = clip.clip_bytes / BPS
    resolved = getattr(backend, "resolved_model", None) if backend is not None else None
    if not isinstance(resolved, str):
        resolved = None
    return {"backend": backend_name, "clip": clip.name, "status": "failed" if failure else "passed",
            "failure": failure, "expected_text": clip.expected, "final_text": text,
            "feed_chunk_ms": chunk_ms,
            "pipeline": getattr(backend, "pipeline", "direct" if backend_name == "deepgram" else "local"),
            "endpointing_ms": getattr(backend, "endpointing_ms", None),
            **word_error_rate(clip.expected, text), "sent_audio_sec": sent_bytes / BPS,
            "clip_duration_sec": clip_seconds, "appended_silence_sec": (len(clip.pcm) - clip.clip_bytes) / BPS,
            "first_partial_from_first_feed_sec": partial_times[0] if partial_times else None,
            "first_correct_prefix_from_first_feed_sec": first_prefix,
            "prefix_words": len(prefix), "partial_updates": len(partial_times), "final_count": len(finals),
            "final_callbacks_from_first_feed_sec": final_times,
            "final_callbacks_relative_to_clip_end_sec": [value - clip_seconds for value in final_times],
            "last_final_relative_to_clip_end_sec": final_times[-1] - clip_seconds if final_times else None,
            "gap_count": len(gaps), "gap_audio_sec": sum(gaps), "connection_failures": connection_failures,
            "flush_completed": flushed, "close_completed": closed,
            "provider_resolved_model": resolved,
            "provider_latency": {name: summary(values) for name, values in latencies.items()}}


async def benchmark(clips, backends, *, environ, repeats=1, startup_timeout=60, factory=None,
                    sleep=asyncio.sleep, monotonic=time.monotonic, chunk_ms=100):
    results = []
    logger = logging.getLogger("bridge")
    was_disabled = logger.disabled
    logger.disabled = True  # Error strings belong to providers; only controlled failure labels escape.
    try:
        for name in backends:
            for repeat in range(repeats):
                for clip in clips:
                    try:
                        bridge.configure({**environ, "OMI_STT_BACKEND": name, "OMI_LANGUAGE": "en",
                                          "OMI_LANGUAGES": "en", "OMI_HOTWORDS": ""})
                        create = factory if factory is not None else bridge.create_transcription_backend
                        result = await trial(name, clip, create, startup_timeout=startup_timeout,
                                             sleep=sleep, monotonic=monotonic, chunk_ms=chunk_ms)
                        result["requested_model"] = bridge.MODEL_LABEL
                    except Exception:
                        result = {"backend": name, "clip": clip.name, "status": "failed", "failure": "ConfigurationFailure"}
                    results.append({"repeat": repeat + 1, **result})
    finally:
        logger.disabled = was_disabled
    by_backend = {}
    for name in backends:
        speech = [item for item in results if item["backend"] == name and item.get("reference_words", 0)]
        words = sum(item["reference_words"] for item in speech)
        by_backend[name] = {
            "normalized_wer": sum(item["word_errors"] for item in speech) / words if words else None,
            "reference_words": words,
            "first_partial_from_first_feed_sec": summary([item["first_partial_from_first_feed_sec"] for item in speech
                                                           if item["first_partial_from_first_feed_sec"] is not None]),
            "last_final_relative_to_clip_end_sec": summary([item["last_final_relative_to_clip_end_sec"] for item in speech
                                                           if item["last_final_relative_to_clip_end_sec"] is not None]),
            "failed_trials": sum(item["status"] == "failed" for item in results if item["backend"] == name),
        }
    return {"schema_version": 1, "status": "failed" if any(item["status"] == "failed" for item in results) else "passed",
            "audio_origin": "macOS synthetic speech and digital silence only", "backends": backends,
            "overrides": {"language": "en", "publication_languages": ["en"], "hotwords": "disabled"},
            "measurement": "Direct backend callbacks; excludes journal, caption delivery and UI painting. Times start at first feed; clip_end is synthesized WAV end, not measured speech end. Negative final offsets are retained.",
            "normalization": "Lexical WER ignores case/punctuation, expands integers below 10000 and decimal digits, and treats API/HTTP as letters. Contractions, dates, homophones and other formatting may still count as errors.",
            "prefix_measurement": "Earliest provisional or final match of the first three normalized reference words; does not establish stability.",
            "limitations": ["Small synthesized English corpus; not wearable, noisy-room, accent, multilingual or production-scale validation.",
                            "Each trial creates a fresh backend; startup warmup is excluded from callback latencies.",
                            "Cloud mode is paid and uploads these synthetic clips; no private wearable audio or saved transcripts are read."],
            "summary": by_backend, "trials": results}


def redacted(value, secrets):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, list):
        return [redacted(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: redacted(item, secrets) for key, item in value.items()}
    return value


def parse_backends(value):
    names = [part.strip().lower() for part in value.split(",")]
    if not names or any(name not in ("whisper", "deepgram") for name in names):
        raise argparse.ArgumentTypeError("Use whisper, deepgram, or whisper,deepgram.")
    return list(dict.fromkeys(names))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", type=parse_backends, default=["whisper"],
                        help="Default whisper; explicitly adding deepgram permits paid upload of synthetic clips only")
    parser.add_argument("--repeats", type=int, default=1, help="Fresh trials per clip/backend (1..3)")
    parser.add_argument("--startup-timeout", type=float, default=60, help="Backend startup timeout in seconds (1..120)")
    parser.add_argument("--pipeline", choices=("direct", "pipecat"), help="Override Deepgram orchestration for this benchmark only")
    parser.add_argument("--endpointing-ms", type=int, help="Override Deepgram pause threshold for this benchmark only (1..5000)")
    parser.add_argument("--chunk-ms", type=int, default=100, help="Paced input packet duration (10..100; Vexo uses 20)")
    parser.add_argument("--report", type=Path, help="Write redacted JSON results")
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 3 or not math.isfinite(args.startup_timeout) or not 1 <= args.startup_timeout <= 120:
        parser.error("Use repeats 1..3 and startup timeout 1..120 seconds.")
    if not 10 <= args.chunk_ms <= 100 or args.endpointing_ms is not None and not 1 <= args.endpointing_ms <= 5000:
        parser.error("Use chunk 10..100 and endpointing 1..5000 milliseconds.")
    try:
        bridge.require_python315()
        env = load_env()
        if args.pipeline:
            env["OMI_DEEPGRAM_PIPELINE"] = args.pipeline
        if args.endpointing_ms is not None:
            env["OMI_DEEPGRAM_ENDPOINTING_MS"] = str(args.endpointing_ms)
        secrets = [secret for key, value in env.items()
                   if any(word in key.upper() for word in ("TOKEN", "KEY", "SECRET", "PASSWORD"))
                   for secret in (value, value.strip()) if secret]
        print("Benchmarking synthetic English audio through direct callbacks; "
              + ("paid Deepgram upload enabled." if "deepgram" in args.backends else "local Whisper only."), flush=True)
        with tempfile.TemporaryDirectory(prefix="omi-benchmark-") as directory:
            clips = synthetic_clips(directory)
            report = asyncio.run(benchmark(clips, args.backends, environ=env, repeats=args.repeats,
                                          startup_timeout=args.startup_timeout, chunk_ms=args.chunk_ms))
        report = redacted(report, secrets)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        for item in report["trials"]:
            wer, draft, final = item.get("wer"), item.get("first_partial_from_first_feed_sec"), item.get("last_final_relative_to_clip_end_sec")
            line = f"{item['backend']} / {item['clip']}: {item['status']}"
            if item.get("failure"):
                line += " (" + item["failure"] + ")"
            line += f"; WER {wer:.3f}" if wer is not None else "; WER n/a"
            line += f"; first partial {draft:.3f}s" if draft is not None else "; first partial n/a"
            line += f"; final from clip end {final:+.3f}s" if final is not None else "; final from clip end n/a"
            print(line)
        return 0 if report["status"] == "passed" else 1
    except KeyboardInterrupt:
        print("Benchmark interrupted; owned backends were asked to close.", file=sys.stderr)
        return 130
    except Exception:
        print("Benchmark failed; check runtime, synthetic audio tools and backend configuration. Provider details are redacted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
