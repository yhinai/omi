#!/usr/bin/env python3.15
"""Check audio -> local Whisper -> captions and saved transcripts in a temporary bridge."""
import argparse
import asyncio
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge
from config import load_env
from aiohttp.test_utils import TestClient, TestServer

PHRASE = "The bridge test is working. Blue lantern seven."
CHUNK_BYTES = 3200


def normalize(text):
    return " ".join("seven" if word == "7" else word for word in re.findall(r"\w+", text.casefold()))


def load_audio(args, directory):
    path = args.wav
    if path is None:
        aiff, path = Path(directory) / "speech.aiff", Path(directory) / "speech.wav"
        subprocess.run(["/usr/bin/say", "-r", "145", "-o", str(aiff), args.expect],
                       check=True, capture_output=True, timeout=15)
        subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                        str(aiff), str(path)], check=True, capture_output=True, timeout=15)
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, "NONE"):
            raise RuntimeError("--wav must contain uncompressed mono PCM16 at 16000 Hz")
        if not 0 < audio.getnframes() <= 16000 * 30:
            raise RuntimeError("Use a speech clip longer than zero and no longer than 30 seconds")
        pcm = audio.readframes(audio.getnframes())
        if len(pcm) != audio.getnframes() * 2:
            raise RuntimeError("The WAV file contains incomplete audio frames")
    return pcm + bytes(32000 * 2)


def verify_saved(directory, finals, expected):
    saved = [json.loads(line) for line in Path(directory, "omi_transcript_events.jsonl").read_text().splitlines()]
    if not saved or any(event.get("type") != "transcript" or event.get("is_final") is not True for event in saved):
        raise RuntimeError("Expected only finalized transcripts in the saved JSONL")
    if not {event["event_id"] for event in finals} <= {event["event_id"] for event in saved}:
        raise RuntimeError("A displayed final was missing from the saved JSONL")
    if normalize(expected) not in normalize(" ".join(event["text"] for event in saved)):
        raise RuntimeError("The saved transcript did not contain the expected phrase")
    markdown = normalize(" ".join(path.read_text() for path in Path(directory).glob("*.md")))
    if any(normalize(event["text"]) not in markdown for event in saved):
        raise RuntimeError("A saved transcript was missing from daily Markdown")


async def check(args):
    env = load_env()
    if getattr(args, "backend", None):
        env["OMI_STT_BACKEND"] = args.backend
    if getattr(args, "pipeline", None):
        env["OMI_DEEPGRAM_PIPELINE"] = args.pipeline
    bridge.configure(env)
    with tempfile.TemporaryDirectory(prefix="omi-smoke-") as directory:
        pcm = load_audio(args, directory)
        bridge.LOG_DIR = directory
        bridge.configure_logging()
        reader = None
        try:
            async with asyncio.timeout(args.timeout), TestClient(TestServer(bridge.create_app())) as client:
                while not getattr(bridge.state.live_stt, "ready", bridge.state.live_stt.connected):
                    await asyncio.sleep(.05)
                partials, finals = [], []
                started = None
                complete = asyncio.Event()
                caption_headers = {"Authorization": "Bearer " + bridge.CAPTION_TOKEN} if bridge.CAPTION_TOKEN else {}
                async with client.ws_connect("/listen?replay=0", headers=caption_headers) as captions:
                    async def receive():
                        async for message in captions:
                            if message.type != bridge.aiohttp.WSMsgType.TEXT:
                                continue
                            event = json.loads(message.data)
                            if event.get("type") == "partial" and event.get("text"):
                                partials.append({**event, "observed_sec": time.monotonic()-started})
                            if event.get("type") == "transcript":
                                finals.append(event)
                                if normalize(args.expect) in normalize(" ".join(item["text"] for item in finals)):
                                    complete.set()
                    reader = asyncio.create_task(receive())
                    headers = {"Authorization": "Bearer " + bridge.STREAM_TOKEN} if bridge.STREAM_TOKEN else {}
                    async with client.ws_connect("/v1/listen", headers=headers) as phone:
                        started = time.monotonic()
                        for offset in range(0, len(pcm), CHUNK_BYTES):
                            await phone.send_bytes(pcm[offset:offset + CHUNK_BYTES])
                            await asyncio.sleep(max(0, started+(offset+CHUNK_BYTES)/32000-time.monotonic()))
                        await phone.send_json({"type": "CloseStream"})
                        await complete.wait()
                    if not partials:
                        raise RuntimeError("No partial captions were received")
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            # Also validate any final events emitted during graceful shutdown.
            verify_saved(directory, finals, args.expect)
            report = {"status": "passed", "backend": bridge.STT_BACKEND,
                "pipeline": getattr(bridge.state.live_stt, "pipeline", "direct" if bridge.STT_BACKEND == "deepgram" else "local"),
                "partial_updates": len(partials), "finals": len(finals),
                "first_partial_sec": partials[0]["observed_sec"],
                "latency": bridge.build_health_dict()["latency"],
                "pipeline_frames": getattr(bridge.state.live_stt, "pipeline_frames", {}),
                "recognition_provenance": {k:v for k,v in finals[0].get("recognition", {}).items()
                    if k in {"provider", "orchestration", "pipecat_version", "deepgram_sdk_version", "requested_model", "resolved_model", "model_version"}},
                "jsonl_verified": True, "markdown_verified": True}
            if getattr(args, "report", None):
                args.report.write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(report), flush=True)
        finally:
            if reader is not None:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            logging.shutdown()


def main():
    bridge.require_python315()
    parser = argparse.ArgumentParser(description=__doc__ + " Runs locally; normal logs stay untouched.")
    parser.add_argument("--report", type=Path, help="Write timing and verification results as JSON")
    parser.add_argument("--backend", choices=("whisper", "deepgram"),
                        help="Select a backend for this temporary test; deepgram sends synthesized/supplied audio to the cloud")
    parser.add_argument("--pipeline", choices=("direct", "pipecat"), help="Deepgram orchestration for this isolated check")
    parser.add_argument("--wav", type=Path, help="Mono PCM16/16000 Hz WAV, at most 30 seconds; otherwise uses macOS speech")
    parser.add_argument("--expect", help="Phrase that must appear in the saved transcript; required with --wav")
    parser.add_argument("--timeout", type=float, default=60, help="Overall streaming timeout in seconds (8-120)")
    args = parser.parse_args()
    if args.wav is not None and not args.expect:
        parser.error("--wav requires --expect")
    args.expect = args.expect or PHRASE
    if not normalize(args.expect) or not 8 < args.timeout <= 120:
        parser.error("Use a nonempty phrase and a timeout greater than 8 and at most 120 seconds")
    print("Running an isolated recognition check; normal logs stay untouched.", flush=True)
    try:
        asyncio.run(check(args))
    except (TimeoutError, bridge.aiohttp.ClientError, ValueError, OSError, RuntimeError,
            subprocess.SubprocessError, wave.Error, EOFError) as error:
        # Provider errors can contain request URLs; avoid exposing credentials or unrelated text.
        sys.exit(f"Smoke check failed ({type(error).__name__}); check configuration, audio, and recognizer readiness.")


if __name__ == "__main__":
    main()
