#!/usr/bin/env python3.15
"""Opt-in real speech -> bridge -> Gemini -> new transcript integration check."""
import argparse
import asyncio
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import tempfile
import time
import wave

if (sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 15)
        or sys.version_info.releaselevel != "final" or sysconfig.get_config_var("Py_GIL_DISABLED")):
    sys.exit("This check requires stable, standard CPython 3.15.x with GIL enabled.")

import aiohttp

PHRASE = "The bridge test is working. Blue lantern seven."
CHUNK_BYTES = 16000 * 2


def normalize(text):
    # Live STT can finalize the two sentences separately; timestamps are not speech.
    text = re.sub(r"(?m)^\[\d{4}-\d\d-\d\d [^\]]+\]\s*", "", text)
    words = re.findall(r"\w+", text.casefold())
    return " ".join("seven" if word == "7" else word for word in words)


class NewTranscript:
    def __init__(self, path):
        self.path, self.identity, self.offset = path, None, 0
        if path.exists():
            stat = path.stat()
            self.identity, self.offset = (stat.st_dev, stat.st_ino), stat.st_size

    def read(self):
        if not self.path.exists():
            return ""
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if (self.identity is not None and identity != self.identity) or stat.st_size < self.offset:
            raise RuntimeError("Transcript rotated/truncated during the check; repeat with an idle bridge.")
        self.identity = identity
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read(256 * 1024)
            self.offset = stream.tell()
        return data.decode("utf-8", errors="replace")


def load_audio(args, directory):
    path = args.wav
    if path is None:
        aiff, path = Path(directory) / "speech.aiff", Path(directory) / "speech.wav"
        try:
            subprocess.run(["/usr/bin/say", "-r", "145", "-o", str(aiff), args.expect],
                           check=True, capture_output=True, timeout=15)
            subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                            str(aiff), str(path)], check=True, capture_output=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            raise RuntimeError("Speech generation needs macOS say/afconvert; supply --wav and --expect instead.") from None
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, "NONE"):
            raise RuntimeError("--wav must contain uncompressed mono PCM16 audio at 16000 Hz.")
        if not 0 < audio.getnframes() <= 16000 * 30:
            raise RuntimeError("Use a speech clip longer than zero and no longer than 30 seconds.")
        pcm = audio.readframes(audio.getnframes())
        if len(pcm) != audio.getnframes() * 2:
            raise RuntimeError("The WAV file contains incomplete audio frames.")
    # Use complete one-second uploads, matching the Omi setting.
    size = max(2 * CHUNK_BYTES, ((len(pcm) + CHUNK_BYTES - 1) // CHUNK_BYTES) * CHUNK_BYTES)
    return pcm + bytes(size - len(pcm))


async def health(session, url):
    async with session.get(url) as response:
        if response.status not in (200, 503):
            raise RuntimeError(f"Health request returned HTTP {response.status}.")
        result = await response.json()
        if not isinstance(result, dict):
            raise RuntimeError("Health must return a JSON object.")
        return result


async def check(args):
    with tempfile.TemporaryDirectory(prefix="omi-speech-check-") as directory:
        pcm = load_audio(args, directory)
        async with asyncio.timeout(args.timeout), aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            before = await health(session, args.health_url)
            if before.get("status") != "ok" or not str(before.get("python_version", "")).startswith("3.15."):
                raise RuntimeError("Bridge health must report ready on Python 3.15 before this check.")
            age = before.get("last_packet_age_sec")
            busy = ((age is not None and age < 8) or before.get("queue_depth", 0)
                    or before.get("buffer_bytes", 0) or before.get("stt_in_flight", False))
            if busy and not args.allow_active:
                raise RuntimeError("Turn Omi off and wait for pending audio to drain; --allow-active explicitly permits mixing.")
            cursor, collected = NewTranscript(args.transcript_log), ""
            deadline = time.monotonic() + args.timeout
            print("Sending controlled speech to the real bridge/Gemini API; this incurs normal API charges.", flush=True)
            for offset in range(0, len(pcm), CHUNK_BYTES):
                if offset:
                    await asyncio.sleep(5)
                async with session.post(args.endpoint, data=pcm[offset:offset + CHUNK_BYTES],
                                        headers={"Content-Type": "application/octet-stream"}, allow_redirects=False) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Audio upload returned HTTP {response.status}; the check did not pass.")
                    await response.read()
            last_upload = time.monotonic()
            while time.monotonic() < deadline:
                collected = (collected + cursor.read())[-256 * 1024:]
                text = re.sub(r"^\[[^\]\n]+\]\s*", "", collected, flags=re.MULTILINE)
                if normalize(args.expect) in normalize(text) and time.monotonic() - last_upload >= 8:
                    after = await health(session, args.health_url)
                    drained = args.allow_active or not any(after.get(key, 0) for key in ("queue_depth", "buffer_bytes", "stt_in_flight"))
                    if drained and after.get("status") == "ok" and after.get("windows_processed", 0) > before.get("windows_processed", 0):
                        print("PASS: controlled speech produced the expected NEW transcript through real Gemini.")
                        print("This verifies injected audio; the physical wearable/BLE/phone hop needs its own check.")
                        return
                await asyncio.sleep(0.5)
            raise RuntimeError("Timed out without a verified new transcript; inspect local health and bridge errors.")


def main():
    parser = argparse.ArgumentParser(description=__doc__ + " Uses the real paid Gemini API.")
    parser.add_argument("--wav", type=Path, help="Uncompressed mono PCM16/16000 Hz speech WAV, at most 30 seconds")
    parser.add_argument("--expect", help="Phrase that must appear in newly appended transcript text")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8787/omi", help="POST URL; use your public Funnel URL to test ingress")
    parser.add_argument("--health-url", default="http://127.0.0.1:8788/health")
    parser.add_argument("--transcript-log", type=Path, default=Path(__file__).resolve().parents[1] / "logs/omi_live_transcript.log")
    parser.add_argument("--timeout", type=float, default=120, help="Overall verification deadline in seconds (8-120)")
    parser.add_argument("--allow-active", action="store_true", help="Permit concurrent wearable audio/pending work; may mix streams")
    args = parser.parse_args()
    if args.wav is not None and not args.expect:
        parser.error("--wav requires --expect")
    args.expect = args.expect or PHRASE
    if not normalize(args.expect) or not 8 < args.timeout <= 120:
        parser.error("Supply a nonempty expected phrase and a timeout greater than 8 and at most 120 seconds")
    try:
        asyncio.run(check(args))
    except TimeoutError:
        sys.exit("Smoke check exceeded its deadline; inspect local health and bridge errors.")
    except aiohttp.ClientError as error:
        sys.exit(f"Smoke check failed ({type(error).__name__}); check endpoint connectivity and local health.")
    except (RuntimeError, OSError, wave.Error, EOFError) as error:
        sys.exit(f"Smoke check failed: {error}")


if __name__ == "__main__":
    main()
