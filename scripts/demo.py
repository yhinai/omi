#!/usr/bin/env python3.15
"""Run the real caption viewer against isolated synthetic fixtures; no ASR or live history."""
import argparse
import asyncio
from contextlib import suppress
from datetime import datetime
import hmac
import json
import logging
import os
from pathlib import Path
import secrets
import signal
import sys
import tempfile
import termios
import time
import wave

import aiohttp
from aiohttp import web

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RECORDED_WAV = PROJECT_DIR / "assets/demo.wav"
SOCKETS = web.AppKey("demo_sockets", set)


def fixture_events(stream_id, started):
    """A finite, explicitly synthetic timeline, with no inference measurements."""
    def final(identity, text, offset, session):
        source = started + offset
        return {"type": "transcript", "event_id": identity, "utterance_id": identity,
                "stream_id": stream_id, "session_id": session, "synthetic": True,
                "is_final": True, "text": text, "audio_start_at": source,
                "sample_offset": int(offset * 16000), "sample_end_offset": int((offset + .2) * 16000),
                "timestamp": datetime.fromtimestamp(source).strftime("%Y-%m-%d %H:%M:%S")}

    status = {"type": "status", "stream_id": stream_id, "synthetic": True,
              "stt_backend": "synthetic-demo", "model": "SYNTHETIC DEMO — no ASR model",
              "backend_ready": True, "provider_connection_state": "synthetic",
              "provider_resolved_model": "fixture-only",
              "workers_alive": True, "stream_connected": False, "audio_active": False,
              "last_packet_age_sec": None, "buffer_sec": 0, "queue_depth": 0,
              "stt_in_flight": False, "last_error": None}
    draft = {"type": "partial", "utterance_id": "demo-turn", "stream_id": stream_id,
             "session_id": 2, "synthetic": True, "audio_start_at": started + 3}
    return [
        (0, final("demo-label", "SYNTHETIC DEMO · Scripted captions; no microphone, "
                  "speech recognition, or live history is connected.", 0, 1)),
        (.2, {**status, "stt_in_flight": True}),
        (.8, {**draft, "revision": 1, "text": "The bridge is getting ready"}),
        (.8, {**draft, "revision": 2, "text": "The bridge is ready to demonstrate"}),
        (1, final("demo-turn", "The bridge is ready to demo.", 3, 2)),
        (1.4, final("demo-unicode", "Unicode stays intact: 👨‍👩‍👧 · می\u200cروم.", 6, 3)),
        (1, {"type": "gap", "event_id": "demo-gap", "stream_id": stream_id,
             "synthetic": True, "text": "Synthetic gap notice · no real audio was lost."}),
        (1.2, final("demo-complete", "Demo complete. Try / to search, Space to hold, "
                   "d for details, and q to exit. All captions here are scripted.", 9, 4)),
        (0, status),
    ]


def create_demo_app(token, *, pace=1):
    """Expose synthetic captions only; no phone ingress, model, or journal routes."""
    app = web.Application()
    app[SOCKETS] = set()
    events = fixture_events(secrets.token_hex(8), time.time())

    async def listen(request):
        candidate = request.headers.get("Authorization", "").encode("utf-8")
        if not hmac.compare_digest(candidate, ("Bearer " + token).encode("utf-8")):
            raise web.HTTPUnauthorized(text="Demo caption token required")
        after = request.query.get("after")
        index = next((i for i, (_, event) in enumerate(events) if event.get("event_id") == after), None) if after else None
        ws = web.WebSocketResponse(heartbeat=20, timeout=1)
        await ws.prepare(request)
        app[SOCKETS].add(ws)

        async def send():
            try:
                await ws.send_json({"type": "replay", "phase": "start", "count": 0,
                                    "reset": bool(after and index is None), "synthetic": True})
                await ws.send_json({"type": "replay", "phase": "end", "synthetic": True})
                for delay, event in events[index + 1 if index is not None else 0:]:
                    await asyncio.sleep(delay * pace)
                    await ws.send_json(event)
                while True:
                    await asyncio.sleep(1)
                    await ws.send_json(events[-1][1])
            finally:
                await ws.close()

        sender = asyncio.create_task(send())
        try:
            async for _ in ws:
                pass
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            app[SOCKETS].discard(ws)
            await ws.close()
        return ws

    async def close_sockets(application):
        await asyncio.gather(*(ws.close() for ws in list(application[SOCKETS])), return_exceptions=True)

    app.router.add_get("/listen", listen)
    app.on_shutdown.append(close_sockets)
    return app


async def run_demo():
    """Own the temporary server and viewer child until q, interruption, or failure."""
    token = secrets.token_urlsafe(32)
    runner = web.AppRunner(create_demo_app(token), access_log=None)
    child = None
    try:
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        port = runner.addresses[0][1]
        with tempfile.TemporaryDirectory(prefix="omi-synthetic-demo-") as directory:
            env = dict(os.environ, OMI_CAPTION_TOKEN=token, PYTHONDONTWRITEBYTECODE="1")
            env.pop("OMI_TUI_MOUSE_LOG", None)
            env.pop("OMI_STREAM_TOKEN", None)
            try:
                launch = asyncio.create_task(asyncio.create_subprocess_exec(
                    sys.executable, "-B", str(PROJECT_DIR / "tui.py"),
                    "--url", f"ws://127.0.0.1:{port}/listen", "--export-dir", directory,
                    env=env, cwd=directory,
                ))
                try:
                    child = await asyncio.shield(launch)
                except asyncio.CancelledError:
                    # A cancelled subprocess launch can still create a process.
                    # Acquire its handle before cleanup so it cannot be orphaned.
                    child = await launch
                    raise
                return await child.wait()
            finally:
                if child is not None and child.returncode is None:
                    with suppress(ProcessLookupError):
                        child.send_signal(signal.SIGINT)
                    try:
                        await asyncio.wait_for(child.wait(), 2)
                    except TimeoutError:
                        with suppress(ProcessLookupError):
                            child.kill()
                        await child.wait()
    finally:
        await runner.cleanup()


def load_recorded_demo_pcm(wav_path):
    chosen = Path(wav_path).expanduser() if wav_path is not None else DEFAULT_RECORDED_WAV
    with wave.open(str(chosen), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError("--wav must be uncompressed mono PCM16 at 16000 Hz")
        frames = audio.getnframes()
        if not 0 < frames <= 16000 * 120:
            raise ValueError("--wav must be between 0 and 120 seconds")
        pcm = audio.readframes(frames)
        if len(pcm) != frames * 2:
            raise ValueError("--wav contains incomplete PCM16 frames")
    return pcm + bytes(16000 * 2), str(chosen.name)


async def run_recorded_demo(wav_path=None, *, headless=False):
    """Run an isolated real local Whisper bridge + viewer fed by recorded 16 kHz PCM16 audio."""
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    import bridge
    from config import load_env

    stream_token = secrets.token_urlsafe(32)
    caption_token = secrets.token_urlsafe(32)
    with tempfile.TemporaryDirectory(prefix="omi-recorded-demo-") as directory:
        pcm, source_label = load_recorded_demo_pcm(wav_path)
        base_env = load_env()
        isolated_env = {
            **base_env,
            "OMI_STT_BACKEND": "whisper",
            "OMI_STREAM_TOKEN": stream_token,
            "OMI_CAPTION_TOKEN": caption_token,
            "OMI_LANGUAGE": base_env.get("OMI_LANGUAGE", "en"),
            "OMI_LANGUAGES": base_env.get("OMI_LANGUAGES", "en"),
        }
        bridge.LOG_DIR = directory
        bridge.state = bridge.BridgeState()
        bridge.configure(isolated_env)
        bridge.configure_logging()
        runner = web.AppRunner(bridge.create_app(start_workers=True), access_log=None)
        child = feeder = None
        try:
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", 0).start()
            port = runner.addresses[0][1]
            async with asyncio.timeout(45):
                while not (bridge.state.live_stt and bridge.state.live_stt.connected):
                    await asyncio.sleep(0.05)

            banner = bridge.make_event(
                "transcript",
                "RECORDED AUDIO FALLBACK · Streaming pre-recorded 16 kHz PCM16 audio through the real local "
                "Silero VAD + MLX Whisper Turbo worker (isolated port & temporary journal; live wearable stream "
                "untouched). Press b to bookmark, [ / ] to jump, d for live MLX latency, e to export, q to exit.",
                audio_start_at=time.time() - 3.0,
                session_id=0,
                utterance_id="recorded-fallback-banner",
            )
            banner["synthetic"] = True
            bridge.write_event(banner)

            async def feed_pcm():
                await asyncio.sleep(0.35)
                timeout = aiohttp.ClientTimeout(total=None, connect=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.ws_connect(
                        f"ws://127.0.0.1:{port}/v1/listen",
                        headers={"Authorization": f"Bearer {stream_token}"},
                    ) as phone_ws:
                        started = time.monotonic()
                        frame_bytes = 320  # 10 ms @ 16 kHz PCM16 (matches real Omi phone cadence)
                        for idx, offset in enumerate(range(0, len(pcm), frame_bytes), 1):
                            await phone_ws.send_bytes(pcm[offset : offset + frame_bytes])
                            target = started + idx * 0.01
                            delay = target - time.monotonic()
                            if delay > 0:
                                await asyncio.sleep(delay)
                        await phone_ws.send_json({"type": "CloseStream"})

            if headless:
                partials, finals = [], []
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        f"ws://127.0.0.1:{port}/listen?replay=100",
                        headers={"Authorization": f"Bearer {caption_token}"},
                    ) as cap_ws:
                        feeder = asyncio.create_task(feed_pcm())
                        stop_collect = asyncio.Event()

                        async def collect():
                            while not stop_collect.is_set():
                                try:
                                    ev = await cap_ws.receive_json(timeout=0.1)
                                except TimeoutError:
                                    continue
                                if ev.get("type") == "partial" and ev.get("text"):
                                    partials.append(ev)
                                elif ev.get("type") == "transcript" and not ev.get("synthetic"):
                                    finals.append(ev)

                        collector = asyncio.create_task(collect())
                        await feeder
                        await bridge.state.live_stt.flush_now(timeout=15)
                        await asyncio.sleep(0.15)
                        stop_collect.set()
                        await collector
                health = bridge.build_health_dict()
                receipt = {
                    "status": "passed" if finals else "failed",
                    "mode": "recorded_audio_real_whisper",
                    "labeled_fallback": True,
                    "audio_source_file": source_label,
                    "audio_duration_sec": round(len(pcm) / 32000, 2),
                    "worker_load_warm_sec": health["worker_load_warm_sec"],
                    "partial_updates": len(partials),
                    "finals": len(finals),
                    "latency": health["latency"],
                    "transcript": " ".join(item["text"] for item in finals),
                }
                print(json.dumps(receipt, indent=2), flush=True)
                return 0 if finals else 1

            feeder = asyncio.create_task(feed_pcm())
            env = dict(os.environ, OMI_CAPTION_TOKEN=caption_token, PYTHONDONTWRITEBYTECODE="1")
            env.pop("OMI_TUI_MOUSE_LOG", None)
            env.pop("OMI_STREAM_TOKEN", None)
            try:
                launch = asyncio.create_task(asyncio.create_subprocess_exec(
                    sys.executable, "-B", str(PROJECT_DIR / "tui.py"),
                    "--url", f"ws://127.0.0.1:{port}/listen", "--export-dir", directory,
                    env=env, cwd=directory,
                ))
                try:
                    child = await asyncio.shield(launch)
                except asyncio.CancelledError:
                    child = await launch
                    raise
                return await child.wait()
            finally:
                if child is not None and child.returncode is None:
                    with suppress(ProcessLookupError):
                        child.send_signal(signal.SIGINT)
                    try:
                        await asyncio.wait_for(child.wait(), 2)
                    except TimeoutError:
                        with suppress(ProcessLookupError):
                            child.kill()
                        await child.wait()
        finally:
            if feeder is not None and not feeder.done():
                feeder.cancel()
                await asyncio.gather(feeder, return_exceptions=True)
            await runner.cleanup()
            for lg in (bridge.logger, bridge.event_logger):
                for handler in lg.handlers[:]:
                    lg.removeHandler(handler)
                    handler.close()
            logging.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="Press q or Ctrl+C to exit. Default mode uses scripted fixtures; --recorded-audio runs real local Whisper on recorded PCM16.")
    parser.add_argument("--recorded-audio", action="store_true",
                        help="Labeled recorded-audio fallback: stream 16 kHz PCM16 WAV into an isolated real local Whisper bridge")
    parser.add_argument("--wav", type=Path,
                        help="Optional 16 kHz mono PCM16 WAV for --recorded-audio (defaults to assets/demo.wav)")
    parser.add_argument("--headless", action="store_true",
                        help="With --recorded-audio, verify real local Whisper captions without opening curses")
    args = parser.parse_args(argv)
    if args.wav is not None and not args.recorded_audio:
        parser.error("--wav requires --recorded-audio")
    if args.headless and not args.recorded_audio:
        parser.error("--headless requires --recorded-audio")
    if sys.version_info[:2] != (3, 15):
        parser.error("Run with the project's .venv/bin/python3.15.")
    if args.recorded_audio and args.headless:
        try:
            return asyncio.run(run_recorded_demo(args.wav, headless=True))
        except KeyboardInterrupt:
            return 130
        except (OSError, RuntimeError, ValueError, TimeoutError, wave.Error):
            parser.exit(1, "Recorded-audio demo check failed; inspect local Whisper configuration and WAV format.\n")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("Open a real terminal to run the synthetic viewer demo.")
    terminal_state = termios.tcgetattr(sys.stdin.fileno())
    try:
        if args.recorded_audio:
            return asyncio.run(run_recorded_demo(args.wav))
        return asyncio.run(run_demo())
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError, TimeoutError, wave.Error):
        label = "Recorded-audio demo" if args.recorded_audio else "Synthetic demo"
        parser.exit(1, f"{label} could not start; check the local Python environment and audio assets.\n")
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal_state)


if __name__ == "__main__":
    sys.exit(main())
