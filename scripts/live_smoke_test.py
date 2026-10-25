#!/usr/bin/env python3.15
"""Opt-in real Gemini streaming check in an isolated bridge with temporary logs."""
import asyncio
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import omi_bridge as bridge
from aiohttp.test_utils import TestClient, TestServer

PHRASE = "The bridge test is working. Blue lantern seven."


def normalized(text):
    return " ".join("seven" if word == "7" else word for word in re.findall(r"\w+", text.casefold()))


async def check():
    bridge.require_python315()
    bridge.configure()
    bridge.STT_MODE = "live"
    with tempfile.TemporaryDirectory(prefix="omi-isolated-live-") as root:
        aiff, wav = Path(root) / "speech.aiff", Path(root) / "speech.wav"
        subprocess.run(["/usr/bin/say", "-r", "145", "-o", str(aiff), PHRASE], check=True, capture_output=True)
        subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                        str(aiff), str(wav)], check=True, capture_output=True)
        with wave.open(str(wav)) as audio:
            pcm = audio.readframes(audio.getnframes()) + bytes(32000 * 2)
        bridge.LOG_DIR = root
        bridge.TRANSCRIPT_LOG = str(Path(root) / "omi_live_transcript.log")
        bridge.configure_logging()
        client = TestClient(TestServer(bridge.create_app()))
        await client.start_server()
        reader = None
        try:
            async with asyncio.timeout(40):
                while not bridge.state.live_stt.connected:
                    await asyncio.sleep(.05)
                async with client.ws_connect("/listen-ws?mode=captions&replay=0") as ws:
                    partials, finals = [], []
                    complete = asyncio.Event()
                    async def receive():
                        async for message in ws:
                            if message.type not in (bridge.aiohttp.WSMsgType.TEXT, bridge.aiohttp.WSMsgType.BINARY):
                                continue
                            event = json.loads(message.data)
                            if event.get("type") == "partial" and event.get("text"):
                                partials.append(event)
                            if event.get("type") == "transcript":
                                finals.append(event)
                                if normalized(PHRASE) in normalized(" ".join(item["text"] for item in finals)):
                                    complete.set()
                    reader = asyncio.create_task(receive())
                    for offset in range(0, len(pcm), 32000):
                        response = await client.post("/omi", data=pcm[offset:offset+32000],
                            headers={"Content-Type": "application/octet-stream"})
                        assert response.status == 200
                        await response.read()
                        await asyncio.sleep(1)
                    await asyncio.wait_for(complete.wait(), 15)
                    assert partials and finals, "Both partial and final events are required"
                    saved = [json.loads(line) for line in Path(root, "omi_transcript_events.jsonl").read_text().splitlines()]
                    assert all(event["type"] != "partial" for event in saved)
                    assert all(event.get("is_final") is True for event in saved if event["type"] == "transcript")
                    assert {event["event_id"] for event in finals} <= {event["event_id"] for event in saved}
                    assert not any(event["type"] == "gap" for event in saved), "Unexpected gap in controlled stream"
                    health = bridge.build_health_dict()
                    print(json.dumps({"status": "passed", "partial_updates": len(partials), "finals": len(finals),
                        "receipt_to_first_partial_sec": health["latency"]["receipt_to_first_partial_sec"],
                        "receipt_to_final_sec": health["latency"]["receipt_to_final_sec"],
                        "final_only_persistence": True}), flush=True)
        finally:
            if reader is not None:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            await client.close()
            logging.shutdown()


if __name__ == "__main__":
    print("Running isolated real-Gemini streaming check; normal API charges apply.", flush=True)
    asyncio.run(check())
