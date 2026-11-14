#!/usr/bin/env python3.15
"""Observe live Omi stages without sending audio or replacing the phone stream.

Default traffic is loopback only. --phrase waits for a new, explicitly named test
utterance; unrelated captions are discarded. --public checks only the fixed public
caption endpoint's unauthenticated HTTP status and never sends credentials there.
"""
import argparse
import asyncio
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.doctor import environment
import tui

HEALTH_URL = "http://127.0.0.1:8788/health"
CAPTION_URL = "ws://127.0.0.1:8787/listen?replay=0"
PUBLIC_URL = "https://omi.alhinai.dev/listen"
MAX_BODY = 65536
MAX_MESSAGE = 262144
MAX_MESSAGES = 1024
MAX_LINE = 524288
MAX_FILE_SCAN = 8 * 1024 * 1024
STAGES = ("configuration", "controlPlane", "backendReady", "phoneConnected", "streamPacketProgress",
          "captionSubscription", "captionStatus", "newEventsOnly", "providerConnected", "resolvedModel",
          "providerFinal", "provenance", "journal", "markdown", "viewerConsumption", "terminalRenderer", "publicCaptionAuth")
NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()


def normalize(text):
    return [NUMBER_WORDS[int(word)] if word.isdigit() and len(word) <= 2 and int(word) <= 20 else word
            for word in re.findall(r"\w+", text.casefold())]


def contains(text, phrase):
    return " " + " ".join(normalize(phrase)) + " " in " " + " ".join(normalize(text)) + " "


def redact(value, secrets):
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: redact(item, secrets) for key, item in value.items()}
    return value


async def read_health(session):
    async with asyncio.timeout(4), session.get(HEALTH_URL, allow_redirects=False) as response:
        if response.status not in (200, 503):
            raise ValueError("Unexpected health status")
        data = bytearray()
        while chunk := await response.content.read(8192):
            data.extend(chunk)
            if len(data) > MAX_BODY:
                raise ValueError("Health response exceeds bound")
        health = json.loads(data)
        if not isinstance(health, dict) or health.get("stt_backend") not in ("whisper", "deepgram"):
            raise ValueError("Invalid health metadata")
        return health


def safe_model(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_. -]{1,160}", value) else None


def source_metadata_valid(event):
    start, end = event.get("sample_offset"), event.get("sample_end_offset")
    session = event.get("session_id")
    return (isinstance(start, int) and not isinstance(start, bool) and isinstance(end, int)
            and not isinstance(end, bool) and 0 <= start < end and isinstance(session, int)
            and not isinstance(session, bool) and session >= 0)


def journal_observation(root, event):
    """Scan bounded lines; decode only candidates containing this matched event ID."""
    identity = event.get("event_id", "")
    if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity):
        return "failed", "Matched event has no usable journal identity."
    needle = identity.encode("ascii")
    complete = True
    for suffix in ("", ".1", ".2", ".3"):
        path = Path(root, "logs", "omi_transcript_events.jsonl" + suffix)
        try:
            if not path.exists():
                continue
            if path.is_symlink() or not path.is_file():
                complete = False
                continue
            with path.open("rb") as source:
                consumed = 0
                while consumed < MAX_FILE_SCAN:
                    line = source.readline(min(MAX_LINE + 1, MAX_FILE_SCAN - consumed))
                    if not line:
                        break
                    consumed += len(line)
                    if len(line) > MAX_LINE or not line.endswith(b"\n"):
                        complete = False
                        continue
                    if needle not in line:
                        continue
                    try:
                        saved = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if not isinstance(saved, dict) or saved.get("event_id") != identity:
                        continue
                    keys = ("type", "text", "sample_offset", "sample_end_offset", "session_id", "stream_id", "recognition")
                    if all(saved.get(key) == event.get(key) for key in keys):
                        return "observed", "Matched final and its source/provenance metadata are present in JSONL."
                    return "failed", "Matched journal identity differs from the received final or its provenance."
                if source.read(1):
                    complete = False
        except OSError:
            complete = False
    return ("failed", "Matched event was absent from the completely scanned journals.") if complete else (
        "not_tested", "Bounded journal scan could not establish persistence of the matched event.")


def markdown_observation(root, event):
    stamp = event.get("timestamp", "")
    try:
        if not isinstance(stamp, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", stamp):
            raise ValueError("Invalid timestamp")
        datetime.fromisoformat(stamp)
        path = Path(root, "logs", stamp[:10] + ".md")
        if path.is_symlink() or not path.is_file():
            return "failed", "Daily Markdown for the matched event is unavailable."
        target = normalize(event.get("text", ""))
        tail = []
        with path.open("rb") as source:
            size = source.seek(0, os.SEEK_END)
            source.seek(max(0, size - MAX_FILE_SCAN))
            if source.tell():
                source.readline(MAX_LINE + 1)
            complete = size <= MAX_FILE_SCAN
            consumed = 0
            while consumed < MAX_FILE_SCAN:
                line = source.readline(min(MAX_LINE + 1, MAX_FILE_SCAN - consumed))
                if not line:
                    break
                consumed += len(line)
                if len(line) > MAX_LINE:
                    complete = False
                    tail = []
                    continue
                words = tail + normalize(line.decode("utf-8", errors="replace"))
                if target and " " + " ".join(target) + " " in " " + " ".join(words) + " ":
                    return "observed", "Daily projection contains the matched final text; Markdown has no event IDs."
                tail = words[-max(0, len(target) - 1):] if len(target) > 1 else []
        return ("failed", "Matched final text was absent from the scanned daily Markdown.") if complete else (
            "not_tested", "Bounded Markdown scan could not establish projection of the matched final.")
    except (OSError, ValueError):
        return "failed", "Could not inspect the matched event's daily Markdown."


def verify_sample(root, event, backend, stages):
    def set_stage(name, status, detail, **data):
        stages[name] = {"status": status, "detail": detail, **data}
    set_stage("providerFinal", "observed", "A new finalized event matched the explicitly requested test phrase.")
    recognition = event.get("recognition")
    recognition = recognition if isinstance(recognition, dict) else {}
    provider = recognition.get("provider")
    language = recognition.get("language")
    provenance = source_metadata_valid(event) and (backend != "deepgram" or (
        provider == "deepgram" and recognition.get("requested_model") == "nova-3" and language in ("en", "en-US", "en-us")))
    set_stage("provenance", "observed" if provenance else "failed",
              ("Matched source timing and provider provenance validated." if provider == backend else
               "Matched source timing/session validated; this event does not attest the configured provider identity.") if provenance else
              "Matched event lacks valid source timing or expected provider provenance.", provider_attested=provider == backend)
    resolved = safe_model(recognition.get("resolved_model"))
    if resolved:
        set_stage("resolvedModel", "observed", "Provider model metadata came from the matched final.", model=resolved)
    set_stage("journal", *journal_observation(root, event))
    set_stage("markdown", *markdown_observation(root, event))
    try:
        viewer = tui.Viewer()
        viewer.add(event)
        rows = tui.rows_for(viewer, 72)
        consumed = viewer.last_event_id == event.get("event_id") and contains(" ".join(row.text for row in rows), event["text"])
        set_stage("viewerConsumption", "observed" if consumed else "failed",
                  "A fresh Viewer ingested and laid out only the matched test event; no curses window was opened."
                  if consumed else "Viewer could not ingest/layout the matched final.", rows=len(rows))
    except Exception:
        set_stage("viewerConsumption", "failed", "Viewer ingestion/layout failed; private details are omitted.")


async def verify(*, phrase=None, timeout=10, public=False, root=ROOT, environ=None,
                 session_factory=aiohttp.ClientSession, sleep=asyncio.sleep, on_ready=None):
    stages = {name: {"status": "not_tested", "detail": "Not observed during this check."} for name in STAGES}
    stages["terminalRenderer"]["detail"] = "Actual terminal painting and human readability were not observed."
    secrets = []
    matched = None
    received = 0
    health = last_health = None
    def set_stage(name, status, detail, **data):
        stages[name] = {"status": status, "detail": detail, **data}
    try:
        if not math.isfinite(timeout) or not 1 <= timeout <= 60:
            raise ValueError("Timeout outside bounds")
        if phrase is not None and (not 3 <= len(normalize(phrase)) <= 40 or len(phrase) > 300):
            raise ValueError("Use a distinctive short test phrase")
        env = environment(Path(root), os.environ if environ is None else environ)
        secrets = [secret for key, value in env.items() if any(word in key.upper() for word in ("TOKEN", "KEY", "SECRET", "PASSWORD"))
                   for secret in (value, value.strip()) if secret]
        token = env.get("OMI_CAPTION_TOKEN", env.get("OMI_STREAM_TOKEN", "")).strip()
        if len(token) > 4096 or any(c.isspace() or not c.isprintable() for c in token):
            raise ValueError("Invalid token")
        set_stage("configuration", "observed", "Read-only verifier configuration resolved; credentials are hidden.")
    except (OSError, ValueError, UnicodeError):
        set_stage("configuration", "failed", "Check dotenv quoting, caption credentials, timeout 1..60 and a distinctive phrase of 3..40 words.")
    else:
        headers = {"Authorization": "Bearer " + token} if token else {}
        client_timeout = aiohttp.ClientTimeout(total=None, connect=3, sock_connect=3, sock_read=4)
        async with session_factory(timeout=client_timeout, trust_env=False) as session:
            try:
                health = await read_health(session)
                set_stage("controlPlane", "observed", "Bounded loopback health JSON was received.", backend=health["stt_backend"])
            except Exception:
                set_stage("controlPlane", "failed", "Loopback health was unavailable or invalid; private error details are omitted.")
            if health is not None:
                replay_started = replay_ended = status_seen = announced = False
                try:
                    async with session.ws_connect(CAPTION_URL, headers=headers, heartbeat=10, max_msg_size=MAX_MESSAGE,
                                                  timeout=aiohttp.ClientWSTimeout(ws_close=1)) as ws:
                        set_stage("captionSubscription", "observed", "Loopback caption subscription opened with replay=0; configured credentials stayed on loopback.")
                        async with asyncio.timeout(timeout):
                            while received < MAX_MESSAGES:
                                message = await ws.receive()
                                received += 1
                                if message.type != aiohttp.WSMsgType.TEXT or not isinstance(message.data, str):
                                    raise ValueError("Caption stream ended or returned unexpected data")
                                if len(message.data.encode("utf-8")) > MAX_MESSAGE:
                                    raise ValueError("Caption message too large")
                                event = json.loads(message.data)
                                if not isinstance(event, dict):
                                    continue
                                kind = event.get("type")
                                if kind == "replay":
                                    if event.get("phase") == "start":
                                        if event.get("count") != 0:
                                            raise ValueError("Unexpected history")
                                        replay_started = True
                                    elif event.get("phase") == "end" and replay_started:
                                        replay_ended = True
                                        set_stage("newEventsOnly", "observed", "Zero-event replay boundary completed before accepting a test final.")
                                elif kind == "status":
                                    stream = event.get("stream_id")
                                    status_seen = (event.get("stt_backend") == health["stt_backend"]
                                                   and isinstance(stream, str) and 0 < len(stream) <= 128)
                                    set_stage("captionStatus", "observed" if status_seen else "failed",
                                              "Caption pipeline status matches loopback backend metadata." if status_seen else "Caption status metadata is inconsistent.")
                                elif phrase and replay_ended and kind == "transcript" and event.get("is_final") is True:
                                    text = event.get("text")
                                    if isinstance(text, str) and contains(text, phrase):
                                        matched = event  # Only this explicitly matched test event survives this iteration.
                                if phrase and status_seen and replay_ended and not announced:
                                    announced = True
                                    if on_ready is not None:
                                        on_ready()
                                if status_seen and replay_ended and (phrase is None or matched is not None):
                                    break
                            else:
                                raise ValueError("Caption message count limit")
                except TimeoutError:
                    if phrase:
                        set_stage("providerFinal", "failed", "No new matching final arrived within the requested observation timeout.")
                    if not status_seen:
                        set_stage("captionStatus", "failed", "Caption status was not observed within the timeout.")
                except Exception:
                    set_stage("captionSubscription", "failed", "Caption stream failed a bounded protocol/authentication check; private details are omitted.")
                if not replay_ended:
                    set_stage("newEventsOnly", "failed", "The zero-history replay boundary was not established.")
                await sleep(1)
                try:
                    last_health = await read_health(session)
                except Exception:
                    set_stage("controlPlane", "failed", "The final loopback health snapshot was unavailable or invalid.")
            if public:
                try:
                    async with asyncio.timeout(4), session.get(PUBLIC_URL, allow_redirects=False, headers={}) as response:
                        set_stage("publicCaptionAuth", "observed" if response.status == 401 else "failed",
                                  "Fixed public caption endpoint rejected an unauthenticated request." if response.status == 401 else
                                  "Fixed public caption endpoint did not return the expected unauthenticated 401.", http_status=response.status)
                except Exception:
                    set_stage("publicCaptionAuth", "failed", "Public authentication check was unavailable; no credentials were sent.")
        current = last_health or health
        if current:
            ready = current.get("status") == "ok" and current.get("backend_ready", current.get("stream_connected")) is True and current.get("workers_alive") is True
            set_stage("backendReady", "observed" if ready else "failed", "Backend orchestration reports ready." if ready else "Backend readiness is degraded.")
            phone = current.get("direct_ws_connected") is True
            set_stage("phoneConnected", "observed" if phone else "failed", "Phone ingress reports connected." if phone else "No phone ingress connection was observed.")
            connected = current.get("stream_connected") is True
            set_stage("providerConnected", "observed" if connected else "not_tested",
                      "Provider transport reports connected." if connected else "Provider is disconnected or lazily idle; this does not verify recognition.")
            resolved = safe_model(current.get("provider_resolved_model"))
            if resolved:
                set_stage("resolvedModel", "observed", "Loopback health included provider-resolved model metadata.", model=resolved)
        stable_identity = (health and last_health and health.get("pid") == last_health.get("pid")
                           and health.get("stt_backend") == last_health.get("stt_backend"))
        if health and last_health and not stable_identity:
            set_stage("controlPlane", "failed", "Bridge identity or backend changed during observation; the stage chain is unresolved.")
        if stable_identity:
            first, last = health.get("direct_ws_packets"), last_health.get("direct_ws_packets")
            if isinstance(first, int) and isinstance(last, int) and not isinstance(first, bool) and not isinstance(last, bool) and last > first:
                set_stage("streamPacketProgress", "observed", "Phone packet counter progressed between loopback snapshots.", packets=last-first)
            else:
                set_stage("streamPacketProgress", "not_tested", "No packet progress was observed; an idle phone is not evidence of audio loss.")
        if matched is not None and stable_identity:
            verify_sample(root, matched, (last_health or health)["stt_backend"], stages)
        elif phrase and stages["providerFinal"]["status"] == "not_tested":
            set_stage("providerFinal", "failed", "A new matching final was not established.")
    statuses = [item["status"] for item in stages.values()]
    report = {"status": "failed" if "failed" in statuses else "partial" if "not_tested" in statuses else "observed",
              "stages": stages, "caption_messages_observed": received,
              "test_phrase_requested": phrase is not None,
              "test_phrase_sha256": hashlib.sha256(phrase.encode()).hexdigest() if phrase is not None else None,
              "limits": {"health_bytes": MAX_BODY, "message_bytes": MAX_MESSAGE, "messages": MAX_MESSAGES,
                         "file_scan_bytes_per_file": MAX_FILE_SCAN},
              "limitations": ["Healthy idle status does not prove ASR, microphone quality, BLE or phone-side buffering.",
                              "Only an explicitly matched new test final is checked in storage and a fresh Viewer; unrelated text is discarded.",
                              "Viewer ingestion/layout is not actual terminal painting; public 401 is not wearable/tunnel audio validation."]}
    return redact(report, secrets)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phrase", help="Wait for this distinctive new wearable test utterance (3..40 words; never injects audio)")
    parser.add_argument("--timeout", type=float, default=10, help="Caption observation timeout, 1..60 seconds")
    parser.add_argument("--public", action="store_true", help="Check the fixed public caption URL returns 401 without credentials")
    parser.add_argument("--json", action="store_true", help="Print safe JSON stage results")
    parser.add_argument("--report", type=Path, help="Save the same safe JSON stage results")
    args = parser.parse_args(argv)
    try:
        def waiting():
            print("Caption subscription ready; waiting for the requested new test phrase.", file=sys.stderr, flush=True)
        report = asyncio.run(verify(phrase=args.phrase, timeout=args.timeout, public=args.public,
                                    on_ready=waiting if args.phrase else None))
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            for name, stage in report["stages"].items():
                print(f"{stage['status']:10} {name}: {stage['detail']}")
        return 1 if report["status"] == "failed" else 0
    except KeyboardInterrupt:
        print("Verification interrupted; the phone stream was not changed.", file=sys.stderr)
        return 130
    except Exception:
        print("Verification failed; private errors, captions and credentials are omitted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
