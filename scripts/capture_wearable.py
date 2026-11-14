#!/usr/bin/env python3.15
"""Explicitly record up to ten seconds of newly received Omi audio locally.

No recording occurs without --record. This client uses loopback health and one
verified local signal; it does not access credentials or upload captured audio.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

HEALTH_URL = "http://127.0.0.1:8788/health"
BRIDGE_PATH = Path(__file__).resolve().parents[1] / "bridge.py"
WAIT_SECONDS = 15
AUDIO_READY_SECONDS = 3


class CaptureError(RuntimeError):
    pass


def fetch_health(timeout=1):
    opener = build_opener(ProxyHandler({}))
    try:
        response = opener.open(HEALTH_URL, timeout=timeout)
    except HTTPError as error:
        if error.code != 503:
            raise CaptureError("Local bridge health is unavailable") from None
        response = error  # A degraded bridge can still expose capture capability.
    with response:
        encoded = response.read(262145)
    if len(encoded) > 262144:
        raise CaptureError("Local health response exceeded its limit")
    health = json.loads(encoded)
    if not isinstance(health, dict):
        raise CaptureError("Invalid local bridge health")
    return health


def inspect_process(pid, *, expected_script=BRIDGE_PATH, runner=None, uid=None):
    """Bind PID to this repository's exact bridge command, owner and birth time."""
    runner = subprocess.run if runner is None else runner
    uid = os.getuid() if uid is None else uid
    result = runner(["ps", "-ww", "-p", str(pid), "-o", "uid=", "-o", "lstart=", "-o", "command="],
                    capture_output=True, text=True, timeout=1, check=False)
    fields = result.stdout.strip().split(None, 6)
    if result.returncode or len(fields) != 7 or not fields[0].isdigit() or int(fields[0]) != uid:
        raise CaptureError("The bridge PID does not belong to this user")
    command = fields[6]
    suffix = " " + str(expected_script)
    if not command.endswith(suffix):
        raise CaptureError("The bridge PID does not run this repository's bridge")
    program = command[:-len(suffix)].strip()
    name = Path(program).name
    if not re.fullmatch(r"Python|python(?:3(?:\.\d+)*)?", name):
        raise CaptureError("The bridge PID has an unexpected interpreter")
    return int(fields[0]), " ".join(fields[1:6]), command


def capability(health):
    pid, capture = health.get("pid"), health.get("audio_capture")
    if (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1
            or not isinstance(capture, dict) or capture.get("supported") is not True
            or type(capture.get("protocol_version")) is not int or capture.get("protocol_version") != 1
            or capture.get("trigger") != "SIGUSR1"):
        raise CaptureError("This bridge does not support the explicit capture signal; no signal sent")
    return pid, capture


def wait_for_audio(first_health, expected_pid, *, get_health, clock, sleep):
    """Observe fresh packet progress before activating any raw-audio capture."""
    baseline = first_health.get("direct_ws_packets")
    baseline = baseline if type(baseline) is int and baseline >= 0 else None
    deadline = clock() + AUDIO_READY_SECONDS
    while clock() < deadline:
        current = get_health(timeout=min(1, max(.001, deadline-clock())))
        pid, capture = capability(current)
        if pid != expected_pid or capture.get("state") == "recording":
            raise CaptureError("Bridge identity or capture state changed; no signal sent")
        count, age = current.get("direct_ws_packets"), current.get("last_packet_age_sec")
        valid_count = type(count) is int and count >= 0
        recent = (isinstance(age, (int, float)) and not isinstance(age, bool)
                  and math.isfinite(age) and 0 <= age <= 1)
        if (current.get("direct_ws_connected") is True and recent and valid_count
                and baseline is not None and count > baseline):
            return current
        if valid_count and (baseline is None or count < baseline):
            baseline = count
        sleep(min(.1, max(0, deadline-clock())))
    raise CaptureError("Phone audio did not advance within 3 seconds; reconnect Omi and try again; no signal sent")


def record_capture(*, get_health=None, process_identity=None, send_signal=None,
                   clock=None, sleep=None, announce=None):
    get_health = fetch_health if get_health is None else get_health
    process_identity = inspect_process if process_identity is None else process_identity
    send_signal = os.kill if send_signal is None else send_signal
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    announce = print if announce is None else announce
    first_health = get_health()
    first_pid, first = capability(first_health)
    if first.get("state") == "recording":
        raise CaptureError("A diagnostic capture is already active; no signal sent")
    identity = process_identity(first_pid)
    second_pid, previous = capability(wait_for_audio(first_health, first_pid,
        get_health=get_health, clock=clock, sleep=sleep))
    if (second_pid != first_pid or process_identity(second_pid) != identity
            or previous.get("state") == "recording"):
        raise CaptureError("Bridge identity or capture state changed; no signal sent")
    previous_id = previous.get("capture_id")
    announce("Recording up to 10 seconds of newly received Omi audio locally.")
    send_signal(second_pid, signal.SIGUSR1)
    deadline = clock() + WAIT_SECONDS
    while clock() < deadline:
        pid, capture = capability(get_health(timeout=min(1, max(.001, deadline-clock()))))
        if pid != second_pid:
            raise CaptureError("The bridge restarted before capture completion")
        if capture.get("capture_id") and capture.get("capture_id") != previous_id:
            state = capture.get("state")
            if state == "completed":
                samples = capture.get("samples")
                if isinstance(samples, bool) or not isinstance(samples, int) or not 0 <= samples <= 160000:
                    raise CaptureError("The completed capture reported an invalid sample count")
                if not samples:
                    raise CaptureError("The capture received no audio; no usable recording was created")
                path = capture.get("path")
                if not isinstance(path, str) or not Path(path).is_absolute():
                    raise CaptureError("The completed capture reported an invalid local path")
                return {"path": path, "samples": samples, "audio_sec": round(samples/16000, 6),
                        "received_packets": capture.get("received_packets", 0), "reason": capture.get("reason")}
            if state in ("error", "cancelled"):
                raise CaptureError("The local diagnostic capture failed or was cancelled")
        sleep(min(.2, max(0, deadline-clock())))
    raise CaptureError("Capture completion was not observed within 15 seconds; no further signal sent")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="Explicitly authorize one bounded local recording")
    parser.add_argument("--wait-for-enter", action="store_true",
                        help="Wait for your Enter key in a real terminal before the countdown")
    parser.add_argument("--countdown", type=int, choices=range(0, 11), default=3, metavar="0..10",
                        help="Seconds to prepare before checking audio and starting (default 3)")
    args = parser.parse_args(argv)
    if not args.record:
        parser.error("--record is required; no audio has been recorded")
    if args.wait_for_enter and not sys.stdin.isatty():
        parser.error("--wait-for-enter requires a real terminal; no recording started")
    try:
        if args.wait_for_enter:
            input("Get the wearable ready, then press Enter. No recording has started: ")
        if args.countdown:
            print(f"Preparing for {args.countdown} seconds; speak after the Recording message.", flush=True)
            time.sleep(args.countdown)
        print(json.dumps(record_capture()), flush=True)
    except KeyboardInterrupt:
        sys.exit("Capture client stopped. Any capture already started remains bounded to ten seconds; no further signal sent.")
    except CaptureError as error:
        sys.exit(str(error))
    except (OSError, ValueError, subprocess.SubprocessError):
        sys.exit("Capture did not complete; check local bridge health and explicit capture support.")


if __name__ == "__main__":
    main()
