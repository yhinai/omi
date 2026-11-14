"""Daily Omi commands, with development tools kept behind an explicit dev command."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

from config import VEXO_BAND

ROOT = Path(__file__).resolve().parent


def nonnegative(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def status_report():
    from scripts.doctor import live_health
    try:
        code, health = live_health()
        if not isinstance(health, dict):
            raise ValueError("Expected local health object")
        backend = health.get("stt_backend")
        if code not in (200, 503) or backend not in {"deepgram", "whisper"}:
            raise ValueError("Unexpected local health response")
        healthy = code == 200 and health.get("status") == "ok" and health.get("backend_ready") is True
        phone = health.get("direct_ws_connected") is True
        packet_age = health.get("last_packet_age_sec")
        provider = health.get("provider_connection_state")
        if provider not in {"connected", "connecting", "idle", "disconnected", "retrying", "reconnecting", "stopped", "closing", "closed"}:
            provider = "unknown" if backend == "deepgram" else "local"
        pending = health.get("pending_saves")
        saving = ("error" if health.get("last_output_error") else
                  "pending" if nonnegative(pending) and pending > 0 else
                  "ready" if nonnegative(pending) and pending == 0 else "unknown")
        age = health.get("last_transcript_age_sec")
        pid = health.get("pid")
        return {"service": "healthy" if healthy else "degraded", "pid": pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
                "source": health.get("source_identity"), "source_loss": health.get("source_loss", {}),
                "backend": backend, "phone": "connected" if phone else "disconnected",
                "pipeline": health.get("stt_pipeline") if health.get("stt_pipeline") in {"direct", "pipecat", "local"}
                            else "direct" if backend == "deepgram" else "local",
                "audio": "flowing" if phone and nonnegative(packet_age) and packet_age <= 2 else "idle",
                "recognizer": provider, "saving": saving,
                "pending_saves": pending if nonnegative(pending) else None,
                "last_final_age_sec": age if nonnegative(age) else None}
    except (OSError, ValueError, TypeError, KeyError):
        return {"service": "unavailable", "detail": "Local health is unavailable; run ./omi check or ./run.sh status."}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Operate Omi live captions.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("start", "Start the service"), ("stop", "Drain and stop the service"),
                            ("restart", "Gracefully restart, preserving history")):
        commands.add_parser(name, help=help_text)
    status = commands.add_parser("status", help="Show service, phone, audio, recognizer and saving states")
    status.add_argument("--json", action="store_true")
    backend = commands.add_parser("backend", help="Validate, switch and verify the selected recognizer")
    backend.add_argument("name", choices=("deepgram", "whisper"))
    for name, help_text in (("check", "Read configuration and live health without sending audio"),
                            ("captions", "Open the terminal viewer"),
                            ("verify", "Observe new live captions and optionally a specific test phrase")):
        commands.add_parser(name, help=help_text, add_help=False)
    source = commands.add_parser("source", help="Run a wearable input in the foreground; Ctrl+C drains")
    source.add_argument("name", choices=("vexo",))
    source.add_argument("--band", type=int, choices=(VEXO_BAND,), required=True)
    source.add_argument("--transport", choices=("serial", "ble"), default="serial")
    voice = commands.add_parser("voice", help="Local voice enrollment, status or deletion")
    voice.add_argument("arguments", nargs=argparse.REMAINDER)
    dev = commands.add_parser("dev", help="Explicit development, demo and diagnostic tools")
    dev.add_argument("tool", choices=("test", "smoke", "benchmark", "faults", "demo", "capture"))
    args, extra = parser.parse_known_args(argv)
    forwarding = args.command in {"check", "captions", "verify", "dev"}
    if extra and not forwarding:
        parser.error("unrecognized arguments: " + " ".join(extra))
    if args.command == "status":
        report = status_report()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            for key, value in report.items():
                if value is not None:
                    label = "Input connection" if key == "phone" else key.replace("_", " ").capitalize()
                    print(f"{label}: {value}")
            if report["service"] == "healthy":
                print("Service health does not establish microphone quality or recognition accuracy.")
        return 0 if report["service"] == "healthy" else 1
    if args.command == "voice":
        executable = sys.executable
        arguments = [str(ROOT/"voice_cli.py"), *args.arguments]
    elif args.command == "source":
        executable = str(ROOT / ".venv-vexo/bin/python")
        arguments = [str(ROOT / "vexo_source.py"), "--band", str(args.band), "--transport", args.transport]
    elif args.command in {"start", "stop", "restart"}:
        executable, arguments = str(ROOT / "run.sh"), [args.command]
    else:
        executable = sys.executable
        if args.command == "backend":
            arguments = [str(ROOT / "scripts/set_backend.py"), args.name]
        elif args.command == "dev":
            if args.tool == "test":
                arguments = ["-m", "unittest", "discover", "-s", str(ROOT / "tests"), *extra]
            else:
                tools = {"smoke": "smoke_test.py", "benchmark": "benchmark_stt.py", "faults": "faults.py",
                         "demo": "demo.py", "capture": "capture_wearable.py"}
                arguments = [str(ROOT / "scripts" / tools[args.tool]), *extra]
        else:
            script = {"check": "scripts/doctor.py", "captions": "tui.py", "verify": "scripts/verify_live.py"}[args.command]
            arguments = [str(ROOT / script), *( ["--live"] if args.command == "check" else []), *extra]
    # Replacing this process preserves terminal signals and the underlying command's exit status.
    os.chdir(ROOT)
    os.execv(executable, [executable, *arguments])


if __name__ == "__main__":
    sys.exit(main())
