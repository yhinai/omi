#!/usr/bin/env python3
"""Read-only Omi preflight; never loads a model, sends audio, or starts a service."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_env, read_env_file
HEALTH_URL = "http://127.0.0.1:8788/health"
LIMITATIONS = [
    "Package metadata and asset readability only; no model or native dependency is imported.",
    "Does not verify inference accuracy, wearable/BLE, phone configuration, public tunnel or terminal rendering.",
    "Live health describes the running process; configuration checks describe files for the next launch.",
]
PROBE = """
import importlib.metadata, json, platform, sys, sysconfig
packages = {}
for name in sys.argv[1:]:
    try: packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: packages[name] = None
print(json.dumps(dict(implementation=sys.implementation.name, version=list(sys.version_info[:3]),
    releaselevel=sys.version_info.releaselevel, gil_disabled=bool(sysconfig.get_config_var('Py_GIL_DISABLED')),
    platform=sys.platform, machine=platform.machine(), packages=packages)))
"""


KNOWN_OMI_KEYS = frozenset({
    "OMI_HOST",
    "OMI_STT_BACKEND",
    "OMI_DEEPGRAM_PIPELINE",
    "OMI_PIPECAT_PYTHON",
    "OMI_DEEPGRAM_MODEL",
    "OMI_DEEPGRAM_ENDPOINTING_MS",
    "OMI_DEEPGRAM_DIARIZE",
    "OMI_STREAM_TOKEN",
    "OMI_CAPTION_TOKEN",
    "OMI_MAX_CAPTION_CLIENTS",
    "OMI_LANGUAGE",
    "OMI_LANGUAGES",
    "OMI_HOTWORDS",
    "OMI_WHISPER_PYTHON",
    "OMI_WHISPER_MODEL",
    "OMI_WHISPER_VAD",
    "OMI_MAX_AUDIO_AGE_SEC",
    "OMI_IDLE_FLUSH_SEC",
    "OMI_PYTHON",
    "OMI_TUI_MOUSE_LOG",
})


def dotenv_values(root):
    return read_env_file(Path(root) / ".env")


def environment(root, environ):
    return load_env(Path(root) / ".env", environ)


def probe(python, packages):
    """Inspect package metadata only: importing MLX here could allocate the GPU."""
    result = subprocess.run([str(python), "-c", PROBE, *packages], capture_output=True,
                            text=True, timeout=8, check=True)
    return json.loads(result.stdout)


def version_tuple(value):
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", value or "")
    return tuple(map(int, match.groups())) if match else ()


def live_health():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        response = opener.open(HEALTH_URL, timeout=3)
    except urllib.error.HTTPError as error:
        response = error  # A 503 still carries useful, bounded health JSON.
    with response:
        payload = response.read(65537)
        if len(payload) > 65536:
            raise ValueError("Oversized health response")
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("Expected health object")
        return response.code, data


def inspect(root=ROOT, *, environ=None, live=False):
    root = Path(root)
    checks = []
    secrets = []

    def add(name, status, detail):
        for secret in secrets:
            detail = detail.replace(secret, "[redacted]")
        checks.append({"check": name, "status": status, "detail": detail})

    try:
        file_env = dotenv_values(root)
        env = dict(file_env)
        env.update(os.environ if environ is None else environ)
    except (OSError, UnicodeError, ValueError):
        add("configuration", "fail", "Cannot parse .env; check file access and quoting.")
        return {"status": "failed", "checks": checks, "limitations": LIMITATIONS}
    secrets = [value for key, value in env.items()
               if value and any(word in key.upper() for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))]
    add("configuration", "pass", "Project .env parsed; process environment takes precedence.")
    unknown_keys = sorted(key for key in file_env if key.startswith("OMI_") and key not in KNOWN_OMI_KEYS)
    if unknown_keys:
        add("unused_dotenv_keys", "warn",
            "Ignored unrecognized .env keys: " + ", ".join(unknown_keys) + ".")
    backend = env.get("OMI_STT_BACKEND", "whisper").strip().lower()
    add("stt_backend", "pass" if backend in ("whisper", "deepgram") else "fail",
        "Configured local Whisper." if backend == "whisper" else "Configured Deepgram cloud recognition." if backend == "deepgram" else
        "OMI_STT_BACKEND must be whisper or deepgram.")
    token = env.get("OMI_STREAM_TOKEN", "").strip()
    add("stream_token", "pass" if token else "warn",
        "Configured (value hidden)." if token else "No stream token; use one before exposing audio ingress publicly.")
    caption_token = env.get("OMI_CAPTION_TOKEN", token).strip()
    add("caption_token", "pass" if caption_token else "warn",
        "Configured (value hidden)." if caption_token else "Caption access is anonymous; use a token before exposing captions publicly.")
    try:
        valid_limit = 1 <= int(env.get("OMI_MAX_CAPTION_CLIENTS", "16")) <= 1024
    except ValueError:
        valid_limit = False
    add("caption_client_limit", "pass" if valid_limit else "fail",
        "Caption client limit must be an integer from 1 to 1024.")
    language = env.get("OMI_LANGUAGE", "en").strip().lower()
    if backend == "whisper":
        languages = {part.strip().lower() for part in env.get("OMI_LANGUAGES", "en").split(",") if part.strip()}
        language_ok = bool(languages) and (language in ("", "auto") or language in languages)
        add("recognition_languages", "pass" if language_ok else "fail",
            "Publication allowlist permits the selected language or automatic detection." if language_ok else
            "OMI_LANGUAGES must be nonempty and include fixed OMI_LANGUAGE; otherwise speech can be silently dropped.")
    elif backend == "deepgram":
        add("deepgram_api_key", "pass" if env.get("DEEPGRAM_API_KEY", "").strip() else "fail",
            "Configured (value hidden); key validity is not tested." if env.get("DEEPGRAM_API_KEY", "").strip() else
            "DEEPGRAM_API_KEY is required for cloud recognition (value hidden).")
        languages = {part.strip().lower() for part in env.get("OMI_LANGUAGES", "en").split(",") if part.strip()}
        english = {"en", "en-us"}
        add("recognition_languages", "pass" if language in english and languages and languages <= english else "fail",
            "Deepgram mode requires fixed English and a nonempty English-only publication allowlist; automatic detection is disabled.")
        try:
            from backend_settings import DeepgramConfig
            DeepgramConfig.from_env(env)
            add("deepgram_configuration", "pass", "Nova-3 model, English language, endpointing and diarization settings are valid.")
        except (ImportError, ValueError):
            add("deepgram_configuration", "fail",
                "Check backend_settings.py and Deepgram configuration: Nova-3, English only, endpointing integer 1..5000, diarization true or false, and API key presence.")
        add("deepgram_data_flow", "warn",
            "Audio leaves this Mac for Deepgram. The adapter requests mip_opt_out=true and redact_usage=true; these flags do not prove absolute privacy or retention behavior.")

    bridge_python = root / ".venv/bin/python3.15"
    runtimes = [("bridge", bridge_python, {"aiohttp": None})]
    if backend == "whisper":
        worker_python = Path(os.path.expanduser(env.get("OMI_WHISPER_PYTHON", str(root / ".venv-whisper/bin/python"))))
        worker_packages = {}
        try:
            for line in (root / "requirements-whisper.txt").read_text().splitlines():
                match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
                if match:
                    worker_packages[match[1]] = match[2]
        except OSError:
            add("worker_requirements", "fail", "Cannot read requirements-whisper.txt.")
        runtimes.append(("worker", worker_python, worker_packages))
    if backend == "deepgram" and env.get("OMI_DEEPGRAM_PIPELINE", "direct").strip().lower() == "pipecat":
        from backend_settings import DeepgramConfig
        try:
            config = DeepgramConfig.from_env(env)
            runtimes.append(("pipecat", Path(config.python), {"pipecat-ai": "1.16.0", "deepgram-sdk": "7.19.0"}))
        except ValueError:
            pass  # Configuration failure was already recorded above.
    for role, python, packages in runtimes:
        if not python.is_file() or not os.access(python, os.X_OK):
            add(role + "_runtime", "fail", f"Missing executable: {python}")
            continue
        try:
            info = probe(python, packages)
            expected = (3, 15) if role == "bridge" else (3, 14)
            runtime_ok = (info["implementation"] == "cpython" and tuple(info["version"][:2]) == expected
                          and info["releaselevel"] == "final" and not info["gil_disabled"])
            if role == "worker":
                runtime_ok &= info["platform"] == "darwin" and info["machine"] == "arm64"
            version = ".".join(map(str, info["version"]))
            add(role + "_runtime", "pass" if runtime_ok else "fail",
                f"Python {version}; {info['platform']} {info['machine']}; "
                + ("supported runtime." if runtime_ok else f"requires stable standard CPython {expected[0]}.{expected[1]}"
                   + (" on arm64 macOS." if role == "worker" else ".")))
            for package, pinned in packages.items():
                installed = info["packages"].get(package)
                valid = installed == pinned if pinned else (3, 14, 5) <= version_tuple(installed) < (4, 0, 0)
                add(role + "_package_" + package, "pass" if valid else "fail",
                    f"{package}: " + ("installed and compatible." if valid else "missing or incompatible; install "
                    + ({"worker": "requirements-whisper.txt.", "bridge": "requirements.txt.", "pipecat": "requirements-pipecat.txt."}[role])))
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
            add(role + "_runtime", "fail", "Runtime probe failed or timed out; no subprocess output is shown.")

    if backend == "whisper":
        assets = Path.home() / "Library/Application Support/OmiWhisper"
        model = Path(os.path.expanduser(env.get("OMI_WHISPER_MODEL", str(assets / "whisper-turbo"))))
        vad = Path(os.path.expanduser(env.get("OMI_WHISPER_VAD", str(assets / "silero_vad.onnx"))))
        for name, path in (("model_config", model / "config.json"), ("model_weights", model / "weights.safetensors"),
                           ("speech_detector", vad)):
            try:
                valid = path.is_file() and path.stat().st_size > 0 and os.access(path, os.R_OK)
            except OSError:
                valid = False
            add(name, "pass" if valid else "fail", ("Readable nonempty file: " if valid else "Missing, empty or unreadable file: ") + str(path))
    try:
        timings = [float(env.get(key, default)) for key, default in
                   (("OMI_MAX_AUDIO_AGE_SEC", "45"), ("OMI_IDLE_FLUSH_SEC", "8"))]
        valid = all(math.isfinite(value) and value > 0 for value in timings)
    except ValueError:
        valid = False
    add("audio_timings", "pass" if valid else "fail", "Audio age and idle timeout must be finite and positive.")
    for name in ("run.sh", "tui.sh"):
        try:
            result = subprocess.run(["/bin/bash", "-n", str(root / name)], capture_output=True, timeout=3)
            valid = result.returncode == 0 and os.access(root / name, os.X_OK)
        except (OSError, subprocess.SubprocessError):
            valid = False
        add(name, "pass" if valid else "fail", "Shell syntax and executable permission " + ("verified." if valid else "need repair."))
    add("port_inspection", "pass" if shutil.which("lsof") else "fail", "lsof is required for safe launcher port checks.")

    if live:
        try:
            code, health = live_health()
            expected_pipeline = env.get("OMI_DEEPGRAM_PIPELINE", "direct").strip().lower() if backend == "deepgram" else "local"
            actual_pipeline = health.get("stt_pipeline", "direct" if health.get("stt_backend") == "deepgram" else "local")
            valid = (code == 200 and health.get("status") == "ok" and health.get("stt_backend") == backend
                     and actual_pipeline == expected_pipeline)
            add("live_health", "pass" if valid else "fail",
                "Loopback bridge reports ready with the selected backend and pipeline." if valid else
                "Loopback bridge is degraded, uses a different backend/pipeline, or returned unexpected health; inspect local health.")
            if valid:
                phone = health.get("direct_ws_connected") is True
                add("live_phone", "pass" if phone else "warn", "Phone audio stream connected." if phone else "No phone stream; healthy idle state does not verify recognition.")
        except (OSError, ValueError, TypeError):
            add("live_health", "fail", "Cannot read bounded JSON health from http://127.0.0.1:8788/health.")
    return {"status": "failed" if any(c["status"] == "fail" for c in checks) else "passed",
            "backend": backend if backend in ("whisper", "deepgram") else "invalid",
            "checks": checks, "limitations": LIMITATIONS + (
                ["Cloud preflight never contacts Deepgram or validates API-key access, billing, request acceptance or recognition quality."]
                if backend == "deepgram" else [])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable, redacted checks")
    parser.add_argument("--live", action="store_true", help="Also read existing loopback bridge health (never starts it)")
    args = parser.parse_args(argv)
    report = inspect(live=args.live)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for check in report["checks"]:
            print(f"{check['status'].upper():4} {check['check']}: {check['detail']}")
        print("Preflight " + report["status"] + ". Model inference, wearable, tunnel and microphone accuracy are not tested.")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
