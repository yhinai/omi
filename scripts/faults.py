#!/usr/bin/env python3.15
"""Offline fault and corruption checks; never loads a model or touches live ports/history."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bridge
import storage
import stt


def verify_journal_and_invariants(log_dir: Path, viewer_finals: list[dict], *, expect_text: bool = False) -> list[str]:
    """Return list of invariant violation strings (empty list == all invariants hold)."""
    errors = []
    journal = log_dir / "omi_transcript_events.jsonl"
    if not journal.is_file():
        if expect_text or viewer_finals:
            errors.append("Missing omi_transcript_events.jsonl despite expected/emitted finals")
        return errors

    raw_bytes = journal.read_bytes()
    if raw_bytes and not raw_bytes.endswith(b"\n"):
        errors.append("Journal has unterminated final line")

    records = []
    seen_event_ids = set()
    prev_end_by_session = {}
    for line_no, raw_line in enumerate(raw_bytes.splitlines(), 1):
        try:
            rec = json.loads(raw_line.decode("utf-8"))
        except Exception as exc:
            errors.append(f"Malformed JSONL at line {line_no}: {type(exc).__name__}")
            continue
        kind = rec.get("type")
        if kind not in ("transcript", "gap"):
            errors.append(f"Illegal non-final event type in JSONL at line {line_no}: {kind!r}")
        eid = rec.get("event_id")
        if not isinstance(eid, str) or not eid:
            errors.append(f"Missing event_id at line {line_no}")
        elif eid in seen_event_ids:
            errors.append(f"Duplicate event_id in JSONL: {eid}")
        else:
            seen_event_ids.add(eid)
        if kind == "transcript":
            if rec.get("is_final") is not True:
                errors.append(f"Transcript event missing is_final=True at line {line_no}")
            s_off = rec.get("sample_offset")
            e_off = rec.get("sample_end_offset")
            sess = (rec.get("stream_id"), rec.get("session_id"))
            if isinstance(s_off, int) and isinstance(e_off, int):
                if e_off < s_off:
                    errors.append(f"Negative sample span ({s_off}..{e_off}) at line {line_no}")
                if sess in prev_end_by_session and s_off < prev_end_by_session[sess]:
                    errors.append(f"Overlapping/regressing sample_offset {s_off} < {prev_end_by_session[sess]} at line {line_no}")
                prev_end_by_session[sess] = e_off
        records.append(rec)

    saved_finals = [r for r in records if r.get("type") == "transcript"]
    if [r["event_id"] for r in saved_finals] != [v["event_id"] for v in viewer_finals]:
        errors.append(
            f"Viewer finals ({len(viewer_finals)}) do not match saved JSONL finals ({len(saved_finals)})"
        )

    md_files = sorted(log_dir.glob("????-??-??.md"))
    md_bytes = b"".join(p.read_bytes() for p in md_files)
    if b"\x00" in md_bytes:
        errors.append("Daily Markdown contains NUL (\\x00) bytes")
    md_text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in md_files)
    for r in saved_finals:
        for word in r["text"].split():
            if word not in md_text:
                errors.append(f"Saved final word {word!r} missing from daily Markdown")
                break
    return errors



def run_checks() -> dict:
    results = {}

    # 1. 10ms real-phone packet cadence during 4.0s worker startup/restart (< 10s byte cap!)
    gaps = []
    adapter = stt.LocalWhisperSTT(
        python=sys.executable, model="m", vad="v",
        on_partial=lambda e: None, on_final=lambda *a, **kw: True,
        on_gap=lambda reason, sec, **kw: gaps.append((reason, sec, kw)),
        on_latency=lambda *a: None, on_connection=lambda *a: None,
    )
    now_wall, now_mono = time.time(), time.monotonic()
    for i in range(400):  # 400 x 10ms = 4.0s = 128,000 bytes (only 40% of 320,000 byte cap!)
        adapter.feed(b"\x00\x10" * 160, now_wall + i * 0.01, now_mono + i * 0.01)
    evicted_4s = round(sum(sec for r, sec, _ in gaps if r == "queue_evicted"), 4)
    results["10ms_cadence_4s_buffer_no_premature_eviction"] = {
        "passed": evicted_4s == 0.0 and adapter.pending_bytes == 128000,
        "pending_bytes": adapter.pending_bytes,
        "evicted_sec": evicted_4s,
        "detail": "400x10ms (4.0s) real-phone packets must fit inside 10s queue without packet-count eviction",
    }

    # 2. 10ms cadence 12.0s true overload -> keeps newest 10.0s, reports exact 2.0s queue_evicted
    gaps2 = []
    adapter2 = stt.LocalWhisperSTT(
        python=sys.executable, model="m", vad="v",
        on_partial=lambda e: None, on_final=lambda *a, **kw: True,
        on_gap=lambda reason, sec, **kw: gaps2.append((reason, sec, kw)),
        on_latency=lambda *a: None, on_connection=lambda *a: None,
    )
    for i in range(1200):  # 12.0s = 384,000 bytes
        adapter2.feed(b"\x00\x10" * 160, now_wall + i * 0.01, now_mono + i * 0.01)
    evicted_12s = round(sum(sec for r, sec, _ in gaps2 if r == "queue_evicted"), 4)
    results["10ms_cadence_12s_overload_exact_2s_eviction"] = {
        "passed": evicted_12s == 2.0 and adapter2.pending_bytes == 320000 and adapter2.queue[0].sample_offset == 32000,
        "pending_bytes": adapter2.pending_bytes,
        "evicted_sec": evicted_12s,
        "oldest_retained_sample_offset": adapter2.queue[0].sample_offset,
    }

    # 3. External Markdown truncate/recreate on 2nd paragraph (offset > header) -> zero NUL (\x00) bytes
    with tempfile.TemporaryDirectory(prefix="omi-md-fault-") as md_dir:
        notes = storage.DailyNotes(md_dir, idle_flush=8)
        notes.add("2026-11-14 03:00:00", "First paragraph in the daily file.")
        notes.close_paragraph()
        notes.add("2026-11-14 03:00:02", "Second paragraph starts at offset > header.")
        md_file = Path(md_dir, "2026-11-14.md")
        md_file.write_bytes(b"")  # External truncate while notes.words is non-empty & notes.offset > header!
        notes.add("2026-11-14 03:00:04", "Continuation after external truncate.")
        raw_md = md_file.read_bytes()
        results["markdown_external_truncate_no_nul_bytes"] = {
            "passed": b"\x00" not in raw_md and b"Continuation after external truncate." in raw_md,
            "nul_byte_count": raw_md.count(b"\x00"),
        }

    # 3b. Rapid same-IP reconnect + stale replaced socket frames + worker crash replay + slow consumer + archive collision
    # Run targeted real-component integration checks via unittest classes & verify 0 failures
    targeted = subprocess.run(
        [
            sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "tests"),
            "-p", "test_session_safety.py", "-q",
        ],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    launcher_ut = subprocess.run(
        [
            sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "tests"),
            "-p", "test_launcher.py", "-q",
        ],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    results["rapid_reconnect_stale_frames_and_slow_consumer"] = {
        "passed": targeted.returncode == 0,
        "returncode": targeted.returncode,
    }
    results["repeated_same_second_archive_uniqueness"] = {
        "passed": launcher_ut.returncode == 0,
        "returncode": launcher_ut.returncode,
    }

    # 4. Storage failure (ENOSPC) + retry + uncertain write deduplication
    with tempfile.TemporaryDirectory(prefix="omi-store-fault-") as st_dir:
        fail_disk = True
        committed = []
        errors = []
        journal_path = Path(st_dir, "omi_transcript_events.jsonl")

        def flaky_append(line: str):
            if fail_disk:
                raise OSError(28, "No space left on device")
            with journal_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        store = storage.TranscriptStore(
            directory=st_dir,
            append=flaky_append,
            create_event=bridge.make_event,
            on_commit=committed.append,
            on_error=lambda target, err: errors.append((target, type(err).__name__)),
            on_recovered=lambda: None,
        )
        store.notes = storage.DailyNotes(st_dir)
        ok1 = store.final("Durable after disk full.", session_id=1, utterance_id="u1",
                          sample_offset=0, sample_end_offset=16000, audio_start_at=1700000000.0)
        fail_disk = False
        ok2 = store.final("Durable after disk full.", session_id=1, utterance_id="u1",
                          sample_offset=0, sample_end_offset=16000, audio_start_at=1700000000.0)
        lines = journal_path.read_text(encoding="utf-8").splitlines()
        results["storage_enospc_retry_single_commit"] = {
            "passed": (ok1 is False and ok2 is True and len(lines) == 1 and len(committed) == 1),
            "first_attempt": ok1,
            "retry_attempt": ok2,
            "journal_lines": len(lines),
            "committed_events": len(committed),
        }

    # 5. Truncated multi-byte UTF-8 journal tail recovery
    with tempfile.TemporaryDirectory(prefix="omi-tail-fault-") as tl_dir:
        jpath = Path(tl_dir, "omi_transcript_events.jsonl")
        valid1 = json.dumps({"event_id": "e1", "type": "transcript", "is_final": True, "text": "Before crash 👨‍👩‍👧"}, ensure_ascii=False).encode("utf-8")
        truncated = b'{"event_id":"broken","type":"transcript","text":"Cut mid-emoji \xf0\x9f'  # incomplete 4-byte UTF-8, no \n
        jpath.write_bytes(valid1 + b"\n" + truncated)
        storage.separate_journal_tail(str(jpath))
        valid2 = json.dumps({"event_id": "e2", "type": "transcript", "is_final": True, "text": "After recovery مرحبا"}, ensure_ascii=False).encode("utf-8")
        with jpath.open("ab") as f:
            f.write(valid2 + b"\n")
        store = storage.TranscriptStore(
            directory=tl_dir, append=lambda s: None, create_event=bridge.make_event,
            on_commit=lambda e: None, on_error=lambda *a: None, on_recovered=lambda: None,
        )
        replayed = list(store.recent_events())
        results["truncated_utf8_journal_tail_recovery"] = {
            "passed": [e["event_id"] for e in replayed] == ["e1", "e2"] and jpath.read_bytes().startswith(valid1 + b"\n" + truncated + b"\n"),
            "replayed_ids": [e["event_id"] for e in replayed],
        }

    # 6. Deliberate corruption negative tests -> confirm verifier CATCHES every single one!
    with tempfile.TemporaryDirectory(prefix="omi-corrupt-") as cr_dir:
        cpath = Path(cr_dir)
        ev1 = {"event_id": "id1", "stream_id": "s", "session_id": 1, "type": "transcript", "is_final": True,
               "text": "Verified words.", "sample_offset": 0, "sample_end_offset": 8000}
        (cpath / "omi_transcript_events.jsonl").write_text(json.dumps(ev1) + "\n", encoding="utf-8")
        (cpath / "2026-11-14.md").write_text("# Omi Transcript\n\nVerified words.\n", encoding="utf-8")
        clean_ok = verify_journal_and_invariants(cpath, [ev1], expect_text=True) == []

        # Corruption A: leaked partial in JSONL
        (cpath / "omi_transcript_events.jsonl").write_text(
            json.dumps(ev1) + "\n" + json.dumps({"event_id": "p1", "type": "partial", "text": "draft"}) + "\n"
        )
        caught_partial = any("Illegal non-final" in e for e in verify_journal_and_invariants(cpath, [ev1]))

        # Corruption B: duplicate event_id in JSONL
        (cpath / "omi_transcript_events.jsonl").write_text(json.dumps(ev1) + "\n" + json.dumps(ev1) + "\n")
        caught_dup = any("Duplicate event_id" in e for e in verify_journal_and_invariants(cpath, [ev1, ev1]))

        # Corruption C: dropped viewer update (JSONL has 1 final, viewer received 0)
        (cpath / "omi_transcript_events.jsonl").write_text(json.dumps(ev1) + "\n")
        caught_dropped_viewer = any("Viewer finals" in e for e in verify_journal_and_invariants(cpath, []))

        # Corruption D: overlapping/regressing sample_offset
        ev2_bad = {**ev1, "event_id": "id2", "sample_offset": 4000, "sample_end_offset": 9000}
        (cpath / "omi_transcript_events.jsonl").write_text(json.dumps(ev1) + "\n" + json.dumps(ev2_bad) + "\n")
        caught_overlap = any("Overlapping" in e for e in verify_journal_and_invariants(cpath, [ev1, ev2_bad]))

        results["negative_corruption_verifiers"] = {
            "passed": all((clean_ok, caught_partial, caught_dup, caught_dropped_viewer, caught_overlap)),
            "clean_control_passed": clean_ok,
            "caught_leaked_partial": caught_partial,
            "caught_duplicate_event_id": caught_dup,
            "caught_dropped_viewer_event": caught_dropped_viewer,
            "caught_overlapping_sample_offset": caught_overlap,
        }

    return {
        "all_passed": all(item["passed"] for item in results.values()),
        "scenarios": results,
    }



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write the verification receipt")
    parser.add_argument("--label", default="offline", help="Label for this check")
    args = parser.parse_args(argv)
    report = {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_root": str(ROOT), "label": args.label,
        "lifecycle_fault_suite": run_checks(),
    }
    payload = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n")
    print(payload)
    return 0 if report["lifecycle_fault_suite"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
