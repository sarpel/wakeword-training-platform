#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wakeword dataset summary reporter.

- Scans these classes by default:
  positive, negative, hard_negative, background, rirs
- Reports per-class:
  file count, total duration, duration stats, sample-rate & channels distribution, total size
- Supports .wav with built-in "wave".
- Optionally supports many formats (flac/ogg/mp3/...) if "soundfile" is installed.

Usage examples:
  python dataset_report.py --positive ./positive --negative ./negative --hard_negative ./hard_negative --background ./background --rirs ./rirs --ext .wav --json_out report.json
  python dataset_report.py --positive /data/pos --negative /data/neg --hard_negative /data/hn --background /data/bg --rirs /data/rirs
  python dataset_report.py --root /path/to/datasets --ext .wav .flac
  python dataset_report.py --root /path/to/datasets --json_out report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import wave
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Optional: soundfile for non-wav formats
try:
    import soundfile as sf  # type: ignore
    HAS_SF = True
except Exception:
    sf = None
    HAS_SF = False


CLASSES_DEFAULT = ["positive", "negative", "hard_negative", "background", "rirs"]


@dataclass
class FileInfo:
    path: str
    duration_s: float
    samplerate: int
    channels: int
    size_bytes: int


@dataclass
class ClassReport:
    class_name: str
    root: str
    file_count: int
    total_duration_s: float
    total_size_bytes: int
    duration_min_s: float
    duration_p50_s: float
    duration_mean_s: float
    duration_p95_s: float
    duration_max_s: float
    samplerate_counts: Dict[str, int]
    channels_counts: Dict[str, int]
    unreadable_files: int
    unreadable_examples: List[str]


def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    units = ["KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        v /= 1024.0
        if v < 1024.0:
            return f"{v:.2f} {u}"
    return f"{v:.2f} PB"


def human_duration(seconds: float) -> str:
    if seconds < 0:
        return "0s"
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}h {m}m {sec}s"
    if m > 0:
        return f"{m}m {sec}s"
    return f"{sec}s"


def percentile(sorted_vals: List[float], p: float) -> float:
    # p in [0,100]
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def read_audio_info(path: Path) -> Optional[Tuple[float, int, int]]:
    """
    Returns (duration_s, samplerate, channels) or None if unreadable.
    """
    suffix = path.suffix.lower()
    if suffix == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                sr = wf.getframerate()
                ch = wf.getnchannels()
                if sr <= 0:
                    return None
                dur = frames / float(sr)
                return (float(dur), int(sr), int(ch))
        except Exception:
            return None

    # Non-wav: try soundfile if available
    if HAS_SF:
        try:
            info = sf.info(str(path))  # type: ignore
            if info.samplerate <= 0:
                return None
            dur = float(info.frames) / float(info.samplerate)
            return (dur, int(info.samplerate), int(info.channels))
        except Exception:
            return None

    return None


def iter_files(root: Path, exts: List[str]) -> Iterable[Path]:
    # Normalize extensions: ".wav" style
    exts_norm = set(e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_norm:
            yield p


def analyze_class(
    class_name: str,
    root: Path,
    exts: List[str],
    workers: int,
    max_unreadable_examples: int = 10,
) -> ClassReport:
    files = list(iter_files(root, exts))
    unreadable = 0
    unreadable_examples: List[str] = []

    infos: List[FileInfo] = []
    sr_counts: Counter[int] = Counter()
    ch_counts: Counter[int] = Counter()
    total_size = 0

    if not files:
        return ClassReport(
            class_name=class_name,
            root=str(root),
            file_count=0,
            total_duration_s=0.0,
            total_size_bytes=0,
            duration_min_s=0.0,
            duration_p50_s=0.0,
            duration_mean_s=0.0,
            duration_p95_s=0.0,
            duration_max_s=0.0,
            samplerate_counts={},
            channels_counts={},
            unreadable_files=0,
            unreadable_examples=[],
        )

    def one(p: Path) -> Tuple[Optional[FileInfo], Optional[str]]:
        size = 0
        try:
            size = p.stat().st_size
        except Exception:
            size = 0

        meta = read_audio_info(p)
        if meta is None:
            return (None, str(p))

        dur, sr, ch = meta
        return (
            FileInfo(
                path=str(p),
                duration_s=dur,
                samplerate=sr,
                channels=ch,
                size_bytes=size,
            ),
            None,
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(one, p) for p in files]
        for fut in as_completed(futures):
            info, bad = fut.result()
            if info is None:
                unreadable += 1
                if bad and len(unreadable_examples) < max_unreadable_examples:
                    unreadable_examples.append(bad)
                continue
            infos.append(info)
            sr_counts[info.samplerate] += 1
            ch_counts[info.channels] += 1
            total_size += info.size_bytes

    durations = sorted([i.duration_s for i in infos])
    total_dur = float(sum(durations))
    if durations:
        dmin = float(durations[0])
        dmax = float(durations[-1])
        dp50 = float(percentile(durations, 50))
        dp95 = float(percentile(durations, 95))
        dmean = float(total_dur / len(durations))
    else:
        dmin = dp50 = dmean = dp95 = dmax = 0.0

    return ClassReport(
        class_name=class_name,
        root=str(root),
        file_count=len(infos),
        total_duration_s=total_dur,
        total_size_bytes=total_size,
        duration_min_s=dmin,
        duration_p50_s=dp50,
        duration_mean_s=dmean,
        duration_p95_s=dp95,
        duration_max_s=dmax,
        samplerate_counts={str(k): int(v) for k, v in sorted(sr_counts.items(), key=lambda x: x[0])},
        channels_counts={str(k): int(v) for k, v in sorted(ch_counts.items(), key=lambda x: x[0])},
        unreadable_files=unreadable,
        unreadable_examples=unreadable_examples,
    )


def print_table(reports: List[ClassReport]) -> None:
    # Compact per-class summary table
    headers = ["Class", "Files", "Total Dur", "Size", "Min", "P50", "Mean", "P95", "Max", "Unreadable"]
    rows = []
    for r in reports:
        rows.append([
            r.class_name,
            str(r.file_count),
            human_duration(r.total_duration_s),
            human_bytes(r.total_size_bytes),
            f"{r.duration_min_s:.3f}s",
            f"{r.duration_p50_s:.3f}s",
            f"{r.duration_mean_s:.3f}s",
            f"{r.duration_p95_s:.3f}s",
            f"{r.duration_max_s:.3f}s",
            str(r.unreadable_files),
        ])

    # Column widths
    cols = list(zip(headers, *rows)) if rows else [(h,) for h in headers]
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt_row(items: List[str]) -> str:
        return " | ".join(str(items[i]).ljust(widths[i]) for i in range(len(items)))

    sep = "-+-".join("-" * w for w in widths)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))


def print_distributions(reports: List[ClassReport]) -> None:
    for r in reports:
        print(f"\n[{r.class_name}] distributions")
        if r.file_count == 0 and r.unreadable_files == 0:
            print("  (no files found)")
            continue

        if r.samplerate_counts:
            sr_str = ", ".join(f"{k}Hz:{v}" for k, v in r.samplerate_counts.items())
            print(f"  samplerates: {sr_str}")
        else:
            print("  samplerates: (none)")

        if r.channels_counts:
            ch_str = ", ".join(f"{k}ch:{v}" for k, v in r.channels_counts.items())
            print(f"  channels:    {ch_str}")
        else:
            print("  channels:    (none)")

        if r.unreadable_files > 0 and r.unreadable_examples:
            print("  unreadable examples:")
            for p in r.unreadable_examples:
                print(f"    - {p}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Wakeword dataset summary reporter")
    ap.add_argument("--root", type=str, default=None,
                    help="Root directory containing class folders (positive/negative/hard_negative/background/rirs)")
    ap.add_argument("--positive", type=str, default=None)
    ap.add_argument("--negative", type=str, default=None)
    ap.add_argument("--hard_negative", type=str, default=None)
    ap.add_argument("--background", type=str, default=None)
    ap.add_argument("--rirs", type=str, default=None)

    ap.add_argument("--ext", nargs="+", default=[".wav"],
                    help="Audio extensions to include, e.g. --ext .wav .flac .ogg (non-wav needs soundfile)")
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) // 2),
                    help="Thread workers for metadata reading")
    ap.add_argument("--json_out", type=str, default=None,
                    help="Optional: write full report to JSON file")
    args = ap.parse_args()

    # Resolve class roots
    class_paths: Dict[str, Optional[Path]] = {}

    if args.root:
        base = Path(args.root).expanduser().resolve()
        for c in CLASSES_DEFAULT:
            class_paths[c] = base / c
    else:
        class_paths["positive"] = Path(args.positive).expanduser().resolve() if args.positive else None
        class_paths["negative"] = Path(args.negative).expanduser().resolve() if args.negative else None
        class_paths["hard_negative"] = Path(args.hard_negative).expanduser().resolve() if args.hard_negative else None
        class_paths["background"] = Path(args.background).expanduser().resolve() if args.background else None
        class_paths["rirs"] = Path(args.rirs).expanduser().resolve() if args.rirs else None

    missing = [k for k, v in class_paths.items() if v is None]
    if missing:
        print("ERROR: Missing paths for:", ", ".join(missing), file=sys.stderr)
        print("Use --root <dir> or provide all of --positive/--negative/--hard_negative/--background/--rirs", file=sys.stderr)
        return 2

    # Validate folders exist
    for k, p in class_paths.items():
        assert p is not None
        if not p.exists() or not p.is_dir():
            print(f"ERROR: Class folder not found or not a directory: {k} -> {p}", file=sys.stderr)
            return 2

    if args.ext and any((e.lower() != ".wav" and not HAS_SF) for e in args.ext):
        print("NOTE: Non-wav formats requested but 'soundfile' not installed.", file=sys.stderr)
        print("      Install: pip install soundfile  (and system libsndfile if needed)", file=sys.stderr)

    t0 = time.time()
    reports: List[ClassReport] = []
    for c in CLASSES_DEFAULT:
        p = class_paths[c]
        assert p is not None
        reports.append(analyze_class(c, p, args.ext, args.workers))

    # Print
    print("\nWakeword Dataset Report")
    print("=======================")
    print(f"Extensions: {', '.join(args.ext)}")
    print(f"soundfile available: {HAS_SF}")
    print(f"Workers: {args.workers}")

    print("\nPer-class summary")
    print("-----------------")
    print_table(reports)

    # Global totals
    total_files = sum(r.file_count for r in reports)
    total_unreadable = sum(r.unreadable_files for r in reports)
    total_dur = sum(r.total_duration_s for r in reports)
    total_size = sum(r.total_size_bytes for r in reports)

    print("\nTotals")
    print("------")
    print(f"Total readable files: {total_files}")
    print(f"Total unreadable:     {total_unreadable}")
    print(f"Total duration:       {human_duration(total_dur)} ({total_dur:.2f}s)")
    print(f"Total size:           {human_bytes(total_size)}")

    print_distributions(reports)

    dt = time.time() - t0
    print(f"\nDone in {dt:.2f}s")

    if args.json_out:
        out = {
            "generated_at_unix": time.time(),
            "extensions": args.ext,
            "soundfile_available": HAS_SF,
            "workers": args.workers,
            "totals": {
                "readable_files": total_files,
                "unreadable_files": total_unreadable,
                "total_duration_s": total_dur,
                "total_size_bytes": total_size,
            },
            "classes": [asdict(r) for r in reports],
        }
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON written: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
