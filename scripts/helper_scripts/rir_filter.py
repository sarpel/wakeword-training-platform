#!/usr/bin/env python3
"""
rir_filter.py

Filter out RIR audio files that are "too silent" or not impulse-like.

Why this works (simple intuition):
- A usable RIR usually has a noticeable peak (the impulse) and a decaying tail.
- Junk files are often near-digital-silence, DC-ish, or lack an impulse peak.

What it does:
- Scans input directory recursively for audio files.
- Loads audio (mono mixdown).
- Computes:
  - peak dBFS
  - RMS dBFS
  - dynamic range proxy (peak - RMS)
  - active duration above a relative threshold
  - DC offset ratio
- Rejects files that trip any rule.
- Copies rejects to a reject directory preserving relative paths.
- Writes a CSV report.

Safe defaults: it only rejects the obvious garbage.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}  # mp3/m4a may depend on your libsndfile build


@dataclass
class Metrics:
    peak_dbfs: float
    rms_dbfs: float
    crest_db: float
    active_ms: float
    dc_ratio: float
    duration_s: float
    sr: int


def dbfs_from_amp(x: float, floor_db: float = -120.0) -> float:
    # amplitude -> dBFS (assuming 1.0 is full scale)
    x = float(abs(x))
    if x <= 0.0:
        return floor_db
    return max(20.0 * np.log10(x), floor_db)


def load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    # Mixdown to mono
    mono = data.mean(axis=1).astype(np.float32, copy=False)
    # Remove NaN/Inf just in case
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    return mono, int(sr)


def compute_metrics(x: np.ndarray, sr: int) -> Metrics:
    if x.size == 0:
        return Metrics(
            peak_dbfs=-120.0, rms_dbfs=-120.0, crest_db=0.0, active_ms=0.0, dc_ratio=1.0, duration_s=0.0, sr=sr
        )

    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)) + 1e-12)

    peak_dbfs = dbfs_from_amp(peak)
    rms_dbfs = dbfs_from_amp(rms)
    crest_db = peak_dbfs - rms_dbfs  # impulse-y signals often have higher crest

    duration_s = x.size / sr

    # Active duration: how long is the signal "meaningfully above noise floor"?
    # We use a RELATIVE threshold: e.g. -40 dB from the peak amplitude.
    # This helps distinguish "pure silence" from "quiet but structured".
    rel_thresh_db = -40.0
    rel_thresh_amp = peak * (10.0 ** (rel_thresh_db / 20.0))
    if peak <= 1e-9:
        active_ms = 0.0
    else:
        active = np.abs(x) >= rel_thresh_amp
        active_ms = 1000.0 * float(active.sum()) / sr

    # DC ratio: if the mean is large compared to RMS, it's suspicious (offset / constant-ish)
    mean = float(np.mean(x))
    dc_ratio = float(abs(mean) / rms)  # higher => more DC-like relative to energy

    return Metrics(
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        crest_db=crest_db,
        active_ms=active_ms,
        dc_ratio=dc_ratio,
        duration_s=duration_s,
        sr=sr,
    )


def decide_reject(
    m: Metrics,
    *,
    min_peak_dbfs: float,
    min_rms_dbfs: float,
    min_crest_db: float,
    min_active_ms: float,
    max_dc_ratio: float,
    min_duration_s: float,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    # Rule 0: extremely short files are rarely meaningful RIRs
    if m.duration_s < min_duration_s:
        reasons.append(f"too_short(<{min_duration_s:.3f}s)")

    # Rule 1: near digital silence (peak too low)
    if m.peak_dbfs < min_peak_dbfs:
        reasons.append(f"peak_too_low({m.peak_dbfs:.1f}dBFS < {min_peak_dbfs:.1f}dBFS)")

    # Rule 2: overall energy too low (RMS too low)
    if m.rms_dbfs < min_rms_dbfs:
        reasons.append(f"rms_too_low({m.rms_dbfs:.1f}dBFS < {min_rms_dbfs:.1f}dBFS)")

    # Rule 3: not impulse-like (crest factor too low)
    # Impulse responses usually have a sharp peak + quieter tail => crest factor tends to be higher.
    if m.crest_db < min_crest_db:
        reasons.append(f"crest_too_low({m.crest_db:.1f}dB < {min_crest_db:.1f}dB)")

    # Rule 4: almost no "active" content relative to its own peak (suggests spike or numerical junk)
    if m.active_ms < min_active_ms:
        reasons.append(f"active_too_short({m.active_ms:.1f}ms < {min_active_ms:.1f}ms)")

    # Rule 5: too DC-ish
    if m.dc_ratio > max_dc_ratio:
        reasons.append(f"dc_ratio_too_high({m.dc_ratio:.3f} > {max_dc_ratio:.3f})")

    reject = len(reasons) > 0
    return reject, reasons


def iter_audio_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            out.append(p)
    return out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_or_move(src: Path, dst: Path, move: bool) -> None:
    ensure_parent(dst)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter silent / useless RIR audio files.")
    ap.add_argument("--input-dir", dest="in_dir", required=True, help="Input directory containing RIR audio files")
    ap.add_argument("--reject", dest="reject_dir", required=True, help="Directory to copy/move rejected files into")
    ap.add_argument("--report", dest="report_csv", default="rir_filter_report.csv", help="CSV report path")

    # Action
    ap.add_argument("--move", action="store_true", help="Move rejected files (default: copy)")

    # Thresholds (safe-ish defaults: reject only obvious garbage)
    ap.add_argument(
        "--min-peak-dbfs", type=float, default=-70.0, help="Reject if peak is below this (default: -70 dBFS)"
    )
    ap.add_argument("--min-rms-dbfs", type=float, default=-85.0, help="Reject if RMS is below this (default: -85 dBFS)")
    ap.add_argument("--min-crest-db", type=float, default=6.0, help="Reject if peak-RMS is below this (default: 6 dB)")
    ap.add_argument(
        "--min-active-ms", type=float, default=3.0, help="Reject if active duration is below this (default: 3 ms)"
    )
    ap.add_argument(
        "--max-dc-ratio", type=float, default=0.25, help="Reject if |mean|/RMS is above this (default: 0.25)"
    )
    ap.add_argument(
        "--min-duration-s", type=float, default=0.005, help="Reject if file is shorter than this (default: 0.005 s)"
    )

    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    reject_dir = Path(args.reject_dir).expanduser().resolve()
    report_csv = Path(args.report_csv).expanduser().resolve()

    if not in_dir.exists():
        raise SystemExit(f"Input directory does not exist: {in_dir}")

    files = iter_audio_files(in_dir)
    files.sort()

    total = 0
    kept = 0
    rejected = 0
    errors = 0

    ensure_parent(report_csv)
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "path",
                "status",
                "reasons",
                "sr",
                "duration_s",
                "peak_dbfs",
                "rms_dbfs",
                "crest_db",
                "active_ms",
                "dc_ratio",
            ]
        )

        for p in files:
            total += 1
            rel = p.relative_to(in_dir)
            try:
                x, sr = load_audio_mono(p)
                m = compute_metrics(x, sr)

                rej, reasons = decide_reject(
                    m,
                    min_peak_dbfs=args.min_peak_dbfs,
                    min_rms_dbfs=args.min_rms_dbfs,
                    min_crest_db=args.min_crest_db,
                    min_active_ms=args.min_active_ms,
                    max_dc_ratio=args.max_dc_ratio,
                    min_duration_s=args.min_duration_s,
                )

                if rej:
                    status = "REJECT"
                    rejected += 1
                    dst = reject_dir / rel
                    copy_or_move(p, dst, move=args.move)
                else:
                    status = "KEEP"
                    kept += 1

                w.writerow(
                    [
                        str(rel),
                        status,
                        ";".join(reasons),
                        m.sr,
                        f"{m.duration_s:.6f}",
                        f"{m.peak_dbfs:.2f}",
                        f"{m.rms_dbfs:.2f}",
                        f"{m.crest_db:.2f}",
                        f"{m.active_ms:.3f}",
                        f"{m.dc_ratio:.6f}",
                    ]
                )

            except Exception as e:
                errors += 1
                w.writerow([str(rel), "ERROR", repr(e), "", "", "", "", "", "", ""])

    print("Done.")
    print(f"Scanned   : {total}")
    print(f"Kept      : {kept}")
    print(f"Rejected  : {rejected}")
    print(f"Errors    : {errors}")
    print(f"Reject dir: {reject_dir}")
    print(f"Report    : {report_csv}")
    print()
    print("Tip: If you fear rejecting useful quiet RIRs, loosen thresholds:")
    print("  e.g. --min-peak-dbfs -80 --min-rms-dbfs -95 --min-active-ms 1")


if __name__ == "__main__":
    main()
