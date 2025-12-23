#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc", ".caf"}  # mp3 garanti değil


def to_mono(x: np.ndarray) -> np.ndarray:
    # x: (n,) or (n, ch)
    if x.ndim == 1:
        return x.astype(np.float32, copy=False)
    return x.mean(axis=1).astype(np.float32, copy=False)


def dbfs_from_rms(x: np.ndarray, eps: float = 1e-12) -> float:
    # Full scale: 1.0 varsayımı (soundfile float okuduğunda genelde -1..1)
    rms = float(np.sqrt(np.mean(x * x)) + eps)
    return 20.0 * np.log10(rms)


def dbfs_from_peak(x: np.ndarray, eps: float = 1e-12) -> float:
    peak = float(np.max(np.abs(x)) + eps)
    return 20.0 * np.log10(peak)


def safe_move(src: Path, dst: Path, keep_structure: bool, root: Path) -> Path:
    if keep_structure:
        rel = src.relative_to(root)
        target = dst / rel
    else:
        target = dst / src.name

    target.parent.mkdir(parents=True, exist_ok=True)

    # Çakışma olursa isim sonuna _1, _2 ekle
    if target.exists():
        stem, suf = target.stem, target.suffix
        i = 1
        while True:
            cand = target.with_name(f"{stem}_{i}{suf}")
            if not cand.exists():
                target = cand
                break
            i += 1

    shutil.move(str(src), str(target))
    return target


def safe_copy(src: Path, dst: Path, keep_structure: bool, root: Path) -> Path:
    if keep_structure:
        rel = src.relative_to(root)
        target = dst / rel
    else:
        target = dst / src.name

    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        stem, suf = target.stem, target.suffix
        i = 1
        while True:
            cand = target.with_name(f"{stem}_{i}{suf}")
            if not cand.exists():
                target = cand
                break
            i += 1

    shutil.copy2(str(src), str(target))
    return target


def is_silent_file(
    path: Path,
    rms_dbfs_threshold: float,
    peak_dbfs_threshold: float | None,
    min_duration_sec: float,
) -> tuple[bool, dict]:
    info = {}
    try:
        with sf.SoundFile(str(path)) as f:
            sr = f.samplerate
            frames = len(f)
            dur = frames / float(sr) if sr else 0.0

            info["sr"] = sr
            info["frames"] = frames
            info["duration_sec"] = dur

            if dur < min_duration_sec:
                info["reason"] = f"duration<{min_duration_sec}"
                return False, info  # çok kısa dosyayı otomatik silent saymıyoruz

            # tamamını oku (dataset çok büyükse "blok okuma" versiyon yazarım)
            x = f.read(dtype="float32", always_2d=False)

        x = to_mono(np.asarray(x))
        if x.size == 0:
            info["reason"] = "empty"
            return True, info  # boş dosya -> silent kabul

        rms_dbfs = dbfs_from_rms(x)
        peak_dbfs = dbfs_from_peak(x)

        info["rms_dbfs"] = rms_dbfs
        info["peak_dbfs"] = peak_dbfs

        silent = rms_dbfs <= rms_dbfs_threshold
        if peak_dbfs_threshold is not None:
            silent = silent and (peak_dbfs <= peak_dbfs_threshold)

        return silent, info

    except Exception as e:
        info["error"] = repr(e)
        # Hatalı okunamayanları silent diye taşımak riskli; default: taşıma
        return False, info


def iter_audio_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def main():
    ap = argparse.ArgumentParser(
        description="Dataset içindeki (çok) sessiz ses dosyalarını bulur ve ayrı klasöre taşır/kopyalar."
    )
    ap.add_argument("input_dir", type=str, help="Taranacak kök klasör")
    ap.add_argument("output_dir", type=str, help="Sessiz bulunanların gideceği klasör")

    ap.add_argument("--rms-th", type=float, default=-50.0,
                    help="RMS dBFS eşiği. Daha küçük/negatif = daha katı. Örn: -55, -60 (varsayılan: -50)")
    ap.add_argument("--peak-th", type=float, default=None,
                    help="Opsiyonel peak dBFS eşiği. Verirsen her ikisini de sağlamalı. Örn: -35")
    ap.add_argument("--min-dur", type=float, default=0.25,
                    help="Bu süreden kısa dosyaları silent sayma (varsayılan: 0.25s)")

    ap.add_argument("--copy", action="store_true",
                    help="Taşımak yerine kopyala (default: move)")
    ap.add_argument("--keep-structure", action="store_true",
                    help="Alt klasör yapısını output içinde koru")
    ap.add_argument("--dry-run", action="store_true",
                    help="Hiçbir şey taşımadan sadece raporla")
    ap.add_argument("--log", type=str, default="silent_report.tsv",
                    help="Rapor dosyası adı (output_dir içine yazılır)")

    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / args.log
    moved = 0
    scanned = 0

    with report_path.open("w", encoding="utf-8") as rep:
        rep.write("status\tpath\tduration_sec\trms_dbfs\tpeak_dbfs\tnote\n")

        for fpath in iter_audio_files(in_dir):
            scanned += 1
            silent, info = is_silent_file(
                fpath,
                rms_dbfs_threshold=args.rms_th,
                peak_dbfs_threshold=args.peak_th,
                min_duration_sec=args.min_dur,
            )

            dur = info.get("duration_sec", "")
            rms = info.get("rms_dbfs", "")
            peak = info.get("peak_dbfs", "")
            note = info.get("reason", info.get("error", ""))

            if silent:
                rep.write(f"silent\t{fpath}\t{dur}\t{rms}\t{peak}\t{note}\n")
                if not args.dry_run:
                    if args.copy:
                        safe_copy(fpath, out_dir, args.keep_structure, in_dir)
                    else:
                        safe_move(fpath, out_dir, args.keep_structure, in_dir)
                    moved += 1
            else:
                rep.write(f"keep\t{fpath}\t{dur}\t{rms}\t{peak}\t{note}\n")

    print(f"Scanned: {scanned}")
    print(f"Silent moved/copied: {moved}  (dry-run={args.dry_run}, copy={args.copy})")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
