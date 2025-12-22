#!/usr/bin/env python3
"""
dataset_sampler.py

Utilities to:
1) Subsample a huge features .npy (e.g., openwakeword_features_ACAV100M_2000_hrs_16bit.npy)
   by target hours or by fraction, using memmap and chunked writes.
2) Randomly sample audio files from datasets (LibriSpeech, LJSpeech, Microsoft SDNS, MUSAN)
   by target hours or file count, optionally copying or writing a manifest.

Design notes:
- Features mode assumes a frame-based feature matrix with a constant hop (e.g., 10 ms).
- You can sample by FRACTION or by TARGET_HOURS. If both given, TARGET_HOURS wins.
- For features, to preserve locality, we sample contiguous segments of length `segment_seconds`.
- Audio mode computes durations with soundfile without loading full audio into RAM.
- Reproducible with --seed.

Install deps (if needed):
  pip install numpy soundfile tqdm

Usage examples:
  # 1) Subsample features to ~300 hours, assuming 10 ms hop and 2.0 s segments
  python dataset_sampler.py sample_features \
      --input /data/openwakeword_features_ACAV100M_2000_hrs_16bit.npy \
      --output /data/openwakeword_features_subset_300h.npy \
      --target_hours 300 \
      --frame_hop_ms 10 \
      --segment_seconds 2.0 \
      --seed 42

  # 2) Subsample features by 0.2 fraction (~20% of frames)
  python dataset_sampler.py sample_features \
      --input /data/features.npy \
      --output /data/features_20pct.npy \
      --fraction 0.2 \
      --frame_hop_ms 10 \
      --segment_seconds 1.5

  # 3) Sample ~40 hours from LibriSpeech + 20 hours from LJSpeech
  python dataset_sampler.py sample_audio \
      --roots /data/LibriSpeech/train-clean-100 /data/LJSpeech-1.1/wavs \
      --output_manifest /data/sample_manifest.csv \
      --target_hours 60 \
      --copy_to /data/sampled_audio \
      --max_files_per_dir 100000 \
      --seed 123

  # 4) Sample 8 hours from MUSAN only, do not copy, just manifest
  python dataset_sampler.py sample_audio \
      --roots /data/musan \
      --output_manifest /data/musan8h_manifest.csv \
      --target_hours 8 \
      --seed 7
"""
import argparse
import csv
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np

try:
    import soundfile as sf
except Exception as e:
    sf = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# -----------------------------
# Helpers
# -----------------------------

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wma", ".aac", ".opus"}


def find_audio_files(roots: List[Path], max_files_per_dir: int = 1_000_000) -> List[Path]:
    files = []
    for r in roots:
        count = 0
        for p in r.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                files.append(p)
                count += 1
                if count >= max_files_per_dir:
                    break
    return files


def get_audio_duration_seconds(path: Path) -> Optional[float]:
    if sf is None:
        return None
    try:
        info = sf.info(str(path))
        if info.frames > 0 and info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
        return None
    except Exception:
        return None


def write_manifest(manifest_path: Path, rows: List[Tuple[str, float]]):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "duration_s"])
        for p, d in rows:
            w.writerow([p, f"{d:.3f}"])


def human_hours(seconds: float) -> str:
    return f"{seconds/3600.0:.2f} h"


# -----------------------------
# Features sampler
# -----------------------------

def sample_features_memmap(
    input_npy: Path,
    output_npy: Path,
    frame_hop_ms: float,
    segment_seconds: float,
    target_hours: Optional[float] = None,
    fraction: Optional[float] = None,
    seed: int = 42,
):
    """
    Subsample a large features .npy by selecting random contiguous segments.

    Assumptions:
    - Input is a 2D or 3D array:
        2D: (num_frames, feat_dim)
        3D: (num_frames, time, feat_dim)  # Rare; we handle by flattening first dimension only across frames.
    - Constant feature hop size given by frame_hop_ms.

    Strategy:
    - Compute total seconds from num_frames * hop.
    - Determine number of frames to extract from target_hours or fraction.
    - Draw random start indices and take contiguous windows of segment_frames until quota is met.
    - Write out in chunks to avoid high RAM usage.
    """
    rng = random.Random(seed)

    # Read shape without loading all into RAM
    arr_in = np.load(str(input_npy), mmap_mode="r")
    shape = arr_in.shape
    if len(shape) == 2:
        num_frames, feat_dim = shape
        frame_shape = (feat_dim,)
    elif len(shape) == 3:
        num_frames, time_dim, feat_dim = shape
        frame_shape = (time_dim, feat_dim)
    else:
        raise ValueError(f"Unsupported input shape: {shape}. Expect 2D or 3D.")

    hop_s = frame_hop_ms / 1000.0
    total_seconds = num_frames * hop_s

    if target_hours is None and fraction is None:
        raise ValueError("Provide either --target_hours or --fraction")

    if target_hours is not None:
        target_seconds = target_hours * 3600.0
        fraction = min(1.0, max(0.0, target_seconds / total_seconds))
    else:
        fraction = float(fraction)
        fraction = min(1.0, max(0.0, fraction))

    target_frames = int(round(num_frames * fraction))
    if target_frames <= 0:
        raise ValueError("Target frames computed as 0. Check inputs.")

    segment_frames = max(1, int(round(segment_seconds / hop_s)))
    # Number of segments needed
    n_segments = max(1, int(math.ceil(target_frames / segment_frames)))

    # Prepare output memmap with conservative over-allocation then truncate at end
    # We cannot know exact final count due to rounding, so allocate n_segments*segment_frames
    out_frames_cap = n_segments * segment_frames
    out_shape = (out_frames_cap,) + frame_shape
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    arr_out = np.memmap(str(output_npy), dtype=arr_in.dtype, mode="w+", shape=out_shape)

    filled = 0
    for _ in tqdm(range(n_segments), desc="Sampling segments"):
        start = rng.randint(0, max(0, num_frames - segment_frames))
        end = start + segment_frames
        # Slice and write
        chunk = arr_in[start:end]
        arr_out[filled:filled + len(chunk)] = chunk
        filled += len(chunk)
        if filled >= target_frames:
            break

    # sample_features_memmap fonksiyonunun sonunda
        final_frames = min(filled, target_frames)

        # memmap bağlantısını kapat
        del arr_out

        # Geçici isim belirle
        tmp_path = str(output_npy) + ".tmp.npy"

        # Oversized memmap dosyasını yeniden oku
        arr_out = np.memmap(str(output_npy), dtype=arr_in.dtype, mode="r", shape=out_shape)

        # Gerçek veriyi kopyala
        compact = np.array(arr_out[:final_frames])

        # Geçici dosyaya yaz
        np.save(tmp_path, compact)

        # Eski memmap dosyasını sil ve geçiciyi asıl isimle değiştir
        os.remove(str(output_npy))
        os.rename(tmp_path, str(output_npy))

        print(f"[features] Input frames: {num_frames:,}  -> Output frames: {final_frames:,}")
        print(f"[features] Output seconds ≈ {final_frames * hop_s:.1f} s ({human_hours(final_frames * hop_s)})")
        print(f"[features] Saved: {output_npy}")

    # Clean temp memmap
    try:
        # remove the old oversized file if different from final .npy
        # np.save overwrote the same path with a new file. nothing else to remove.
        pass
    except Exception:
        pass

    print(f"[features] Input frames: {num_frames:,}  -> Output frames: {final_frames:,}")
    print(f"[features] Effective fraction: {final_frames/num_frames:.4f}")
    print(f"[features] Output seconds ≈ {final_frames * hop_s:.1f} s ({human_hours(final_frames * hop_s)})")
    print(f"[features] Saved: {output_npy}")


# -----------------------------
# Audio sampler
# -----------------------------

def sample_audio_files(
    roots: List[Path],
    target_hours: Optional[float],
    target_count: Optional[int],
    output_manifest: Path,
    copy_to: Optional[Path],
    seed: int,
    max_files_per_dir: int = 1_000_000,
):
    """
    Randomly sample audio files until reaching target_hours OR target_count.
    Writes a manifest CSV; optionally copies files to a flat folder tree.
    """
    if sf is None:
        print("soundfile is not available. Install with `pip install soundfile`. Exiting.", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(seed)
    files = find_audio_files(roots, max_files_per_dir=max_files_per_dir)
    if not files:
        print("No audio files found under the given roots.", file=sys.stderr)
        sys.exit(2)

    # Shuffle for randomness
    rng.shuffle(files)

    selected: List[Tuple[str, float]] = []
    total_sec = 0.0

    def meets_goal() -> bool:
        goal_count = (target_count is not None and len(selected) >= target_count)
        goal_hours = (target_hours is not None and total_sec >= target_hours * 3600.0)
        # If both given, stop when either satisfied
        return goal_count or goal_hours

    for p in tqdm(files, desc="Scanning durations"):
        dur = get_audio_duration_seconds(p)
        if dur is None or dur <= 0:
            continue
        selected.append((str(p), dur))
        total_sec += dur
        if meets_goal():
            break

    if not selected:
        print("No valid audio durations found.", file=sys.stderr)
        sys.exit(3)

    # If we overshot hours, do minimal trim at the end
    if target_hours is not None and total_sec > target_hours * 3600.0:
        # Trim last file if desired. Simpler: keep as is; difference is small.
        pass

    write_manifest(output_manifest, selected)
    print(f"[audio] Selected {len(selected)} files. Total ≈ {human_hours(total_sec)}")
    print(f"[audio] Manifest: {output_manifest}")

    if copy_to:
        copy_to.mkdir(parents=True, exist_ok=True)
        # Keep folder structure shallow: copy with unique numeric prefix to avoid collisions
        for i, (src, _) in enumerate(tqdm(selected, desc="Copying")):
            src_path = Path(src)
            dst = copy_to / f"{i:06d}_{src_path.name}"
            try:
                shutil.copy2(src_path, dst)
            except Exception as e:
                print(f"Copy failed for {src_path}: {e}", file=sys.stderr)
        print(f"[audio] Copied files to: {copy_to}")


# -----------------------------
# CLI
# -----------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Dataset sampling utilities for wakeword training.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # sample_features
    pf = sub.add_parser("sample_features", help="Subsample a large features .npy by hours or fraction.")
    pf.add_argument("--input", required=True, type=Path, help="Path to input .npy (memmap-supported).")
    pf.add_argument("--output", required=True, type=Path, help="Path to output .npy")
    pf.add_argument("--frame_hop_ms", type=float, default=10.0, help="Feature hop in milliseconds. Default 10 ms.")
    pf.add_argument("--segment_seconds", type=float, default=2.0, help="Length of contiguous segments to sample.")
    pf.add_argument("--target_hours", type=float, default=None, help="Target hours to extract.")
    pf.add_argument("--fraction", type=float, default=None, help="Fraction of frames to extract (0..1).")
    pf.add_argument("--seed", type=int, default=42)

    # sample_audio
    pa = sub.add_parser("sample_audio", help="Randomly sample audio files by hours or count.")
    pa.add_argument("--roots", nargs="+", required=True, type=Path, help="Dataset root directories to search recursively.")
    pa.add_argument("--output_manifest", required=True, type=Path, help="CSV path to write selected files and durations.")
    pa.add_argument("--target_hours", type=float, default=None, help="Target total hours across all selected files.")
    pa.add_argument("--target_count", type=int, default=None, help="Target number of files.")
    pa.add_argument("--copy_to", type=Path, default=None, help="Optional directory to copy selected files into.")
    pa.add_argument("--max_files_per_dir", type=int, default=1_000_000, help="Safety cap while scanning.")
    pa.add_argument("--seed", type=int, default=42)

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "sample_features":
        sample_features_memmap(
            input_npy=args.input,
            output_npy=args.output,
            frame_hop_ms=args.frame_hop_ms,
            segment_seconds=args.segment_seconds,
            target_hours=args.target_hours,
            fraction=args.fraction,
            seed=args.seed,
        )
    elif args.cmd == "sample_audio":
        sample_audio_files(
            roots=args.roots,
            target_hours=args.target_hours,
            target_count=args.target_count,
            output_manifest=args.output_manifest,
            copy_to=args.copy_to,
            seed=args.seed,
            max_files_per_dir=args.max_files_per_dir,
        )
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
