#!/usr/bin/env python3
# (content identical to previous message; recreated after kernel reset)
import argparse
import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    from tqdm import tqdm
except Exception:

    def tqdm(x: Any, **kwargs: Any) -> Any:
        return x


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wma", ".aac", ".opus"}
TARGET_SR = 16000


def find_audio_files(root: Optional[Path]) -> List[Path]:
    if root is None:
        return []
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return files


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x.mean(axis=1)


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    n_in = x.shape[0]
    n_out = int(round(n_in * sr_out / sr_in))
    if n_out <= 1:
        return x[:1]
    t_in = np.linspace(0.0, 1.0, num=n_in, endpoint=False, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)
    return np.interp(t_out, t_in, x).astype(np.float32)


def read_random_crop(path: Path, crop_s: float, seed: int) -> Tuple[np.ndarray, int]:
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise RuntimeError("Invalid audio metadata")
    total_s = info.frames / info.samplerate
    if total_s < crop_s + 0.01:
        raise RuntimeError("Too short for crop")
    rng = random.Random(seed)
    max_start = total_s - crop_s
    start_s = rng.uniform(0.0, max_start)
    start_frame = int(start_s * info.samplerate)
    n_frames = int(crop_s * info.samplerate)
    audio, sr = sf.read(str(path), start=start_frame, frames=n_frames, dtype="float32", always_2d=False)
    if audio.size == 0:
        raise RuntimeError("Empty read")
    audio = to_mono(audio)
    if sr != TARGET_SR:
        audio = resample_linear(audio, sr, TARGET_SR)
        sr = TARGET_SR
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 0:
        audio = (audio / max(peak, 1e-6) * 0.8).astype(np.float32)
    return audio, sr


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def sample_dataset(
    name: str,
    roots: List[Path],
    out_dir: Path,
    target_count: int,
    crop_min: float,
    crop_max: float,
    max_per_file: int,
    seed: int,
) -> List[Tuple[str, float, str]]:
    files = []
    for r in roots:
        files.extend(find_audio_files(r))
    if not files:
        print(f"[{name}] no files found. skipping.")
        return []

    rng = random.Random(seed)
    rng.shuffle(files)
    per_file_counter: Dict[str, int] = {}
    selected: List[Tuple[str, float, str]] = []

    i = 0
    while len(selected) < target_count and i < len(files):
        p = files[i]
        i += 1
        key = str(p)
        count_for_file = per_file_counter.get(key, 0)
        if count_for_file >= max_per_file:
            continue
        crop_s = rng.uniform(crop_min, crop_max)
        try:
            audio, sr = read_random_crop(p, crop_s, seed=rng.randint(0, 2**31 - 1))
        except Exception:
            continue
        idx = len(selected)
        out_path = out_dir / name / f"{idx:06d}.wav"
        write_wav(out_path, audio, sr)
        selected.append((str(out_path), len(audio) / sr, name))
        per_file_counter[key] = count_for_file + 1

    print(f"[{name}] wrote {len(selected)} crops to {out_dir/name}")
    return selected


def write_manifest(manifest_path: Path, rows: List[Tuple[str, float, str]]):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "duration_s", "source"])
        for p, d, src in rows:
            w.writerow([p, f"{d:.3f}", src])


def main() -> None:
    ap = argparse.ArgumentParser(description="Crop 1–2 s negatives from AudioSet/FSD50K/FMA.")
    ap.add_argument("--audioset_root", type=Path, default=None)
    ap.add_argument("--fsd50k_root", type=Path, default=None)
    ap.add_argument("--fma_root", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--target_audioset", type=int, default=3000)
    ap.add_argument("--target_fsd50k", type=int, default=3000)
    ap.add_argument("--target_fma", type=int, default=2000)
    ap.add_argument("--crop_min", type=float, default=1.0)
    ap.add_argument("--crop_max", type=float, default=2.0)
    ap.add_argument("--max_per_file", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", type=Path, default=None, help="CSV path; default <out_dir>/manifest_easyneg.csv")
    args = ap.parse_args()

    if args.crop_min <= 0 or args.crop_max <= 0 or args.crop_max < args.crop_min:
        raise SystemExit("Invalid crop_min/crop_max")

    rows: List[Tuple[str, float, str]] = []
    if args.audioset_root:
        rows += sample_dataset(
            name="audioset",
            roots=[args.audioset_root],
            out_dir=args.out_dir,
            target_count=args.target_audioset,
            crop_min=args.crop_min,
            crop_max=args.crop_max,
            max_per_file=args.max_per_file,
            seed=args.seed,
        )
    if args.fsd50k_root:
        rows += sample_dataset(
            name="fsd50k",
            roots=[args.fsd50k_root],
            out_dir=args.out_dir,
            target_count=args.target_fsd50k,
            crop_min=args.crop_min,
            crop_max=args.crop_max,
            max_per_file=args.max_per_file,
            seed=args.seed,
        )
    if args.fma_root:
        rows += sample_dataset(
            name="fma",
            roots=[args.fma_root],
            out_dir=args.out_dir,
            target_count=args.target_fma,
            crop_min=args.crop_min,
            crop_max=args.crop_max,
            max_per_file=args.max_per_file,
            seed=args.seed,
        )

    if not rows:
        print("No datasets provided or no files found.")
        return

    manifest_path = args.manifest or (args.out_dir / "manifest_easyneg.csv")
    write_manifest(manifest_path, rows)
    total_h = sum(d for _, d, _ in rows) / 3600.0
    print(f"[all] total crops: {len(rows)}  total ≈ {total_h:.2f} h  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
