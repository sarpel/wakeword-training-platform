#!/usr/bin/env python3
"""
speech_commands_cvs_sampler.py

Goal
----
Build a NORMAL negative pool (not hard-negatives) from:
  - Google Speech Commands (GSC)
  - Common Voice Single Word (CVSW)
Copy audio clips into a single target folder, normalize to 16 kHz mono WAV,
and write a manifest CSV.

Assumptions
-----------
- GSC layout: <gsc_root>/<word>/<files.wav>
- CVSW layout: either <cvsw_root>/<word>/<files> OR a flat folder. We infer labels
  from parent folder name when available; otherwise label="unknown".
- Any audio format readable by soundfile is accepted; output is WAV PCM16 @ 16k.

Install
-------
pip install soundfile numpy tqdm

Examples
--------
python speech_commands_cvs_sampler.py \
  --gsc_root "D:/data/speech_commands" \
  --cvsw_root "D:/data/common_voice_single_word" \
  --out_dir "D:/data/negatives_normal" \
  --target_gsc 8000 --target_cvsw 6000 \
  --per_class_cap 200 \
  --deny_words wake,word,hey_katya \
  --min_sec 0.4 --max_sec 1.5 \
  --seed 42
"""
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import random

import numpy as np
import soundfile as sf

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wma", ".aac", ".opus"}
TARGET_SR = 16000


def list_audio(root: Optional[Path]) -> List[Path]:
    if not root:
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]


def label_from_path(p: Path, dataset_name: str) -> str:
    # Prefer immediate parent folder as label
    if p.parent and p.parent != p.anchor:
        return p.parent.name
    return f"{dataset_name}_unknown"


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x.mean(axis=1)


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    n_in = x.shape[0]
    n_out = int(round(n_in * sr_out / sr_in))
    if n_out <= 1:
        return x[:1].astype(np.float32)
    t_in = np.linspace(0.0, 1.0, num=n_in, endpoint=False, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)
    return np.interp(t_out, t_in, x).astype(np.float32)


def read_clip(path: Path) -> Tuple[np.ndarray, int, float]:
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise RuntimeError("invalid audio metadata")
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.size == 0:
        raise RuntimeError("empty audio")
    audio = to_mono(audio)
    dur = audio.shape[0] / sr
    if sr != TARGET_SR:
        audio = resample_linear(audio, sr, TARGET_SR)
        sr = TARGET_SR
        dur = audio.shape[0] / sr
    # soft normalize
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = (audio / max(peak, 1e-6) * 0.8).astype(np.float32)
    return audio, sr, dur


def sample_dataset(
    name: str,
    root: Optional[Path],
    out_dir: Path,
    target_count: int,
    per_class_cap: int,
    deny_words: List[str],
    min_sec: float,
    max_sec: float,
    seed: int,
) -> List[Tuple[str, float, str]]:
    if root is None:
        return []

    files = list_audio(root)
    if not files:
        print(f"[{name}] no audio files under {root}. skipping.")
        return []

    rng = random.Random(seed)
    rng.shuffle(files)

    per_label_counter: Dict[str, int] = {}
    selected: List[Tuple[str, float, str]] = []

    for p in tqdm(files, desc=f"{name}: scanning"):
        if len(selected) >= target_count:
            break
        label = label_from_path(p, dataset_name=name).lower()
        if label in deny_words:
            continue
        try:
            audio, sr, dur = read_clip(p)
        except Exception:
            continue
        if dur < min_sec or dur > max_sec:
            continue
        # balance by label
        cnt = per_label_counter.get(label, 0)
        if cnt >= per_class_cap:
            continue
        # write
        idx = len(selected)
        out_path = out_dir / name / f"{idx:06d}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, sr, subtype="PCM_16")
        selected.append((str(out_path), dur, name))
        per_label_counter[label] = cnt + 1

    print(f"[{name}] kept {len(selected)} clips. labels used: {len(per_label_counter)}")
    return selected


def write_manifest(path: Path, rows: List[Tuple[str, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "duration_s", "source"])
        for p, d, s in rows:
            w.writerow([p, f"{d:.3f}", s])


def main():
    ap = argparse.ArgumentParser(description="Collect NORMAL negatives from GSC and CV Single Word.")
    ap.add_argument("--gsc_root", type=Path, default=None, help="Google Speech Commands root folder")
    ap.add_argument("--cvsw_root", type=Path, default=None, help="Common Voice Single Word root folder")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--target_gsc", type=int, default=10000)
    ap.add_argument("--target_cvsw", type=int, default=5000)
    ap.add_argument("--per_class_cap", type=int, default=200, help="Max clips per word to keep for balance")
    ap.add_argument("--deny_words", type=str, default="", help="Comma-separated list to exclude (e.g., wakeword variants)")
    ap.add_argument("--min_sec", type=float, default=0.4)
    ap.add_argument("--max_sec", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", type=Path, default=None, help="CSV; default <out_dir>/manifest_normalneg.csv")
    args = ap.parse_args()

    deny = [w.strip().lower() for w in args.deny_words.split(",") if w.strip()]

    rows: List[Tuple[str, float, str]] = []
    if args.gsc_root:
        rows += sample_dataset(
            name="gsc",
            root=args.gsc_root,
            out_dir=args.out_dir,
            target_count=args.target_gsc,
            per_class_cap=args.per_class_cap,
            deny_words=deny,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
            seed=args.seed,
        )
    if args.cvsw_root:
        rows += sample_dataset(
            name="cvsw",
            root=args.cvsw_root,
            out_dir=args.out_dir,
            target_count=args.target_cvsw,
            per_class_cap=args.per_class_cap,
            deny_words=deny,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
            seed=args.seed,
        )

    if not rows:
        print("No data selected.")
        return

    manifest = args.manifest or (args.out_dir / "manifest_normalneg.csv")
    write_manifest(manifest, rows)
    tot = sum(d for _, d, _ in rows)
    print(f"[all] total files: {len(rows)}  total duration ≈ {tot/3600.0:.2f} h  manifest: {manifest}")


if __name__ == "__main__":
    main()
