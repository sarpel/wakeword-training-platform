#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import json
import subprocess
from pathlib import Path

DESIRED_SR = 16000
DESIRED_CHANNELS = 1
DESIRED_BIT_DEPTH = 16  # PCM s16le
OUTPUT_EXT = ".wav"
SUFFIX = "_converted"

AUDIO_EXTS = {
    ".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus",
    ".m4a", ".aac", ".wma", ".aiff", ".aif", ".aifc"
}

def which(cmd: str) -> bool:
    try:
        subprocess.run([cmd, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except FileNotFoundError:
        return False

def ffprobe_info(path: Path):
    """
    Return tuple (sample_rate:int|None, channels:int|None, bit_depth:int|None, codec:str|None)
    Uses ffprobe if available. On failure, returns (None, None, None, None).
    """
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels,bit_depth,codec_name",
            "-of", "json", str(path)
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        if p.returncode != 0:
            return (None, None, None, None)
        data = json.loads(p.stdout)
        streams = data.get("streams", [])
        if not streams:
            return (None, None, None, None)
        s = streams[0]
        sr = int(s["sample_rate"]) if s.get("sample_rate") else None
        ch = int(s["channels"]) if s.get("channels") else None
        bd = int(s["bit_depth"]) if s.get("bit_depth") else None
        codec = s.get("codec_name")
        return (sr, ch, bd, codec)
    except Exception:
        return (None, None, None, None)

def needs_conversion(path: Path) -> bool:
    """
    Decide if a file needs conversion to target format.
    If ffprobe missing or cannot parse, conservatively convert unless it is a .wav named *_converted.wav.
    """
    if path.suffix.lower() != ".wav":
        return True
    sr, ch, bd, codec = ffprobe_info(path)
    # If probe failed, assume needs conversion unless name already indicates converted
    if sr is None or ch is None:
        return not path.name.lower().endswith(f"{SUFFIX}.wav")
    # bit depth may be None for some wavs; treat None as incompatible to be safe
    if bd is None:
        bd_ok = False
    else:
        bd_ok = (bd == DESIRED_BIT_DEPTH)
    codec_ok = (codec in ("pcm_s16le", "pcm_s16be")) or bd_ok  # tolerate endian ambiguity via bd check
    return not (sr == DESIRED_SR and ch == DESIRED_CHANNELS and codec_ok)

def safe_output_path(in_path: Path) -> Path:
    base = in_path.stem
    parent = in_path.parent
    out = parent / f"{base}{SUFFIX}{OUTPUT_EXT}"
    # Avoid overwrite
    i = 1
    while out.exists():
        out = parent / f"{base}{SUFFIX}_{i}{OUTPUT_EXT}"
        i += 1
    return out

def convert_with_ffmpeg(in_path: Path, out_path: Path) -> bool:
    """
    Convert to 16kHz, mono, 16-bit PCM WAV using ffmpeg.
    Returns True on success.
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(in_path),
        "-ar", str(DESIRED_SR),    # sample rate
        "-ac", str(DESIRED_CHANNELS),  # channels
        "-c:a", "pcm_s16le",       # 16-bit PCM
        str(out_path)
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode == 0

def move_original(root: Path, in_path: Path):
    """
    Move original file to <root>/not_compatible preserving relative structure.
    """
    rel = in_path.relative_to(root)
    quarantine_root = root / "not_compatible"
    dest = quarantine_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(in_path), str(dest))

def process_root(root_dir: Path):
    have_ffmpeg = which("ffmpeg") and which("ffprobe")
    if not have_ffmpeg:
        print("Uyarı: ffmpeg/ffprobe bulunamadı. Lütfen ffmpeg kurun ve PATH'e ekleyin.", file=sys.stderr)

    converted = 0
    skipped = 0
    failed = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            src = Path(dirpath) / name
            if src.suffix.lower() not in AUDIO_EXTS:
                continue
            # Skip already converted outputs
            if src.name.lower().endswith(f"{SUFFIX}.wav"):
                continue

            must_convert = needs_conversion(src) if have_ffmpeg else True
            if not must_convert:
                skipped += 1
                continue

            out_path = safe_output_path(src)
            out_ok = convert_with_ffmpeg(src, out_path) if have_ffmpeg else False

            if out_ok and out_path.exists():
                try:
                    move_original(root_dir, src)
                except Exception as e:
                    print(f"HATA: Orijinali tasiyamadim: {src} -> {e}", file=sys.stderr)
                    # Keep going; consider as converted though
                converted += 1
                print(f"[OK] {src} -> {out_path.name}")
            else:
                failed += 1
                print(f"[FAIL] {src} (donusturulemedi)")

    print(f"\nÖzet: converted={converted}, skipped={skipped}, failed={failed}")

def main():
    if len(sys.argv) < 2:
        print("Kullanım: python audio_convert_16k16bit_mono.py <root_folder>", file=sys.stderr)
        sys.exit(1)
    root = Path(sys.argv[1]).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Hata: Klasör yok: {root}", file=sys.stderr)
        sys.exit(2)
    process_root(root)

if __name__ == "__main__":
    main()
