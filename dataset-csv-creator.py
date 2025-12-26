import csv
from pathlib import Path
import librosa

ROOT = Path("data/raw")

LABEL_MAP = {
    "positive": "wakeword",
    "negative": "non_wakeword",
    "hard_negative": "non_wakeword",
    "background": "background",
    "rirs": "rir"
}

with open("wakeword_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "label", "duration", "source"])

    for folder, label in LABEL_MAP.items():
        base = ROOT / folder
        if not base.exists():
            continue

        for wav in base.rglob("*.wav"):
            try:
                duration = librosa.get_duration(path=wav)
            except Exception:
                continue

            writer.writerow([
                wav.as_posix(),
                label,
                round(duration, 3),
                folder
            ])
