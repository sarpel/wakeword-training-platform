import os

import soundfile as sf


def analyze_audio_files(root_dir=".", output_file="flagged_files.txt"):
    flagged = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith((".wav", ".flac", ".mp3")):
                continue

            path = os.path.join(subdir, file)
            try:
                info = sf.info(path)
                sample_rate = info.samplerate
                channels = info.channels
                duration = info.frames / sample_rate

                # Flag conditions
                if sample_rate < 8000 or sample_rate < 16000 or duration < 0.4 or duration > 4.0 or channels > 1:
                    flagged.append(f"{path} | {duration:.2f}s | {sample_rate}Hz | {channels}ch")

            except RuntimeError:
                flagged.append(f"{path} | unreadable file")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Flagged audio files\n\n")
        for line in flagged:
            f.write(f"- {line}\n")

    print(f"{len(flagged)} files flagged. Results saved to {output_file}")


if __name__ == "__main__":
    analyze_audio_files(".")
