import os
import shutil

import soundfile as sf


def move_22050_files(root_dir="."):
    dest_dir = os.path.join(root_dir, "22050")
    os.makedirs(dest_dir, exist_ok=True)

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith((".wav", ".mp3", ".flac")):
                continue

            file_path = os.path.join(subdir, file)
            try:
                info = sf.info(file_path)
                if info.samplerate == 22050:
                    target_path = os.path.join(dest_dir, file)
                    shutil.move(file_path, target_path)
                    print(f"Moved: {file_path} -> {target_path}")
            except RuntimeError:
                print(f"Skipped (unreadable): {file_path}")


if __name__ == "__main__":
    move_22050_files(".")
