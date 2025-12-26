import os

import librosa
import soundfile as sf


def convert_22050_to_16000(root_dir="."):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith((".wav", ".flac", ".mp3")):
                continue

            file_path = os.path.join(subdir, file)
            try:
                info = sf.info(file_path)
                if info.samplerate == 22050:
                    data, sr = librosa.load(file_path, sr=16000)
                    new_path = os.path.join(subdir, f"{os.path.splitext(file)[0]}_16k.wav")
                    sf.write(new_path, data, 16000)
                    print(f"Converted: {file_path} -> {new_path}")
            except Exception as e:
                print(f"Error on {file_path}: {e}")


if __name__ == "__main__":
    convert_22050_to_16000(".")
