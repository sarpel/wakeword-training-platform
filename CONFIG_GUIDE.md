# Wakeword Training Configuration Guide

This guide explains all the configurable parameters in the training system.

## 📁 Data Configuration (`data`)
*   **sample_rate**: Audio quality (Hz). 16000 is standard for speech.
*   **audio_duration**: Length of audio clips in seconds. 1.0s is usually enough for a short wake word.
*   **feature_type**: How audio is converted for the AI. "mel" (Mel Spectrogram) is best for most cases.
*   **n_mels**: Detail level of the spectrogram. 64 is standard, 40 is faster/smaller for edge devices.
*   **n_mfcc**: Alternative feature type. Set to 0 if using "mel".
*   **normalize_audio**: Keeps volume consistent across samples. Keep this True.

## 🧠 Model Configuration (`model`)
*   **architecture**: The "brain" structure.
    *   `resnet18`: Very accurate, but large. Good for PC/Server.
    *   `mobilenetv3`: Good balance of speed and accuracy.
    *   `tiny_conv`: Extremely small, for microcontrollers (ESP32).
*   **num_classes**: 2 for Wakeword (Wake Word vs. Not Wake Word).
*   **dropout**: Randomly ignores parts of the brain during training to prevent memorization. 0.2-0.5 is typical.
*   **hidden_size**: Size of internal memory (for RNNs like LSTM/GRU).
*   **bidirectional**: If True, processes audio forwards and backwards (better accuracy, 2x slower).

## 🏋️ Training Configuration (`training`)
*   **batch_size**: How many samples to learn from at once. Higher = faster but needs more GPU memory.
*   **epochs**: How many times to go through the entire dataset.
*   **learning_rate**: How fast the model learns. Too high = unstable, too low = slow.
*   **early_stopping_patience**: Stop if model doesn't improve for this many epochs.
*   **num_workers**: CPU cores used to load data. Set to 4-16 depending on your PC.

## 🔊 Augmentation (`augmentation`)
*   **time_stretch**: Speed up or slow down audio (e.g., 0.8 to 1.2x speed).
*   **pitch_shift**: Make voice higher or deeper.
*   **background_noise_prob**: Chance to add background noise (rain, cafe, etc.).
*   **noise_snr**: How loud the noise is (Signal-to-Noise Ratio). Lower = louder noise.
*   **rir_prob**: Chance to add reverb (Room Impulse Response) to simulate different rooms.
*   **time_shift_prob**: Chance to shift the audio in time (left/right).

## 🔧 Optimizer (`optimizer`)
*   **optimizer**: The math used to update the brain. `adamw` is generally best.
*   **weight_decay**: Prevents the model from becoming too complex (regularization).
*   **mixed_precision**: Uses less memory and runs faster on modern GPUs (RTX 2000+).

## 📉 Loss Function (`loss`)
*   **loss_function**: How the model measures its mistakes.
    *   `cross_entropy`: Standard.
    *   `focal_loss`: Focuses more on hard-to-classify examples.
*   **class_weights**: "balanced" makes the model pay equal attention to rare classes.
*   **hard_negative_weight**: Extra penalty for mistaking a similar word for the wake word.

## ⚡ Advanced
*   **qat**: Quantization Aware Training. Prepares model for running on low-power chips (int8).
*   **distillation**: Teaches a small student model from a large teacher model.
