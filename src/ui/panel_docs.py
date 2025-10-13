"""
Panel 6: Documentation & Knowledge Base
- Comprehensive wakeword training guide
- Best practices and industry standards
- Troubleshooting guide
"""
import gradio as gr


def create_docs_panel() -> gr.Blocks:
    """
    Create Panel 6: Documentation & Knowledge Base

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks() as panel:
        gr.Markdown("# 📚 Documentation & Knowledge Base")
        gr.Markdown("Complete guide to wakeword training, best practices, and troubleshooting.")

        with gr.Tabs():
            # Introduction
            with gr.TabItem("🏠 Introduction"):
                gr.Markdown("""
# Welcome to Wakeword Training Platform

## What is a Wakeword?

A **wakeword** (or wake word) is a specific word or phrase that activates a voice-controlled device.
Examples include "Hey Siri", "Alexa", "OK Google". When detected, the device starts listening for commands.

## Use Cases

- **Smart Home Devices**: Voice assistants, smart speakers
- **Mobile Applications**: Voice-activated apps
- **Automotive**: Hands-free car systems
- **Accessibility**: Voice control for users with disabilities
- **IoT Devices**: Voice-controlled appliances

## How This Platform Works

This platform provides a complete pipeline for training custom wakeword detection models:

1. **Dataset Management** - Organize and validate your audio datasets
2. **Configuration** - Set training parameters with industry-standard defaults
3. **Training** - Train models with GPU acceleration and live monitoring
4. **Evaluation** - Test models with files or live microphone
5. **Export** - Convert models to ONNX for deployment
6. **Documentation** - Learn best practices (you are here!)

## System Requirements

### Mandatory
- **NVIDIA GPU** with CUDA support (Compute Capability 6.0+)
- **CUDA Toolkit** 11.8 or 12.x
- **cuDNN** 8.x
- **Python** 3.8-3.11

### Recommended
- **GPU Memory**: 8GB+ (10GB+ for large datasets)
- **RAM**: 16GB+
- **Storage**: 100GB+ for datasets
- **CPU**: Multi-core for data loading

## Quick Start

1. Prepare your datasets (positive, negative, background, etc.)
2. Load datasets in Panel 1 and split them
3. Configure training parameters in Panel 2
4. Start training in Panel 3
5. Evaluate your model in Panel 4
6. Export to ONNX in Panel 5

Let's get started! 🚀
                """)

            # Dataset Preparation
            with gr.TabItem("📊 Dataset Preparation"):
                gr.Markdown("""
# Dataset Preparation Guide

## Dataset Types

### 1. Positive Samples ✅
- **What**: Actual wakeword utterances
- **Examples**: "Hey Assistant", "Wake Up", your custom phrase
- **Requirements**:
  - Minimum: 1,000 samples (5,000+ recommended)
  - Multiple speakers: 50+ unique voices
  - Varied environments: quiet, noisy, reverberant
  - Natural variations: speed, pitch, accent

### 2. Negative Samples ❌
- **What**: Non-wakeword speech
- **Examples**: General conversation, random phrases
- **Requirements**:
  - 8-10x more than positive samples
  - Diverse vocabulary and speakers
  - Similar acoustic characteristics to positive

### 3. Hard Negative Samples ⚠️
- **What**: Phrases that sound similar to wakeword
- **Examples**: If wakeword is "Hey Assistant"
  - "Hey system"
  - "Hey assistance"
  - "A assistant"
- **Requirements**:
  - 20-30% of total negative samples
  - Critical for reducing false positives

### 4. Background Noise 🔊
- **What**: Environmental sounds without speech
- **Examples**:
  - White noise, pink noise
  - Traffic, wind, rain
  - TV, music, crowd noise
- **Usage**: Mixed with positive/negative at 30-50% probability

### 5. Room Impulse Responses (RIRs) 🏠
- **What**: Acoustic characteristics of rooms
- **Usage**: Convolved with audio at 20-30% probability
- **Effect**: Adds realistic reverberation

### 6. .npy Files 💾
- **What**: Pre-computed features (MFCC, spectrograms)
- **Format**: NumPy arrays
- **Supported shapes**:
  - Raw audio: (N, samples)
  - Spectrograms: (N, freq_bins, time_steps)
  - MFCC: (N, n_mfcc, time_steps)

## Audio Format Requirements

- **Sample Rate**: 16kHz (recommended, 8-48kHz supported)
- **Bit Depth**: 16-bit
- **Channels**: Mono (stereo will be converted)
- **Duration**: 1.5-2 seconds typical
- **Format**: WAV, MP3, FLAC, OGG

## Dataset Organization

```
dataset_root/
├── positive/
│   ├── speaker1/
│   │   ├── sample_001.wav
│   │   └── sample_002.wav
│   └── speaker2/
│       └── sample_001.wav
├── negative/
│   ├── conversation/
│   └── commands/
├── hard_negative/
│   └── similar_phrases/
├── background/
│   ├── noise/
│   └── music/
└── rirs/
    ├── room1.wav
    └── room2.wav
```

**Note**: Subfolders are automatically scanned!

## Recording Best Practices

### For Positive Samples
1. **Multiple Speakers**: Aim for age, gender, accent diversity
2. **Natural Delivery**: Not robotic, include hesitations
3. **Varied Speed**: Slow, normal, fast delivery
4. **Varied Volume**: Whisper to normal speaking
5. **Varied Distance**: Near (30cm) to far (3m) from mic

### For Negative Samples
1. **Phonetically Rich**: Cover diverse speech sounds
2. **Conversational**: Natural speech patterns
3. **Varied Topics**: General conversation, news, etc.

### General Guidelines
- Use good quality microphone
- Record in quiet environment (add noise via augmentation)
- Avoid clipping (peak below -3dB)
- Remove silence padding
- Label files consistently

## Data Quality Checklist

✅ No corrupted files
✅ Consistent sample rate
✅ Appropriate duration (1-3 seconds)
✅ No excessive silence
✅ Balanced speaker distribution
✅ Clear audio (SNR > 20dB for clean samples)
✅ Proper labeling

## Industry Standards

| Metric | Minimum | Recommended | Excellent |
|--------|---------|-------------|-----------|
| Positive Samples | 1,000 | 5,000 | 20,000+ |
| Negative Samples | 8,000 | 40,000 | 200,000+ |
| Unique Speakers | 20 | 50 | 200+ |
| Recording Hours | 2h | 10h | 50h+ |
| Pos:Neg Ratio | 1:5 | 1:8 | 1:10 |
                """)

            # Training Configuration
            with gr.TabItem("⚙️ Configuration Guide"):
                gr.Markdown("""
# Training Configuration Guide

## Basic Parameters

### Data Parameters

#### Sample Rate
- **Default**: 16,000 Hz
- **Why**: Balance between quality and computational cost
- **Options**: 8kHz (low quality, fast), 16kHz (recommended), 22kHz/44kHz (high quality, slower)

#### Audio Duration
- **Default**: 1.5 seconds
- **Why**: Typical wakeword length with padding
- **Consideration**: Longer = more context, more memory

#### MFCC Coefficients
- **Default**: 40
- **Range**: 13-80
- **Why**: Captures important frequency characteristics
- **Trade-off**: More = richer features but more computation

#### FFT Size
- **Default**: 512
- **Options**: 256, 512, 1024, 2048
- **Why**: Frequency resolution vs time resolution
- **Rule**: Should be < hop_length * sample_rate / 1000

### Training Parameters

#### Batch Size
- **Default**: 32
- **Range**: 8-256
- **Factors**: GPU memory, dataset size
- **Rule**: Larger = faster training, more memory
- **Tip**: Use automatic batch size finder if OOM errors

#### Epochs
- **Default**: 50
- **Range**: 10-200
- **Why**: Sufficient for convergence with early stopping
- **Note**: Early stopping prevents overfitting

#### Learning Rate
- **Default**: 0.001 (1e-3)
- **Range**: 1e-5 to 1e-2
- **Why**: Balanced learning speed
- **Scheduling**: Cosine annealing recommended

#### Early Stopping Patience
- **Default**: 10 epochs
- **Why**: Stops training when validation loss plateaus
- **Trade-off**: Higher = more chances to improve, longer training

### Model Parameters

#### Architecture Comparison

| Architecture | Accuracy | Speed | Size | Use Case |
|--------------|----------|-------|------|----------|
| ResNet-18 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 44MB | Best accuracy |
| MobileNetV3 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 5MB | Edge deployment |
| LSTM | ⭐⭐⭐ | ⭐⭐ | 20MB | Sequential data |
| GRU | ⭐⭐⭐ | ⭐⭐⭐ | 15MB | Faster LSTM |
| TCN | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 25MB | Modern choice |

**Recommendation**: Start with ResNet-18 for best results

## Advanced Parameters

### Augmentation

#### Time Stretch (0.8-1.2x)
- Simulates faster/slower speech
- Preserves pitch

#### Pitch Shift (±2 semitones)
- Simulates different voice pitches
- Important for speaker variation

#### Background Noise (30-50% probability)
- SNR: 5-20 dB
- Adds realism
- Reduces overfitting to clean audio

#### RIR Convolution (20-30% probability)
- Adds room reverberation
- Critical for real-world performance

### Optimizer & Scheduler

#### Optimizers
- **Adam** (recommended): Adaptive learning, works well with default settings
- **SGD**: Classic, requires careful tuning
- **AdamW**: Adam with weight decay decoupling

#### Schedulers
- **Cosine**: Smooth decay, no tuning needed (recommended)
- **Step**: Drops LR at milestones
- **Plateau**: Reduces LR when metric plateaus
- **None**: Constant learning rate

#### Weight Decay
- **Default**: 1e-4
- **Purpose**: L2 regularization, prevents overfitting

#### Gradient Clipping
- **Default**: 1.0
- **Purpose**: Prevents exploding gradients
- **Rare Issue**: Usually not triggered

### Loss & Sampling

#### Loss Functions
- **Cross Entropy**: Standard for classification (recommended)
- **Focal Loss**: Better for imbalanced datasets
  - Alpha: 0.25 (class weight)
  - Gamma: 2.0 (focus on hard examples)

#### Label Smoothing
- **Default**: 0.1
- **Effect**: Softens targets (0.9/0.1 instead of 1.0/0.0)
- **Benefit**: Better generalization

#### Class Weights
- **Balanced**: Automatic weight calculation
- **None**: No weighting
- **Custom**: Manual weight specification

#### Hard Negative Weight
- **Default**: 2.0
- **Effect**: Give more importance to hard negatives
- **Critical**: Reduces false positives

### Checkpointing

#### Frequency Options
- **Best Only**: Save only when validation improves (recommended)
- **Every Epoch**: All checkpoints (for analysis)
- **Every N Epochs**: Periodic saving
- **Trade-off**: Storage vs flexibility

## Presets

### Small Dataset (<10k samples)
- Aggressive augmentation
- Smaller model (MobileNet)
- More regularization
- Lower batch size

### Large Dataset (>100k samples)
- Moderate augmentation
- Larger model (ResNet)
- Higher batch size
- Faster convergence

### Fast Training
- Minimal augmentation
- Larger batch size
- Fewer epochs
- Quick iteration

### High Accuracy
- Aggressive augmentation
- ResNet-18
- More epochs
- Patience for convergence

### Edge Deployment
- MobileNetV3
- Optimized for size
- Quantization-aware training
                """)

            # Training Process
            with gr.TabItem("🚀 Training Process"):
                gr.Markdown("""
# Training Process Guide

## Metrics Explained

### Loss
- **What**: Error measurement (lower is better)
- **Train Loss**: Error on training data
- **Val Loss**: Error on validation data
- **Healthy Pattern**: Both decreasing, val_loss slightly higher
- **Warning Signs**:
  - Val_loss increasing while train_loss decreasing = **Overfitting**
  - Both very high = **Underfitting**

### Accuracy
- **What**: Percentage of correct predictions
- **Target**: >95% on test set
- **Note**: Can be misleading with imbalanced data

### False Positive Rate (FPR) 🚨
- **What**: % of negatives incorrectly classified as positive
- **Impact**: False alarms, user frustration
- **Target**: <5% (ideally <2%)
- **Critical Metric**: Most important for user experience

### False Negative Rate (FNR)
- **What**: % of positives incorrectly classified as negative
- **Impact**: Missed activations
- **Target**: <5%
- **Note**: Less critical than FPR (user will repeat)

### Precision & Recall
- **Precision**: When model says "wakeword", how often is it right?
- **Recall**: Of all actual wakewords, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall

### Training Speed
- **Samples/sec**: Throughput (higher is better)
- **Target**: >1000 samples/sec on modern GPU
- **Factors**: Batch size, augmentation, model architecture

## Training Stages

### Stage 1: Initial (Epochs 1-5)
- **Behavior**: Rapid loss decrease
- **Metrics**: Accuracy quickly rises from 50% to 80%+
- **Normal**: High variability

### Stage 2: Learning (Epochs 5-30)
- **Behavior**: Steady improvement
- **Metrics**: Reaching 90%+ accuracy
- **Monitoring**: Watch for overfitting signs

### Stage 3: Convergence (Epochs 30-50)
- **Behavior**: Slow improvements
- **Metrics**: Fine-tuning to 95%+
- **Decision**: Early stopping may trigger

### Stage 4: Plateau (After convergence)
- **Behavior**: No improvement
- **Metrics**: Oscillating around best values
- **Action**: Early stopping terminates training

## Common Training Patterns

### 🟢 Healthy Training
```
Epoch | Train Loss | Val Loss | Train Acc | Val Acc
------|------------|----------|-----------|--------
1     | 0.6500     | 0.6700   | 65%       | 63%
10    | 0.2000     | 0.2300   | 92%       | 90%
20    | 0.0800     | 0.1200   | 97%       | 95%
30    | 0.0400     | 0.0900   | 98%       | 96%
```
✅ Both losses decreasing
✅ Val_loss close to train_loss
✅ High accuracy on both sets

### 🟡 Overfitting
```
Epoch | Train Loss | Val Loss | Train Acc | Val Acc
------|------------|----------|-----------|--------
1     | 0.6500     | 0.6700   | 65%       | 63%
10    | 0.2000     | 0.2500   | 92%       | 88%
20    | 0.0500     | 0.3000   | 98%       | 87%
30    | 0.0100     | 0.4000   | 99%       | 85%
```
❌ Train_loss decreasing, val_loss increasing
❌ High train_acc, lower val_acc
**Solution**: More regularization, augmentation, early stopping

### 🔴 Underfitting
```
Epoch | Train Loss | Val Loss | Train Acc | Val Acc
------|------------|----------|-----------|--------
1     | 0.6900     | 0.6900   | 52%       | 51%
10    | 0.6500     | 0.6600   | 60%       | 59%
20    | 0.6200     | 0.6300   | 65%       | 64%
30    | 0.6000     | 0.6100   | 68%       | 67%
```
❌ Both losses high and barely decreasing
❌ Low accuracy on both sets
**Solution**: Larger model, more epochs, lower regularization

## Troubleshooting

### Problem: Training is Slow
- **Check GPU Utilization**: Should be >80%
- **Solutions**:
  - Increase batch size
  - Reduce num_workers if CPU bottleneck
  - Enable mixed precision (FP16)
  - Check data loading speed

### Problem: GPU Out of Memory
- **Solutions**:
  - Reduce batch size
  - Shorter audio duration
  - Smaller model
  - Enable gradient accumulation
  - Clear CUDA cache

### Problem: Loss is NaN
- **Causes**: Exploding gradients, too high learning rate
- **Solutions**:
  - Lower learning rate (try 1e-4)
  - Enable gradient clipping
  - Check for corrupted data
  - Use mixed precision carefully

### Problem: Poor Validation Performance
- **If Overfitting**:
  - More augmentation
  - More regularization (weight decay, label smoothing)
  - Smaller model
  - More diverse data

- **If Underfitting**:
  - Larger model
  - Less regularization
  - More epochs
  - Better features

### Problem: High False Positive Rate
- **Solutions**:
  - Add more hard negatives
  - Increase hard_negative_weight
  - Lower detection threshold
  - More diverse negative samples
  - Check for data leakage

### Problem: High False Negative Rate
- **Solutions**:
  - Add more positive samples
  - More augmentation on positives
  - Higher detection threshold
  - Check for very hard cases in data

## Best Practices

1. **Start Simple**: Use default configuration first
2. **Monitor Closely**: Watch metrics every few epochs
3. **Save Checkpoints**: Enable frequent checkpointing during experimentation
4. **Validate Often**: Check validation metrics, not just loss
5. **Test Thoroughly**: Final test on unseen test set
6. **Real-World Testing**: Test with actual use cases
7. **Iterate**: Training is iterative, expect multiple runs
                """)

            # Evaluation & Deployment
            with gr.TabItem("🎯 Evaluation Guide"):
                gr.Markdown("""
# Evaluation & Deployment Guide

## Evaluation Strategies

### 1. Test Set Evaluation
- **When**: After training completes
- **Purpose**: Unbiased performance estimate
- **Metrics**: Accuracy, FPR, FNR, F1
- **Requirement**: Test set never used during training

### 2. File-Based Testing
- **When**: Testing specific scenarios
- **Purpose**: Controlled evaluation
- **Use Cases**:
  - Different accents
  - Various noise conditions
  - Edge cases

### 3. Live Microphone Testing
- **When**: Final validation
- **Purpose**: Real-world performance
- **Critical**: Tests actual deployment scenario

## Threshold Selection

### Understanding Threshold
- Model outputs confidence (0.0-1.0)
- Threshold determines positive classification
- **High threshold** (0.7-0.9): Fewer false positives, more false negatives
- **Low threshold** (0.3-0.5): More false positives, fewer false negatives

### Finding Optimal Threshold

1. **Plot ROC Curve**: Trade-off visualization
2. **Calculate Metrics**: At different thresholds
3. **Business Decision**:
   - Voice assistant: Prioritize low FPR (fewer false alarms)
   - Accessibility: Prioritize low FNR (don't miss activations)

### Recommended Thresholds

| Use Case | Threshold | FPR Target | FNR Target |
|----------|-----------|------------|------------|
| Consumer Device | 0.7-0.8 | <2% | <5% |
| Accessibility | 0.4-0.5 | <10% | <2% |
| Security | 0.85-0.95 | <0.5% | <10% |

## Real-World Testing

### Test Scenarios

#### 1. Acoustic Conditions
- Quiet room
- TV/music playing
- Multiple speakers
- Outdoor environment
- Car interior

#### 2. Speaker Variations
- Different ages
- Gender diversity
- Accent variations
- Speech impediments
- Emotional states (excited, tired, etc.)

#### 3. Distance & Position
- Near (30cm)
- Medium (1-2m)
- Far (3-5m)
- Off-axis angles

#### 4. Challenging Cases
- Whisper
- Shouting
- Fast speech
- Hesitant speech
- Partial utterance

### Testing Checklist

✅ Test set metrics (Acc, FPR, FNR)
✅ Confusion matrix analysis
✅ Per-class performance
✅ Real-world file testing
✅ Live microphone testing
✅ Various acoustic conditions
✅ Multiple speakers
✅ Edge cases
✅ Latency measurement
✅ Error analysis

## Deployment Considerations

### Model Format
- **PyTorch (.pt)**: Training and Python deployment
- **ONNX (.onnx)**: Cross-platform, faster inference
- **Quantized ONNX**: Smaller, faster, slight accuracy loss

### Optimization Techniques

#### FP16 Quantization
- **Size**: ~50% reduction
- **Speed**: 1.5-2x faster on modern GPUs
- **Accuracy**: Minimal loss (<1%)
- **Recommended**: Yes for most deployments

#### INT8 Quantization
- **Size**: ~75% reduction
- **Speed**: 2-4x faster
- **Accuracy**: 1-3% loss
- **Consideration**: Requires calibration data

### Edge Deployment

#### Target Devices
- Raspberry Pi
- NVIDIA Jetson
- Mobile phones (iOS/Android)
- Custom hardware

#### Requirements
- **Model Size**: <10MB (INT8 quantized)
- **Inference Time**: <50ms
- **Memory**: <100MB RAM
- **Power**: Low-power optimized

#### Recommendations
- Use MobileNetV3 architecture
- INT8 quantization
- Optimize for target hardware
- Test on actual device

### Inference Pipeline

1. **Audio Capture**: Continuous or triggered
2. **Preprocessing**: Resample, normalize
3. **Feature Extraction**: MFCC/Mel-spectrogram
4. **Model Inference**: Forward pass
5. **Post-processing**: Threshold, smoothing
6. **Action**: Trigger downstream logic

### Latency Budget

| Component | Time (ms) | Optimization |
|-----------|-----------|--------------|
| Audio Buffer | 10-50 | Minimize buffer size |
| Preprocessing | 5-10 | Optimize transforms |
| Model Inference | 10-30 | Quantization, optimization |
| Post-processing | 1-5 | Efficient algorithms |
| **Total** | **30-100** | **<100ms target** |

## Production Checklist

✅ Model meets accuracy targets (>95%)
✅ FPR is acceptable (<5%)
✅ FNR is acceptable (<5%)
✅ Inference latency <100ms
✅ Model size fits deployment constraints
✅ Real-world testing passed
✅ Error handling implemented
✅ Monitoring & logging in place
✅ A/B testing plan ready
✅ Rollback strategy defined
                """)

            # Troubleshooting
            with gr.TabItem("🔧 Troubleshooting"):
                gr.Markdown("""
# Troubleshooting Guide

## Installation Issues

### CUDA Not Available
**Symptoms**: "CUDA is not available" error on startup

**Solutions**:
1. Check GPU is NVIDIA (AMD not supported)
2. Install CUDA Toolkit (11.8 or 12.x)
3. Install PyTorch with CUDA:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
4. Verify installation:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

### Package Conflicts
**Symptoms**: Import errors, version conflicts

**Solutions**:
1. Use fresh virtual environment
2. Install exact versions from requirements.txt
3. Clear pip cache: `pip cache purge`
4. Reinstall: `pip install -r requirements.txt --force-reinstall`

## Data Issues

### Corrupted Audio Files
**Symptoms**: Loading errors, NaN loss

**Solutions**:
1. Run dataset validation
2. Check audio with: `soundfile.read(path)`
3. Remove corrupted files
4. Re-encode problematic files:
   ```
   ffmpeg -i input.wav -ar 16000 -ac 1 output.wav
   ```

### Imbalanced Dataset
**Symptoms**: High accuracy but poor FPR/FNR

**Solutions**:
1. Check class distribution in health report
2. Collect more minority class samples
3. Use class weighting: `class_weights='balanced'`
4. Adjust sampling strategy
5. Try focal loss

### Data Leakage
**Symptoms**: Perfect training, poor real-world performance

**Check For**:
- Same speaker in train/val/test
- Duplicate files across splits
- Overlapping recordings

**Solutions**:
- Speaker-independent splitting
- Remove duplicates
- Re-split datasets

## Training Issues

### GPU Out of Memory
**Symptoms**: "CUDA out of memory" error

**Solutions** (in order):
1. Reduce batch_size (try 16, 8, 4)
2. Reduce audio_duration
3. Smaller model architecture
4. Enable gradient accumulation
5. Clear CUDA cache: `torch.cuda.empty_cache()`
6. Close other GPU programs

### Loss is NaN
**Symptoms**: Loss becomes NaN after few iterations

**Solutions**:
1. Lower learning rate (try 1e-4 or 1e-5)
2. Check for corrupted data
3. Enable gradient clipping
4. Use smaller batch size
5. Check for extreme values in data

### Training is Stuck
**Symptoms**: Loss not decreasing, accuracy ~50%

**Solutions**:
1. Check data loading (verify labels)
2. Increase learning rate
3. Try different optimizer
4. Simplify model first
5. Check for bugs in data pipeline

### Overfitting
**Symptoms**: Train acc >> Val acc, increasing val_loss

**Solutions**:
1. More data augmentation
2. Increase weight_decay (try 1e-3)
3. Use label_smoothing (0.1-0.2)
4. Smaller model
5. Early stopping (lower patience)
6. Dropout layers
7. More training data

### Underfitting
**Symptoms**: Both train/val accuracy low

**Solutions**:
1. Larger model
2. More epochs
3. Higher learning rate
4. Less regularization
5. Better features
6. Check data quality

### Very Slow Training
**Symptoms**: <100 samples/sec, low GPU utilization

**Solutions**:
1. Increase batch_size
2. Reduce num_workers if CPU bottleneck
3. Enable mixed_precision=True
4. Check data loading speed
5. Profile code to find bottleneck
6. Use faster data augmentation

## Evaluation Issues

### High False Positive Rate
**Symptoms**: Many false alarms in testing

**Solutions**:
1. **More hard negatives**: Critical!
2. Increase hard_negative_weight
3. Lower detection threshold
4. Analyze false positives
5. Add those cases to training data
6. Check for confusable phonemes

### High False Negative Rate
**Symptoms**: Missing actual wakewords

**Solutions**:
1. More positive samples (especially edge cases)
2. More aggressive augmentation
3. Higher detection threshold
4. Check if model is too aggressive
5. Analyze missed cases
6. Add those cases to training data

### Poor Real-World Performance
**Symptoms**: Good test metrics, poor in practice

**Check For**:
- Train/test mismatch (clean train, noisy reality)
- Missing acoustic conditions
- Different microphone characteristics
- Data leakage

**Solutions**:
1. Collect real-world data
2. More diverse training data
3. Better augmentation
4. Test in actual deployment environment

### Inconsistent Results
**Symptoms**: Performance varies across runs

**Solutions**:
1. Set random seeds
2. Larger test set
3. Multiple evaluation runs
4. Check for dataset size
5. Ensure deterministic operations

## Export Issues

### ONNX Export Fails
**Symptoms**: Export error, incompatible operations

**Solutions**:
1. Use lower opset version (try 13, 12)
2. Simplify model architecture
3. Check for unsupported operations
4. Update onnx package
5. Export to TorchScript first

### ONNX Model Incorrect
**Symptoms**: Different outputs from PyTorch vs ONNX

**Solutions**:
1. Check input preprocessing
2. Verify shapes and dtypes
3. Compare outputs numerically
4. Check for dynamic operations
5. Validate with test inputs

### Quantized Model Poor Accuracy
**Symptoms**: Significant accuracy drop after INT8

**Solutions**:
1. Use calibration data
2. Try FP16 instead
3. Quantization-aware training
4. Check quantization configuration
5. Some models don't quantize well

## Microphone Issues

### No Audio Captured
**Symptoms**: Silence when recording

**Solutions**:
1. Check microphone permissions
2. Select correct microphone device
3. Test with system audio recorder
4. Check sample rate compatibility
5. Restart application

### Poor Detection with Microphone
**Symptoms**: Works with files, not with mic

**Possible Causes**:
- Different audio characteristics
- Microphone noise
- Wrong sample rate
- Buffer size issues

**Solutions**:
1. Match training conditions
2. Add more real microphone data
3. Adjust buffer size
4. Check preprocessing pipeline

## General Debugging

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check GPU Status
```python
from src.config.cuda_utils import get_cuda_validator
validator = get_cuda_validator()
print(validator.get_device_info())
print(validator.get_memory_info())
```

### Profile Code
```python
import torch.profiler as profiler
# Profile training loop
```

### Validate Dataset
- Run health check in Panel 1
- Manually inspect samples
- Check statistics

## Getting Help

If issues persist:
1. Check logs in `logs/` directory
2. Review configuration in Panel 2
3. Verify GPU status
4. Collect error messages
5. Document reproduction steps

## Common Error Messages

### "Expected tensor to be on CUDA but got CPU"
- **Cause**: Tensor on wrong device
- **Fix**: Ensure all tensors moved to GPU

### "RuntimeError: CUDA error: device-side assert triggered"
- **Cause**: Invalid index, NaN values
- **Fix**: Check labels, data ranges

### "RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED"
- **Cause**: cuDNN not properly installed
- **Fix**: Reinstall CUDA and cuDNN

### "ValueError: Expected input to be 2D, got 3D"
- **Cause**: Shape mismatch
- **Fix**: Check model input requirements
                """)

            # Glossary
            with gr.TabItem("📖 Glossary"):
                gr.Markdown("""
# Glossary

## A

**Accuracy**: Percentage of correct predictions (TP+TN)/(TP+TN+FP+FN)

**Augmentation**: Transforming training data to increase diversity

## B

**Batch Size**: Number of samples processed together

**Background Noise**: Environmental sounds without speech

## C

**Checkpoint**: Saved model state during training

**Class Imbalance**: Unequal number of samples per class

**CUDA**: NVIDIA's parallel computing platform

**Confusion Matrix**: Table showing prediction vs actual labels

## D

**Dataset Split**: Division into train/validation/test sets

**dB (Decibel)**: Logarithmic unit for sound intensity

## E

**Early Stopping**: Halting training when validation stops improving

**Epoch**: One complete pass through training dataset

## F

**False Positive (FP)**: Negative sample classified as positive

**False Negative (FN)**: Positive sample classified as negative

**F1-Score**: Harmonic mean of precision and recall

**FPR (False Positive Rate)**: FP / (FP + TN)

**FNR (False Negative Rate)**: FN / (TP + FN)

## G

**Gradient Clipping**: Limiting gradient magnitude to prevent exploding gradients

**GPU**: Graphics Processing Unit (required for this platform)

## H

**Hard Negative**: Negative sample similar to positive

**Hyperparameter**: Configuration parameter set before training

## L

**Label Smoothing**: Softening hard targets (0.9 instead of 1.0)

**Learning Rate**: Step size for weight updates

**Loss Function**: Measures prediction error

## M

**MFCC**: Mel-Frequency Cepstral Coefficients (audio features)

**Mixed Precision**: Using FP16 and FP32 together

## O

**ONNX**: Open Neural Network Exchange (model format)

**Optimizer**: Algorithm for updating model weights (Adam, SGD)

**Overfitting**: Model memorizes training data, poor generalization

## P

**Precision**: TP / (TP + FP) - When model says positive, how often correct?

## Q

**Quantization**: Reducing model precision (FP32→FP16→INT8)

## R

**Recall**: TP / (TP + FN) - Of all positives, how many caught?

**Regularization**: Techniques to prevent overfitting

**RIR**: Room Impulse Response (acoustic room characteristics)

**ROC Curve**: Receiver Operating Characteristic curve

## S

**Sample Rate**: Audio samples per second (Hz)

**Scheduler**: Adjusts learning rate during training

**SNR**: Signal-to-Noise Ratio

**Spectrogram**: Visual representation of audio frequencies over time

## T

**Threshold**: Confidence cutoff for positive classification

**True Positive (TP)**: Positive sample correctly classified

**True Negative (TN)**: Negative sample correctly classified

## U

**Underfitting**: Model too simple, poor performance on train and val

## V

**Validation Set**: Data for tuning hyperparameters (not used in training)

## W

**Wakeword**: Trigger phrase for voice activation

**Weight Decay**: L2 regularization strength

## Acronyms

- **AI**: Artificial Intelligence
- **CNN**: Convolutional Neural Network
- **cuDNN**: CUDA Deep Neural Network library
- **DNN**: Deep Neural Network
- **FP16**: 16-bit Floating Point
- **FP32**: 32-bit Floating Point
- **INT8**: 8-bit Integer
- **ML**: Machine Learning
- **NN**: Neural Network
- **OOM**: Out Of Memory
- **TCN**: Temporal Convolutional Network
- **RNN**: Recurrent Neural Network
- **LSTM**: Long Short-Term Memory
- **GRU**: Gated Recurrent Unit
                """)

        gr.Markdown("---")
        gr.Markdown("*This documentation is comprehensive. Use the tabs above to navigate topics.*")

    return panel


if __name__ == "__main__":
    # Test the panel
    demo = create_docs_panel()
    demo.launch()