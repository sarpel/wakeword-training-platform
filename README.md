# Wakeword Training Platform

**Production-Ready Wakeword Detection Training with GPU Acceleration and Gradio UI**

A complete, user-friendly platform for training custom wakeword detection models with state-of-the-art features and a beautiful web interface. No command-line experience needed!

---

## What is This Project?

Train your own "Hey Siri" or "Alexa" style wakeword detector! This platform lets you:
- 📊 Manage your audio datasets through a web interface
- 🚀 Train high-accuracy models with GPU acceleration
- 🎯 Test your model in real-time with your microphone
- 📈 Evaluate performance with production-ready metrics
- 💾 Export models for deployment on any device

---

## Features at a Glance

### 🎨 Beautiful Web Interface
- Six intuitive panels guide you through the complete workflow
- Real-time training visualization with live metrics
- Interactive evaluation with microphone testing
- No command-line required - everything in your browser!

### 🚀 Production-Ready Training
- **CMVN Normalization**: Better accuracy across different devices (+2-4%)
- **EMA**: More stable models for real-world use (+1-2% validation accuracy)
- **Balanced Sampling**: Handle imbalanced datasets automatically
- **LR Finder**: Automatically find the best learning rate
- **GPU Acceleration**: Train 3-5× faster with mixed precision

### 📊 Advanced Metrics
- **FAH (False Alarms per Hour)**: Production metric that matters
- **EER (Equal Error Rate)**: Research-standard metric
- **Operating Point Selection**: Find the best threshold for your use case
- ROC curves, confusion matrices, and more!

### 💪 Real-World Ready
- Real-time microphone testing
- Streaming detection with smart voting
- ONNX export for mobile/edge deployment
- Temperature calibration for better confidence

---

## Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **GPU**: NVIDIA GPU with 6GB+ VRAM recommended
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sarpel/wakeword-training-platform.git
   cd wakeword-training-platform
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: For PyTorch with CUDA 11.8, use:
   ```bash
   pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Launch the Application**
   ```bash
   python run.py
   ```

   The application will open in your web browser at `http://localhost:7860`

---

## How to Use

### Panel 1: Dataset Management

**Scan & Organize Your Audio Files**

1. Click **"Scan Dataset"** to find all audio files in your `data/raw` folder
2. Review detected samples:
   - **Positive**: Files in `data/raw/positive/` folder (your wakeword)
   - **Negative**: Files in `data/raw/negative/` folder (other speech/noise)
3. Click **"Split Dataset"** to create train/validation/test splits (default: 70/15/15)

**Where to Put Your Audio**:
```
data/raw
├── positive/          # Your wakeword recordings
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── negative/          # Other speech, noise, silence
    ├── noise_001.wav
    ├── speech_001.wav
    └── ...
```

**Tips**:
- Use at least 500+ positive samples and 1000+ negative samples for good results
- Each audio file should be 1-3 seconds long
- Supported formats: WAV, MP3, FLAC, OGG

---

### Panel 2: Configuration

**Choose Your Model & Settings**

1. **Select Preset**: Choose from quick presets (recommended for beginners)
   - **Quick**: Fast training, good accuracy (ResNet18, 50 epochs)
   - **Balanced**: Best balance of speed and accuracy (ResNet18, 100 epochs)
   - **Maximum**: Best accuracy, longer training (ResNet34, 150 epochs)

2. **Or Customize**:
   - **Model Architecture**: ResNet18 (fast), ResNet34 (accurate)
   - **Training Epochs**: How many times to see the data (50-150)
   - **Batch Size**: How many samples per step (16-64)
   - **Learning Rate**: How fast to learn (auto-detected if you enable LR Finder)

3. **Enable Features** (in Advanced section):
   - ✅ **CMVN**: Always recommended (+2-4% accuracy)
   - ✅ **EMA**: Always recommended (more stable)
   - ⚡ **LR Finder**: Recommended for first training
   - 🎯 **Balanced Sampler**: Enable if you have imbalanced data

**Default Settings for Beginners**:
```
✅ Use "Balanced" preset
✅ Enable CMVN
✅ Enable EMA
✅ Enable LR Finder (first time only)
✅ Keep other settings as default
```

---

### Panel 3: Training

**Train Your Model with One Click**

1. **Review Advanced Features** (optional, in collapsible section):
   - **CMVN**: Normalizes features for better accuracy ✓ (Enable by default)
   - **EMA**: Smooths model weights for stability ✓ (Enable by default)
   - **Balanced Sampler**: Controls class ratios in batches (Enable if imbalanced)
   - **LR Finder**: Finds optimal learning rate automatically (Enable for first training)

2. **Click "Start Training"** and watch the magic happen!

3. **Monitor Progress**:
   - **Training Status**: Current epoch and batch
   - **Current Metrics**: Loss, accuracy, FPR, FNR
   - **Live Plots**: Loss curves, accuracy curves, metrics over time
   - **Training Log**: Detailed progress messages
   - **Best Model Info**: Tracks best performing checkpoint

**What to Expect**:
- **Training Time**:
  - 50 epochs: ~2-4 hours (depending on GPU and dataset size)
  - 100 epochs: ~4-8 hours
- **GPU Usage**: 70-95% (good!)
- **Memory Usage**: 2-6 GB VRAM
- **Best model saved automatically** to `models/checkpoints/best_model.pt`

**Tips**:
- Don't close the browser during training (but you can minimize it)
- Check training log for confirmation that advanced features are active
- Training loss should decrease steadily
- Validation accuracy should increase (might fluctuate slightly)

---

### Panel 4: Evaluation

**Test Your Model**

Three ways to evaluate:

#### 📁 File Evaluation
1. Upload audio files (WAV, MP3, FLAC)
2. Set detection threshold (0.5 = balanced, 0.7 = more strict)
3. Click **"Evaluate Files"**
4. See results table with predictions and confidence
5. Export results to CSV

#### 🎤 Live Microphone Test
1. Click **"Start Recording"**
2. Speak your wakeword!
3. See real-time detection with confidence scores
4. See live waveform visualization
5. View detection history

**Note**: Requires microphone access. If unavailable, simulated mode activates.

#### 📊 Test Set Evaluation
1. Load your trained model
2. Enable **"Advanced Production Metrics"** ✓
3. Set **Target FAH** (False Alarms per Hour):
   - `1.0` = One false alarm per hour (balanced)
   - `0.5` = One false alarm every 2 hours (strict)
   - `2.0` = Two false alarms per hour (aggressive)
4. Click **"Run Test Evaluation"**

**Results You'll See**:
- **Basic Metrics**: Accuracy, Precision, Recall, F1 Score
- **Confusion Matrix**: Visual error breakdown
- **ROC Curve**: Performance at all thresholds
- **Advanced Metrics** (if enabled):
  - **ROC-AUC**: Overall performance score (higher is better, aim for >0.95)
  - **EER**: Equal Error Rate (lower is better, aim for <0.10)
  - **pAUC**: Performance at low false alarm rates
  - **Operating Point**: Recommended threshold for your target FAH

**Understanding Results**:
```
Good Model:
  Accuracy: >95%
  F1 Score: >94%
  EER: <0.10 (10%)
  ROC-AUC: >0.95
  FAH: ≤1.0 at TPR >90%

Excellent Model:
  Accuracy: >97%
  F1 Score: >96%
  EER: <0.05 (5%)
  ROC-AUC: >0.98
  FAH: ≤0.5 at TPR >95%
```

---

### Panel 5: Model Export

**Deploy Your Model**

1. **Load your trained model**
2. **Choose export format**:
   - **ONNX**: Universal format, works everywhere
   - **TorchScript**: Optimized PyTorch format
   - **Quantized INT8**: 4× smaller, faster on CPU

3. **Click "Export Model"**

**Where to Use Exported Models**:
- 📱 **Mobile Apps**: iOS (Core ML), Android (TFLite after conversion)
- 🌐 **Web**: ONNX.js for in-browser inference
- 🖥️ **Edge Devices**: Raspberry Pi, NVIDIA Jetson, etc.
- ☁️ **Cloud**: Deploy as API endpoint

---

## Understanding the Features

### What is CMVN?
**Simple**: Makes your model work better across different microphones and recording conditions.

**Technical**: Normalizes features so different recording devices produce similar outputs.

**Impact**: +2-4% accuracy improvement, especially important for real-world deployment.

**Default**: ✓ Enabled (Always recommended)

---

### What is EMA?
**Simple**: Creates a smoother, more stable version of your model.

**Technical**: Maintains shadow weights by averaging model weights over time.

**Impact**: +1-2% validation accuracy, more consistent predictions.

**Default**: ✓ Enabled (Always recommended)

---

### What is Balanced Sampling?
**Simple**: Ensures the model sees equal amounts of positive and negative examples.

**Technical**: Maintains fixed class ratios within each mini-batch during training.

**Impact**: 20-30% faster convergence on imbalanced datasets, 5-15% reduction in false positives.

**Default**: ☐ Disabled (Enable if you have 2× more negatives than positives)

---

### What is LR Finder?
**Simple**: Automatically finds the best learning rate for your dataset.

**Technical**: Runs exponential range test to find optimal learning rate.

**Impact**: 10-15% faster training, eliminates manual tuning.

**Default**: ☐ Disabled (Enable for first training, then disable to save 2-5 minutes)

---

### What is FAH (False Alarms per Hour)?
**Simple**: How many times per hour the model incorrectly thinks it heard your wakeword.

**Technical**: FPR converted to real-world time scale: `FAH = (FP / total_seconds) × 3600`

**Impact**: The production metric that matters most for user experience.

**Target**:
- Strict: FAH ≤ 0.5 (1 false alarm every 2 hours)
- Balanced: FAH ≤ 1.0 (1 false alarm per hour)
- Aggressive: FAH ≤ 2.0 (2 false alarms per hour)

---

## Default Configuration

**Recommended Starting Configuration** (for beginners):

```
Dataset:
  - Minimum 500+ positive samples
  - Minimum 1000+ negative samples
  - 70% train / 15% validation / 15% test split

Model:
  - Architecture: ResNet18 (fast, good accuracy)
  - Epochs: 100 (balanced)
  - Batch Size: 32
  - Learning Rate: Auto (enable LR Finder)

Advanced Features:
  ✓ CMVN Enabled (Always)
  ✓ EMA Enabled (Always)
  ☐ Balanced Sampler (Enable if imbalanced)
  ✓ LR Finder (First training only)

Evaluation:
  ✓ Advanced Metrics Enabled
  Target FAH: 1.0 (balanced)
```

---

## Tips for Best Results

### Data Collection
1. **Positive Samples** (Your Wakeword):
   - Record in different locations (quiet room, kitchen, outdoors)
   - Use different microphones (phone, laptop, USB mic)
   - Different speakers (male, female, children)
   - Different distances (near, far)
   - Aim for 500-2000 samples

2. **Negative Samples**:
   - General speech (not your wakeword)
   - Background noise (TV, music, traffic)
   - Silence/ambient sound
   - Similar sounding words
   - Aim for 1000-5000 samples

### Training Tips
1. **Start Small**: Try 50 epochs first to verify everything works
2. **Enable CMVN & EMA**: Always! These give free performance boosts
3. **Use LR Finder**: For first training to find optimal learning rate
4. **Monitor Validation**: If val loss stops decreasing, training is done
5. **Be Patient**: Good models take time (4-8 hours typical)

### Evaluation Tips
1. **Test with Real Audio**: Don't just rely on test set
2. **Try Live Microphone**: See how it performs in real-time
3. **Tune Threshold**: Use "Operating Point" recommendation from advanced metrics
4. **Consider Use Case**: Strict threshold for bedside device, aggressive for kitchen

---

## Troubleshooting

### "CUDA out of memory"
**Solution**: Reduce batch size in Panel 2 (try 24, then 16, then 8)

### "No GPU detected"
**Solution**:
1. Check CUDA installation: `nvidia-smi`
2. Reinstall PyTorch with CUDA: `pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118`

### Training is very slow
**Solution**:
1. ✓ Enable Mixed Precision (in configuration)
2. ✓ Increase num_workers (in configuration, try 8-16)
3. ✓ Use precomputed features (Panel 1, "Precompute Features" button)

### Model has high false alarm rate
**Solution**:
1. Collect more negative samples (diversity is key!)
2. Enable Balanced Sampler if dataset is imbalanced
3. Increase detection threshold (0.5 → 0.65 → 0.75)
4. Train with hard negative mining (see TECHNICAL_FEATURES.md)

### Model doesn't detect wakeword well
**Solution**:
1. Collect more positive samples (different conditions)
2. Check if training accuracy is high (>95%)
3. Enable EMA if not already enabled
4. Decrease detection threshold (0.5 → 0.4 → 0.35)

### Microphone not working
**Solution**:
1. Check browser permissions (allow microphone access)
2. If on Windows, install `sounddevice`: `pip install sounddevice`
3. Simulated mode activates automatically if microphone unavailable

---

## Project Structure

```
wakeword-training-platform/
├── README.md                 # This file (Simple guide)
├── TECHNICAL_FEATURES.md     # Detailed technical documentation
├── run.py                    # Quick launcher (double-click to start)
├── requirements.txt          # Python dependencies
├── setup.py                  # Installation setup
│
├── src/                      # Source code
│   ├── ui/                   # Gradio web interface
│   ├── data/                 # Data processing
│   ├── models/               # Model architectures
│   ├── training/             # Training loop & optimizations
│   ├── evaluation/           # Inference & evaluation
│   └── config/               # Configuration management
│
├── data/                     # Your audio files (you create this)
│   ├── positive/             # Wakeword recordings
│   ├── negative/             # Other audio
│   └── splits/               # Auto-generated train/val/test splits
│
├── models/                   # Trained models (auto-created)
│   └── checkpoints/          # Model checkpoints
│
├── examples/                 # Example scripts
│   └── complete_training_pipeline.py  # Full training example
│
└── docs/                     # Additional documentation
    └── ...
```

---

## FAQ

**Q: How much data do I need?**
A: Minimum 500 positive + 1000 negative samples. More is always better! Aim for 1000-2000 positive and 3000-5000 negative for production quality.

**Q: How long does training take?**
A: Typically 2-8 hours depending on:
  - Dataset size (more samples = longer)
  - Number of epochs (50 vs 100 vs 150)
  - GPU power (RTX 3090 vs GTX 1660)
  - Model architecture (ResNet18 vs ResNet34)

**Q: What GPU do I need?**
A: Minimum: GTX 1060 (6GB) or equivalent
   Recommended: RTX 3060 (12GB) or better
   Ideal: RTX 3090/4090 (24GB) for large datasets

**Q: Can I train without a GPU?**
A: Technically yes, but it will be 10-20× slower. Not recommended for datasets over 10k samples.

**Q: How do I improve accuracy?**
A:
  1. Collect more diverse data (different conditions, speakers)
  2. Enable CMVN and EMA (always!)
  3. Train longer (100-150 epochs)
  4. Try larger model (ResNet34 instead of ResNet18)
  5. Use data augmentation (automatically enabled)

**Q: How do I reduce false alarms?**
A:
  1. Collect more negative samples (especially similar-sounding)
  2. Use hard negative mining (see TECHNICAL_FEATURES.md)
  3. Increase detection threshold
  4. Enable Balanced Sampler
  5. Use streaming detector with voting (automatically enabled in production)

**Q: Can I train multiple wakewords?**
A: Currently binary classification (wakeword vs non-wakeword). For multiple wakewords, train separate models or see multi-class extension in TECHNICAL_FEATURES.md.

**Q: Where are my models saved?**
A: `models/checkpoints/best_model.pt` (best model during training)

**Q: How do I deploy my model?**
A:
  1. Export to ONNX (Panel 5)
  2. Use ONNX Runtime for inference
  3. See deployment examples in `examples/` folder
  4. Read deployment section in TECHNICAL_FEATURES.md

---

## Next Steps

1. **Collect Your Data**: Record positive and negative audio samples
2. **Launch the App**: Run `python run.py`
3. **Follow the Panels**: Dataset → Configuration → Training → Evaluation → Export
4. **Read Technical Docs**: See `TECHNICAL_FEATURES.md` for advanced usage
5. **Join the Community**: Star the repo, report issues, contribute!

---

## System Requirements

**Minimum**:
- Python 3.8+
- 8GB RAM
- NVIDIA GPU with 4GB VRAM
- 10GB free disk space
- Windows 10 / Ubuntu 18.04 / macOS 10.15

**Recommended**:
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- 50GB+ free disk space (for large datasets)
- Windows 11 / Ubuntu 20.04+ / macOS 12+

---

## Support & Resources

- **Technical Documentation**: See `TECHNICAL_FEATURES.md` for in-depth details
- **Example Scripts**: Check `examples/` folder for code examples
- **Configuration Guide**: See `docs/` folder for advanced configurations
- **Issue Tracker**: Report bugs and request features on GitHub

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

Built with:
- **PyTorch**: Deep learning framework
- **Gradio**: Web UI framework
- **Librosa**: Audio processing
- **ONNX**: Model export standard

Special thanks to the open-source community!

---

**Happy Training! 🚀**

*Build amazing wakeword detectors with ease!*
