# RIR & NPY Feature Enhancement Implementation Plan

**Project**: Wakeword Training Platform
**Version**: 1.0
**Date**: 2025-10-12
**Status**: Design Phase - Ready for Implementation

---

## Executive Summary

This implementation plan addresses two critical enhancements identified in the codebase audit:

1. **RIR (Room Impulse Response) Enhancement**: Add dry/wet mixing and quality control to achieve industry-standard reverberation augmentation
2. **NPY Feature Integration**: Enable precomputed feature consumption in the training pipeline for significant performance gains

**Expected Impact**:
- **RIR Enhancement**: 15-20% improvement in model robustness across different acoustic environments
- **NPY Integration**: 40-60% reduction in training time for large datasets (>50k samples)

---

## Phase 1: RIR Enhancement

### 1.1 Current State Analysis

**Existing Implementation** (`src/data/augmentation.py:263-310`):
- ✅ Basic convolution with energy normalization
- ✅ DC offset removal and gain clamping
- ✅ Random RIR selection
- ❌ No dry/wet mixing (100% wet signal)
- ❌ No RIR quality validation
- ❌ Limited file format support (WAV only)
- ❌ Low RIR count limit (50 files)

### 1.2 Enhancement Design

#### 1.2.1 Dry/Wet Mixing

**Objective**: Allow control over reverberation intensity by mixing original (dry) and reverberant (wet) signals.

**Mathematical Foundation**:
```
output = (dry_ratio * original_signal) + (wet_ratio * reverberant_signal)
where: dry_ratio + wet_ratio = 1.0
```

**Industry Standards**:
- Light reverb: 70% dry, 30% wet (dry_ratio=0.7)
- Medium reverb: 50% dry, 50% wet (dry_ratio=0.5)
- Heavy reverb: 30% dry, 70% wet (dry_ratio=0.3)

**Configuration Parameters**:
```python
@dataclass
class AugmentationConfig:
    # Existing parameters...
    rir_prob: float = 0.25

    # NEW: Dry/wet mixing parameters
    rir_dry_wet_min: float = 0.3  # Minimum dry ratio (30% dry, 70% wet)
    rir_dry_wet_max: float = 0.7  # Maximum dry ratio (70% dry, 30% wet)
    rir_dry_wet_strategy: str = "random"  # random, fixed, adaptive
```

**Implementation Location**: `src/data/augmentation.py:263-310`

**Modified Method Signature**:
```python
def apply_rir(
    self,
    waveform: torch.Tensor,
    dry_wet_ratio: Optional[float] = None
) -> torch.Tensor:
    """
    Apply Room Impulse Response with dry/wet mixing

    Args:
        waveform: Input waveform (channels, samples)
        dry_wet_ratio: Dry signal ratio (0.0=full wet, 1.0=full dry)
                      If None, random value from config range

    Returns:
        Mixed waveform (dry + wet)
    """
```

**Processing Steps**:
1. Store original waveform (dry signal)
2. Apply RIR convolution to create wet signal
3. Apply energy normalization to wet signal
4. Mix: `output = dry_ratio * dry + (1 - dry_ratio) * wet`
5. Final normalization if needed

#### 1.2.2 RIR Quality Control

**Objective**: Filter out invalid or poor-quality RIR files during loading.

**Quality Criteria**:

| Criterion | Validation | Action if Failed |
|-----------|-----------|------------------|
| Duration | 0.1s ≤ duration ≤ 5.0s | Skip file, log warning |
| Energy | Total energy > 1e-6 | Skip file (silent RIR) |
| NaN/Inf | No NaN or Inf values | Skip file, log error |
| Peak location | First 10% contains peak | Pass (but log warning if late) |
| Decay | Exponential decay present | Pass (informational only) |

**Implementation Location**: `src/data/augmentation.py:106-132` (_load_rirs method)

**New Validation Function**:
```python
def _validate_rir(
    self,
    waveform: torch.Tensor,
    file_path: Path
) -> Tuple[bool, Optional[str]]:
    """
    Validate RIR quality

    Args:
        waveform: RIR waveform tensor
        file_path: Path to RIR file (for logging)

    Returns:
        (is_valid, warning_message)
    """
    warnings = []

    # Duration check
    duration = waveform.shape[-1] / self.sample_rate
    if duration < 0.1 or duration > 5.0:
        return False, f"Invalid duration: {duration:.2f}s (expected 0.1-5.0s)"

    # Energy check
    energy = torch.sum(waveform ** 2).item()
    if energy < 1e-6:
        return False, "RIR has near-zero energy (silent)"

    # NaN/Inf check
    if not torch.isfinite(waveform).all():
        return False, "RIR contains NaN or Inf values"

    # Peak location check (first 10%)
    peak_idx = torch.argmax(torch.abs(waveform)).item()
    peak_position = peak_idx / waveform.shape[-1]
    if peak_position > 0.1:
        warnings.append(f"Peak at {peak_position*100:.1f}% (expected <10%)")

    warning_msg = "; ".join(warnings) if warnings else None
    return True, warning_msg
```

#### 1.2.3 Extended File Format Support

**Objective**: Support multiple RIR file formats beyond WAV.

**Supported Formats**:
- `.wav` (existing)
- `.flac` (lossless, preferred for RIRs)
- `.mp3` (acceptable but lossy)

**Implementation**: Modify line 117 in `augmentation.py`:
```python
# OLD:
rir_files = list(Path(rir_dir).rglob("*.wav"))

# NEW:
rir_files = (
    list(Path(rir_dir).rglob("*.wav")) +
    list(Path(rir_dir).rglob("*.flac")) +
    list(Path(rir_dir).rglob("*.mp3"))
)
```

#### 1.2.4 Increased RIR Capacity

**Objective**: Increase RIR file limit for better diversity.

**Change**: Line 110 in `augmentation.py`:
```python
# OLD: Limit to 50 RIRs
for rir_file in rir_files[:50]:

# NEW: Limit to 200 RIRs (with memory warning)
max_rirs = min(len(rir_files), 200)
logger.info(f"Loading up to {max_rirs} RIRs (found {len(rir_files)})")
for rir_file in rir_files[:max_rirs]:
```

**Memory Consideration**: Log warning if estimated memory usage > 500MB.

### 1.3 Implementation Tasks (RIR)

| Task ID | Description | File | Priority | Est. Time |
|---------|-------------|------|----------|-----------|
| RIR-1 | Add dry/wet config parameters | `src/config/defaults.py` | High | 15 min |
| RIR-2 | Implement `_validate_rir()` method | `src/data/augmentation.py` | High | 30 min |
| RIR-3 | Modify `_load_rirs()` with validation | `src/data/augmentation.py` | High | 30 min |
| RIR-4 | Add dry/wet mixing to `apply_rir()` | `src/data/augmentation.py` | High | 45 min |
| RIR-5 | Extend file format support | `src/data/augmentation.py` | Medium | 10 min |
| RIR-6 | Increase RIR capacity to 200 | `src/data/augmentation.py` | Low | 5 min |
| RIR-7 | Update presets with new params | `src/config/presets.py` | Medium | 20 min |
| RIR-8 | Add unit tests for validation | `tests/test_augmentation.py` | High | 45 min |
| RIR-9 | Add unit tests for dry/wet mixing | `tests/test_augmentation.py` | High | 30 min |

**Total Estimated Time**: ~3.5 hours

---

## Phase 2: NPY Feature Integration

### 2.1 Current State Analysis

**Existing Components**:
- ✅ `NpyExtractor` (src/data/npy_extractor.py) - fully functional
- ✅ Panel 1 UI integration for NPY extraction
- ✅ Memory-mapped loading support
- ✅ Feature type inference
- ❌ **NO consumption in training pipeline**
- ❌ **NO manifest integration**
- ❌ **NO dataset support for NPY files**

**Critical Gap**: Extracted NPY features are never used during training. All training loads raw audio and recomputes features every time.

### 2.2 Enhancement Design

#### 2.2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    NPY Feature Pipeline                      │
└─────────────────────────────────────────────────────────────┘

Option A: Use Precomputed Features (NPY)
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Raw      │ --> │ Feature  │ --> │ Save     │ --> │ Training │
│ Audio    │     │ Extract  │     │ .npy     │     │ (Load    │
│ Files    │     │ (Batch)  │     │ Files    │     │  NPY)    │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
   (once)           (once)           (once)          (fast!)

Option B: On-the-Fly (Current, Slow)
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Raw      │ --> │ Feature  │ --> │ Training │
│ Audio    │     │ Extract  │     │ (Every   │
│ Files    │     │ (Every   │     │  Epoch)  │
└──────────┘     │  Epoch!) │     └──────────┘
                 └──────────┘
```

**Key Benefits**:
- **Speed**: 40-60% faster training (no repeated feature extraction)
- **Consistency**: Same features across epochs (reproducibility)
- **Flexibility**: Easy to experiment with different augmentations

#### 2.2.2 Configuration Extension

**Add to `DataConfig`** (`src/config/defaults.py:12-28`):

```python
@dataclass
class DataConfig:
    """Data processing configuration"""
    # Existing parameters...
    sample_rate: int = 16000
    audio_duration: float = 2.5
    # ... other existing params ...

    # NEW: NPY feature parameters
    use_precomputed_features: bool = False  # Enable NPY loading
    npy_feature_dir: Optional[str] = None   # Directory with .npy files
    npy_feature_type: str = "mel"           # mel, mfcc (must match extraction)
    npy_cache_features: bool = True         # Cache loaded features in RAM
    fallback_to_audio: bool = True          # If NPY missing, load raw audio
```

#### 2.2.3 NPY Manifest Integration

**Objective**: Extend dataset manifest to include NPY file mappings.

**Manifest Schema Extension**:

```json
{
  "files": [
    {
      "path": "data/raw/positive/sample_001.wav",
      "category": "positive",
      "duration": 2.5,
      "sample_rate": 16000,
      "npy_path": "data/npy/positive/sample_001.npy"  // NEW FIELD
    }
  ]
}
```

**Implementation Location**: `src/data/splitter.py:407-433`

**Modified Method**:
```python
def split_datasets(
    self,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
    npy_dir: Optional[Path] = None  # NEW PARAMETER
) -> Dict:
```

**NPY Path Mapping Logic**:
```python
def _find_npy_path(self, audio_path: Path, npy_dir: Path) -> Optional[str]:
    """
    Find corresponding .npy file for audio file

    Args:
        audio_path: Path to audio file
        npy_dir: Root directory containing .npy files

    Returns:
        Path to .npy file or None if not found
    """
    # Try matching by relative path structure
    # Example: data/raw/positive/sample_001.wav
    #       -> data/npy/positive/sample_001.npy

    relative_path = audio_path.relative_to(audio_path.parents[2])  # Get from category level
    npy_path = npy_dir / relative_path.with_suffix('.npy')

    if npy_path.exists():
        return str(npy_path)

    # Try matching by filename only (less precise)
    filename = audio_path.stem  # Without extension
    npy_candidates = list(npy_dir.rglob(f"{filename}.npy"))

    if len(npy_candidates) == 1:
        return str(npy_candidates[0])
    elif len(npy_candidates) > 1:
        logger.warning(f"Multiple .npy candidates for {audio_path.name}, using first")
        return str(npy_candidates[0])

    return None
```

#### 2.2.4 Dataset NPY Loading

**Objective**: Modify `WakewordDataset` to load NPY features when available.

**Implementation Location**: `src/data/dataset.py:20-272`

**Constructor Changes**:
```python
def __init__(
    self,
    manifest_path: Path,
    sample_rate: int = 16000,
    audio_duration: float = 2.5,
    augment: bool = False,
    cache_audio: bool = False,
    augmentation_config: Optional[Dict] = None,
    background_noise_dir: Optional[Path] = None,
    rir_dir: Optional[Path] = None,
    device: str = 'cpu',
    feature_type: str = 'mel',
    n_mels: int = 128,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 160,
    # NEW PARAMETERS:
    use_precomputed_features: bool = False,
    npy_cache_features: bool = True,
    fallback_to_audio: bool = True
):
```

**New Instance Variables**:
```python
self.use_precomputed_features = use_precomputed_features
self.npy_cache_features = npy_cache_features
self.fallback_to_audio = fallback_to_audio

# Feature cache (separate from audio cache)
self.feature_cache = {} if npy_cache_features else None
```

**Modified `__getitem__` Method** (Line 164-210):

```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
    """
    Get item by index

    Returns:
        Tuple of (features_tensor, label, metadata)
    """
    file_info = self.files[idx]
    file_path = Path(file_info['path'])
    category = file_info['category']
    label = self.label_map[category]

    # NEW: Try loading from NPY first if enabled
    if self.use_precomputed_features:
        features = self._load_from_npy(file_info, idx)

        if features is not None:
            # Successfully loaded from NPY
            metadata = {
                'path': str(file_path),
                'category': category,
                'label': label,
                'source': 'npy',  # NEW: Track data source
                'sample_rate': self.sample_rate,
                'duration': self.audio_duration
            }
            return features, label, metadata

        elif not self.fallback_to_audio:
            raise FileNotFoundError(
                f"NPY file not found for {file_path} and fallback disabled"
            )

        # If NPY not found, fall through to audio loading

    # EXISTING: Load from raw audio (unchanged logic)
    if self.audio_cache is not None and idx in self.audio_cache:
        audio = self.audio_cache[idx]
    else:
        audio = self.audio_processor.process_audio(file_path)
        if self.audio_cache is not None:
            self.audio_cache[idx] = audio

    # Apply augmentation if enabled
    if self.augment:
        audio = self._apply_augmentation(audio)

    # Convert to tensor and extract features
    audio_tensor = torch.from_numpy(audio).float()
    features = self.feature_extractor(audio_tensor)

    metadata = {
        'path': str(file_path),
        'category': category,
        'label': label,
        'source': 'audio',  # NEW: Track data source
        'sample_rate': self.sample_rate,
        'duration': self.audio_duration
    }

    return features, label, metadata
```

**New Helper Method**:
```python
def _load_from_npy(
    self,
    file_info: Dict,
    idx: int
) -> Optional[torch.Tensor]:
    """
    Load precomputed features from .npy file

    Args:
        file_info: File metadata from manifest
        idx: Sample index (for caching)

    Returns:
        Feature tensor or None if not found
    """
    # Check cache first
    if self.feature_cache is not None and idx in self.feature_cache:
        return self.feature_cache[idx]

    # Get NPY path from manifest
    npy_path = file_info.get('npy_path')

    if not npy_path or not Path(npy_path).exists():
        return None

    try:
        # Load NPY file (memory-mapped for efficiency)
        features = np.load(npy_path, mmap_mode='r')

        # Convert to tensor
        features_tensor = torch.from_numpy(np.array(features)).float()

        # Validate shape
        expected_shape = self.feature_extractor.get_output_shape(
            int(self.audio_duration * self.sample_rate)
        )

        if features_tensor.shape != expected_shape:
            logger.warning(
                f"Shape mismatch for {npy_path}: "
                f"expected {expected_shape}, got {features_tensor.shape}"
            )
            return None

        # Cache if enabled
        if self.feature_cache is not None:
            self.feature_cache[idx] = features_tensor

        return features_tensor

    except Exception as e:
        logger.error(f"Error loading NPY {npy_path}: {e}")
        return None
```

#### 2.2.5 Batch Feature Extraction Tool

**Objective**: Provide utility to precompute features for entire dataset.

**New File**: `src/data/batch_feature_extractor.py`

**Class Design**:
```python
class BatchFeatureExtractor:
    """
    Batch extract and save features for entire dataset
    """

    def __init__(
        self,
        config: DataConfig,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = device
        self.feature_extractor = FeatureExtractor(
            sample_rate=config.sample_rate,
            feature_type=config.feature_type,
            n_mels=config.n_mels,
            n_mfcc=config.n_mfcc,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            device=device
        )
        self.audio_processor = AudioProcessor(
            target_sr=config.sample_rate,
            target_duration=config.audio_duration
        )

    def extract_dataset(
        self,
        audio_files: List[Path],
        output_dir: Path,
        batch_size: int = 32,
        preserve_structure: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Extract features for all audio files

        Args:
            audio_files: List of audio file paths
            output_dir: Directory to save .npy files
            batch_size: Batch size for GPU processing
            preserve_structure: Preserve directory structure in output
            progress_callback: Progress callback(current, total, message)

        Returns:
            Dictionary with extraction results
        """
```

**Processing Strategy**:
1. Group audio files into batches
2. Load audio batch into memory
3. Process batch on GPU (parallel feature extraction)
4. Save features as individual .npy files
5. Track success/failure for each file

#### 2.2.6 UI Integration (Panel 1)

**Objective**: Add "Batch Extract Features" button to Panel 1.

**Implementation Location**: `src/ui/panel_dataset.py`

**New UI Component**:
```python
with gr.Row():
    gr.Markdown("### Batch Feature Extraction")

with gr.Row():
    with gr.Column():
        extract_config = gr.Dropdown(
            label="Feature Type",
            choices=["mel", "mfcc"],
            value="mel"
        )
        extract_batch_size = gr.Slider(
            minimum=16, maximum=128, value=32, step=16,
            label="Batch Size (GPU)"
        )
        extract_button = gr.Button(
            "⚡ Extract Features to NPY",
            variant="primary"
        )

    with gr.Column():
        extract_log = gr.Textbox(
            label="Extraction Log",
            lines=8,
            value="Ready to extract features...",
            interactive=False
        )
```

**Handler Function**:
```python
def batch_extract_features_handler(
    root_path: str,
    feature_type: str,
    batch_size: int,
    progress=gr.Progress()
) -> str:
    """Extract features for all audio files"""
    try:
        # Validate dataset scanned
        if _current_dataset_info is None:
            return "❌ Please scan datasets first"

        # Initialize extractor
        config = DataConfig(
            feature_type=feature_type,
            # ... other params ...
        )

        extractor = BatchFeatureExtractor(
            config=config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Collect all audio files
        all_files = []
        for category_data in _current_dataset_info['categories'].values():
            all_files.extend([Path(f['path']) for f in category_data['files']])

        # Extract
        output_dir = Path(root_path) / "npy"
        results = extractor.extract_dataset(
            audio_files=all_files,
            output_dir=output_dir,
            batch_size=batch_size,
            progress_callback=lambda c, t, m: progress(c/t, desc=m)
        )

        return f"✅ Extracted {results['success_count']} features to {output_dir}"

    except Exception as e:
        return f"❌ Error: {str(e)}"
```

### 2.3 Implementation Tasks (NPY)

| Task ID | Description | File | Priority | Est. Time |
|---------|-------------|------|----------|-----------|
| NPY-1 | Add NPY config parameters | `src/config/defaults.py` | High | 20 min |
| NPY-2 | Add `_find_npy_path()` to splitter | `src/data/splitter.py` | High | 45 min |
| NPY-3 | Modify `split_datasets()` with NPY mapping | `src/data/splitter.py` | High | 30 min |
| NPY-4 | Add NPY parameters to Dataset `__init__` | `src/data/dataset.py` | High | 15 min |
| NPY-5 | Implement `_load_from_npy()` method | `src/data/dataset.py` | High | 45 min |
| NPY-6 | Modify `__getitem__()` with NPY loading | `src/data/dataset.py` | High | 30 min |
| NPY-7 | Create `BatchFeatureExtractor` class | `src/data/batch_feature_extractor.py` | High | 2 hours |
| NPY-8 | Add UI component to Panel 1 | `src/ui/panel_dataset.py` | Medium | 45 min |
| NPY-9 | Implement UI handler | `src/ui/panel_dataset.py` | Medium | 30 min |
| NPY-10 | Update `load_dataset_splits()` function | `src/data/dataset.py` | High | 20 min |
| NPY-11 | Add unit tests for NPY loading | `tests/test_dataset.py` | High | 1 hour |
| NPY-12 | Add integration test | `tests/test_npy_integration.py` | High | 1 hour |
| NPY-13 | Update presets with NPY params | `src/config/presets.py` | Low | 15 min |
| NPY-14 | Update CLAUDE.md documentation | `CLAUDE.md` | Medium | 30 min |

**Total Estimated Time**: ~9.5 hours

---

## Phase 3: Testing & Validation

### 3.1 RIR Enhancement Testing

**Test Suite**: `tests/test_rir_enhancement.py`

**Test Cases**:

1. **Test Dry/Wet Mixing**:
   - Verify output is weighted sum of dry and wet
   - Test edge cases (100% dry, 100% wet)
   - Validate energy conservation

2. **Test RIR Validation**:
   - Test duration checks (too short, too long, valid)
   - Test energy checks (silent, normal)
   - Test NaN/Inf detection
   - Test peak location detection

3. **Test Multiple Formats**:
   - Load WAV, FLAC, MP3 RIRs
   - Verify same processing for all formats

4. **Test Capacity Increase**:
   - Load 200 RIRs
   - Verify memory usage is reasonable
   - Test random selection from larger pool

**Validation Metrics**:
- All tests pass with 100% success rate
- No memory leaks detected
- Processing time < 50ms per RIR application

### 3.2 NPY Integration Testing

**Test Suite**: `tests/test_npy_integration.py`

**Test Cases**:

1. **Test NPY Loading**:
   - Load precomputed features from NPY
   - Verify shape matches expected
   - Test caching mechanism

2. **Test Fallback**:
   - Test fallback to audio when NPY missing
   - Test error when fallback disabled

3. **Test Manifest Integration**:
   - Create manifest with NPY paths
   - Load dataset with NPY paths
   - Verify correct file loading

4. **Test Batch Extraction**:
   - Extract features for small dataset
   - Verify all files processed
   - Validate output NPY files

5. **Test Training Pipeline**:
   - Train for 1 epoch with NPY features
   - Train for 1 epoch with audio
   - Compare accuracy (should be identical)
   - Measure time savings

**Validation Metrics**:
- NPY loading speed: >50% faster than audio loading
- Training speed: 40-60% faster with NPY
- Model accuracy: Same ±0.5% between NPY and audio
- No memory leaks with caching enabled

### 3.3 Integration Testing

**Test Scenario**: Complete workflow with both enhancements

**Steps**:
1. Scan dataset with RIRs
2. Extract NPY features with RIR augmentation
3. Split dataset with NPY paths
4. Train model using NPY features
5. Validate model performance

**Success Criteria**:
- All pipeline steps complete without errors
- Training time reduced by 40-60%
- Model FPR < 5%, FNR < 5%, Accuracy > 95%
- RIR augmentation improves robustness by 15-20%

---

## Phase 4: Documentation & Deployment

### 4.1 Code Documentation

**Files to Update**:

1. **CLAUDE.md**: Add NPY and RIR sections
2. **README.md**: Update feature list and quick start
3. **Implementation_plan.md**: Mark features as complete
4. **Sprint documentation**: Create Sprint 8 completion doc

**Documentation Content**:

#### RIR Enhancement Section:
```markdown
### RIR (Room Impulse Response) Augmentation

**Dry/Wet Mixing**: Controls reverberation intensity.
- Light reverb: `rir_dry_wet_ratio=0.7`
- Medium reverb: `rir_dry_wet_ratio=0.5`
- Heavy reverb: `rir_dry_wet_ratio=0.3`

**Quality Control**: Automatic validation of RIR files.
- Duration: 0.1-5.0 seconds
- Energy threshold: > 1e-6
- Format support: WAV, FLAC, MP3

**Configuration**:
```python
config.augmentation.rir_prob = 0.25
config.augmentation.rir_dry_wet_min = 0.3
config.augmentation.rir_dry_wet_max = 0.7
```
```

#### NPY Feature Section:
```markdown
### NPY Precomputed Features

**Performance**: 40-60% faster training for large datasets.

**Workflow**:
1. Scan dataset in Panel 1
2. Click "⚡ Extract Features to NPY"
3. Enable in config: `use_precomputed_features=True`
4. Train normally (features loaded from NPY)

**Configuration**:
```python
config.data.use_precomputed_features = True
config.data.npy_feature_dir = "data/npy"
config.data.fallback_to_audio = True
```

**Batch Extraction**:
```bash
# Via UI: Panel 1 > Batch Feature Extraction
# Via CLI:
python -m src.data.batch_feature_extractor \
    --audio-dir data/raw \
    --output-dir data/npy \
    --feature-type mel \
    --batch-size 32
```
```

### 4.2 User Guide Updates

**New Sections**:

1. **Advanced Augmentation Guide**:
   - When to use dry/wet mixing
   - RIR dataset preparation tips
   - Troubleshooting RIR issues

2. **Performance Optimization Guide**:
   - When to use NPY features
   - Memory vs speed tradeoffs
   - Batch size recommendations

3. **Common Workflows**:
   - Scenario: Large dataset (>50k samples) → Use NPY
   - Scenario: Small dataset (<10k samples) → Skip NPY
   - Scenario: Multiple experiments → Extract NPY once

### 4.3 Configuration Presets Update

**Modify**: `src/config/presets.py`

**Add New Preset**: "Fast Training (NPY)"

```python
def fast_training_npy_preset() -> WakewordConfig:
    """
    Fast training using precomputed NPY features
    Best for: Large datasets, multiple experiments
    """
    config = get_default_config()

    # Enable NPY features
    config.data.use_precomputed_features = True
    config.data.npy_cache_features = True
    config.data.fallback_to_audio = False

    # Minimal augmentation (already baked into NPY)
    config.augmentation.background_noise_prob = 0.3
    config.augmentation.rir_prob = 0.15

    # Higher batch size (features pre-loaded)
    config.training.batch_size = 256

    config.config_name = "fast_training_npy"
    config.description = "Fast training with precomputed features"

    return config
```

**Update Existing Presets**: Add RIR dry/wet parameters to all.

---

## Phase 5: Rollout Strategy

### 5.1 Implementation Order

**Priority 1: Critical Path (Week 1)**
1. RIR-1, RIR-2, RIR-3, RIR-4 (Core RIR enhancement)
2. NPY-1, NPY-2, NPY-3 (Manifest integration)
3. NPY-4, NPY-5, NPY-6 (Dataset NPY loading)

**Priority 2: Feature Complete (Week 2)**
4. NPY-7, NPY-8, NPY-9 (Batch extraction tool)
5. RIR-5, RIR-6 (Extended RIR support)
6. NPY-10 (Update dataset loader)

**Priority 3: Testing & Polish (Week 3)**
7. RIR-8, RIR-9 (RIR tests)
8. NPY-11, NPY-12 (NPY tests)
9. RIR-7, NPY-13 (Preset updates)

**Priority 4: Documentation (Week 3-4)**
10. NPY-14 (Documentation)
11. User guide updates
12. Sprint 8 completion doc

### 5.2 Risk Mitigation

**Risk 1**: NPY shape mismatch issues
- **Mitigation**: Strict validation in `_load_from_npy()`
- **Fallback**: Auto-fallback to audio loading

**Risk 2**: Memory usage with NPY caching
- **Mitigation**: Make caching optional
- **Monitor**: Add memory usage logging

**Risk 3**: RIR quality validation too strict
- **Mitigation**: Use warnings instead of errors for minor issues
- **Flexibility**: Allow override with `strict_rir_validation=False`

**Risk 4**: Breaking changes in existing workflows
- **Mitigation**: All new features are opt-in (disabled by default)
- **Backward compatibility**: Existing configs work without changes

### 5.3 Performance Benchmarks

**Before Implementation**:
- Run baseline training on sample dataset
- Measure: Time per epoch, memory usage, model accuracy

**After Implementation**:
- Run same training with NPY features
- Run same training with RIR enhancements
- Compare metrics

**Expected Results**:
| Metric | Baseline | With NPY | With RIR | Both |
|--------|----------|----------|----------|------|
| Time/epoch | 100% | 40-50% | 95% | 40-50% |
| Memory | 100% | 110% | 100% | 110% |
| Accuracy | 95% | 95% | 95% | 95% |
| FPR @ 95% TPR | 5% | 5% | 3-4% | 3-4% |

---

## Phase 6: Maintenance & Future Work

### 6.1 Monitoring

**Metrics to Track**:
- NPY feature cache hit rate
- RIR validation failure rate
- Training speed improvement
- Model performance consistency

**Logging Additions**:
```python
logger.info(f"NPY cache hit rate: {hit_rate:.1%}")
logger.info(f"RIR validation: {valid}/{total} passed")
logger.info(f"Average features load time: {avg_time:.2f}ms")
```

### 6.2 Future Enhancements

**RIR Improvements**:
1. Adaptive dry/wet based on SNR
2. RIR database with metadata (room size, RT60)
3. Automatic RIR quality scoring

**NPY Improvements**:
1. Compressed NPY format (zarr, HDF5)
2. Augmentation-aware NPY (multiple versions)
3. Distributed NPY loading (multi-GPU)

**Integration Opportunities**:
1. Panel 2: Show NPY status in config UI
2. Panel 3: Display data source (NPY vs audio) in metrics
3. Panel 6: Add performance comparison charts

---

## Appendix A: File Change Summary

### Modified Files

| File | Changes | Lines Changed |
|------|---------|---------------|
| `src/config/defaults.py` | Add NPY and RIR params | +15 |
| `src/data/augmentation.py` | RIR validation and dry/wet | +120 |
| `src/data/splitter.py` | NPY path mapping | +60 |
| `src/data/dataset.py` | NPY loading logic | +100 |
| `src/ui/panel_dataset.py` | Batch extraction UI | +80 |
| `src/config/presets.py` | Update all presets | +40 |
| `CLAUDE.md` | Documentation updates | +150 |

### New Files

| File | Purpose | Lines |
|------|---------|-------|
| `src/data/batch_feature_extractor.py` | Batch NPY extraction | ~300 |
| `tests/test_rir_enhancement.py` | RIR test suite | ~200 |
| `tests/test_npy_integration.py` | NPY test suite | ~300 |
| `docs/RIR_NPY_Enhancement_Plan.md` | This document | ~1500 |

**Total Code Changes**: ~1,065 lines (excluding this plan)

---

## Appendix B: Configuration Examples

### Example 1: Training with NPY Features

```python
from src.config.defaults import WakewordConfig

config = WakewordConfig()

# Enable NPY features
config.data.use_precomputed_features = True
config.data.npy_feature_dir = "data/npy"
config.data.fallback_to_audio = True

# Save config
config.save("configs/training_with_npy.yaml")
```

### Example 2: RIR Enhancement Configuration

```python
config = WakewordConfig()

# RIR settings
config.augmentation.rir_prob = 0.3
config.augmentation.rir_dry_wet_min = 0.4
config.augmentation.rir_dry_wet_max = 0.7

# Increase RIR diversity
# (modify augmentation.py line 110 to load more RIRs)

config.save("configs/enhanced_rir.yaml")
```

### Example 3: Combined Optimized Config

```python
config = WakewordConfig()

# NPY for speed
config.data.use_precomputed_features = True
config.data.npy_cache_features = True

# RIR for robustness
config.augmentation.rir_prob = 0.25
config.augmentation.rir_dry_wet_min = 0.3
config.augmentation.rir_dry_wet_max = 0.7

# Faster training with larger batches
config.training.batch_size = 256
config.training.num_workers = 8

config.save("configs/optimized.yaml")
```

---

## Appendix C: CLI Commands

### Extract Features

```bash
# Create batch extractor CLI
python -m src.data.batch_feature_extractor \
    --audio-dir data/raw \
    --output-dir data/npy \
    --feature-type mel \
    --n-mels 128 \
    --batch-size 32 \
    --device cuda \
    --preserve-structure
```

### Validate RIRs

```bash
# Create RIR validation CLI
python -m src.data.augmentation validate_rirs \
    --rir-dir data/raw/rirs \
    --output-report rirs_validation.txt
```

### Train with NPY

```bash
# Standard training command (auto-detects NPY if configured)
python src/ui/app.py
# Then use Panel 2 to load config with NPY enabled
# Then use Panel 3 to train
```

---

## Appendix D: Troubleshooting Guide

### Issue 1: NPY Shape Mismatch

**Symptom**: `Shape mismatch for .npy file`

**Causes**:
- Feature type mismatch (mel vs mfcc)
- Different feature extraction parameters
- Corrupted NPY file

**Solutions**:
1. Re-extract features with current config
2. Enable `fallback_to_audio=True`
3. Delete and re-generate NPY files

### Issue 2: RIR Validation Failures

**Symptom**: `RIR validation failed: Invalid duration`

**Causes**:
- RIR files too short or too long
- Wrong file format
- Corrupted audio

**Solutions**:
1. Check RIR files manually
2. Use standard RIR databases (e.g., MIT RIR dataset)
3. Adjust validation thresholds if needed

### Issue 3: Slow NPY Loading

**Symptom**: Training still slow with NPY

**Causes**:
- NPY files on slow storage (network drive)
- Caching disabled
- Memory-mapped mode issues

**Solutions**:
1. Move NPY files to fast local SSD
2. Enable `npy_cache_features=True`
3. Increase `num_workers`
4. Use regular loading instead of mmap

### Issue 4: High Memory Usage

**Symptom**: OOM errors with NPY caching

**Causes**:
- Too many features cached in RAM
- Large batch size
- Feature cache enabled for large dataset

**Solutions**:
1. Disable caching: `npy_cache_features=False`
2. Reduce batch size
3. Use memory-mapped loading only

---

## Conclusion

This implementation plan provides a comprehensive roadmap for integrating industry-standard RIR enhancements and efficient NPY feature consumption into the Wakeword Training Platform.

**Key Deliverables**:
- ✅ Detailed technical specifications
- ✅ Task breakdown with time estimates
- ✅ Testing strategy and validation metrics
- ✅ Documentation plan
- ✅ Risk mitigation strategies

**Expected Impact**:
- **Performance**: 40-60% faster training with NPY
- **Quality**: 15-20% better robustness with enhanced RIR
- **Usability**: Clear UI for feature extraction and configuration

**Next Steps**:
1. Review and approve this plan
2. Begin implementation with Priority 1 tasks
3. Continuous testing during development
4. Documentation and rollout

**Estimated Total Time**: 2-3 weeks (1 developer, part-time)

---

**Document Status**: Ready for Implementation
**Approval Required**: Yes
**Implementation Start**: Pending approval
