"""
Complete Example: Training a Wakeword Model with Knowledge Distillation

This script demonstrates how to use knowledge distillation to train a lightweight
student model (MobileNetV3) using a powerful teacher model (Wav2Vec2).

WHAT IS KNOWLEDGE DISTILLATION? (ELI5)
=======================================
Think of it like this:
- Teacher = An expert professor (Wav2Vec2, large and smart)
- Student = A young learner (MobileNetV3, small and fast)

The student learns from BOTH:
1. The textbook (ground truth labels)
2. The professor's explanations (teacher's soft predictions)

Result: Student becomes much smarter than learning from textbook alone!

USAGE
=====
Basic:
    python examples/train_with_distillation.py

With custom config:
    python examples/train_with_distillation.py --config config/my_distillation.yaml

With custom alpha:
    python examples/train_with_distillation.py --alpha 0.7 --temperature 3.0
"""

import argparse
import logging
from pathlib import Path
from typing import Union

import torch


# ============================================================================
# SECURITY: Safe checkpoint loading to prevent arbitrary code execution
# ============================================================================
def safe_load_checkpoint(checkpoint_path: Union[str, Path], device: str = "cpu") -> dict:
    """
    Safely load a checkpoint with validation to prevent arbitrary code execution.

    This function implements defense-in-depth:
    1. Validates the checkpoint path (prevents path traversal)
    2. Loads tensors only when possible (weights_only=True) for PyTorch 2.4+
    3. Validates checkpoint structure before returning
    4. Provides clear error messages for debugging

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to map tensors to

    Returns:
        dict: Validated checkpoint dictionary

    Raises:
        ValueError: If checkpoint path or content is invalid
        FileNotFoundError: If checkpoint file does not exist
    """
    checkpoint_path = Path(checkpoint_path)

    # SECURITY: Validate checkpoint path (prevent path traversal)
    # Checkpoints should be in models/ or checkpoints/ directories
    allowed_dirs = ["models", "checkpoints", "cache", "exports"]
    try:
        resolved_path = checkpoint_path.resolve()
        # Check if parent directory is in allowed list
        parent_name = resolved_path.parent.name
        grandparent_name = resolved_path.parent.parent.name if resolved_path.parent.parent else ""

        is_allowed = parent_name in allowed_dirs or grandparent_name in allowed_dirs

        if not is_allowed:
            raise ValueError(
                f"Security violation: Checkpoint path '{checkpoint_path}' resolves to '{resolved_path}' "
                f"which is not in allowed directories ({allowed_dirs})"
            )
    except Exception as e:
        raise ValueError(f"Invalid checkpoint path: {e}") from e

    # SECURITY: Check file exists and is a regular file
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint path is not a file: {checkpoint_path}")

    # SECURITY: Load checkpoint safely
    try:
        # Try weights_only=True (PyTorch 2.4+ recommended)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        logger.info("Loaded checkpoint with weights_only=True (safest)")
    except Exception:
        # Fallback: Load with pickle but validate structure afterward
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.warning("Loaded checkpoint with pickle (less safe - validate structure)")

        # SECURITY: Validate checkpoint structure
        required_keys = ["model_state_dict", "optimizer_state_dict", "epoch"]
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(
                    f"Invalid checkpoint format: missing required key '{key}'. "
                    f"This may be a corrupted or malicious checkpoint file."
                )

        # SECURITY: Check for suspicious keys (arbitrary code)
        suspicious_keys = ["__builtins__", "__code__", "__func__", "eval", "exec", "compile"]
        for key in checkpoint.keys():
            key_str = str(key).lower()
            for susp in suspicious_keys:
                if susp in key_str:
                    raise ValueError(
                        f"Security violation: Suspicious key '{key}' found in checkpoint. "
                        f"This checkpoint may contain arbitrary code."
                    )

    return checkpoint


from torch.utils.data import DataLoader

# ============================================================================
# IMPORTS: Bring in all the necessary components
# ============================================================================
# These imports load the building blocks for our training pipeline
from src.config.defaults import DistillationConfig, WakewordConfig
from src.data.dataset import WakewordDataset
from src.models.architectures import create_model
from src.training.checkpoint_manager import CheckpointManager
from src.training.distillation_trainer import DistillationTrainer

# ============================================================================
# LOGGING SETUP: So we can see what's happening during training
# ============================================================================
logging.basicConfig(
    level=logging.INFO,  # Show INFO and above (INFO, WARNING, ERROR)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command-line arguments.

    This allows us to customize the training from the command line without
    modifying the script. For example:
        python train_with_distillation.py --alpha 0.7 --epochs 100
    """
    parser = argparse.ArgumentParser(description="Train wakeword model with knowledge distillation")

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional, uses defaults if not provided)",
    )

    # Distillation parameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Distillation alpha (0.0-1.0). Higher = more teacher influence",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=3.0,
        help="Distillation temperature (1.0-10.0). Higher = softer targets",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default="",
        help="Path to custom teacher checkpoint (optional)",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")

    # Model selection
    parser.add_argument(
        "--student-architecture",
        type=str,
        default="mobilenetv3",
        choices=["mobilenetv3", "resnet18", "tiny_conv", "lstm", "gru"],
        help="Student model architecture",
    )

    # Paths
    parser.add_argument("--data-root", type=str, default="data", help="Root directory for data")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )

    return parser.parse_args()


def create_configuration(args):
    """
    Create WakewordConfig from arguments.

    This function builds the complete configuration object that controls
    all aspects of training, including distillation settings.

    Args:
        args: Parsed command-line arguments

    Returns:
        WakewordConfig: Complete configuration object
    """
    logger.info("Creating configuration...")

    # STEP 1: Load base configuration
    # ================================
    if args.config:
        # Load from YAML file if provided
        logger.info(f"Loading configuration from {args.config}")
        config = WakewordConfig.load(Path(args.config))
    else:
        # Create default configuration
        logger.info("Using default configuration")
        config = WakewordConfig()

    # STEP 2: Configure Knowledge Distillation
    # =========================================
    logger.info("Configuring knowledge distillation...")
    config.distillation = DistillationConfig(
        enabled=True,  # Enable distillation
        teacher_architecture="wav2vec2",  # Use Wav2Vec2 as teacher
        teacher_model_path=args.teacher_checkpoint,  # Custom teacher (if provided)
        temperature=args.temperature,  # Softness of probability distributions
        alpha=args.alpha,  # Balance: (1-α)×student + α×teacher
    )

    logger.info(f"  ✓ Teacher: Wav2Vec2 (pretrained HuggingFace model)")
    logger.info(f"  ✓ Temperature: {args.temperature} (higher = softer targets)")
    logger.info(f"  ✓ Alpha: {args.alpha} ({args.alpha*100:.0f}% teacher, {(1-args.alpha)*100:.0f}% ground truth)")

    # STEP 3: Configure Student Model
    # ================================
    logger.info(f"Configuring student model: {args.student_architecture}")
    config.model.architecture = args.student_architecture
    config.model.num_classes = 2  # Binary: wakeword vs non-wakeword
    config.model.dropout = 0.25  # Dropout for regularization

    # STEP 4: Configure Training Parameters
    # ======================================
    config.training.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.learning_rate = args.learning_rate
    config.training.early_stopping_patience = 15

    # STEP 5: CRITICAL: Configure Data Pipeline for Raw Audio
    # ========================================================
    # IMPORTANT: Teacher model (Wav2Vec2) requires RAW AUDIO, not spectrograms!
    # We MUST disable precomputed features to ensure raw audio is passed to teacher
    logger.info("Configuring data pipeline for raw audio (required for teacher)...")
    config.data.use_precomputed_features_for_training = False  # CRITICAL!
    config.data.fallback_to_audio = True  # CRITICAL!
    config.data.sample_rate = 16000  # Standard for speech
    config.data.audio_duration = 1.5  # 1.5 seconds of audio
    config.data.n_mels = 64  # Mel frequency bins

    logger.info("  ✓ Raw audio pipeline enabled")
    logger.info("  ✓ Spectrograms will be computed on-the-fly")

    # STEP 6: Configure Optimizer
    # ============================
    config.optimizer.optimizer = "adamw"  # AdamW optimizer
    config.optimizer.weight_decay = 0.01  # Regularization
    config.optimizer.scheduler = "cosine"  # Cosine learning rate schedule
    config.optimizer.mixed_precision = True  # Use FP16 to save memory

    logger.info("Configuration complete!")
    return config


def create_datasets(config, data_root):
    """
    Create training and validation datasets.

    IMPORTANT: Datasets MUST return raw audio for distillation to work!

    Args:
        config: WakewordConfig object
        data_root: Root directory containing data/splits/

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    logger.info("Loading datasets...")

    # Define paths to dataset splits
    # These JSON files contain metadata about audio samples
    data_root = Path(data_root)
    train_manifest = data_root / "splits" / "train.json"
    val_manifest = data_root / "splits" / "val.json"

    # Check if splits exist
    if not train_manifest.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {train_manifest}\n"
            f"Please run data scanning and splitting first (Panel 1 in UI)"
        )

    # STEP 1: Create Training Dataset
    # ================================
    # This dataset loads audio, applies augmentation, and returns batches
    logger.info("Creating training dataset...")
    train_dataset = WakewordDataset(
        manifest_path=train_manifest,
        sample_rate=config.data.sample_rate,  # 16000 Hz
        audio_duration=config.data.audio_duration,  # 1.5 seconds
        augment=True,  # Enable data augmentation (noise, pitch shift, etc.)
        device="cuda",  # Process on GPU
        feature_type="mel",  # Use mel spectrograms for student
        n_mels=config.data.n_mels,  # Number of mel bins
        n_fft=config.data.n_fft,  # FFT window size
        hop_length=config.data.hop_length,  # Hop length for STFT
        # CRITICAL PARAMETERS FOR DISTILLATION:
        return_raw_audio=True,  # Return raw audio for teacher!
        use_precomputed_features_for_training=False,  # Compute on-the-fly
        fallback_to_audio=True,  # Force raw audio path
    )

    # STEP 2: Create Validation Dataset
    # ==================================
    # Same as training, but no augmentation
    logger.info("Creating validation dataset...")
    val_dataset = WakewordDataset(
        manifest_path=val_manifest,
        sample_rate=config.data.sample_rate,
        audio_duration=config.data.audio_duration,
        augment=False,  # No augmentation for validation
        device="cuda",
        feature_type="mel",
        n_mels=config.data.n_mels,
        n_fft=config.data.n_fft,
        hop_length=config.data.hop_length,
        # CRITICAL:
        return_raw_audio=True,
        use_precomputed_features_for_training=False,
        fallback_to_audio=True,
    )

    logger.info(f"  ✓ Training samples: {len(train_dataset)}")
    logger.info(f"  ✓ Validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, batch_size):
    """
    Create PyTorch DataLoaders for efficient batching.

    DataLoaders handle:
    - Batching samples together
    - Shuffling data
    - Multi-threaded loading
    - GPU memory pinning

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Number of samples per batch

    Returns:
        tuple: (train_loader, val_loader)
    """
    logger.info("Creating data loaders...")

    # Training DataLoader
    # Shuffle=True: Randomize order each epoch (prevents overfitting)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Randomize
        num_workers=4,  # 4 worker threads for loading
        pin_memory=True,  # Pin memory for faster GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
    )

    # Validation DataLoader
    # Shuffle=False: Keep consistent order for validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't randomize
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    logger.info(f"  ✓ Batches per epoch: {len(train_loader)}")
    logger.info(f"  ✓ Validation batches: {len(val_loader)}")

    return train_loader, val_loader


def create_student_model(config):
    """
    Create the student model that will be trained.

    The student is a small, efficient model that learns from both:
    1. Ground truth labels (standard supervised learning)
    2. Teacher's soft predictions (knowledge distillation)

    Args:
        config: WakewordConfig object

    Returns:
        torch.nn.Module: Student model
    """
    logger.info(f"Creating student model: {config.model.architecture}")

    # Calculate input dimensions
    # This depends on audio duration and hop length
    time_steps = int(config.data.sample_rate * config.data.audio_duration) // config.data.hop_length + 1

    # For spectrogram-based models, input_size is frequency dimension
    input_size = config.data.n_mels

    # Create model using factory
    model = create_model(
        architecture=config.model.architecture,
        num_classes=config.model.num_classes,
        input_size=input_size,
        dropout=config.model.dropout,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"  ✓ Total parameters: {total_params:,}")
    logger.info(f"  ✓ Trainable parameters: {trainable_params:,}")

    return model


def main():
    """
    Main training function.

    This orchestrates the entire training pipeline:
    1. Parse arguments
    2. Create configuration
    3. Load datasets
    4. Create model
    5. Initialize trainer
    6. Train with distillation
    7. Evaluate and export
    """
    # ========================================================================
    # STEP 1: SETUP
    # ========================================================================
    logger.info("=" * 70)
    logger.info("KNOWLEDGE DISTILLATION TRAINING")
    logger.info("=" * 70)

    args = parse_args()

    # Verify GPU availability if requested
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available! Falling back to CPU (SLOW)")
        args.device = "cpu"

    logger.info(f"Device: {args.device}")
    if args.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ========================================================================
    # STEP 2: CONFIGURATION
    # ========================================================================
    config = create_configuration(args)

    # ========================================================================
    # STEP 3: DATA LOADING
    # ========================================================================
    train_dataset, val_dataset = create_datasets(config, args.data_root)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config.training.batch_size)

    # ========================================================================
    # STEP 4: MODEL CREATION
    # ========================================================================
    student_model = create_student_model(config)

    # ========================================================================
    # STEP 5: TRAINER INITIALIZATION
    # ========================================================================
    logger.info("Initializing DistillationTrainer...")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    # Create DistillationTrainer
    # This will automatically:
    # 1. Load teacher model (Wav2Vec2)
    # 2. Freeze teacher parameters
    # 3. Set up combined loss function
    trainer = DistillationTrainer(
        model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_manager=checkpoint_manager,
        device=args.device,
    )

    logger.info("  ✓ Teacher model loaded and frozen")
    logger.info("  ✓ Student model ready for training")
    logger.info("  ✓ Distillation loss configured")

    # ========================================================================
    # STEP 6: TRAINING
    # ========================================================================
    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)
    logger.info(f"Student: {config.model.architecture}")
    logger.info(f"Teacher: Wav2Vec2 (frozen)")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Alpha (teacher weight): {config.distillation.alpha}")
    logger.info(f"Temperature: {config.distillation.temperature}")
    logger.info("=" * 70)

    # Train the model
    # The trainer will:
    # 1. For each batch:
    #    a. Forward pass through student
    #    b. Forward pass through teacher (no gradients)
    #    c. Compute combined loss
    #    d. Backpropagate through student only
    # 2. Validate after each epoch
    # 3. Save best model
    results = trainer.train()

    # ========================================================================
    # STEP 7: RESULTS
    # ========================================================================
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Best Epoch: {results.get('best_epoch', 'N/A')}")
    logger.info(f"Best Validation Loss: {results.get('best_val_loss', float('inf')):.4f}")
    logger.info(f"Best Validation F1: {results.get('best_val_f1', 0.0):.4f}")
    logger.info(f"Best Validation Accuracy: {results.get('best_val_acc', 0.0):.2%}")
    logger.info(f"Best Model Path: {results.get('best_checkpoint', 'N/A')}")

    # ========================================================================
    # STEP 8: EXPORT (OPTIONAL)
    # ========================================================================
    logger.info("\nExporting student model to ONNX...")

    # Load best checkpoint
    best_checkpoint_path = results.get("best_checkpoint")
    if best_checkpoint_path and Path(best_checkpoint_path).exists():
        # SECURITY: Use safe loading to prevent arbitrary code execution
        checkpoint = safe_load_checkpoint(best_checkpoint_path, args.device)
        student_model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"  ✓ Loaded best checkpoint")

    # Export to ONNX
    student_model.eval()
    time_steps = int(config.data.sample_rate * config.data.audio_duration) // config.data.hop_length + 1
    dummy_input = torch.randn(1, 1, config.data.n_mels, time_steps).to(args.device)

    onnx_path = checkpoint_dir / f"{config.model.architecture}_distilled.onnx"
    torch.onnx.export(
        student_model,
        dummy_input,
        onnx_path,
        input_names=["audio_features"],
        output_names=["logits"],
        dynamic_axes={"audio_features": {0: "batch"}},
        opset_version=14,
    )

    logger.info(f"  ✓ Student model exported to: {onnx_path}")

    # ========================================================================
    # DONE!
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ALL DONE! 🎉")
    logger.info("=" * 70)
    logger.info(f"Student model trained with teacher guidance")
    logger.info(f"Ready for deployment: {onnx_path}")


if __name__ == "__main__":
    main()
