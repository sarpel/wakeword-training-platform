"""
Complete Training Pipeline Example
Demonstrates all newly integrated production features working together
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import load_dataset_splits
from src.data.cmvn import compute_cmvn_from_dataset, CMVN
from src.data.balanced_sampler import create_balanced_sampler_from_dataset
from src.models.architectures import create_model
from src.training.ema import EMA, EMAScheduler
from src.training.lr_finder import LRFinder
from src.training.trainer import Trainer
from src.models.temperature_scaling import calibrate_model
from src.training.advanced_metrics import calculate_comprehensive_metrics
from src.evaluation.evaluator import ModelEvaluator
from src.config.defaults import get_default_config


def main():
    """
    Complete training pipeline with all production features
    """
    print("=" * 80)
    print("COMPLETE TRAINING PIPELINE - Production Features Integration")
    print("=" * 80)

    # ============================================================================
    # STEP 1: Load Configuration
    # ============================================================================
    print("\n[1/9] Loading Configuration...")

    config = get_default_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"  Device: {device}")
    print(f"  Model: {config.model.architecture}")
    print(f"  Batch size: {config.training.batch_size}")

    # ============================================================================
    # STEP 2: Load Datasets
    # ============================================================================
    print("\n[2/9] Loading Datasets...")

    splits_dir = Path("data/splits")
    if not splits_dir.exists():
        print(f"  ❌ Splits directory not found: {splits_dir}")
        print(f"  Please run dataset scanning and splitting first")
        return

    train_ds, val_ds, test_ds = load_dataset_splits(
        splits_dir=splits_dir,
        sample_rate=config.data.sample_rate,
        audio_duration=config.data.audio_duration,
        augment_train=True,
        use_precomputed_features=True,
        apply_cmvn=False  # Will apply after computing stats
    )

    print(f"  ✅ Datasets loaded:")
    print(f"     Train: {len(train_ds)} samples")
    print(f"     Val:   {len(val_ds)} samples")
    print(f"     Test:  {len(test_ds)} samples")

    # ============================================================================
    # STEP 3: Compute CMVN Statistics
    # ============================================================================
    print("\n[3/9] Computing CMVN Statistics...")

    cmvn_path = Path("data/cmvn_stats.json")

    if not cmvn_path.exists():
        print(f"  Computing CMVN from training set...")
        cmvn = compute_cmvn_from_dataset(
            dataset=train_ds,
            stats_path=cmvn_path,
            max_samples=1000  # Use subset for speed
        )
        print(f"  ✅ CMVN stats saved to {cmvn_path}")
    else:
        print(f"  ✅ Loading existing CMVN stats from {cmvn_path}")
        cmvn = CMVN(stats_path=cmvn_path)

    # Reload datasets with CMVN enabled
    train_ds, val_ds, test_ds = load_dataset_splits(
        splits_dir=splits_dir,
        sample_rate=config.data.sample_rate,
        audio_duration=config.data.audio_duration,
        augment_train=True,
        use_precomputed_features=True,
        cmvn_path=cmvn_path,
        apply_cmvn=True  # Enable CMVN
    )

    print(f"  ✅ Datasets reloaded with CMVN normalization")

    # ============================================================================
    # STEP 4: Create Balanced Batch Sampler
    # ============================================================================
    print("\n[4/9] Creating Balanced Batch Sampler...")

    try:
        train_sampler = create_balanced_sampler_from_dataset(
            dataset=train_ds,
            batch_size=config.training.batch_size,
            ratio=(1, 1, 1),  # pos:neg:hard_neg
            drop_last=True
        )

        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=config.training.persistent_workers
        )

        print(f"  ✅ Balanced sampler created:")
        print(f"     Ratio: 1:1:1 (pos:neg:hard_neg)")
        print(f"     Total batches: {len(train_sampler)}")

    except Exception as e:
        print(f"  ⚠️  Could not create balanced sampler: {e}")
        print(f"  Using standard DataLoader instead...")

        train_loader = DataLoader(
            train_ds,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory
        )

    # Validation loader (standard)
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # ============================================================================
    # STEP 5: Create Model and Find Optimal LR
    # ============================================================================
    print("\n[5/9] Creating Model and Finding Optimal LR...")

    # Calculate input size for model
    input_samples = int(config.data.sample_rate * config.data.audio_duration)
    time_steps = input_samples // config.data.hop_length + 1
    
    feature_dim = config.data.n_mels if config.data.feature_type == "mel_spectrogram" or config.data.feature_type == "mel" else config.data.n_mfcc
    
    if config.model.architecture == "cd_dnn":
        input_size = feature_dim * time_steps
    else:
        input_size = feature_dim

    model = create_model(
        architecture=config.model.architecture,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
        input_size=input_size,
        input_channels=1,
    )

    model = model.to(device)
    print(f"  ✅ Model created: {config.model.architecture}")

    # Find optimal learning rate
    print(f"  Running LR finder (this may take a few minutes)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    lr_finder = LRFinder(model, optimizer, criterion, device=device)

    lrs, losses = lr_finder.range_test(
        train_loader,
        start_lr=1e-6,
        end_lr=1e-2,
        num_iter=100
    )

    optimal_lr = lr_finder.suggest_lr()

    print(f"  ✅ LR finder complete:")
    print(f"     Suggested LR: {optimal_lr:.2e}")
    print(f"     Original LR: {config.training.learning_rate:.2e}")

    # Use optimal LR (or keep original if very different)
    if 1e-5 <= optimal_lr <= 1e-2:
        config.training.learning_rate = optimal_lr
        print(f"  Using suggested LR: {optimal_lr:.2e}")
    else:
        print(f"  Keeping original LR: {config.training.learning_rate:.2e}")

    # ============================================================================
    # STEP 6: Create Trainer with EMA
    # ============================================================================
    print("\n[6/9] Creating Trainer with EMA...")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=Path("checkpoints"),
        device=device,
        use_ema=True,  # Enable EMA
        ema_decay=0.999
    )

    print(f"  ✅ Trainer initialized:")
    print(f"     EMA enabled: Yes")
    print(f"     EMA decay: 0.999 → 0.9995 (final 10 epochs)")
    print(f"     Mixed precision: {config.optimizer.mixed_precision}")
    print(f"     Gradient clipping: {config.optimizer.gradient_clip}")

    # ============================================================================
    # STEP 7: Train Model
    # ============================================================================
    print("\n[7/9] Training Model...")
    print(f"  Starting training for {config.training.epochs} epochs...")
    print(f"  (Set epochs to lower value for testing, e.g., config.training.epochs = 5)")

    # For demo purposes, train only a few epochs
    # config.training.epochs = 5  # Uncomment for quick demo

    try:
        results = trainer.train(
            start_epoch=0,
            seed=42,
            deterministic=False
        )

        print(f"\n  ✅ Training complete:")
        print(f"     Final epoch: {results['final_epoch']}")
        print(f"     Best val loss: {results['best_val_loss']:.4f}")
        print(f"     Best val F1: {results['best_val_f1']:.4f}")
        print(f"     Training time: {results['training_time'] / 3600:.2f} hours")

    except KeyboardInterrupt:
        print(f"\n  ⚠️  Training interrupted by user")
        return

    # ============================================================================
    # STEP 8: Calibrate with Temperature Scaling
    # ============================================================================
    print("\n[8/9] Calibrating Model with Temperature Scaling...")

    temp_scaling = calibrate_model(
        model=model,
        val_loader=val_loader,
        device=device,
        lr=0.01,
        max_iter=50
    )

    print(f"  ✅ Temperature scaling fitted:")
    print(f"     Optimal temperature: {temp_scaling.get_temperature():.4f}")

    # ============================================================================
    # STEP 9: Evaluate with Advanced Metrics
    # ============================================================================
    print("\n[9/9] Evaluating with Advanced Metrics...")

    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        sample_rate=config.data.sample_rate,
        audio_duration=config.data.audio_duration,
        device=device,
        feature_type=config.data.feature_type,
        n_mels=config.data.n_mels,
        n_mfcc=config.data.n_mfcc,
        n_fft=config.data.n_fft,
        hop_length=config.data.hop_length
    )

    # Calculate total audio duration for FAH
    total_seconds = len(test_ds) * config.data.audio_duration

    # Run comprehensive evaluation
    metrics = evaluator.evaluate_with_advanced_metrics(
        dataset=test_ds,
        total_seconds=total_seconds,
        target_fah=1.0,  # Target: 1 false alarm per hour
        batch_size=32
    )

    print(f"\n  ✅ Advanced Metrics:")
    print(f"     ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"     EER: {metrics['eer']:.4f} @ {metrics['eer_threshold']:.4f}")
    print(f"     pAUC (FPR≤0.1): {metrics['pauc_at_fpr_0.1']:.4f}")
    print(f"\n     Operating Point (FAH ≤ 1.0):")
    print(f"       Threshold: {metrics['operating_point']['threshold']:.4f}")
    print(f"       TPR: {metrics['operating_point']['tpr']:.4f}")
    print(f"       FPR: {metrics['operating_point']['fpr']:.4f}")
    print(f"       Precision: {metrics['operating_point']['precision']:.4f}")
    print(f"       F1: {metrics['operating_point']['f1_score']:.4f}")
    print(f"       FAH: {metrics['operating_point']['fah']:.2f}")

    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE - All Production Features Integrated Successfully!")
    print("=" * 80)

    print("\nFeatures Demonstrated:")
    print("  ✅ CMVN (Corpus-level normalization)")
    print("  ✅ Balanced batch sampling (pos:neg:hard_neg)")
    print("  ✅ LR Finder (automated learning rate discovery)")
    print("  ✅ EMA (Exponential Moving Average with adaptive decay)")
    print("  ✅ Temperature Scaling (model calibration)")
    print("  ✅ Advanced Metrics (FAH, EER, pAUC, operating point)")

    print("\nNext Steps:")
    print("  1. Review checkpoints in: checkpoints/")
    print("  2. Check best model: checkpoints/best_model.pt")
    print("  3. Use detected operating threshold in production")
    print("  4. Integrate streaming detector for real-time detection")

    print("\nSee IMPLEMENTATION_GUIDE.md for detailed documentation")
    print("=" * 80)


if __name__ == "__main__":
    main()
