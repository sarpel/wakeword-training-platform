"""
Ablation Study: Compare Training With vs Without Knowledge Distillation

This script trains the same student model twice:
1. WITHOUT distillation (baseline)
2. WITH distillation (teacher-guided)

Then compares the results to quantify the improvement from distillation.

USAGE:
    python examples/compare_with_without_distillation.py

EXPECTED RESULTS:
    Typical improvement from distillation: +2% to +5% F1 score
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.config.defaults import DistillationConfig, WakewordConfig
from src.data.dataset import WakewordDataset
from src.models.factory import create_model
from src.training.checkpoint_manager import CheckpointManager
from src.training.distillation_trainer import DistillationTrainer
from src.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_base_config():
    """Create base configuration shared by both experiments."""
    config = WakewordConfig()

    # Student model: MobileNetV3 (lightweight)
    config.model.architecture = "mobilenetv3"
    config.model.num_classes = 2
    config.model.dropout = 0.25

    # Training
    config.training.batch_size = 32
    config.training.epochs = 50  # Shorter for quick comparison
    config.training.learning_rate = 0.001
    config.training.early_stopping_patience = 10

    # Data (raw audio for distillation compatibility)
    config.data.sample_rate = 16000
    config.data.audio_duration = 1.5
    config.data.n_mels = 64
    config.data.use_precomputed_features_for_training = False
    config.data.fallback_to_audio = True

    # Optimizer
    config.optimizer.optimizer = "adamw"
    config.optimizer.weight_decay = 0.01
    config.optimizer.scheduler = "cosine"
    config.optimizer.mixed_precision = True

    return config


def load_datasets(config, data_root="data"):
    """Load training and validation datasets."""
    logger.info("Loading datasets...")

    data_root = Path(data_root)
    train_manifest = data_root / "splits" / "train.json"
    val_manifest = data_root / "splits" / "val.json"

    if not train_manifest.exists():
        raise FileNotFoundError(f"Training manifest not found: {train_manifest}\n" f"Please run data scanning first.")

    # Training dataset
    train_dataset = WakewordDataset(
        manifest_path=train_manifest,
        sample_rate=config.data.sample_rate,
        audio_duration=config.data.audio_duration,
        augment=True,
        device="cuda",
        feature_type="mel",
        n_mels=config.data.n_mels,
        n_fft=config.data.n_fft,
        hop_length=config.data.hop_length,
        return_raw_audio=True,
        use_precomputed_features_for_training=False,
        fallback_to_audio=True,
    )

    # Validation dataset
    val_dataset = WakewordDataset(
        manifest_path=val_manifest,
        sample_rate=config.data.sample_rate,
        audio_duration=config.data.audio_duration,
        augment=False,
        device="cuda",
        feature_type="mel",
        n_mels=config.data.n_mels,
        n_fft=config.data.n_fft,
        hop_length=config.data.hop_length,
        return_raw_audio=True,
        use_precomputed_features_for_training=False,
        fallback_to_audio=True,
    )

    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_student_model(config):
    """Create fresh student model."""
    time_steps = int(config.data.sample_rate * config.data.audio_duration) // config.data.hop_length + 1
    input_size = config.data.n_mels

    model = create_model(
        architecture=config.model.architecture,
        num_classes=config.model.num_classes,
        input_size=input_size,
        dropout=config.model.dropout,
    )

    return model


def train_baseline(config, train_loader, val_loader):
    """
    Train WITHOUT distillation (baseline).

    Returns:
        dict: Training results
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: BASELINE (No Distillation)")
    logger.info("=" * 70)

    # Ensure distillation is DISABLED
    config.distillation.enabled = False

    # Create fresh model
    model = create_student_model(config)
    logger.info(f"Model: {config.model.architecture}")
    logger.info(f"Distillation: DISABLED")

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/baseline")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    # Create STANDARD trainer (no distillation)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_manager=checkpoint_manager,
        device="cuda",
    )

    logger.info("Starting baseline training...")
    results = trainer.train()

    logger.info("\nBaseline Results:")
    logger.info(f"  Best Epoch: {results.get('best_epoch', 'N/A')}")
    logger.info(f"  Best Val Loss: {results.get('best_val_loss', float('inf')):.4f}")
    logger.info(f"  Best Val F1: {results.get('best_val_f1', 0.0):.4f}")
    logger.info(f"  Best Val Acc: {results.get('best_val_acc', 0.0):.2%}")

    return {
        "name": "Baseline (No Distillation)",
        "best_epoch": results.get("best_epoch"),
        "best_val_loss": results.get("best_val_loss"),
        "best_val_f1": results.get("best_val_f1"),
        "best_val_acc": results.get("best_val_acc"),
        "checkpoint_path": results.get("best_checkpoint"),
    }


def train_with_distillation(config, train_loader, val_loader, alpha=0.6, temperature=3.0):
    """
    Train WITH distillation (experimental).

    Args:
        config: Base configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        alpha: Distillation alpha (teacher weight)
        temperature: Distillation temperature

    Returns:
        dict: Training results
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: WITH DISTILLATION")
    logger.info("=" * 70)

    # Enable distillation
    config.distillation = DistillationConfig(
        enabled=True,
        teacher_architecture="wav2vec2",
        temperature=temperature,
        alpha=alpha,
    )

    # Create fresh model (same architecture)
    model = create_student_model(config)
    logger.info(f"Model: {config.model.architecture}")
    logger.info(f"Distillation: ENABLED")
    logger.info(f"  Teacher: Wav2Vec2")
    logger.info(f"  Alpha: {alpha} ({alpha*100:.0f}% teacher)")
    logger.info(f"  Temperature: {temperature}")

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/distillation")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    # Create DISTILLATION trainer
    trainer = DistillationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_manager=checkpoint_manager,
        device="cuda",
    )

    logger.info("Starting distillation training...")
    results = trainer.train()

    logger.info("\nDistillation Results:")
    logger.info(f"  Best Epoch: {results.get('best_epoch', 'N/A')}")
    logger.info(f"  Best Val Loss: {results.get('best_val_loss', float('inf')):.4f}")
    logger.info(f"  Best Val F1: {results.get('best_val_f1', 0.0):.4f}")
    logger.info(f"  Best Val Acc: {results.get('best_val_acc', 0.0):.2%}")

    return {
        "name": "With Distillation",
        "best_epoch": results.get("best_epoch"),
        "best_val_loss": results.get("best_val_loss"),
        "best_val_f1": results.get("best_val_f1"),
        "best_val_acc": results.get("best_val_acc"),
        "checkpoint_path": results.get("best_checkpoint"),
        "alpha": alpha,
        "temperature": temperature,
    }


def compare_results(baseline, distillation):
    """
    Compare and visualize results.

    Args:
        baseline: Baseline results dict
        distillation: Distillation results dict
    """
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON")
    logger.info("=" * 70)

    # Calculate improvements
    f1_improvement = distillation["best_val_f1"] - baseline["best_val_f1"]
    acc_improvement = distillation["best_val_acc"] - baseline["best_val_acc"]
    loss_improvement = baseline["best_val_loss"] - distillation["best_val_loss"]

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Metric':<30} {'Baseline':<15} {'Distillation':<15} {'Δ':<10}")
    print("=" * 70)
    print(
        f"{'Validation F1':<30} {baseline['best_val_f1']:<15.4f} {distillation['best_val_f1']:<15.4f} {f1_improvement:+.4f}"
    )
    print(
        f"{'Validation Accuracy':<30} {baseline['best_val_acc']:<15.2%} {distillation['best_val_acc']:<15.2%} {acc_improvement:+.2%}"
    )
    print(
        f"{'Validation Loss':<30} {baseline['best_val_loss']:<15.4f} {distillation['best_val_loss']:<15.4f} {loss_improvement:+.4f}"
    )
    print(f"{'Best Epoch':<30} {baseline['best_epoch']:<15} {distillation['best_epoch']:<15}")
    print("=" * 70)

    # Interpretation
    print("\n📊 RESULTS INTERPRETATION:")
    print("-" * 70)

    if f1_improvement > 0.02:
        print("✅ SIGNIFICANT IMPROVEMENT from distillation (+2% or more F1)")
        print("   → Distillation is highly effective for this task")
    elif f1_improvement > 0:
        print("✅ MODEST IMPROVEMENT from distillation")
        print("   → Distillation provides some benefit")
    else:
        print("⚠️  NO IMPROVEMENT from distillation")
        print("   → Consider: (1) Teacher may not be strong enough")
        print("              (2) Alpha/temperature may need tuning")
        print("              (3) Dataset may be too simple")

    print(f"\nF1 Improvement: {f1_improvement:+.2%} ({f1_improvement*100:+.2f} percentage points)")
    print(f"Accuracy Improvement: {acc_improvement:+.2%}")

    # Create visualization
    create_comparison_plot(baseline, distillation)

    # Save results
    save_comparison_results(baseline, distillation, f1_improvement, acc_improvement)


def create_comparison_plot(baseline, distillation):
    """Create bar chart comparing metrics."""
    logger.info("\nCreating comparison plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: F1 Score
    models = ["Baseline", "Distillation"]
    f1_scores = [baseline["best_val_f1"], distillation["best_val_f1"]]
    colors = ["#ff6b6b", "#51cf66"]

    ax1.bar(models, f1_scores, color=colors, alpha=0.7, edgecolor="black")
    ax1.set_ylabel("F1 Score", fontsize=12)
    ax1.set_title("F1 Score Comparison", fontsize=14, fontweight="bold")
    ax1.set_ylim([min(f1_scores) - 0.05, max(f1_scores) + 0.05])
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, v in enumerate(f1_scores):
        ax1.text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")

    # Plot 2: Accuracy
    accuracies = [baseline["best_val_acc"] * 100, distillation["best_val_acc"] * 100]

    ax2.bar(models, accuracies, color=colors, alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy Comparison", fontsize=14, fontweight="bold")
    ax2.set_ylim([min(accuracies) - 2, max(accuracies) + 2])
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, v in enumerate(accuracies):
        ax2.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontweight="bold")

    plt.tight_layout()

    # Save plot
    plot_path = Path("results/distillation_comparison.png")
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"  ✓ Plot saved to: {plot_path}")

    plt.close()


def save_comparison_results(baseline, distillation, f1_improvement, acc_improvement):
    """Save comparison results to JSON."""
    logger.info("\nSaving results...")

    results = {
        "baseline": baseline,
        "distillation": distillation,
        "improvements": {
            "f1": f1_improvement,
            "accuracy": acc_improvement,
            "f1_percent": f1_improvement * 100,
            "accuracy_percent": acc_improvement * 100,
        },
    }

    results_path = Path("results/distillation_comparison.json")
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"  ✓ Results saved to: {results_path}")


def main():
    """Run complete comparison experiment."""
    logger.info("=" * 70)
    logger.info("ABLATION STUDY: WITH vs WITHOUT DISTILLATION")
    logger.info("=" * 70)

    # Check GPU
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Training will be VERY slow on CPU")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            return

    # Setup
    config = create_base_config()
    train_loader, val_loader = load_datasets(config)

    # Experiment 1: Baseline (no distillation)
    baseline_results = train_baseline(config, train_loader, val_loader)

    # Experiment 2: With distillation
    distillation_results = train_with_distillation(config, train_loader, val_loader, alpha=0.6, temperature=3.0)

    # Compare
    compare_results(baseline_results, distillation_results)

    # Done!
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE! 🎉")
    logger.info("=" * 70)
    logger.info("Check results/distillation_comparison.png for visualization")
    logger.info("Check results/distillation_comparison.json for detailed metrics")


if __name__ == "__main__":
    main()
