#!/usr/bin/env python3
"""
Training Insights Analyzer

Analyzes training logs and checkpoints to detect patterns,
provide recommendations, and identify potential issues.

Usage:
    python scripts/training_insights.py --checkpoint models/best_model.pt
    python scripts/training_insights.py --log logs/training.log
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TrainingInsight:
    """Single training insight/recommendation"""

    category: str  # "warning", "improvement", "success", "critical"
    title: str
    message: str
    metric: Optional[str] = None
    value: Optional[float] = None
    recommendation: Optional[str] = None


class TrainingInsightsAnalyzer:
    """Analyze training patterns and provide actionable insights"""

    def __init__(self):
        self.insights: List[TrainingInsight] = []
        self.metrics_history: Dict[str, List[float]] = {}
        self.config: Optional[Dict] = None

    def analyze_checkpoint(self, checkpoint_path: Path) -> List[TrainingInsight]:
        """Analyze a model checkpoint for insights"""
        self.insights = []

        if not checkpoint_path.exists():
            self.insights.append(
                TrainingInsight(
                    category="critical",
                    title="Checkpoint Not Found",
                    message=f"Checkpoint file not found: {checkpoint_path}",
                )
            )
            return self.insights

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as e:
            self.insights.append(
                TrainingInsight(
                    category="critical",
                    title="Checkpoint Load Error",
                    message=f"Failed to load checkpoint: {e}",
                )
            )
            return self.insights

        # Extract config
        if "config" in checkpoint:
            config = checkpoint["config"]
            if isinstance(config, dict):
                self.config = config
            else:
                self.config = config.model_dump() if hasattr(config, "model_dump") else vars(config)

        # Extract training history
        if "history" in checkpoint:
            history = checkpoint["history"]
            self._analyze_training_history(history)

        # Analyze model state
        if "model_state_dict" in checkpoint:
            self._analyze_model_weights(checkpoint["model_state_dict"])

        # Analyze config
        if self.config:
            self._analyze_config()

        # Epoch analysis
        if "epoch" in checkpoint:
            epoch = checkpoint["epoch"]
            self._analyze_epoch_count(epoch)

        # Best metrics
        if "best_val_loss" in checkpoint:
            self._analyze_best_metrics(checkpoint)

        return self.insights

    def _analyze_training_history(self, history: Dict[str, List[float]]):
        """Analyze training history for patterns"""

        # Check for val_loss patterns
        if "val_loss" in history and len(history["val_loss"]) > 5:
            val_losses = history["val_loss"]

            # Fluctuation analysis
            std = np.std(val_losses[-10:]) if len(val_losses) >= 10 else np.std(val_losses)
            mean = np.mean(val_losses[-10:]) if len(val_losses) >= 10 else np.mean(val_losses)
            cv = std / mean if mean > 0 else 0  # Coefficient of variation

            if cv > 0.15:
                self.insights.append(
                    TrainingInsight(
                        category="warning",
                        title="High Validation Loss Fluctuation",
                        message=f"Val loss has high variance (CV={cv:.2%}). This may indicate unstable training.",
                        metric="val_loss_cv",
                        value=float(cv),
                        recommendation="Try: Lower learning rate, increase batch size, add warmup epochs, or use EMA.",
                    )
                )

            # Overfitting detection
            if "train_loss" in history:
                train_losses = history["train_loss"]
                if len(train_losses) > 10 and len(val_losses) > 10:
                    train_trend = train_losses[-1] - train_losses[-10]
                    val_trend = val_losses[-1] - val_losses[-10]

                    if train_trend < -0.05 and val_trend > 0.05:
                        self.insights.append(
                            TrainingInsight(
                                category="warning",
                                title="Potential Overfitting Detected",
                                message="Training loss decreasing while validation loss increasing.",
                                recommendation="Add more regularization (dropout, weight decay), use data augmentation, or enable early stopping.",
                            )
                        )

            # Plateau detection
            if len(val_losses) > 20:
                recent = val_losses[-10:]
                older = val_losses[-20:-10]
                improvement = np.mean(older) - np.mean(recent)

                if abs(improvement) < 0.001:
                    self.insights.append(
                        TrainingInsight(
                            category="improvement",
                            title="Learning Plateau",
                            message="Model appears to have plateaued - minimal improvement in last 10 epochs.",
                            recommendation="Try: Learning rate scheduling, different optimizer, or increase model capacity.",
                        )
                    )

        # Check accuracy
        if "val_accuracy" in history:
            val_acc = history["val_accuracy"]
            if val_acc and val_acc[-1] < 0.85:
                self.insights.append(
                    TrainingInsight(
                        category="warning",
                        title="Low Validation Accuracy",
                        message=f"Final validation accuracy is {val_acc[-1]:.2%}",
                        metric="val_accuracy",
                        value=val_acc[-1],
                        recommendation="Check data quality, increase model capacity, or adjust class weights.",
                    )
                )
            elif val_acc and val_acc[-1] > 0.98:
                self.insights.append(
                    TrainingInsight(
                        category="warning",
                        title="Suspiciously High Accuracy",
                        message=f"Validation accuracy is {val_acc[-1]:.2%} - might indicate data leakage.",
                        metric="val_accuracy",
                        value=val_acc[-1],
                        recommendation="Verify train/val split doesn't have overlapping samples.",
                    )
                )

    def _analyze_model_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Analyze model weights for issues"""

        # Check for NaN or Inf
        for name, param in state_dict.items():
            if torch.isnan(param).any():
                self.insights.append(
                    TrainingInsight(
                        category="critical",
                        title="NaN Weights Detected",
                        message=f"Layer '{name}' contains NaN values.",
                        recommendation="Training diverged. Use gradient clipping, lower learning rate.",
                    )
                )
                break

            if torch.isinf(param).any():
                self.insights.append(
                    TrainingInsight(
                        category="critical",
                        title="Infinite Weights Detected",
                        message=f"Layer '{name}' contains Inf values.",
                        recommendation="Training diverged. Use gradient clipping, lower learning rate.",
                    )
                )
                break

        # Check weight magnitudes
        weight_mags = []
        for name, param in state_dict.items():
            if "weight" in name and param.dim() >= 2:
                weight_mags.append((name, param.abs().mean().item()))

        if weight_mags:
            avg_mag = np.mean([m for _, m in weight_mags])
            if avg_mag > 10:
                self.insights.append(
                    TrainingInsight(
                        category="warning",
                        title="Large Weight Magnitudes",
                        message=f"Average weight magnitude is {avg_mag:.2f}",
                        recommendation="Consider stronger L2 regularization (weight_decay).",
                    )
                )

    def _analyze_config(self):
        """Analyze configuration for common issues"""

        if not self.config:
            return

        # Learning rate analysis
        training = self.config.get("training", {})
        lr = training.get("learning_rate", 0.001)
        batch_size = training.get("batch_size", 32)

        if lr > 0.01:
            self.insights.append(
                TrainingInsight(
                    category="warning",
                    title="High Learning Rate",
                    message=f"Learning rate {lr} is high for most wakeword tasks.",
                    recommendation="Consider lr=5e-4 to 1e-3 for stable training.",
                )
            )

        if batch_size < 32:
            self.insights.append(
                TrainingInsight(
                    category="improvement",
                    title="Small Batch Size",
                    message=f"Batch size {batch_size} may cause noisy gradients.",
                    recommendation="Increase to 64-128 if GPU memory allows.",
                )
            )

        # QAT analysis
        qat = self.config.get("qat", {})
        if qat.get("enabled", False):
            start_epoch = qat.get("start_epoch", 0)
            if start_epoch < 5:
                self.insights.append(
                    TrainingInsight(
                        category="improvement",
                        title="Early QAT Start",
                        message=f"QAT starts at epoch {start_epoch}.",
                        recommendation="Start QAT after epoch 5-10 for better convergence.",
                    )
                )

            self.insights.append(
                TrainingInsight(
                    category="success",
                    title="QAT Enabled",
                    message="Quantization-aware training is enabled for INT8 deployment.",
                )
            )

        # Distillation analysis
        distillation = self.config.get("distillation", {})
        if distillation.get("enabled", False):
            alpha = distillation.get("alpha", 0.5)
            if alpha > 0.5:
                self.insights.append(
                    TrainingInsight(
                        category="improvement",
                        title="High Distillation Alpha",
                        message=f"Distillation alpha={alpha} heavily favors teacher.",
                        recommendation="Try alpha=0.3 to balance hard labels and teacher knowledge.",
                    )
                )

            self.insights.append(
                TrainingInsight(
                    category="success",
                    title="Knowledge Distillation Enabled",
                    message=f"Using {distillation.get('teacher_architecture', 'wav2vec2')} as teacher.",
                )
            )

    def _analyze_epoch_count(self, epoch: int):
        """Analyze training duration"""

        if epoch < 20:
            self.insights.append(
                TrainingInsight(
                    category="improvement",
                    title="Short Training",
                    message=f"Training stopped at epoch {epoch}.",
                    recommendation="Consider training for more epochs (100+) for better convergence.",
                )
            )
        elif epoch > 200:
            self.insights.append(
                TrainingInsight(
                    category="improvement",
                    title="Long Training",
                    message=f"Training ran for {epoch} epochs.",
                    recommendation="Check if early stopping is enabled to prevent overfitting.",
                )
            )

    def _analyze_best_metrics(self, checkpoint: Dict):
        """Analyze best achieved metrics"""

        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if best_val_loss < 0.1:
            self.insights.append(
                TrainingInsight(
                    category="success",
                    title="Excellent Validation Loss",
                    message=f"Best val loss: {best_val_loss:.4f}",
                    metric="best_val_loss",
                    value=best_val_loss,
                )
            )
        elif best_val_loss > 0.5:
            self.insights.append(
                TrainingInsight(
                    category="warning",
                    title="High Validation Loss",
                    message=f"Best val loss: {best_val_loss:.4f}",
                    metric="best_val_loss",
                    value=best_val_loss,
                    recommendation="Check data quality, model architecture, or hyperparameters.",
                )
            )

    def analyze_log_file(self, log_path: Path) -> List[TrainingInsight]:
        """Analyze training log file for patterns"""
        self.insights = []

        if not log_path.exists():
            self.insights.append(
                TrainingInsight(
                    category="critical", title="Log File Not Found", message=f"Log file not found: {log_path}"
                )
            )
            return self.insights

        with open(log_path, "r") as f:
            content = f.read()

        # Check for common error patterns
        if "CUDA out of memory" in content:
            self.insights.append(
                TrainingInsight(
                    category="critical",
                    title="GPU Memory Error",
                    message="CUDA out of memory error detected.",
                    recommendation="Reduce batch size, use gradient accumulation, or use mixed precision.",
                )
            )

        if "NaN" in content.lower() and "loss" in content.lower():
            self.insights.append(
                TrainingInsight(
                    category="critical",
                    title="NaN Loss Detected",
                    message="NaN loss detected in training logs.",
                    recommendation="Lower learning rate, add gradient clipping, check data for anomalies.",
                )
            )

        # Check for early stopping
        if "early stopping" in content.lower():
            self.insights.append(
                TrainingInsight(
                    category="success",
                    title="Early Stopping Triggered",
                    message="Training stopped early due to no improvement.",
                )
            )

        return self.insights

    def generate_report(self) -> str:
        """Generate a formatted report of insights"""

        if not self.insights:
            return "✅ No issues detected. Training appears healthy."

        report = ["=" * 60, "TRAINING INSIGHTS REPORT", "=" * 60, ""]

        # Group by category
        categories = {
            "critical": "🚨 CRITICAL",
            "warning": "⚠️ WARNINGS",
            "improvement": "💡 IMPROVEMENTS",
            "success": "✅ SUCCESS",
        }

        for cat, label in categories.items():
            cat_insights = [i for i in self.insights if i.category == cat]
            if cat_insights:
                report.append(f"\n{label}")
                report.append("-" * 40)
                for insight in cat_insights:
                    report.append(f"\n📌 {insight.title}")
                    report.append(f"   {insight.message}")
                    if insight.recommendation:
                        report.append(f"   → {insight.recommendation}")

        report.append("\n" + "=" * 60)
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Training Insights Analyzer")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--log", type=str, help="Path to training log file")
    parser.add_argument("--output", type=str, help="Output report file (optional)")
    args = parser.parse_args()

    analyzer = TrainingInsightsAnalyzer()

    if args.checkpoint:
        print(f"Analyzing checkpoint: {args.checkpoint}")
        analyzer.analyze_checkpoint(Path(args.checkpoint))

    if args.log:
        print(f"Analyzing log file: {args.log}")
        analyzer.analyze_log_file(Path(args.log))

    if not args.checkpoint and not args.log:
        # Default: analyze best_model.pt
        default_path = Path("models/best_model.pt")
        if default_path.exists():
            print(f"Analyzing default checkpoint: {default_path}")
            analyzer.analyze_checkpoint(default_path)
        else:
            print("No checkpoint or log file specified. Use --checkpoint or --log.")
            return

    report = analyzer.generate_report()
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
