"""
Dataset Health Checker and Briefing System
Analyzes dataset quality and provides recommendations
"""
from typing import Dict, List, Any

import structlog

logger = structlog.get_logger(__name__)


class DatasetHealthChecker:
    """Analyzes dataset health and provides recommendations"""

    # Industry standards
    MIN_POSITIVE_SAMPLES = 1000
    RECOMMENDED_POSITIVE_SAMPLES = 5000
    MIN_NEGATIVE_TO_POSITIVE_RATIO = 8.0
    MAX_NEGATIVE_TO_POSITIVE_RATIO = 10.0
    HARD_NEGATIVE_RATIO = 0.25  # 25% of negatives
    MIN_SPEAKERS = 20
    RECOMMENDED_SPEAKERS = 50
    MIN_SAMPLE_RATE = 16000
    RECOMMENDED_DURATION_MIN = 1.0
    RECOMMENDED_DURATION_MAX = 2.0

    def __init__(self, dataset_stats: Dict):
        """
        Initialize health checker

        Args:
            dataset_stats: Dataset statistics from scanner
        """
        self.stats = dataset_stats
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.recommendations: List[str] = []
        self.health_score = 100.0

    def analyze(self) -> Dict:
        """
        Analyze dataset health

        Returns:
            Dictionary with health analysis results
        """
        logger.info("Analyzing dataset health...")

        # Check positive samples
        self._check_positive_samples()

        # Check class balance
        self._check_class_balance()

        # Check hard negatives
        self._check_hard_negatives()

        # Check sample rates
        self._check_sample_rates()

        # Check durations
        self._check_durations()

        # Check data diversity
        self._check_diversity()

        # Generate overall assessment
        assessment = self._generate_assessment()

        return {
            "health_score": max(0, self.health_score),
            "assessment": assessment,
            "issues": self.issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "statistics": self._get_key_statistics(),
        }

    def _check_positive_samples(self) -> None:
        """Check if sufficient positive samples"""
        categories = self.stats.get("categories", {})
        positive_count = categories.get("positive", {}).get("file_count", 0)

        if positive_count == 0:
            self.issues.append("❌ CRITICAL: No positive samples found")
            self.health_score -= 50
            self.recommendations.append(
                "• Record wakeword utterances from multiple speakers"
            )
        elif positive_count < self.MIN_POSITIVE_SAMPLES:
            self.warnings.append(
                f"⚠️  Low positive samples: {positive_count} (minimum: {self.MIN_POSITIVE_SAMPLES})"
            )
            self.health_score -= 20
            self.recommendations.append(
                f"• Collect at least {self.MIN_POSITIVE_SAMPLES - positive_count} more positive samples"
            )
        elif positive_count < self.RECOMMENDED_POSITIVE_SAMPLES:
            self.warnings.append(
                f"⚠️  Below recommended positive samples: {positive_count} (recommended: {self.RECOMMENDED_POSITIVE_SAMPLES})"
            )
            self.health_score -= 10
            self.recommendations.append(
                f"• Consider collecting {self.RECOMMENDED_POSITIVE_SAMPLES - positive_count} more positive samples for better accuracy"
            )

    def _check_class_balance(self) -> None:
        """Check positive to negative ratio"""
        categories = self.stats.get("categories", {})
        positive_count = categories.get("positive", {}).get("file_count", 0)
        negative_count = categories.get("negative", {}).get("file_count", 0)

        if positive_count == 0:
            return  # Already handled in positive check

        if negative_count == 0:
            self.issues.append("❌ CRITICAL: No negative samples found")
            self.health_score -= 30
            self.recommendations.append(
                "• Collect general speech and non-wakeword utterances"
            )
            return

        ratio = negative_count / positive_count

        if ratio < self.MIN_NEGATIVE_TO_POSITIVE_RATIO:
            self.warnings.append(
                f"⚠️  Insufficient negative samples: ratio {ratio:.1f}:1 "
                f"(recommended: {self.MIN_NEGATIVE_TO_POSITIVE_RATIO:.0f}-{self.MAX_NEGATIVE_TO_POSITIVE_RATIO:.0f}:1)"
            )
            self.health_score -= 15
            needed = int(
                positive_count * self.MIN_NEGATIVE_TO_POSITIVE_RATIO - negative_count
            )
            self.recommendations.append(
                f"• Collect approximately {needed} more negative samples"
            )
        elif ratio > self.MAX_NEGATIVE_TO_POSITIVE_RATIO * 2:
            self.warnings.append(
                f"⚠️  Very high negative ratio: {ratio:.1f}:1 (may slow training)"
            )
            self.health_score -= 5
            self.recommendations.append(
                "• Consider reducing negative samples or adding more positives"
            )

    def _check_hard_negatives(self) -> None:
        """Check hard negative samples"""
        categories = self.stats.get("categories", {})
        negative_count = categories.get("negative", {}).get("file_count", 0)
        hard_negative_count = categories.get("hard_negative", {}).get("file_count", 0)

        if negative_count == 0:
            return  # Already handled

        if hard_negative_count == 0:
            self.warnings.append("⚠️  No hard negative samples found")
            self.health_score -= 10
            self.recommendations.append(
                "• Record phrases similar to wakeword to reduce false positives\n"
                "  Hard negatives are critical for high precision"
            )
        else:
            hard_ratio = hard_negative_count / negative_count

            if hard_ratio < self.HARD_NEGATIVE_RATIO * 0.5:
                self.warnings.append(
                    f"⚠️  Low hard negative ratio: {hard_ratio*100:.1f}% "
                    f"(recommended: {self.HARD_NEGATIVE_RATIO*100:.0f}%)"
                )
                self.health_score -= 8
                needed = int(
                    negative_count * self.HARD_NEGATIVE_RATIO - hard_negative_count
                )
                self.recommendations.append(
                    f"• Collect approximately {needed} more hard negative samples"
                )

    def _check_sample_rates(self) -> None:
        """Check sample rate consistency"""
        categories = self.stats.get("categories", {})

        all_sample_rates: List[int] = []
        inconsistent_categories = []

        for category, data in categories.items():
            sample_rates = data.get("sample_rates", {})
            if sample_rates:
                all_sample_rates.extend(int(sr) for sr in sample_rates.keys())
                if len(sample_rates) > 1:
                    inconsistent_categories.append(category)

        if not all_sample_rates:
            return

        # Check for low sample rates
        low_rate_found = any(sr < self.MIN_SAMPLE_RATE for sr in all_sample_rates)
        if low_rate_found:
            self.warnings.append(
                f"⚠️  Low sample rate detected (< {self.MIN_SAMPLE_RATE}Hz)\n"
                "  Audio quality may be insufficient"
            )
            self.health_score -= 10
            self.recommendations.append(
                f"• Re-record audio at {self.MIN_SAMPLE_RATE}Hz or higher"
            )

        # Check for inconsistency
        if inconsistent_categories:
            self.warnings.append(
                f"⚠️  Inconsistent sample rates in: {', '.join(inconsistent_categories)}\n"
                "  Will be resampled during training"
            )
            self.health_score -= 5

    def _check_durations(self) -> None:
        """Check audio duration consistency"""
        categories = self.stats.get("categories", {})

        for category, data in categories.items():
            if category in ["background", "rirs"]:
                continue  # Skip background and RIRs

            avg_duration = data.get("avg_duration", 0)

            if avg_duration > 0:
                if avg_duration < self.RECOMMENDED_DURATION_MIN:
                    self.warnings.append(
                        f"⚠️  Short average duration in {category}: {avg_duration:.2f}s "
                        f"(recommended: {self.RECOMMENDED_DURATION_MIN}-{self.RECOMMENDED_DURATION_MAX}s)"
                    )
                    self.health_score -= 5
                elif avg_duration > self.RECOMMENDED_DURATION_MAX * 2:
                    self.warnings.append(
                        f"⚠️  Long average duration in {category}: {avg_duration:.2f}s "
                        f"(recommended: {self.RECOMMENDED_DURATION_MIN}-{self.RECOMMENDED_DURATION_MAX}s)\n"
                        "  Consider trimming to wakeword portion"
                    )
                    self.health_score -= 5

    def _check_diversity(self) -> None:
        """Check data diversity"""
        categories = self.stats.get("categories", {})

        # Check if background noise exists
        if (
            "background" not in categories
            or categories["background"].get("file_count", 0) == 0
        ):
            self.warnings.append("⚠️  No background noise samples found")
            self.health_score -= 8
            self.recommendations.append(
                "• Collect background noise (white noise, environmental sounds)\n"
                "  for realistic augmentation"
            )

        # Check if RIRs exist
        if "rirs" not in categories or categories["rirs"].get("file_count", 0) == 0:
            self.warnings.append("⚠️  No Room Impulse Response (RIR) samples found")
            self.health_score -= 5
            self.recommendations.append(
                "• Collect or download RIR samples for reverberation simulation\n"
                "  Improves real-world performance"
            )

    def _generate_assessment(self) -> str:
        """
        Generate overall assessment

        Returns:
            Assessment string with emoji indicator
        """
        score = self.health_score

        if score >= 90:
            return "🟢 EXCELLENT - Dataset is ready for high-quality training"
        elif score >= 75:
            return "🟡 GOOD - Dataset is usable with minor improvements recommended"
        elif score >= 60:
            return "🟠 FAIR - Dataset needs improvement before training"
        elif score >= 40:
            return "🔴 POOR - Significant issues detected, training not recommended"
        else:
            return "⛔ CRITICAL - Dataset insufficient for training"

    def _get_key_statistics(self) -> Dict:
        """
        Get key statistics for display

        Returns:
            Dictionary with key stats
        """
        categories = self.stats.get("categories", {})

        positive_count = categories.get("positive", {}).get("file_count", 0)
        negative_count = categories.get("negative", {}).get("file_count", 0)
        hard_negative_count = categories.get("hard_negative", {}).get("file_count", 0)
        background_count = categories.get("background", {}).get("file_count", 0)
        rir_count = categories.get("rirs", {}).get("file_count", 0)

        ratio = negative_count / positive_count if positive_count > 0 else 0

        return {
            "positive_samples": positive_count,
            "negative_samples": negative_count,
            "hard_negative_samples": hard_negative_count,
            "background_samples": background_count,
            "rir_samples": rir_count,
            "total_samples": self.stats.get("total_files", 0),
            "total_duration_minutes": self.stats.get("total_duration_minutes", 0),
            "negative_to_positive_ratio": f"{ratio:.1f}:1" if ratio > 0 else "N/A",
        }

    def generate_report(self) -> str:
        """
        Generate human-readable health report

        Returns:
            Formatted report string
        """
        analysis = self.analyze()

        report = []
        report.append("=" * 60)
        report.append("DATASET HEALTH REPORT")
        report.append("=" * 60)
        report.append("")

        # Overall assessment
        report.append(f"Overall Assessment: {analysis['assessment']}")
        report.append(f"Health Score: {analysis['health_score']:.1f}/100")
        report.append("")

        # Key statistics
        report.append("Key Statistics:")
        report.append("-" * 60)
        stats = analysis["statistics"]
        report.append(f"  Positive Samples:      {stats['positive_samples']:,}")
        report.append(f"  Negative Samples:      {stats['negative_samples']:,}")
        report.append(f"  Hard Negative Samples: {stats['hard_negative_samples']:,}")
        report.append(f"  Background Samples:    {stats['background_samples']:,}")
        report.append(f"  RIR Samples:           {stats['rir_samples']:,}")
        report.append(f"  Total Samples:         {stats['total_samples']:,}")
        report.append(
            f"  Total Duration:        {stats['total_duration_minutes']:.1f} minutes"
        )
        report.append(f"  Neg:Pos Ratio:         {stats['negative_to_positive_ratio']}")
        report.append("")

        # Issues
        if analysis["issues"]:
            report.append("Critical Issues:")
            report.append("-" * 60)
            for issue in analysis["issues"]:
                report.append(f"  {issue}")
            report.append("")

        # Warnings
        if analysis["warnings"]:
            report.append("Warnings:")
            report.append("-" * 60)
            for warning in analysis["warnings"]:
                report.append(f"  {warning}")
            report.append("")

        # Recommendations
        if analysis["recommendations"]:
            report.append("Recommendations:")
            report.append("-" * 60)
            for rec in analysis["recommendations"]:
                report.append(f"  {rec}")
            report.append("")

        # Training readiness
        report.append("Training Readiness:")
        report.append("-" * 60)
        if analysis["health_score"] >= 75:
            report.append("  ✅ Dataset is ready for training")
            report.append("  • Proceed to configuration and training")
        elif analysis["health_score"] >= 60:
            report.append("  ⚠️  Dataset can be used but improvements recommended")
            report.append("  • Consider addressing warnings before training")
        else:
            report.append("  ❌ Dataset not ready for training")
            report.append("  • Address critical issues before proceeding")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)


if __name__ == "__main__":
    # Test health checker
    print("Dataset Health Checker Test")
    print("=" * 60)

    # Example stats
    example_stats = {
        "total_files": 15000,
        "total_duration_minutes": 500,
        "categories": {
            "positive": {"file_count": 1500, "avg_duration": 1.5},
            "negative": {"file_count": 12000, "avg_duration": 1.8},
            "hard_negative": {"file_count": 3000, "avg_duration": 1.6},
            "background": {"file_count": 500, "avg_duration": 5.0},
            "rirs": {"file_count": 100, "avg_duration": 1.0},
        },
    }

    checker = DatasetHealthChecker(example_stats)
    report = checker.generate_report()
    print(report)
