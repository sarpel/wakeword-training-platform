"""
Model Evaluator for File-Based and Test Set Evaluation
GPU-accelerated batch evaluation with comprehensive metrics
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import time
import logging
from dataclasses import dataclass

from src.data.audio_utils import AudioProcessor
from src.data.feature_extraction import FeatureExtractor
from src.training.metrics import MetricsCalculator, MetricResults
from src.training.advanced_metrics import (
    calculate_comprehensive_metrics,
    calculate_fah,
    find_operating_point
)
from src.config.cuda_utils import enforce_cuda

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Single file evaluation result"""
    filename: str
    prediction: str  # "Positive" or "Negative"
    confidence: float
    latency_ms: float
    logits: np.ndarray


class ModelEvaluator:
    """
    Model evaluator for file-based and batch evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        sample_rate: int = 16000,
        audio_duration: float = 1.5,
        device: str = 'cuda',
        feature_type: str = 'mel',
        n_mels: int = 128,
        n_mfcc: int = 40,
        n_fft: int = 1024,
        hop_length: int = 160
    ):
        """
        Initialize model evaluator

        Args:
            model: Trained PyTorch model
            sample_rate: Audio sample rate
            audio_duration: Audio duration in seconds
            device: Device for inference ('cuda' or 'cpu')
            feature_type: Feature type ('mel' or 'mfcc')
            n_mels: Number of mel filterbanks
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        # Enforce CUDA
        enforce_cuda()

        self.model = model
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.device = device

        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()

        # Audio processor
        self.audio_processor = AudioProcessor(
            target_sr=sample_rate,
            target_duration=audio_duration
        )

        # Normalize feature type (handle legacy 'mel_spectrogram')
        if feature_type == 'mel_spectrogram':
            feature_type = 'mel'

        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            feature_type=feature_type,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            device=device
        )

        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(device=device)

        logger.info(f"ModelEvaluator initialized on {device}")

    def evaluate_file(
        self,
        audio_path: Path,
        threshold: float = 0.5
    ) -> EvaluationResult:
        """
        Evaluate single audio file

        Args:
            audio_path: Path to audio file
            threshold: Classification threshold

        Returns:
            EvaluationResult with prediction and metrics
        """
        start_time = time.time()

        # Load and process audio
        audio = self.audio_processor.process_audio(audio_path)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Extract features
        features = self.feature_extractor(audio_tensor)

        # Add batch dimension
        features = features.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = self.model(features)
            # Convert to float32 immediately after inference to ensure compatibility
            logits = logits.float()

            # FIX: Swap logits because model was trained with reversed class order
            # During training, the model learned class_0=positive, class_1=negative
            # But standard convention is class_0=negative, class_1=positive
            logits = logits[:, [1, 0]]

        # Get prediction
        probabilities = torch.softmax(logits, dim=1)
        confidence = probabilities[0, 1].item()  # Probability of positive class
        predicted_class = 1 if confidence >= threshold else 0

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Create result
        result = EvaluationResult(
            filename=audio_path.name,
            prediction="Positive" if predicted_class == 1 else "Negative",
            confidence=confidence,
            latency_ms=latency_ms,
            logits=logits.cpu().numpy()[0]
        )

        return result

    def evaluate_files(
        self,
        audio_paths: List[Path],
        threshold: float = 0.5,
        batch_size: int = 32
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple audio files in batches

        Args:
            audio_paths: List of paths to audio files
            threshold: Classification threshold
            batch_size: Batch size for processing

        Returns:
            List of EvaluationResult for each file
        """
        results = []

        # Process in batches
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]

            # Load batch
            batch_audio = []
            valid_paths = []

            for path in batch_paths:
                try:
                    audio = self.audio_processor.process_audio(path)
                    batch_audio.append(audio)
                    valid_paths.append(path)
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")
                    # Add error result
                    results.append(EvaluationResult(
                        filename=path.name,
                        prediction="Error",
                        confidence=0.0,
                        latency_ms=0.0,
                        logits=np.array([0.0, 0.0])
                    ))

            if not batch_audio:
                continue

            # Convert to tensor and extract features
            batch_features = []
            for audio in batch_audio:
                audio_tensor = torch.from_numpy(audio).float()
                features = self.feature_extractor(audio_tensor)
                batch_features.append(features)

            # Stack batch and move to device
            batch_tensor = torch.stack(batch_features).to(self.device)

            # Batch inference
            start_time = time.time()

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits = self.model(batch_tensor)
                # Convert to float32 immediately after inference to ensure compatibility
                logits = logits.float()

                # FIX: Swap logits because model was trained with reversed class order
                logits = logits[:, [1, 0]]

            batch_latency = (time.time() - start_time) * 1000 / len(valid_paths)

            # Get predictions
            probabilities = torch.softmax(logits, dim=1)
            confidences = probabilities[:, 1].cpu().numpy()
            predicted_classes = (confidences >= threshold).astype(int)

            # Create results
            for path, confidence, pred_class, logit in zip(
                valid_paths, confidences, predicted_classes, logits.cpu().numpy()
            ):
                results.append(EvaluationResult(
                    filename=path.name,
                    prediction="Positive" if pred_class == 1 else "Negative",
                    confidence=float(confidence),
                    latency_ms=batch_latency,
                    logits=logit
                ))

        return results

    def evaluate_dataset(
        self,
        dataset,
        threshold: float = 0.5,
        batch_size: int = 32
    ) -> Tuple[MetricResults, List[EvaluationResult]]:
        """
        Evaluate entire dataset with ground truth labels

        Args:
            dataset: PyTorch Dataset with ground truth
            threshold: Classification threshold
            batch_size: Batch size for processing

        Returns:
            Tuple of (MetricResults, List[EvaluationResult])
        """
        from torch.utils.data import DataLoader

        def collate_fn(batch):
            """Custom collate function to handle metadata"""
            features, labels, metadata_list = zip(*batch)

            # Stack features and labels
            features = torch.stack(features)
            labels = torch.tensor(labels)

            # Keep metadata as list of dicts
            return features, labels, list(metadata_list)

        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Single process for evaluation
            pin_memory=True,
            collate_fn=collate_fn
        )

        all_predictions = []
        all_targets = []
        all_logits = []
        results = []

        logger.info(f"Evaluating dataset with {len(dataset)} samples...")

        # Evaluate
        with torch.no_grad():
            for batch_idx, (inputs, targets, metadata) in enumerate(loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Inference
                start_time = time.time()

                with torch.cuda.amp.autocast():
                    logits = self.model(inputs)
                # Convert to float32 immediately after inference to ensure compatibility
                logits = logits.float()

                # FIX: Swap logits because model was trained with reversed class order
                logits = logits[:, [1, 0]]

                batch_latency = (time.time() - start_time) * 1000 / len(inputs)

                # Collect for metrics
                all_predictions.append(logits.cpu())
                all_targets.append(targets.cpu())
                all_logits.append(logits.cpu())

                # Create individual results
                probabilities = torch.softmax(logits, dim=1)
                confidences = probabilities[:, 1].cpu().numpy()
                predicted_classes = (confidences >= threshold).astype(int)

                for i, (confidence, pred_class, logit, meta) in enumerate(
                    zip(confidences, predicted_classes, logits.cpu().numpy(), metadata)
                ):
                    results.append(EvaluationResult(
                        filename=Path(meta['path']).name if 'path' in meta else f"sample_{batch_idx}_{i}",
                        prediction="Positive" if pred_class == 1 else "Negative",
                        confidence=float(confidence),
                        latency_ms=batch_latency,
                        logits=logit
                    ))

        # Calculate overall metrics
        all_preds = torch.cat(all_predictions, dim=0)
        all_targs = torch.cat(all_targets, dim=0)

        metrics = self.metrics_calculator.calculate(all_preds, all_targs, threshold=threshold)

        logger.info(f"Evaluation complete: {metrics}")

        return metrics, results

    def get_roc_curve_data(
        self,
        dataset,
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve data

        Args:
            dataset: PyTorch Dataset with ground truth
            batch_size: Batch size for processing

        Returns:
            Tuple of (fpr_array, tpr_array, thresholds)
        """
        from torch.utils.data import DataLoader

        def collate_fn(batch):
            """Custom collate function to handle metadata"""
            features, labels, metadata_list = zip(*batch)

            # Stack features and labels
            features = torch.stack(features)
            labels = torch.tensor(labels)

            # Keep metadata as list of dicts
            return features, labels, list(metadata_list)

        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )

        all_confidences = []
        all_targets = []

        # Collect predictions
        with torch.no_grad():
            for inputs, targets, _ in loader:
                inputs = inputs.to(self.device)

                with torch.cuda.amp.autocast():
                    logits = self.model(inputs)
                # Convert to float32 immediately after inference to ensure compatibility
                logits = logits.float()

                # FIX: Swap logits because model was trained with reversed class order
                logits = logits[:, [1, 0]]

                # Calculate probabilities
                probabilities = torch.softmax(logits, dim=1)
                confidences = probabilities[:, 1].cpu().numpy()

                all_confidences.extend(confidences)
                all_targets.extend(targets.cpu().numpy())

        all_confidences = np.array(all_confidences)
        all_targets = np.array(all_targets)

        # Calculate ROC curve
        thresholds = np.linspace(0, 1, 100)
        fpr_list = []
        tpr_list = []

        for threshold in thresholds:
            predictions = (all_confidences >= threshold).astype(int)

            tp = ((predictions == 1) & (all_targets == 1)).sum()
            tn = ((predictions == 0) & (all_targets == 0)).sum()
            fp = ((predictions == 1) & (all_targets == 0)).sum()
            fn = ((predictions == 0) & (all_targets == 1)).sum()

            # True Positive Rate (Recall)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # False Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            fpr_list.append(fpr)
            tpr_list.append(tpr)

        return np.array(fpr_list), np.array(tpr_list), thresholds

    def evaluate_with_advanced_metrics(
        self,
        dataset,
        total_seconds: float,
        target_fah: float = 1.0,
        batch_size: int = 32
    ) -> Dict:
        """
        Evaluate dataset with comprehensive production metrics

        Args:
            dataset: PyTorch Dataset with ground truth
            total_seconds: Total audio duration in dataset (for FAH calculation)
            target_fah: Target false alarms per hour for operating point
            batch_size: Batch size for processing

        Returns:
            Dictionary with comprehensive metrics including FAH, EER, pAUC
        """
        from torch.utils.data import DataLoader

        def collate_fn(batch):
            """Custom collate function to handle metadata"""
            features, labels, metadata_list = zip(*batch)
            features = torch.stack(features)
            labels = torch.tensor(labels)
            return features, labels, list(metadata_list)

        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )

        all_logits = []
        all_targets = []

        logger.info(f"Running comprehensive evaluation on {len(dataset)} samples...")

        # Collect all predictions
        with torch.no_grad():
            for inputs, targets, _ in loader:
                inputs = inputs.to(self.device)

                with torch.cuda.amp.autocast():
                    logits = self.model(inputs)
                # Convert to float32 immediately after inference to ensure compatibility
                logits = logits.float()

                # FIX: Swap logits because model was trained with reversed class order
                logits = logits[:, [1, 0]]

                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())

        # Concatenate all batches
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(
            logits=all_logits,
            labels=all_targets,
            total_seconds=total_seconds,
            target_fah=target_fah
        )

        logger.info("Advanced evaluation complete:")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  EER: {metrics['eer']:.4f}")
        logger.info(f"  pAUC (FPR≤0.1): {metrics['pauc_at_fpr_0.1']:.4f}")
        logger.info(f"  Operating Point:")
        logger.info(f"    Threshold: {metrics['operating_point']['threshold']:.4f}")
        logger.info(f"    TPR: {metrics['operating_point']['tpr']:.4f}")
        logger.info(f"    FPR: {metrics['operating_point']['fpr']:.4f}")
        logger.info(f"    FAH: {metrics['operating_point']['fah']:.2f}")

        return metrics


def load_model_for_evaluation(
    checkpoint_path: Path,
    device: str = 'cuda'
) -> Tuple[nn.Module, Dict]:
    """
    Load trained model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config_dict)
    """
    logger.info(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain configuration")

    config_data = checkpoint['config']

    # Convert config dict to WakewordConfig object if needed
    from src.config.defaults import WakewordConfig

    if isinstance(config_data, dict):
        # Config was saved as dict, need to reconstruct
        config = WakewordConfig.from_dict(config_data)
        logger.info("Converted config dict to WakewordConfig object")
    else:
        # Config is already a WakewordConfig object
        config = config_data

    # Create model
    from src.models.architectures import create_model

    model = create_model(
        architecture=config.model.architecture,
        num_classes=config.model.num_classes,
        pretrained=False,
        dropout=config.model.dropout
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move to device and set eval mode
    model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully: {config.model.architecture}")

    # Get additional info
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'val_loss': checkpoint.get('val_loss', 0.0),
        'val_metrics': checkpoint.get('val_metrics', {}),
        'config': config
    }

    return model, info


if __name__ == "__main__":
    # Test model loading and evaluation
    print("Model Evaluator Test")
    print("=" * 60)

    checkpoint_path = Path("models/checkpoints/best_model.pt")

    if checkpoint_path.exists():
        try:
            model, info = load_model_for_evaluation(checkpoint_path)
            print(f"✅ Model loaded: {info['config'].model.architecture}")
            print(f"   Epoch: {info['epoch']}")
            print(f"   Val Loss: {info['val_loss']:.4f}")

            # Create evaluator
            evaluator = ModelEvaluator(
                model=model,
                sample_rate=info['config'].data.sample_rate,
                audio_duration=info['config'].data.audio_duration,
                feature_type=info['config'].data.feature_type,
                n_mels=info['config'].data.n_mels,
                n_mfcc=info['config'].data.n_mfcc,
                n_fft=info['config'].data.n_fft,
                hop_length=info['config'].data.hop_length
            )

            print(f"✅ Evaluator created")
            print("\nEvaluator module loaded successfully")

        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"⚠️  No checkpoint found at: {checkpoint_path}")
        print("Train a model first (Panel 3)")

    print("\nEvaluation module ready")
