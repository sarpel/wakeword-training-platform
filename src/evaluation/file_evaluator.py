import time
from pathlib import Path
from typing import List

import numpy as np
import structlog
import torch

from src.evaluation.types import EvaluationResult

logger = structlog.get_logger(__name__)


def evaluate_file(
    evaluator, audio_path: Path, threshold: float = 0.5
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
    audio = evaluator.audio_processor.process_audio(audio_path)

    # Convert to tensor
    audio_tensor = torch.from_numpy(audio).float()

    # Extract features
    features = evaluator.feature_extractor(audio_tensor)

    # Add batch dimension
    features = features.unsqueeze(0).to(evaluator.device)

    # Inference
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            logits = evaluator.model(features)
        # Convert to float32 immediately after inference to ensure compatibility
        logits = logits.float()

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
        logits=logits.cpu().numpy()[0],
    )

    return result


def evaluate_files(
    evaluator, audio_paths: List[Path], threshold: float = 0.5, batch_size: int = 32
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
        batch_paths = audio_paths[i : i + batch_size]

        # Load batch
        batch_audio = []
        valid_paths = []

        for path in batch_paths:
            try:
                audio = evaluator.audio_processor.process_audio(path)
                batch_audio.append(audio)
                valid_paths.append(path)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                # Add error result
                results.append(
                    EvaluationResult(
                        filename=path.name,
                        prediction="Error",
                        confidence=0.0,
                        latency_ms=0.0,
                        logits=np.array([0.0, 0.0]),
                    )
                )

        if not batch_audio:
            continue

        # Convert to tensor and extract features
        batch_features = []
        for audio in batch_audio:
            audio_tensor = torch.from_numpy(audio).float()
            features = evaluator.feature_extractor(audio_tensor)
            batch_features.append(features)

        # Stack batch and move to device
        batch_tensor = torch.stack(batch_features).to(evaluator.device)

        # Batch inference
        start_time = time.time()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = evaluator.model(batch_tensor)
            # Convert to float32 immediately after inference to ensure compatibility
            logits = logits.float()

        batch_latency = (time.time() - start_time) * 1000 / len(valid_paths)

        # Get predictions
        probabilities = torch.softmax(logits, dim=1)
        confidences = probabilities[:, 1].cpu().numpy()
        predicted_classes = (confidences >= threshold).astype(int)

        # Create results
        for path, confidence, pred_class, logit in zip(
            valid_paths, confidences, predicted_classes, logits.cpu().numpy()
        ):
            results.append(
                EvaluationResult(
                    filename=path.name,
                    prediction="Positive" if pred_class == 1 else "Negative",
                    confidence=float(confidence),
                    latency_ms=batch_latency,
                    logits=logit,
                )
            )

    return results
