"""
Panel 4: Model Evaluation
- File-based evaluation with batch processing
- Real-time microphone testing
- Test set evaluation with comprehensive metrics
- Confusion matrix and ROC curve visualization
"""
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import structlog

from src.evaluation.evaluator import (
    ModelEvaluator,
    load_model_for_evaluation,
)
from src.evaluation.types import EvaluationResult
from src.evaluation.inference import MicrophoneInference, SimulatedMicrophoneInference
from src.data.dataset import WakewordDataset
from src.training.metrics import MetricResults
from src.exceptions import WakewordException

logger = structlog.get_logger(__name__)


class EvaluationState:
    """Global evaluation state manager"""

    def __init__(self):
        self.model = None
        self.model_info = None
        self.evaluator: Optional[ModelEvaluator] = None
        self.mic_inference: Optional[MicrophoneInference] = None
        self.is_mic_recording = False

        # File evaluation results
        self.file_results: List[EvaluationResult] = []

        # Test set results
        self.test_metrics = None
        self.test_results: List[EvaluationResult] = []

        # Microphone history
        self.mic_history = []

        self.waveform_fig = None
        self.waveform_ax = None
        self.waveform_line = None
        self.waveform_sr = 16000  # modelden set edilecek
        self.window_sec = 1.0  # ekranda gösterilecek süre


# Global state
eval_state = EvaluationState()


def get_available_models() -> List[str]:
    """
    Get list of available trained models

    Returns:
        List of model checkpoint paths
    """
    checkpoint_dir = Path("models/checkpoints")

    if not checkpoint_dir.exists():
        return ["No models available"]

    # Find all checkpoint files
    checkpoints = list(checkpoint_dir.glob("*.pt"))

    if not checkpoints:
        return ["No models available"]

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return [str(p) for p in checkpoints]


def load_model(model_path: str) -> str:
    """
    Load model for evaluation

    Args:
        model_path: Path to model checkpoint

    Returns:
        Status message
    """
    if model_path == "No models available":
        return "❌ No models available. Train a model first (Panel 3)."

    try:
        logger.info(f"Loading model: {model_path}")

        # Load model
        model, info = load_model_for_evaluation(Path(model_path), device="cuda")

        # Create evaluator
        evaluator = ModelEvaluator(
            model=model,
            sample_rate=info["config"].data.sample_rate,
            audio_duration=info["config"].data.audio_duration,
            device="cuda",
            feature_type=info["config"].data.feature_type,
            n_mels=info["config"].data.n_mels,
            n_mfcc=info["config"].data.n_mfcc,
            n_fft=info["config"].data.n_fft,
            hop_length=info["config"].data.hop_length,
        )

        # Update state
        eval_state.model = model
        eval_state.model_info = info
        eval_state.evaluator = evaluator
        eval_state.waveform_sr = info["config"].data.sample_rate

        # Format status message
        status = f"✅ Model Loaded Successfully\n"
        status += f"Architecture: {info['config'].model.architecture}\n"
        status += f"Training Epoch: {info['epoch'] + 1}\n"
        status += f"Val Loss: {info['val_loss']:.4f}\n"

        if "val_metrics" in info and info["val_metrics"]:
            metrics = info["val_metrics"]
            if isinstance(metrics, dict):
                status += f"Val Accuracy: {metrics.get('accuracy', 0) * 100:.2f}%\n"
                status += f"FPR: {metrics.get('fpr', 0) * 100:.2f}%\n"
                status += f"FNR: {metrics.get('fnr', 0) * 100:.2f}%"

        logger.info("Model loaded successfully")
        return status

    except WakewordException as e:
        error_msg = f"❌ Model Error: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        return f"{error_msg}\n\nActionable suggestion: Please check your model for the following error: {e}"
    except Exception as e:
        error_msg = f"❌ Failed to load model: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        return error_msg


def _ensure_waveform_fig():
    if (
        eval_state.waveform_fig is None
        or eval_state.waveform_ax is None
        or eval_state.waveform_line is None
    ):
        # Eski fig varsa kapat
        if eval_state.waveform_fig is not None:
            try:
                plt.close(eval_state.waveform_fig)
            except:
                pass
        fig, ax = plt.subplots(figsize=(10, 3))
        # Başlangıç boş veri
        x = np.linspace(
            0,
            eval_state.window_sec,
            int(eval_state.waveform_sr * eval_state.window_sec),
        )
        y = np.zeros_like(x)
        (line,) = ax.plot(x, y, linewidth=0.5)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.set_title("Live Audio Waveform", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-1.0, 1.0])
        ax.set_xlim([0, eval_state.window_sec])
        plt.tight_layout()
        eval_state.waveform_fig, eval_state.waveform_ax, eval_state.waveform_line = (
            fig,
            ax,
            line,
        )


def _update_waveform_plot(audio: np.ndarray):
    _ensure_waveform_fig()
    sr = eval_state.waveform_sr
    # Son window_sec kadarını göster
    max_len = int(sr * eval_state.window_sec)
    if audio.shape[0] > max_len:
        audio = audio[-max_len:]
    x = np.linspace(0, eval_state.window_sec, max_len)
    # Kısa gelirse sağa hizala
    y = np.zeros(max_len, dtype=audio.dtype)
    y[-len(audio) :] = audio
    eval_state.waveform_line.set_data(x, y)
    # Limitleri sabit tut. redraw için fig’i döndür.
    return eval_state.waveform_fig


def evaluate_uploaded_files(files: List, threshold: float) -> Tuple[pd.DataFrame, str]:
    """
    Evaluate uploaded audio files

    Args:
        files: List of uploaded file objects
        threshold: Detection threshold

    Returns:
        Tuple of (results_dataframe, log_message)
    """
    if eval_state.evaluator is None:
        return None, "❌ Please load a model first"

    if files is None or len(files) == 0:
        return None, "❌ Please upload audio files"

    try:
        logger.info(f"Evaluating {len(files)} files...")

        # Get file paths
        file_paths = [Path(f.name) for f in files]

        # Evaluate
        results = eval_state.evaluator.evaluate_files(
            file_paths, threshold=threshold, batch_size=32
        )

        # Store results
        eval_state.file_results = results

        # Convert to dataframe
        data = []
        for result in results:
            data.append(
                {
                    "Filename": result.filename,
                    "Prediction": result.prediction,
                    "Confidence": f"{result.confidence:.2%}",
                    "Latency (ms)": f"{result.latency_ms:.2f}",
                }
            )

        df = pd.DataFrame(data)

        # Calculate summary
        positive_count = sum(1 for r in results if r.prediction == "Positive")
        negative_count = len(results) - positive_count
        avg_confidence = np.mean([r.confidence for r in results])
        avg_latency = np.mean([r.latency_ms for r in results])

        log_msg = f"✅ Evaluation Complete\n"
        log_msg += f"Files evaluated: {len(results)}\n"
        log_msg += f"Positive detections: {positive_count}\n"
        log_msg += f"Negative detections: {negative_count}\n"
        log_msg += f"Avg confidence: {avg_confidence:.2%}\n"
        log_msg += f"Avg latency: {avg_latency:.2f} ms"

        logger.info(log_msg)

        return df, log_msg

    except WakewordException as e:
        error_msg = f"❌ Evaluation Error: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        return (
            None,
            f"{error_msg}\n\nActionable suggestion: Please check your audio files for the following error: {e}",
        )
    except Exception as e:
        error_msg = f"❌ Evaluation failed: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        return None, error_msg


def export_results_to_csv() -> str:
    """
    Export evaluation results to CSV

    Returns:
        Status message
    """
    if not eval_state.file_results:
        return "❌ No results to export"

    try:
        # Create exports directory
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)

        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.csv"
        filepath = export_dir / filename

        # Convert to dataframe and save
        data = []
        for result in eval_state.file_results:
            data.append(
                {
                    "filename": result.filename,
                    "prediction": result.prediction,
                    "confidence": result.confidence,
                    "latency_ms": result.latency_ms,
                }
            )

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

        logger.info(f"Results exported to: {filepath}")

        return f"✅ Results exported to: {filepath}"

    except WakewordException as e:
        error_msg = f"❌ Export Error: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Export failed: {str(e)}"
        logger.error(error_msg)
        return error_msg


def start_microphone():
    """Start microphone inference"""
    if eval_state.evaluator is None:
        return "❌ Please load a model first", 0.0, None, ""

    if eval_state.is_mic_recording:
        return "⚠️ Already recording", 0.0, None, ""

    try:
        logger.info("Starting microphone inference...")

        # Create microphone inference
        try:
            mic_inf = MicrophoneInference(
                model=eval_state.model,
                sample_rate=eval_state.model_info["config"].data.sample_rate,
                audio_duration=eval_state.model_info["config"].data.audio_duration,
                threshold=0.5,
                device="cuda",
                feature_type=eval_state.model_info["config"].data.feature_type,
                n_mels=eval_state.model_info["config"].data.n_mels,
                n_mfcc=eval_state.model_info["config"].data.n_mfcc,
                n_fft=eval_state.model_info["config"].data.n_fft,
                hop_length=eval_state.model_info["config"].data.hop_length,
            )
        except ImportError:
            # Fallback to simulated
            logger.warning("sounddevice not available, using simulated microphone")
            mic_inf = SimulatedMicrophoneInference(
                model=eval_state.model,
                sample_rate=eval_state.model_info["config"].data.sample_rate,
                threshold=0.5,
                device="cuda",
            )

        mic_inf.start()

        eval_state.mic_inference = mic_inf
        eval_state.is_mic_recording = True
        eval_state.mic_history = []

        return "🟢 Recording... Speak your wakeword!", 0.0, None, ""

    except WakewordException as e:
        error_msg = f"❌ Microphone Error: {str(e)}"
        logger.error(error_msg)
        return (
            error_msg,
            0.0,
            None,
            f"{error_msg}\n\nActionable suggestion: Please check your microphone for the following error: {e}",
        )
    except Exception as e:
        error_msg = f"❌ Failed to start microphone: {str(e)}"
        logger.error(error_msg)
        return error_msg, 0.0, None, ""


def stop_microphone():
    """Stop microphone inference"""
    if not eval_state.is_mic_recording:
        return "⚠️ Not recording", 0.0, None, ""

    try:
        logger.info("Stopping microphone inference...")

        if eval_state.mic_inference:
            eval_state.mic_inference.stop()

        eval_state.is_mic_recording = False

        # Get stats
        if eval_state.mic_inference:
            stats = eval_state.mic_inference.get_stats()
            history_msg = f"\nSession summary:\n"
            history_msg += f"Total detections: {stats['detection_count']}\n"
            history_msg += f"False alarms: {stats['false_alarm_count']}"

            eval_state.mic_history.append(history_msg)

        return "🔴 Not Detecting", 0.0, None, "\n".join(eval_state.mic_history)

    except WakewordException as e:
        error_msg = f"❌ Microphone Error: {str(e)}"
        logger.error(error_msg)
        return (
            error_msg,
            0.0,
            None,
            f"{error_msg}\n\nActionable suggestion: Please check your microphone for the following error: {e}",
        )
    except Exception as e:
        error_msg = f"❌ Failed to stop microphone: {str(e)}"
        logger.error(error_msg)
        return error_msg, 0.0, None, ""


def get_microphone_status() -> Tuple:
    """Get current microphone status for live updates"""
    if not eval_state.is_mic_recording or eval_state.mic_inference is None:
        return "🔴 Not Detecting", 0.0, None, "\n".join(eval_state.mic_history)

    try:
        # Get latest result
        result = eval_state.mic_inference.get_latest_result()

        if result is not None:
            confidence, is_positive, audio_chunk = result

            # Update history
            timestamp = time.strftime("%H:%M:%S")
            if is_positive:
                msg = f"[{timestamp}] ✅ WAKEWORD DETECTED! Confidence: {confidence:.2%}"
                status = f"🟢 WAKEWORD DETECTED! ({confidence:.2%})"
            else:
                msg = f"[{timestamp}] Listening... ({confidence:.2%})"
                status = "🟢 Recording... Speak your wakeword!"

            eval_state.mic_history.append(msg)

            # Keep only last 50 messages
            if len(eval_state.mic_history) > 50:
                eval_state.mic_history = eval_state.mic_history[-50:]

            # Create waveform plot
            fig = _update_waveform_plot(audio_chunk)

            return (
                status,
                round(confidence * 100, 2),
                fig,
                "\n".join(eval_state.mic_history),
            )

        return (
            "🟢 Recording... Speak your wakeword!",
            0.0,
            None,
            "\n".join(eval_state.mic_history),
        )

    except Exception as e:
        logger.error(f"Microphone status error: {e}")
        return "🔴 Error", 0.0, None, str(e)


# def create_waveform_plot(audio: np.ndarray) -> plt.Figure:
# """Create waveform visualization"""
# fig, ax = plt.subplots(figsize=(10, 3))

# time_axis = np.arange(len(audio)) / 16000  # Assuming 16kHz
# ax.plot(time_axis, audio, linewidth=0.5, color='blue')
# ax.set_xlabel('Time (s)', fontsize=10)
# ax.set_ylabel('Amplitude', fontsize=10)
# ax.set_title('Live Audio Waveform', fontsize=12, fontweight='bold')
# ax.grid(True, alpha=0.3)
# ax.set_ylim([-1.0, 1.0])

# plt.tight_layout()
# return fig


def evaluate_test_set(
    data_root: str,
    test_split_path: str,
    threshold: float,
    target_fah: float,
    use_advanced_metrics: bool,
) -> Tuple:
    """
    Evaluate on test dataset

    Args:
        test_split_path: Path to test split
        threshold: Detection threshold
        target_fah: Target false alarms per hour
        use_advanced_metrics: Whether to compute advanced metrics (FAH, EER, pAUC)

    Returns:
        Tuple of (metrics_dict, confusion_matrix_plot, roc_plot, advanced_metrics_dict)
    """
    if eval_state.evaluator is None:
        return {"status": "❌ Please load a model first"}, None, None, {}

    try:
        # Default to data/splits/test.json if not provided
        if not test_split_path or test_split_path.strip() == "":
            test_split_path = str(Path(data_root) / "splits" / "test.json")

        test_path = Path(test_split_path)

        if not test_path.exists():
            return (
                {"status": f"❌ Test split not found: {test_split_path}"},
                None,
                None,
                {},
            )

        logger.info(f"Evaluating test set: {test_path}")

        # Load test dataset
        test_dataset = WakewordDataset(
            manifest_path=test_path,
            sample_rate=eval_state.model_info["config"].data.sample_rate,
            audio_duration=eval_state.model_info["config"].data.audio_duration,
            augment=False,
            device="cuda",
            feature_type=eval_state.model_info["config"].data.feature_type,
            n_mels=eval_state.model_info["config"].data.n_mels,
            n_mfcc=eval_state.model_info["config"].data.n_mfcc,
            n_fft=eval_state.model_info["config"].data.n_fft,
            hop_length=eval_state.model_info["config"].data.hop_length,
        )

        logger.info(f"Loaded {len(test_dataset)} test samples")

        # Evaluate with basic metrics
        metrics, results = eval_state.evaluator.evaluate_dataset(
            test_dataset, threshold=threshold, batch_size=32
        )

        # Store results
        eval_state.test_metrics = metrics
        eval_state.test_results = results

        # Create basic metrics dict
        metrics_dict = {
            "Accuracy": f"{metrics.accuracy:.2%}",
            "Precision": f"{metrics.precision:.2%}",
            "Recall": f"{metrics.recall:.2%}",
            "F1 Score": f"{metrics.f1_score:.2%}",
            "False Positive Rate (FPR)": f"{metrics.fpr:.2%}",
            "False Negative Rate (FNR)": f"{metrics.fnr:.2%}",
            "---": "---",
            "True Positives": str(metrics.true_positives),
            "True Negatives": str(metrics.true_negatives),
            "False Positives": str(metrics.false_positives),
            "False Negatives": str(metrics.false_negatives),
            "Total Samples": str(metrics.total_samples),
        }

        # Compute advanced metrics if enabled
        advanced_metrics_dict = {}
        if use_advanced_metrics:
            logger.info("Computing advanced production metrics...")
            total_seconds = (
                len(test_dataset) * eval_state.model_info["config"].data.audio_duration
            )

            advanced_metrics = eval_state.evaluator.evaluate_with_advanced_metrics(
                dataset=test_dataset,
                total_seconds=total_seconds,
                target_fah=target_fah,
                batch_size=32,
            )

            # Format advanced metrics for display
            advanced_metrics_dict = {
                "📊 Advanced Metrics": "---",
                "ROC-AUC": f"{advanced_metrics['roc_auc']:.4f}",
                "EER (Equal Error Rate)": f"{advanced_metrics['eer']:.4f}",
                "EER Threshold": f"{advanced_metrics['eer_threshold']:.4f}",
                "pAUC (FPR≤0.1)": f"{advanced_metrics['pauc_at_fpr_0.1']:.4f}",
                "": "---",
                "🎯 Operating Point (Target FAH)": "---",
                "Target FAH": f"{target_fah:.1f} per hour",
                "Achieved FAH": f"{advanced_metrics['operating_point']['fah']:.2f} per hour",
                "Threshold": f"{advanced_metrics['operating_point']['threshold']:.4f}",
                "True Positive Rate (TPR)": f"{advanced_metrics['operating_point']['tpr']:.2%}",
                "False Positive Rate (FPR)": f"{advanced_metrics['operating_point']['fpr']:.4%}",
                "Precision": f"{advanced_metrics['operating_point']['precision']:.2%}",
                "F1 Score": f"{advanced_metrics['operating_point']['f1_score']:.2%}",
            }

        # Create confusion matrix plot
        conf_matrix_plot = create_confusion_matrix_plot(metrics)

        # Create ROC curve
        logger.info("Calculating ROC curve...")
        roc_plot = create_roc_curve_plot(test_dataset)

        logger.info("Test set evaluation complete")

        return metrics_dict, conf_matrix_plot, roc_plot, advanced_metrics_dict

    except WakewordException as e:
        error_msg = f"❌ Test Set Error: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        return (
            {
                "status": f"{error_msg}\n\nActionable suggestion: Please check your test set for the following error: {e}"
            },
            None,
            None,
            {},
        )
    except Exception as e:
        error_msg = f"❌ Test set evaluation failed: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        return {"status": error_msg}, None, None, {}


def create_confusion_matrix_plot(metrics: "MetricResults") -> plt.Figure:
    """Create confusion matrix visualization"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Confusion matrix values
    cm = np.array(
        [
            [metrics.true_negatives, metrics.false_positives],
            [metrics.false_negatives, metrics.true_positives],
        ]
    )

    # Plot
    im = ax.imshow(cm, cmap="Blues")

    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="black",
                fontsize=16,
            )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", fontsize=10)

    plt.tight_layout()
    return fig


def create_roc_curve_plot(test_dataset) -> plt.Figure:
    """Create ROC curve visualization"""
    try:
        # Get ROC curve data
        fpr_array, tpr_array, thresholds = eval_state.evaluator.get_roc_curve_data(
            test_dataset, batch_size=32
        )

        # Remove duplicate points for cleaner curve
        unique_fpr = []
        unique_tpr = []
        for fpr, tpr in zip(fpr_array, tpr_array):
            if not unique_fpr or (fpr != unique_fpr[-1] or tpr != unique_tpr[-1]):
                unique_fpr.append(fpr)
                unique_tpr.append(tpr)

        unique_fpr = np.array(unique_fpr)
        unique_tpr = np.array(unique_tpr)

        # Calculate AUC
        if len(unique_fpr) >= 2:
            # Use trapezoidal rule for AUC
            auc = np.trapz(unique_tpr, unique_fpr)
        else:
            # Fallback: estimate AUC based on single point
            # If model predicts perfectly, AUC = 1.0
            if len(unique_tpr) == 1:
                auc = unique_tpr[0] * (1.0 - unique_fpr[0])
            else:
                auc = 0.5  # Random baseline

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        if len(unique_fpr) >= 2:
            ax.plot(
                unique_fpr,
                unique_tpr,
                linewidth=2,
                label=f"ROC Curve (AUC = {auc:.3f})",
            )
        else:
            # Plot single point
            ax.scatter(
                unique_fpr,
                unique_tpr,
                s=100,
                c="blue",
                zorder=3,
                label=f"Operating Point (AUC ≈ {auc:.3f})",
            )
            # Draw connecting lines to show perfect classifier
            ax.plot(
                [0, unique_fpr[0], unique_fpr[0], 1],
                [0, unique_tpr[0], 1, 1],
                "b--",
                linewidth=1,
                alpha=0.5,
                label="Implied ROC Curve",
            )

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"ROC curve generation failed: {e}")
        logger.exception(e)
        # Return empty plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            f"ROC curve generation failed:\n{str(e)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig


def create_evaluation_panel(state: gr.State) -> gr.Blocks:
    """
    Create Panel 4: Model Evaluation

    Args:
        state: Gradio State containing shared application data (config, etc.)

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks() as panel:
        gr.Markdown("# 🎯 Model Evaluation")
        gr.Markdown(
            "Test your trained model with audio files, live microphone, or test dataset."
        )

        with gr.Row():
            model_selector = gr.Dropdown(
                choices=get_available_models(),
                label="Select Trained Model",
                info="Choose a checkpoint to evaluate",
                value=get_available_models()[0]
                if get_available_models()[0] != "No models available"
                else None,
            )
            refresh_models_btn = gr.Button("🔄 Refresh", scale=0)
            load_model_btn = gr.Button("📥 Load Model", variant="primary", scale=1)

        model_status = gr.Textbox(
            label="Model Status",
            value="No model loaded. Select a model and click Load.",
            lines=6,
            interactive=False,
        )

        gr.Markdown("---")

        with gr.Tabs():
            # File-based evaluation
            with gr.TabItem("📁 File Evaluation"):
                gr.Markdown("### Upload Audio Files for Batch Evaluation")

                with gr.Row():
                    with gr.Column():
                        audio_files = gr.File(
                            label="Upload Audio Files (.wav, .mp3, .flac)",
                            file_count="multiple",
                            file_types=[".wav", ".mp3", ".flac", ".ogg"],
                        )

                        threshold_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.05,
                            label="Detection Threshold",
                            info="Confidence threshold for positive detection",
                        )

                        with gr.Row():
                            evaluate_files_btn = gr.Button(
                                "🔍 Evaluate Files", variant="primary", scale=2
                            )
                            export_results_btn = gr.Button("💾 Export CSV", scale=1)

                    with gr.Column():
                        gr.Markdown("### Results")
                        results_table = gr.Dataframe(
                            headers=[
                                "Filename",
                                "Prediction",
                                "Confidence",
                                "Latency (ms)",
                            ],
                            label="Evaluation Results",
                            interactive=False,
                        )

                with gr.Row():
                    evaluation_log = gr.Textbox(
                        label="Evaluation Summary",
                        lines=6,
                        value="Ready to evaluate files...",
                        interactive=False,
                    )

            # Microphone testing
            with gr.TabItem("🎤 Live Microphone Test"):
                gr.Markdown("### Real-Time Wakeword Detection")
                gr.Markdown(
                    "**Note**: Requires microphone access and `sounddevice` package."
                )

                with gr.Row():
                    with gr.Column():
                        sensitivity_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.05,
                            label="Detection Sensitivity (Threshold)",
                        )

                        with gr.Row():
                            start_mic_btn = gr.Button(
                                "🎙️ Start Recording", variant="primary", scale=2
                            )
                            stop_mic_btn = gr.Button(
                                "⏹️ Stop Recording", variant="stop", scale=1
                            )

                    with gr.Column():
                        gr.Markdown("### Detection Status")

                        detection_indicator = gr.Textbox(
                            label="Status",
                            value="🔴 Not Detecting",
                            lines=2,
                            interactive=False,
                        )

                        confidence_display = gr.Number(
                            label="Confidence (%)", value=0.0, interactive=False
                        )

                        waveform_plot = gr.Plot(label="Live Waveform")

                with gr.Row():
                    detection_history = gr.Textbox(
                        label="Detection History",
                        lines=10,
                        value="Start recording to see detections...\n",
                        interactive=False,
                        autoscroll=True,
                    )

            # Test set evaluation
            with gr.TabItem("📊 Test Set Evaluation"):
                gr.Markdown("### Evaluate on Test Dataset with Comprehensive Metrics")

                with gr.Row():
                    test_split_path = gr.Textbox(
                        label="Test Split Path",
                        placeholder="data/splits/test.json (default)",
                        value="data/splits/test.json",
                        lines=1,
                    )

                    test_threshold_slider = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.5,
                        step=0.05,
                        label="Detection Threshold",
                    )

                with gr.Row():
                    with gr.Column():
                        use_advanced_metrics = gr.Checkbox(
                            label="📊 Enable Advanced Production Metrics",
                            value=True,
                            info="Compute FAH, EER, pAUC, and optimal operating point",
                        )

                    with gr.Column():
                        target_fah_slider = gr.Slider(
                            minimum=0.1,
                            maximum=5.0,
                            value=1.0,
                            step=0.1,
                            label="Target FAH (False Alarms per Hour)",
                            info="Desired false alarm rate for production threshold",
                        )

                evaluate_testset_btn = gr.Button(
                    "📈 Run Test Evaluation", variant="primary"
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Basic Metrics")
                        test_metrics = gr.JSON(
                            label="Test Set Metrics",
                            value={"status": "Click 'Run Test Evaluation' to start"},
                        )

                    with gr.Column():
                        gr.Markdown("### Confusion Matrix")
                        confusion_matrix = gr.Plot(label="Confusion Matrix")

                with gr.Row():
                    roc_curve = gr.Plot(
                        label="ROC Curve (Receiver Operating Characteristic)"
                    )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎯 Advanced Production Metrics")
                        advanced_metrics = gr.JSON(
                            label="Production Metrics (FAH, EER, pAUC)",
                            value={
                                "status": "Enable advanced metrics and run evaluation"
                            },
                        )

        # Event handlers
        def refresh_models_handler():
            models = get_available_models()
            return gr.update(
                choices=models,
                value=models[0] if models[0] != "No models available" else None,
            )

        refresh_models_btn.click(fn=refresh_models_handler, outputs=[model_selector])

        load_model_btn.click(
            fn=load_model, inputs=[model_selector], outputs=[model_status]
        )

        evaluate_files_btn.click(
            fn=evaluate_uploaded_files,
            inputs=[audio_files, threshold_slider],
            outputs=[results_table, evaluation_log],
        )

        export_results_btn.click(fn=export_results_to_csv, outputs=[evaluation_log])

        start_mic_btn.click(
            fn=start_microphone,
            outputs=[
                detection_indicator,
                confidence_display,
                waveform_plot,
                detection_history,
            ],
        )

        stop_mic_btn.click(
            fn=stop_microphone,
            outputs=[
                detection_indicator,
                confidence_display,
                waveform_plot,
                detection_history,
            ],
        )

        # Auto-refresh for microphone updates
        mic_refresh = gr.Timer(value=0.5, active=True)  # Update every 0.5 seconds

        mic_refresh.tick(
            fn=get_microphone_status,
            outputs=[
                detection_indicator,
                confidence_display,
                waveform_plot,
                detection_history,
            ],
        )

        evaluate_testset_btn.click(
            fn=lambda test_split_path, threshold, target_fah, use_advanced_metrics, config_state: evaluate_test_set(
                config_state.get("config").data.data_root
                if config_state.get("config")
                else "data",
                test_split_path,
                threshold,
                target_fah,
                use_advanced_metrics,
            ),
            inputs=[
                test_split_path,
                test_threshold_slider,
                target_fah_slider,
                use_advanced_metrics,
                state,
            ],
            outputs=[test_metrics, confusion_matrix, roc_curve, advanced_metrics],
        )

    return panel


if __name__ == "__main__":
    # Test the panel
    demo_state = gr.State(value={"config": None})
    demo = create_evaluation_panel(demo_state)
    demo.launch()
