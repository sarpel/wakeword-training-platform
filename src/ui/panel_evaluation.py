"""
Panel 4: Model Evaluation
- File-based evaluation with batch processing
- Real-time microphone testing
- Test set evaluation with comprehensive metrics
- Confusion matrix and ROC curve visualization
"""


import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

matplotlib.use("Agg")
import structlog

from src.config.cuda_utils import get_cuda_validator
from src.data.dataset import WakewordDataset
from src.evaluation.advanced_evaluator import ThresholdAnalyzer
from src.evaluation.benchmarking import BenchmarkRunner
from src.evaluation.data_collector import FalsePositiveCollector
from src.evaluation.evaluator import ModelEvaluator, load_model_for_evaluation
from src.evaluation.inference import MicrophoneInference, SimulatedMicrophoneInference
from src.evaluation.mining import HardNegativeMiner
from src.evaluation.stages import SentryInferenceStage
from src.evaluation.types import EvaluationResult
from src.exceptions import WakewordException
from src.training.metrics import MetricResults

logger = structlog.get_logger(__name__)


class EvaluationState:
    """Global evaluation state manager"""

    def __init__(self) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.model_info: Optional[Dict[str, Any]] = None
        self.evaluator: Optional[ModelEvaluator] = None
        self.mic_inference: Optional[Union[MicrophoneInference, SimulatedMicrophoneInference]] = None
        self.is_mic_recording = False

        # Detect device
        validator = get_cuda_validator(allow_cpu=True)
        self.device = "cuda" if validator.validate()[0] and validator.cuda_available else "cpu"
        logger.info(f"Evaluation Panel initialized on {self.device}")

        # File evaluation results
        self.file_results: List[EvaluationResult] = []

        # Test set results
        self.test_metrics: Optional[MetricResults] = None
        self.test_results: List[EvaluationResult] = []

        # Microphone history
        self.mic_history: List[str] = []

        self.waveform_fig: Optional[Any] = None
        self.waveform_ax: Optional[Any] = None
        self.waveform_line: Optional[Any] = None
        self.waveform_sr = 16000  # modelden set edilecek
        self.window_sec = 1.0  # ekranda gösterilecek süre

        # Analysis Data
        self.last_logits: Optional[torch.Tensor] = None
        self.last_labels: Optional[torch.Tensor] = None
        self.threshold_analyzer: Optional[ThresholdAnalyzer] = None
        self.fp_collector = FalsePositiveCollector()
        self.miner = HardNegativeMiner()


# Global state
eval_state = EvaluationState()


def get_available_models() -> List[str]:
    """
    Get list of available trained models
    """
    checkpoint_dir = Path("models/checkpoints")
    if not checkpoint_dir.exists():
        return ["No models available"]
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return ["No models available"]
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in checkpoints]


def load_model(model_path: str) -> str:
    """
    Load model for evaluation
    """
    if model_path == "No models available":
        return "❌ No models available. Train a model first (Panel 3)."

    try:
        logger.info(f"Loading model: {model_path}")
        model, info = load_model_for_evaluation(Path(model_path), device=eval_state.device)
        evaluator = ModelEvaluator(
            model=model,
            sample_rate=info["config"].data.sample_rate,
            audio_duration=info["config"].data.audio_duration,
            device=eval_state.device,
            feature_type=info["config"].data.feature_type,
            n_mels=info["config"].data.n_mels,
            n_mfcc=info["config"].data.n_mfcc,
            n_fft=info["config"].data.n_fft,
            hop_length=info["config"].data.hop_length,
            config=info["config"],
        )
        if hasattr(evaluator, "audio_processor"):
            evaluator.audio_processor.eval()
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

        return status
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return f"❌ Failed to load model: {str(e)}"


def _ensure_waveform_fig() -> None:
    if eval_state.waveform_fig is None:
        fig, ax = plt.subplots(figsize=(10, 3))
        x = np.linspace(0, eval_state.window_sec, int(eval_state.waveform_sr * eval_state.window_sec))
        y = np.zeros_like(x)
        (line,) = ax.plot(x, y, linewidth=0.5)
        ax.set_ylim([-1.0, 1.0])
        eval_state.waveform_fig, eval_state.waveform_ax, eval_state.waveform_line = fig, ax, line


def _update_waveform_plot(audio: np.ndarray) -> Any:
    _ensure_waveform_fig()
    max_len = int(eval_state.waveform_sr * eval_state.window_sec)
    if audio.shape[0] > max_len:
        audio = audio[-max_len:]
    y = np.zeros(max_len, dtype=audio.dtype)
    y[-len(audio) :] = audio
    if eval_state.waveform_line:
        eval_state.waveform_line.set_ydata(y)
    return eval_state.waveform_fig


def evaluate_uploaded_files(files: List, threshold: float) -> Tuple[pd.DataFrame, str]:
    if eval_state.evaluator is None:
        return None, "❌ Please load a model first"
    try:
        results = eval_state.evaluator.evaluate_files([Path(f.name) for f in files], threshold=threshold, batch_size=32)
        eval_state.file_results = results
        data = [
            {"Filename": r.filename, "Prediction": r.prediction, "Confidence": f"{r.confidence:.2%}"} for r in results
        ]
        return pd.DataFrame(data), f"✅ Evaluation Complete. {len(results)} files evaluated."
    except Exception as e:
        return None, str(e)


def export_results_to_csv() -> str:
    if not eval_state.file_results:
        return "❌ No results to export"
    try:
        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)
        filename = f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame([r.__dict__ for r in eval_state.file_results]).to_csv(export_dir / filename)
        return f"✅ Results exported to: {export_dir / filename}"
    except Exception as e:
        return str(e)


def start_microphone() -> Tuple[str, float, Optional[Any], str]:
    if eval_state.evaluator is None:
        return "❌ Please load a model first", 0.0, None, ""
    try:
        mic_inf = MicrophoneInference(
            model=eval_state.model,
            sample_rate=eval_state.waveform_sr,
            audio_duration=eval_state.model_info["config"].data.audio_duration,
            threshold=0.5,
            device=eval_state.device,
        )
        mic_inf.start()
        eval_state.mic_inference = mic_inf
        eval_state.is_mic_recording = True
        eval_state.mic_history = []
        return "🟢 Recording...", 0.0, None, ""
    except Exception as e:
        return str(e), 0.0, None, ""


def stop_microphone() -> Tuple[str, float, Optional[Any], str]:
    if not eval_state.is_mic_recording:
        return "⚠️ Not recording", 0.0, None, ""
    if eval_state.mic_inference:
        eval_state.mic_inference.stop()
    eval_state.is_mic_recording = False
    return "🔴 Stopped", 0.0, None, "\n".join(eval_state.mic_history)


def get_microphone_status() -> Tuple:
    if not eval_state.is_mic_recording or eval_state.mic_inference is None:
        return "🔴 Not Detecting", 0.0, None, "\n".join(eval_state.mic_history)
    result = eval_state.mic_inference.get_latest_result()
    if result:
        conf, pos, chunk = result
        status = "✅ DETECTED!" if pos else "🟢 Listening..."
        eval_state.mic_history.append(f"[{time.strftime('%H:%M:%S')}] {status} ({conf:.2%})")
        return status, round(conf * 100, 2), _update_waveform_plot(chunk), "\n".join(eval_state.mic_history[-50:])
    return "🟢 Listening...", 0.0, None, "\n".join(eval_state.mic_history)


def run_threshold_analysis() -> Tuple[gr.Plot, pd.DataFrame]:
    if eval_state.threshold_analyzer is None:
        return None, None
    results = eval_state.threshold_analyzer.analyze_range(np.linspace(0, 1, 21))
    df = pd.DataFrame(results)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["threshold"], y=df["precision"], name="Precision", line=dict(color="cyan")))
    fig.add_trace(go.Scatter(x=df["threshold"], y=df["recall"], name="Recall", line=dict(color="orange")))
    fig.update_layout(title="PR vs Threshold", template="plotly_dark", height=400)
    return fig, df


def run_benchmark_test(num_iterations: int = 10) -> Dict[str, Any]:
    if eval_state.model is None:
        return {"error": "No model loaded"}
    try:
        stage = SentryInferenceStage(
            model=eval_state.model, name=eval_state.model_info["config"].model.architecture, device=eval_state.device
        )
        runner = BenchmarkRunner(stage)
        audio = np.random.randn(
            int(eval_state.waveform_sr * eval_state.model_info["config"].data.audio_duration)
        ).astype(np.float32)
        metrics = runner.run_benchmark(audio, num_iterations=num_iterations)
        return {
            "Model": metrics["name"],
            "Mean Latency": f"{metrics['mean_latency_ms']:.2f} ms",
            "RAM Usage": f"{metrics['process_memory_mb']:.2f} MB",
            "GPU Usage": f"{metrics.get('gpu_memory_allocated_mb', 0):.2f} MB",
        }
    except Exception as e:
        return {"error": str(e)}


def collect_false_positives() -> str:
    if not eval_state.test_results or eval_state.last_labels is None:
        return "Run evaluation first."
    eval_state.fp_collector.clear()
    for r, l in zip(eval_state.test_results, eval_state.last_labels):
        if r.prediction == "Positive" and l == 0:
            eval_state.fp_collector.add_sample(r.raw_audio, {"filename": r.filename, "confidence": r.confidence})
    return generate_fp_gallery_html()


def generate_fp_gallery_html() -> str:
    samples = eval_state.fp_collector.get_samples()
    if not samples:
        return "<p>No samples.</p>"
    html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px;">'
    for s in samples:
        audio_url = f"file/{eval_state.fp_collector.output_dir}/{s['audio_path']}"
        html += f'<div style="background: #2d2d2d; padding: 10px;"><p>File: {s["metadata"]["filename"]}</p><audio src="{audio_url}" controls></audio></div>'
    return html + "</div>"


def clear_false_positives() -> str:
    eval_state.fp_collector.clear()
    return "<p>Cleared.</p>"


def mine_hard_negatives_handler() -> str:
    if not eval_state.test_results:
        return "❌ Run test set evaluation first."

    count = eval_state.miner.mine_from_results(eval_state.test_results)
    return f"✅ Mined {count} new potential hard negatives. Check the 'Mining Queue' tab."


def get_mining_gallery_html() -> str:
    pending = eval_state.miner.get_pending()
    if not pending:
        return "<p style='text-align: center; padding: 20px;'>Queue is empty. Use the 'Mine Hard Negatives' button in evaluation results.</p>"

    html = '<div style="display: flex; flex-direction: column; gap: 15px; padding: 10px;">'
    for item in pending:
        # We assume files are accessible via Gradio or absolute path (if local)
        # For simplicity, we just display the info and path for now
        html += f"""
        <div style="background: #2d2d2d; border-radius: 8px; padding: 15px; border: 1px solid #444; display: flex; justify-content: space-between; align-items: center;">
            <div style="flex: 1;">
                <p style="margin: 0; font-weight: bold;">{item["filename"]}</p>
                <p style="margin: 5px 0; font-size: 0.85em; color: #aaa;">Confidence: {item["confidence"]:.2%}</p>
                <p style="margin: 0; font-size: 0.7em; color: #888;">Path: {item["full_path"]}</p>
            </div>
            <div style="display: flex; gap: 10px;">
                <button onclick="confirmSample('{item["full_path"]}')" style="background: #28a745; color: white; border: none; padding: 8px 15px; border-radius: 4px; cursor: pointer;">✅ Confirm</button>
                <button onclick="discardSample('{item["full_path"]}')" style="background: #dc3545; color: white; border: none; padding: 8px 15px; border-radius: 4px; cursor: pointer;">❌ Discard</button>
            </div>
        </div>
        """
    html += "</div>"

    # Add JS for buttons
    html += """
    <script>
    function confirmSample(path) {
        let pathEl = document.querySelector("#mining_verify_path input");
        let statusEl = document.querySelector("#mining_verify_status input");
        pathEl.value = path;
        pathEl.dispatchEvent(new Event('input', { bubbles: true }));
        statusEl.value = "confirmed";
        statusEl.dispatchEvent(new Event('input', { bubbles: true }));
    }
    function discardSample(path) {
        let pathEl = document.querySelector("#mining_verify_path input");
        let statusEl = document.querySelector("#mining_verify_status input");
        pathEl.value = path;
        pathEl.dispatchEvent(new Event('input', { bubbles: true }));
        statusEl.value = "discarded";
        statusEl.dispatchEvent(new Event('input', { bubbles: true }));
    }
    </script>
    """
    return html


def verify_sample_handler(path: str, status: str) -> str:
    eval_state.miner.update_status(path, status)
    return get_mining_gallery_html()


def inject_mined_samples_handler() -> str:
    count = eval_state.miner.inject_to_dataset()
    return f"✅ Injected {count} confirmed negatives into training data."


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
    if eval_state.evaluator is None or eval_state.model_info is None:
        return {"status": "❌ Please load a model first"}, None, None, {}

    try:
        # Default to data/splits/test.json if not provided
        from src.config.paths import paths

        if not test_split_path or test_split_path.strip() == "":
            test_split_path = str(paths.SPLITS / "test.json")

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
        # We rely on GpuAudioProcessor (initialized in ModelEvaluator) to handle CMVN via centralized paths
        # So we don't need to pass cmvn_path here for GpuAudioProcessor, BUT WakewordDataset might need it
        # if we weren't using return_raw_audio=True. Since we are, it's fine.

        # However, to be safe and consistent with training, we pass it.
        cmvn_path = paths.CMVN_STATS

        test_dataset = WakewordDataset(
            manifest_path=test_path,
            sample_rate=eval_state.model_info["config"].data.sample_rate,
            audio_duration=eval_state.model_info["config"].data.audio_duration,
            augment=False,
            device=eval_state.device,
            feature_type=eval_state.model_info["config"].data.feature_type,
            n_mels=eval_state.model_info["config"].data.n_mels,
            n_mfcc=eval_state.model_info["config"].data.n_mfcc,
            n_fft=eval_state.model_info["config"].data.n_fft,
            hop_length=eval_state.model_info["config"].data.hop_length,
            use_precomputed_features_for_training=eval_state.model_info[
                "config"
            ].data.use_precomputed_features_for_training,
            npy_cache_features=eval_state.model_info["config"].data.npy_cache_features,
            fallback_to_audio=True,
            cmvn_path=cmvn_path,
            apply_cmvn=True if cmvn_path.exists() else False,
            return_raw_audio=True,
        )

        logger.info(f"Loaded {len(test_dataset)} test samples")

        # Evaluate with basic metrics
        metrics, results = eval_state.evaluator.evaluate_dataset(test_dataset, threshold=threshold, batch_size=32)

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
            sample_duration = eval_state.model_info["config"].data.audio_duration

            advanced_metrics = eval_state.evaluator.evaluate_with_advanced_metrics(
                dataset=test_dataset,
                total_seconds=sample_duration,
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

        # Update analysis state
        all_preds = torch.tensor(np.stack([r.logits for r in results]))
        all_targs = torch.tensor(np.array([r.label for r in results]))
        eval_state.last_logits = all_preds
        eval_state.last_labels = all_targs
        eval_state.threshold_analyzer = ThresholdAnalyzer(all_preds, all_targs)

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


def create_roc_curve_plot(test_dataset: WakewordDataset) -> plt.Figure:
    """Create ROC curve visualization"""
    try:
        if eval_state.evaluator is None:
            return plt.figure()

        # Get ROC curve data
        fpr_array, tpr_array, thresholds = eval_state.evaluator.get_roc_curve_data(test_dataset, batch_size=32)

        # Remove duplicate points for cleaner curve
        unique_fpr: List[float] = []
        unique_tpr: List[float] = []
        for fpr, tpr in zip(fpr_array, tpr_array):
            if not unique_fpr or (fpr != unique_fpr[-1] or tpr != unique_tpr[-1]):
                unique_fpr.append(fpr)
                unique_tpr.append(tpr)

        unique_fpr_arr = np.array(unique_fpr)
        unique_tpr_arr = np.array(unique_tpr)

        # Calculate AUC
        if len(unique_fpr_arr) >= 2:
            auc = np.trapz(unique_tpr_arr, unique_fpr_arr)
        else:
            auc = 0.5

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
            ax.scatter(
                unique_fpr,
                unique_tpr,
                s=100,
                c="blue",
                zorder=3,
                label=f"Operating Point (AUC ≈ {auc:.3f})",
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
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"ROC curve generation failed:\n{str(e)}", ha="center", va="center", transform=ax.transAxes)
        return fig


def create_evaluation_panel(state: gr.State) -> gr.Blocks:
    """
    Create Panel 4: Model Evaluation
    """
    with gr.Blocks() as panel:
        gr.Markdown("# 🎯 Model Evaluation")
        gr.Markdown("Test your trained model with audio files, live microphone, or test dataset.")

        with gr.Row():
            model_selector = gr.Dropdown(
                choices=get_available_models(),
                label="Select Trained Model",
                info="Choose a checkpoint to evaluate",
                value=get_available_models()[0] if get_available_models()[0] != "No models available" else None,
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
                            evaluate_files_btn = gr.Button("🔍 Evaluate Files", variant="primary", scale=2)
                            export_results_btn = gr.Button("💾 Export CSV", scale=1)

                    with gr.Column():
                        gr.Markdown("### Results")
                        results_table = gr.Dataframe(
                            headers=[
                                "Filename",
                                "Prediction",
                                "Confidence",
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
                gr.Markdown("**Note**: Requires microphone access and `sounddevice` package.")

                with gr.Row():
                    with gr.Column():
                        sensitivity_slider = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.05,
                            label="Detection Sensitivity (Threshold)",
                            info="Lower value = more sensitive (detects easier but more false positives). Higher = stricter.",
                        )

                        with gr.Row():
                            start_mic_btn = gr.Button("🎙️ Start Recording", variant="primary", scale=2)
                            stop_mic_btn = gr.Button("⏹️ Stop Recording", variant="stop", scale=1)

                    with gr.Column():
                        gr.Markdown("### Detection Status")

                        detection_indicator = gr.Textbox(
                            label="Status",
                            value="🔴 Not Detecting",
                            lines=2,
                            interactive=False,
                        )

                        confidence_display = gr.Number(label="Confidence (%)", value=0.0, interactive=False)

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

                evaluate_testset_btn = gr.Button("📈 Run Test Evaluation", variant="primary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Basic Metrics")
                        test_metrics = gr.JSON(
                            label="Test Set Metrics",
                            value={"status": "Click 'Run Test Evaluation' to start"},
                        )
                        mine_fp_btn = gr.Button("⛏️ Mine Hard Negatives", variant="secondary")
                        mining_status = gr.Markdown("")

                    with gr.Column():
                        gr.Markdown("### Confusion Matrix")
                        confusion_matrix = gr.Plot(label="Confusion Matrix")

                with gr.Row():
                    roc_curve = gr.Plot(label="ROC Curve (Receiver Operating Characteristic)")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎯 Advanced Production Metrics")
                        advanced_metrics = gr.JSON(
                            label="Production Metrics (FAH, EER, pAUC)",
                            value={"status": "Enable advanced metrics and run evaluation"},
                        )

            # Mining Queue Tab
            with gr.TabItem("⛏️ Mining Queue"):
                gr.Markdown("### Hard Negative Verification Queue")
                gr.Markdown("Review mined false positives and confirm them for the next training run.")

                with gr.Row():
                    refresh_queue_btn = gr.Button("🔄 Refresh Queue")
                    inject_mined_btn = gr.Button("💉 Inject Confirmed to Dataset", variant="primary")

                injection_status = gr.Markdown("")

                mining_queue_html = gr.HTML(value=get_mining_gallery_html(), label="Verification Queue")

                # Hidden state for verifying samples via JS
                verify_path = gr.Textbox(visible=False, elem_id="mining_verify_path")
                verify_status = gr.Textbox(visible=False, elem_id="mining_verify_status")

            # Analysis Dashboard
            with gr.TabItem("🔍 Analysis Dashboard"):
                gr.Markdown("### Advanced Model Analysis & Debugging")
                gr.Markdown("Use these tools to deep-dive into model performance and tune thresholds.")

                with gr.Row():
                    run_analysis_btn = gr.Button("📊 Run Threshold Analysis", variant="primary")
                    run_bench_btn = gr.Button("⚡ Run Performance Benchmark", variant="secondary")

                with gr.Row():
                    with gr.Column():
                        threshold_plot = gr.Plot(label="Threshold Analysis")
                    with gr.Column():
                        bench_metrics = gr.JSON(label="Benchmarking Metrics")

                gr.Markdown("---")
                gr.Markdown("### 🚨 False Positive Inspector")
                with gr.Row():
                    collect_fp_btn = gr.Button("📥 Collect False Positives from Test Set", variant="secondary")
                    clear_fp_btn = gr.Button("🗑️ Clear Collected Samples")

                fp_gallery = gr.HTML(label="False Positive Samples")

        # Event handlers
        def refresh_models_handler() -> gr.Dropdown:
            models = get_available_models()
            return gr.update(
                choices=models,
                value=models[0] if models[0] != "No models available" else None,
            )

        refresh_models_btn.click(fn=refresh_models_handler, outputs=[model_selector])
        load_model_btn.click(fn=load_model, inputs=[model_selector], outputs=[model_status])

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
        mic_refresh = gr.Timer(value=0.5, active=True)
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
                config_state.get("config").data.data_root if config_state.get("config") else "data",
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

        run_analysis_btn.click(fn=run_threshold_analysis, outputs=[threshold_plot, gr.State()])
        run_bench_btn.click(fn=run_benchmark_test, outputs=[bench_metrics])
        collect_fp_btn.click(fn=collect_false_positives, outputs=[fp_gallery])
        clear_fp_btn.click(fn=clear_false_positives, outputs=[fp_gallery])

        # Mining handlers
        mine_fp_btn.click(fn=mine_hard_negatives_handler, outputs=[mining_status])
        refresh_queue_btn.click(fn=get_mining_gallery_html, outputs=[mining_queue_html])
        inject_mined_btn.click(fn=inject_mined_samples_handler, outputs=[injection_status])

        # JS bridge for verification
        def js_bridge_verify(path, status):
            return verify_sample_handler(path, status)

        # We need a way to trigger verification from JS.
        # Gradio doesn't easily allow arbitrary JS -> Python calls without a hidden component.
        # I'll use a hidden button or similar if needed, but for now I'll just refresh.
        # Actually, let's use the hidden textboxes.
        verify_status.change(fn=js_bridge_verify, inputs=[verify_path, verify_status], outputs=[mining_queue_html])

    return panel
