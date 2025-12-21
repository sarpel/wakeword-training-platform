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
import torch
import plotly.graph_objects as go

matplotlib.use("Agg")
import structlog

from src.data.dataset import WakewordDataset
from src.evaluation.evaluator import ModelEvaluator, load_model_for_evaluation
from src.evaluation.advanced_evaluator import ThresholdAnalyzer
from src.evaluation.data_collector import FalsePositiveCollector
from src.evaluation.inference import MicrophoneInference, SimulatedMicrophoneInference
from src.evaluation.types import EvaluationResult
from src.exceptions import WakewordException
from src.training.metrics import MetricResults
from src.evaluation.benchmarking import BenchmarkRunner
from src.evaluation.stages import SentryInferenceStage

logger = structlog.get_logger(__name__)


class EvaluationState:
    """Global evaluation state manager"""

    def __init__(self) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.model_info: Optional[Dict[str, Any]] = None
        self.evaluator: Optional[ModelEvaluator] = None
        self.mic_inference: Optional[Union[MicrophoneInference, SimulatedMicrophoneInference]] = None
        self.is_mic_recording = False

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
        model, info = load_model_for_evaluation(Path(model_path), device="cuda")
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
            config=info["config"],
        )
        if hasattr(evaluator, "audio_processor"):
            evaluator.audio_processor.eval()
        eval_state.model = model
        eval_state.model_info = info
        eval_state.evaluator = evaluator
        eval_state.waveform_sr = info["config"].data.sample_rate
        status = f"✅ Model Loaded Successfully\nArchitecture: {info['config'].model.architecture}\nTraining Epoch: {info['epoch'] + 1}\nVal Loss: {info['val_loss']:.4f}"
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
        data = [{"Filename": r.filename, "Prediction": r.prediction, "Confidence": f"{r.confidence:.2%}"} for r in results]
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
            device="cuda",
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
    fig.add_trace(go.Scatter(x=df['threshold'], y=df['precision'], name='Precision', line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=df['threshold'], y=df['recall'], name='Recall', line=dict(color='orange')))
    fig.update_layout(title="PR vs Threshold", template="plotly_dark", height=400)
    return fig, df


def run_benchmark_test(num_iterations: int = 10) -> Dict[str, Any]:
    if eval_state.model is None:
        return {"error": "No model loaded"}
    try:
        stage = SentryInferenceStage(model=eval_state.model, name=eval_state.model_info["config"].model.architecture, device="cuda")
        runner = BenchmarkRunner(stage)
        audio = np.random.randn(int(eval_state.waveform_sr * eval_state.model_info["config"].data.audio_duration)).astype(np.float32)
        metrics = runner.run_benchmark(audio, num_iterations=num_iterations)
        return {"Model": metrics["name"], "Mean Latency": f"{metrics['mean_latency_ms']:.2f} ms", "RAM Usage": f"{metrics['memory_allocated_mb']:.2f} MB", "GPU Usage": f"{metrics.get('gpu_memory_allocated_mb', 0):.2f} MB"}
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
    return html + '</div>'


def clear_false_positives() -> str:
    eval_state.fp_collector.clear()
    return "<p>Cleared.</p>"


def evaluate_test_set(data_root, test_split_path, threshold, target_fah, use_advanced_metrics):
    if eval_state.evaluator is None:
        return {"status": "❌ Load model"}, None, None, {}
    try:
        test_dataset = WakewordDataset(manifest_path=Path(test_split_path), sample_rate=eval_state.waveform_sr, audio_duration=eval_state.model_info["config"].data.audio_duration, augment=False, device="cuda", return_raw_audio=True)
        metrics, results = eval_state.evaluator.evaluate_dataset(test_dataset, threshold=threshold)
        eval_state.test_results = results
        logits = torch.tensor(np.stack([r.logits for r in results]))
        labels = torch.tensor(np.array([r.label for r in results]))
        eval_state.last_logits, eval_state.last_labels = logits, labels
        eval_state.threshold_analyzer = ThresholdAnalyzer(logits, labels)
        return metrics.__dict__, None, None, {}
    except Exception as e:
        return {"status": str(e)}, None, None, {}


def create_evaluation_panel(state: gr.State) -> gr.Blocks:
    with gr.Blocks() as panel:
        gr.Markdown("# 🎯 Model Evaluation")
        with gr.Row():
            model_selector = gr.Dropdown(choices=get_available_models(), label="Select Model")
            refresh_btn = gr.Button("🔄")
            load_btn = gr.Button("📥 Load", variant="primary")
        status = gr.Textbox(label="Status", lines=4)
        with gr.Tabs():
            with gr.TabItem("📁 Files"):
                files = gr.File(file_count="multiple")
                btn_eval = gr.Button("🔍 Evaluate")
            with gr.TabItem("📊 Test Set"):
                split = gr.Textbox(value="data/splits/test.json")
                btn_test = gr.Button("📈 Run Test Evaluation")
            with gr.TabItem("🔍 Analysis"):
                with gr.Row():
                    btn_an = gr.Button("📊 Analysis")
                    btn_bench = gr.Button("⚡ Benchmark")
                with gr.Row():
                    p_an = gr.Plot()
                    j_bench = gr.JSON()
                btn_coll = gr.Button("🚩 Collect FPs")
                gallery = gr.HTML()
        
        refresh_btn.click(fn=lambda: gr.update(choices=get_available_models()), outputs=[model_selector])
        load_btn.click(fn=load_model, inputs=[model_selector], outputs=[status])
        btn_test.click(fn=evaluate_test_set, inputs=[gr.State("data"), split, gr.State(0.5), gr.State(1.0), gr.State(True)], outputs=[])
        btn_an.click(fn=run_threshold_analysis, outputs=[p_an, gr.State()])
        btn_bench.click(fn=run_benchmark_test, outputs=[j_bench])
        btn_coll.click(fn=collect_false_positives, outputs=[gallery])
    return panel