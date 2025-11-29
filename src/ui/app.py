"""
Main Gradio Application
Wakeword Training Platform with 6 panels
"""
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import asyncio
import sys
from pathlib import Path

import gradio as gr

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.cuda_utils import enforce_cuda
from src.config.logger import get_data_logger, setup_logging
from src.ui.panel_config import create_config_panel
from src.ui.panel_dataset import create_dataset_panel
from src.ui.panel_docs import create_docs_panel
from src.ui.panel_evaluation import create_evaluation_panel
from src.ui.panel_export import create_export_panel
from src.ui.panel_training import create_training_panel


def suppress_windows_asyncio_errors():
    """
    Suppress harmless Windows asyncio errors from closed connections.

    On Windows, ProactorEventLoop raises ConnectionResetError when trying to
    shutdown already-closed sockets (common with WebSocket disconnects).
    This handler suppresses these specific harmless errors to keep terminal clean.
    """
    if sys.platform == "win32":

        def handle_exception(loop, context):
            # Check if this is the specific harmless error
            exception = context.get("exception")
            
            # Check for ConnectionResetError (WinError 10054)
            # This is very common on Windows when clients disconnect abruptly
            if isinstance(exception, ConnectionResetError):
                return
                
            # Check for OSError with 10054
            if isinstance(exception, OSError) and getattr(exception, 'winerror', 0) == 10054:
                return
                
            # Check message for ProactorBasePipeTransport
            message = context.get("message", "")
            if "_ProactorBasePipeTransport" in message:
                return
                
            # Check handle string for ProactorBasePipeTransport
            # The error often comes from _call_connection_lost
            handle = context.get("handle")
            if handle and "_ProactorBasePipeTransport" in str(handle):
                return
                
            if "source_traceback" in context:
                 if "_call_connection_lost" in str(context["source_traceback"]):
                     return

            # For other exceptions, use default handling
            loop.default_exception_handler(context)

        # Set the custom exception handler
        try:
            loop = asyncio.get_event_loop()
            loop.set_exception_handler(handle_exception)
        except RuntimeError:
            # If no event loop exists yet, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.set_exception_handler(handle_exception)


def find_available_port(start_port: int = 7860, end_port: int = 7870) -> int:
    """
    Find an available port in the specified range

    Args:
        start_port: Starting port number
        end_port: Ending port number

    Returns:
        Available port number or start_port if none found
    """
    import socket

    for port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue

    # If no port available, return start_port and let it fail with clear message
    return start_port


def create_app() -> gr.Blocks:
    """
    Create the main Gradio application with all panels

    Returns:
        Gradio Blocks app
    """
    # Validate CUDA
    logger = get_data_logger("app")
    logger.info("Starting Wakeword Training Platform")

    cuda_validator = enforce_cuda()
    logger.info("CUDA validation passed")

    # Create main app with theme
    with gr.Blocks(
        title="Wakeword Training Platform",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        """,
    ) as app:
        # Global state for sharing data between panels
        global_state = gr.State(value={"config": None})

        # Header
        gr.Markdown(
            """
        # 🎙️ Wakeword Training Platform
        ### GPU-Accelerated Custom Wakeword Detection Model Training

        Complete pipeline from dataset management to model deployment.
        """
        )

        # Display GPU info
        gpu_info = cuda_validator.get_device_info()
        gr.Markdown(
            f"""
        **GPU Status**: ✅ {gpu_info['device_count']} GPU(s) available |
        **CUDA Version**: {gpu_info['cuda_version']} |
        **Active Device**: {gpu_info['devices'][0]['name']} ({gpu_info['devices'][0]['total_memory_gb']} GB)
        """
        )

        gr.Markdown("---")

        # Create tabs for 6 panels
        with gr.Tabs():
            with gr.TabItem("📊 1. Dataset Management", id=1):
                panel_dataset = create_dataset_panel()

            with gr.TabItem("⚙️ 2. Configuration", id=2):
                panel_config = create_config_panel(global_state)

            with gr.TabItem("🚀 3. Training", id=3):
                panel_training = create_training_panel(global_state)

            with gr.TabItem("🎯 4. Evaluation", id=4):
                panel_evaluation = create_evaluation_panel(global_state)

            with gr.TabItem("📦 5. ONNX Export", id=5):
                panel_export = create_export_panel()

            with gr.TabItem("📚 6. Documentation", id=6):
                panel_docs = create_docs_panel()

        # Footer
        gr.Markdown("---")
        gr.Markdown(
            """
        **Wakeword Training Platform v1.0** | Reliability-focused implementation |
        GPU-accelerated with PyTorch & CUDA
        """
        )
        
        # Wire up the Auto-Start Pipeline button from Panel 1
        # We access the button and handler exposed by create_dataset_panel
        # and connect inputs from other panels that we now have access to.
        # Note: create_training_panel returns a Block, but we didn't modify it to expose inputs.
        # However, we can access components if they were assigned to the block object or returned.
        
        # Actually, create_training_panel just returns the `panel` object. 
        # We cannot easily access internal components of `panel_training` unless we modify it too.
        # BUT, for the "Auto-Start" button, we assumed default values or passed state.
        # 
        # Let's look at `panel_dataset.auto_start_handler` signature.
        # It requires Training Params. 
        # 
        # Since we cannot get the *live* values from Panel 3 components (because we don't have references to them),
        # we will use sensible defaults for the pipeline, OR rely on the "Loaded Config" step in the handler
        # which tries to load `configs/wakeword_config.yaml`.
        #
        # So we just need to wire the button to the handler, passing the Panel 1 inputs we DO have references to (via panel_dataset.inputs)
        # and hardcode/default the others, trusting the handler's config loading logic.
        
        if hasattr(panel_dataset, "auto_start_btn"):
            ds_inputs = panel_dataset.inputs
            
            # Define default states for training parameters
            s_use_cmvn = gr.State(True)
            s_use_ema = gr.State(True)
            s_ema_decay = gr.State(0.999)
            s_use_balanced = gr.State(True)
            s_pos_ratio = gr.State(1)
            s_neg_ratio = gr.State(1)
            s_hard_ratio = gr.State(1)
            s_run_lr = gr.State(False)
            s_use_wandb = gr.State(False)
            s_wandb_proj = gr.State("wakeword-training")

            panel_dataset.auto_start_btn.click(
                fn=panel_dataset.auto_start_handler,
                inputs=[
                    ds_inputs["root_path"],
                    ds_inputs["skip_val"],
                    ds_inputs["move_unqualified"],
                    ds_inputs["feature_type"],
                    ds_inputs["sample_rate"],
                    ds_inputs["audio_duration"],
                    ds_inputs["n_mels"],
                    ds_inputs["hop_length"],
                    ds_inputs["n_fft"],
                    ds_inputs["batch_size"],
                    ds_inputs["output_dir"],
                    ds_inputs["train"],
                    ds_inputs["val"],
                    ds_inputs["test"],
                    # Training defaults
                    s_use_cmvn,
                    s_use_ema,
                    s_ema_decay,
                    s_use_balanced,
                    s_pos_ratio,
                    s_neg_ratio,
                    s_hard_ratio,
                    s_run_lr,
                    s_use_wandb,
                    s_wandb_proj,
                    global_state,    # The global state dict
                ],
                outputs=[panel_dataset.auto_log]
            )

    logger.info("Application created successfully")
    return app


def launch_app(
    server_name: str = "0.0.0.0",
    server_port: int = None,
    share: bool = False,
    inbrowser: bool = True,
):
    """
    Launch the Gradio application

    Args:
        server_name: Server host
        server_port: Server port (None = auto-find between 7860-7870)
        share: Create public share link
        inbrowser: Open browser automatically
    """
    # Setup logging
    setup_logging()
    logger = get_data_logger("app")
    suppress_windows_asyncio_errors()

    # Find available port if not specified
    if server_port is None:
        server_port = find_available_port(7860, 7870)
        logger.info(f"Auto-selected port: {server_port}")

    # Create and launch app
    app = create_app()

    logger.info(f"Launching app on {server_name}:{server_port}")
    logger.info(f"Share mode: {share}")
    logger.info(f"Open browser: {inbrowser}")

    try:
        app.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            inbrowser=inbrowser,
            show_error=True,
            show_api=False,
            quiet=False,
        )
    except OSError as e:
        if "address already in use" in str(e).lower():
            logger.error(f"Port {server_port} is already in use")
            logger.info("Trying to find another available port...")
            new_port = find_available_port(server_port + 1, 7870)
            if new_port != server_port:
                logger.info(f"Retrying with port {new_port}")
                app.launch(
                    server_name=server_name,
                    server_port=new_port,
                    share=share,
                    inbrowser=inbrowser,
                    show_error=True,
                    quiet=False,
                )
            else:
                logger.error("No available ports found in range 7860-7870")
                raise
        else:
            raise


if __name__ == "__main__":
    # Launch with default settings
    launch_app()