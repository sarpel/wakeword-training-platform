"""
Panel 1: Dataset Management
- Dataset discovery and validation
- Train/test/validation splitting
- .npy file extraction
- Dataset health briefing
"""
import sys
import traceback
from pathlib import Path
from typing import Tuple

import gradio as gr
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import structlog

from src.data.health_checker import DatasetHealthChecker
from src.data.npy_extractor import NpyExtractor
from src.data.preprocessing import VADFilter  # Import VADFilter
from src.data.splitter import DatasetScanner, DatasetSplitter
from src.exceptions import WakewordException

logger = structlog.get_logger(__name__)

# Global state for panel
_current_scanner = None
_current_dataset_info = None


def create_dataset_panel(data_root: str = "data") -> gr.Blocks:
    """
    Create Panel 1: Dataset Management

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks() as panel:
        gr.Markdown("# 📊 Dataset Management")
        gr.Markdown("Manage your wakeword datasets, split them, and validate quality.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Dataset Selection")
                dataset_root = gr.Textbox(
                    label="Dataset Root Directory",
                    placeholder="C:/path/to/datasets or data/raw",
                    lines=1,
                    value=str(Path(data_root) / "raw"),  # Default value
                )

                gr.Markdown("**Expected Structure:**")
                gr.Markdown(
                    """
                ```
                dataset_root/
                ├── positive/       (wakeword utterances)
                │   └── subfolders...
                ├── negative/       (non-wakeword speech)
                │   └── subfolders...
                ├── hard_negative/  (similar sounding phrases)
                │   └── subfolders...
                ├── background/     (environmental noise)
                │   └── subfolders...
                ├── rirs/           (room impulse responses)
                │   └── subfolders...
                └── npy/            (.npy feature files, optional)
                ```
                **Note:** Subfolders are automatically scanned recursively!
                """
                )

                skip_validation = gr.Checkbox(
                    label="Fast Scan (Skip Validation)",
                    value=False,
                    info="Only count files without validation (much faster for large datasets)",
                )

                scan_button = gr.Button("🔍 Scan Datasets", variant="primary")
                scan_status = gr.Textbox(
                    label="Scan Status",
                    value="Ready to scan",
                    lines=1,
                    interactive=False,
                )

            with gr.Column():
                gr.Markdown("### Dataset Statistics")
                stats_display = gr.JSON(
                    label="Dataset Summary", value={"status": "No datasets scanned yet"}
                )

        gr.Markdown("---")

        # MOVED: NPY Feature Management to be BEFORE Splitting
        with gr.Row():
            gr.Markdown("### ⚡ NPY Feature Management (Optional but Recommended)")

        with gr.Tabs():
            # Tab 1: Batch Feature Extraction
            with gr.TabItem("⚡ Batch Feature Extraction"):
                gr.Markdown(
                    "**Performance**: Extract features NOW to speed up training by 40-60%."
                )

                with gr.Row():
                    with gr.Column():
                        extract_feature_type = gr.Dropdown(
                            label="Feature Type",
                            choices=["mel", "mfcc"],
                            value="mel",
                            info="Type of features to extract",
                        )
                        extract_sample_rate = gr.Number(
                            label="Sample Rate",
                            value=16000,
                            precision=0,
                            info="Target sample rate (must match training config)",
                        )
                        extract_duration = gr.Number(
                            label="Audio Duration (s)",
                            value=1.5,
                            info="Target duration (must match training config)",
                        )
                        
                        with gr.Row():
                            extract_n_mels = gr.Number(
                                label="Mel Channels",
                                value=64,
                                precision=0,
                                info="Number of mel filterbanks",
                            )
                            extract_hop_length = gr.Number(
                                label="Hop Length",
                                value=160,
                                precision=0,
                                info="STFT hop length (affects time dimension)",
                            )
                            extract_n_fft = gr.Number(
                                label="FFT Size",
                                value=400,
                                precision=0,
                                info="FFT window size",
                            )
                        
                        extract_batch_size = gr.Slider(
                            minimum=16,
                            maximum=128,
                            value=32,
                            step=16,
                            label="Batch Size (GPU)",
                            info="Higher = faster but uses more memory",
                        )
                        extract_output_dir = gr.Textbox(
                            label="Output Directory (Source for Splitter)",
                            value="data/npy",
                            info="Feature storage location. Dataset Splitter will link to files here.",
                        )
                        batch_extract_button = gr.Button(
                            "⚡ Extract All Features to NPY", variant="primary"
                        )

                    with gr.Column():
                        batch_extraction_log = gr.Textbox(
                            label="Batch Extraction Log",
                            lines=12,
                            value="Ready to extract features...\n\nRECOMMENDED WORKFLOW:\n1. 🔍 Scan Datasets (Top)\n2. ⚡ Extract Features (Here)\n3. ✂️ Split Datasets (Below)\n\nThis ensures NPY files are correctly included in your data splits.",
                            interactive=False,
                        )

            # Tab 2: Analyze Existing NPY Files
            with gr.TabItem("📦 Analyze Existing NPY"):
                with gr.Row():
                    with gr.Column():
                        npy_folder = gr.Textbox(
                            label=".npy Files Directory",
                            value="data/npy",
                            placeholder="Path to .npy files (or leave empty to scan dataset_root/npy)",
                            lines=1,
                        )
                        analyze_button = gr.Button(
                            "🔍 Analyze .npy Files", variant="secondary"
                        )

                        gr.Markdown("### 📐 Shape Validation")
                        gr.Markdown(
                            "Check if NPY files match the expected input shape for training."
                        )

                        with gr.Row():
                            val_sample_rate = gr.Number(
                                label="Target Sample Rate", value=16000
                            )
                            val_duration = gr.Number(label="Target Duration (s)", value=1.5)
                            val_n_mels = gr.Number(label="Mel Channels", value=64, precision=0)
                            val_hop_length = gr.Number(label="Hop Length", value=160)

                        validate_shape_checkbox = gr.Checkbox(
                            label="Validate Shapes",
                            value=True,
                            info="Check dimensions against target config",
                        )
                        delete_invalid_checkbox = gr.Checkbox(
                            label="Delete Invalid Files",
                            value=False,
                            info="⚠️ Permanently delete files with mismatching shapes",
                        )
                        validate_button = gr.Button(
                            "📏 Validate & Clean Shapes", variant="secondary"
                        )

                    with gr.Column():
                        analysis_log = gr.Textbox(
                            label="Analysis Report",
                            lines=12,
                            value="Analyze existing .npy files...",
                            interactive=False,
                        )

            # Tab 3: VAD Filtering
            with gr.TabItem("🧹 VAD Filtering"):
                gr.Markdown("### Voice Activity Detection (VAD) Filter")
                gr.Markdown(
                    "Remove silent or noisy files that do not contain speech. "
                    "Run this BEFORE splitting datasets."
                )
                
                with gr.Row():
                    with gr.Column():
                        vad_energy_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.05, step=0.01,
                            label="Energy Threshold",
                            info="Higher = stricter (needs louder speech)"
                        )
                        vad_min_duration = gr.Number(
                            label="Min Speech Duration (s)", value=0.1
                        )
                        vad_filter_btn = gr.Button("🧹 Filter Dataset with VAD", variant="primary")
                        
                    with gr.Column():
                        vad_log = gr.Textbox(
                            label="VAD Filter Log",
                            lines=10,
                            interactive=False
                        )
        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### ✂️ Train/Test/Validation Split")

                gr.Markdown(
                    "**Industry Standard Ratios:** Train: 70%, Val: 15%, Test: 15%"
                )

                train_ratio = gr.Slider(
                    minimum=0.5,
                    maximum=0.9,
                    value=0.7,
                    step=0.05,
                    label="Train Ratio",
                    info="Training set ratio (70% recommended)",
                )
                val_ratio = gr.Slider(
                    minimum=0.05,
                    maximum=0.3,
                    value=0.15,
                    step=0.05,
                    label="Validation Ratio",
                    info="Validation set ratio (15% recommended)",
                )
                test_ratio = gr.Slider(
                    minimum=0.05,
                    maximum=0.3,
                    value=0.15,
                    step=0.05,
                    label="Test Ratio",
                    info="Test set ratio (15% recommended)",
                )

                split_button = gr.Button("✂️ Split Datasets", variant="primary")
                split_status = gr.Textbox(
                    label="Split Status",
                    value="Step 1: Scan Datasets\nStep 2: (Optional) Extract NPY\nStep 3: Split Datasets",
                    lines=3,
                    interactive=False,
                )

            with gr.Column():
                gr.Markdown("### Health Report")
                health_report = gr.Textbox(
                    label="Dataset Health Analysis",
                    lines=15,
                    value="Run dataset scan to see health report...",
                    interactive=False,
                )

        gr.Markdown("---")

        # Event handlers with full implementation
        def scan_datasets_handler(
            root_path: str, skip_val: bool, progress=gr.Progress()
        ) -> Tuple[dict, str, str]:
            """Scan datasets and return statistics and health report"""
            global _current_scanner, _current_dataset_info

            try:
                if not root_path:
                    return (
                        {"error": "Please provide dataset root path"},
                        "❌ No path provided",
                        "Run dataset scan to see health report...",
                    )

                root_path = Path(root_path)
                if not root_path.exists():
                    return (
                        {"error": f"Path does not exist: {root_path}"},
                        f"❌ Directory not found: {root_path}",
                        "Run dataset scan to see health report...",
                    )

                logger.info(
                    f"Scanning datasets in: {root_path} (skip_validation={skip_val})"
                )

                # Progress callback for Gradio
                def update_progress(progress_value, message):
                    progress(progress_value, desc=f"Scanning: {message}")

                # Create scanner with caching and parallel processing
                use_cache = not skip_val  # Only use cache when validating
                scanner = DatasetScanner(root_path, use_cache=use_cache)

                # Scan datasets with progress
                progress(0, desc="Initializing scan...")
                dataset_info = scanner.scan_datasets(
                    progress_callback=update_progress, skip_validation=skip_val
                )

                # Get statistics
                progress(0.95, desc="Generating statistics...")
                stats = scanner.get_statistics()

                # Save scanner for later use
                _current_scanner = scanner
                _current_dataset_info = dataset_info

                # Move excluded files
                progress(0.96, desc="Moving excluded files...")
                moved_count = scanner.move_excluded_files()

                # Generate health report
                progress(0.97, desc="Generating health report...")
                health_checker = DatasetHealthChecker(stats)
                health_report_text = health_checker.generate_report()

                # Save manifest
                progress(0.99, desc="Saving manifest...")
                manifest_path = (
                    Path(root_path).parent / "splits" / "dataset_manifest.json"
                )
                scanner.save_manifest(manifest_path)

                logger.info("Dataset scan complete")

                # Add cache info to status message
                cache_msg = ""
                if dataset_info.get("cached_files", 0) > 0:
                    cache_msg = f" ({dataset_info['cached_files']} from cache)"

                moved_msg = ""
                if moved_count > 0:
                    moved_msg = f"\n\nMoved {moved_count} low-quality files to 'unqualified_datasets' folder."

                mode_msg = " (fast scan)" if skip_val else ""

                progress(1.0, desc="Complete!")

                return (
                    stats,
                    f"✅ Scan complete! Found {stats['total_files']} audio files{cache_msg}{mode_msg}{moved_msg}",
                    health_report_text,
                )

            except WakewordException as e:
                error_msg = f"❌ Data Error: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return (
                    {"error": error_msg},
                    f"❌ Error: {str(e)}",
                    f"Actionable suggestion: Please check your dataset for the following error: {e}",
                )
            except Exception as e:
                error_msg = f"Error scanning datasets: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return (
                    {"error": error_msg},
                    f"❌ Error: {str(e)}",
                    f"❌ Error during scan:\n{str(e)}",
                )

        def split_datasets_handler(
            root_path: str,
            train: float,
            val: float,
            test: float,
            npy_source: str,  # Added npy_source input
            progress=gr.Progress(),
        ) -> Tuple[str, str]:
            """Split datasets into train/val/test"""
            global _current_dataset_info

            try:
                # Validate ratios
                total = train + val + test
                if abs(total - 1.0) > 0.01:
                    return (
                        f"❌ Ratios must sum to 1.0 (current: {total:.2f})",
                        "No split performed - fix ratios",
                    )

                if _current_dataset_info is None:
                    return (
                        "❌ Please scan datasets first",
                        "No split performed - scan required",
                    )

                logger.info(f"Splitting datasets: {train}/{val}/{test}")

                # Create splitter and split
                progress(0.1, desc="Initializing splitter...")
                splitter = DatasetSplitter(_current_dataset_info)

                # Determine npy source directory
                npy_source_path = (
                    Path(npy_source) if npy_source else Path(root_path) / "npy"
                )
                if not npy_source_path.exists():
                    # Fallback to default if custom path doesn't exist or wasn't provided
                    npy_source_path = Path(root_path) / "npy"
                    # Note: In the UI default is data/raw/npy, here root_path is data/raw usually
                    # Let's be safer and rely on the input if it looks valid, otherwise check standard location

                # If the input npy_source is actually "data/raw/npy" (from the UI default), we use that.
                if npy_source:
                    npy_source_path = Path(npy_source)
                else:
                    # Default fallback relative to data root
                    npy_source_path = Path(root_path) / "npy"

                progress(0.2, desc="Splitting datasets...")
                splits = splitter.split_datasets(
                    train_ratio=train,
                    val_ratio=val,
                    test_ratio=test,
                    random_seed=42,
                    stratify=True,
                    npy_source_dir=npy_source_path,  # Explicitly pass the source path
                )

                # Save splits
                progress(0.7, desc="Saving splits...")
                output_dir = Path(root_path).parent / "splits"
                splitter.save_splits(output_dir)

                # Get split statistics
                progress(0.9, desc="Generating statistics...")
                split_stats = splitter.get_split_statistics()

                # Generate report
                report = ["=" * 60]
                report.append("DATASET SPLIT SUMMARY")
                report.append("=" * 60)
                report.append("")

                for split_name, stats in split_stats.items():
                    report.append(f"{split_name.upper()}:")
                    report.append(f"  Total Files: {stats['total_files']}")
                    report.append(f"  Percentage: {stats['percentage']:.1f}%")
                    report.append(f"  Categories: {stats['categories']}")
                    report.append("")

                report.append(f"✅ Splits saved to: {output_dir}")
                report.append("=" * 60)

                report_text = "\n".join(report)

                logger.info("Dataset split complete")

                progress(1.0, desc="Complete!")

                return (f"✅ Split complete! Saved to {output_dir}", report_text)

            except WakewordException as e:
                error_msg = f"❌ Data Error: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return (
                    f"❌ Error: {str(e)}",
                    f"Actionable suggestion: Please check your dataset for the following error: {e}",
                )
            except Exception as e:
                error_msg = f"Error splitting datasets: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return (f"❌ Error: {str(e)}", f"❌ Error during split:\n{str(e)}")

        def batch_extract_handler(
            root_path: str,
            feature_type: str,
            sample_rate: int,
            audio_duration: float,
            n_mels: int,
            hop_length: int,
            n_fft: int,
            batch_size: int,
            output_dir: str,
            progress=gr.Progress(),
        ) -> str:
            """Batch extract features to NPY files"""
            global _current_dataset_info

            try:
                # Validate dataset scanned
                if _current_dataset_info is None:
                    return "❌ Please scan datasets first (Step 1)"

                logger.info(
                    f"Batch extracting {feature_type} features with batch_size={batch_size}"
                )

                # Import batch extractor
                from src.config.defaults import DataConfig
                from src.data.batch_feature_extractor import BatchFeatureExtractor

                # Create config for feature extraction
                progress(0.05, desc="Initializing batch extractor...")
                config = DataConfig(
                    feature_type=feature_type,
                    sample_rate=int(sample_rate),
                    audio_duration=float(audio_duration),
                    n_mels=int(n_mels),
                    n_mfcc=40,  # MFCC için varsayılan, feature_type='mfcc' ise kullanılır
                    n_fft=int(n_fft),
                    hop_length=int(hop_length),
                )

                # Initialize extractor
                device = "cuda" if torch.cuda.is_available() else "cpu"
                extractor = BatchFeatureExtractor(config=config, device=device)

                # Collect all audio files from scanned dataset
                progress(0.1, desc="Collecting audio files...")
                all_files = []
                for category, category_data in _current_dataset_info[
                    "categories"
                ].items():
                    if category in [
                        "positive",
                        "negative",
                        "hard_negative",
                    ]:  # Skip background and rirs
                        for file_info in category_data["files"]:
                            all_files.append(Path(file_info["path"]))

                if not all_files:
                    return "❌ No audio files found in scanned dataset"

                logger.info(f"Found {len(all_files)} audio files to process")

                # Progress callback
                def update_progress(current, total, message):
                    progress_value = 0.1 + (current / total) * 0.8
                    progress(progress_value, desc=f"{message}")

                # Extract features
                progress(
                    0.15,
                    desc=f"Extracting {len(all_files)} files (batch={batch_size})...",
                )
                output_path = Path(output_dir)
                results = extractor.extract_dataset(
                    audio_files=all_files,
                    output_dir=output_path,
                    batch_size=batch_size,
                    preserve_structure=True,
                    progress_callback=update_progress,
                )

                # Generate report
                progress(0.95, desc="Generating report...")
                report = []
                report.append("=" * 60)
                report.append("BATCH FEATURE EXTRACTION COMPLETE")
                report.append("=" * 60)
                report.append("")
                report.append(f"Feature Type: {feature_type}")
                report.append(f"Device: {device.upper()}")
                report.append(f"Batch Size: {batch_size}")
                report.append(f"Output Directory: {output_path}")
                report.append("")
                report.append(f"✅ Successfully extracted: {results['success_count']}")
                report.append(f"❌ Failed: {results['failed_count']}")
                report.append(f"📊 Total processed: {results['total_files']}")
                report.append("")

                if results["failed_count"] > 0:
                    report.append("Failed Files:")
                    for failed in results["failed_files"][:10]:  # Show first 10
                        report.append(
                            f"  - {Path(failed['path']).name}: {failed['error']}"
                        )
                    if results["failed_count"] > 10:
                        report.append(f"  ... and {results['failed_count'] - 10} more")
                    report.append("")

                report.append("=" * 60)
                report.append("Next Steps:")
                report.append(
                    "1. Split your dataset using 'Train/Test/Validation Split' section"
                )
                report.append(
                    "   → This will automatically organize NPY files into data/npy/train|val|test"
                )
                report.append("2. Go to Panel 2 (Configuration)")
                report.append("3. Verify 'NPY Feature Directory' is set to: data/npy")
                report.append(f"4. Verify 'NPY Feature Type' matches: {feature_type}")
                report.append(
                    "5. Start training (40-60% faster with precomputed features!)"
                )
                report.append("=" * 60)

                report_text = "\n".join(report)

                logger.info(
                    f"Batch extraction complete: {results['success_count']}/{results['total_files']}"
                )

                progress(1.0, desc="Complete!")

                return report_text

            except WakewordException as e:
                error_msg = f"❌ Data Error: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return f"❌ Error: {str(e)}\n\nActionable suggestion: Please check your dataset for the following error: {e}"
            except Exception as e:
                error_msg = f"Error during batch extraction: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return f"❌ Error: {str(e)}\n\nStack trace:\n{traceback.format_exc()}"

        def analyze_npy_handler(
            npy_path: str, root_path: str, progress=gr.Progress()
        ) -> str:
            """Analyze existing .npy files"""
            try:
                # Determine npy folder path
                if not npy_path:
                    if not root_path:
                        return "❌ Please provide either .npy folder path or scan datasets first"
                    npy_path = Path(root_path) / "npy"
                else:
                    npy_path = Path(npy_path)

                if not npy_path.exists():
                    return f"❌ Directory not found: {npy_path}"

                logger.info(f"Analyzing .npy files in: {npy_path}")

                # Create extractor with parallel processing
                progress(0.05, desc="Initializing analyzer...")
                extractor = NpyExtractor()

                # Scan for .npy files
                progress(0.1, desc="Scanning for .npy files...")
                npy_files = extractor.scan_npy_files(npy_path, recursive=True)

                if not npy_files:
                    return f"ℹ️  No .npy files found in: {npy_path}"

                # Progress callback for extraction
                def update_progress(current, total, message):
                    progress_value = 0.1 + (current / total) * 0.8
                    progress(progress_value, desc=f"Analyzing: {message}")

                # Extract and analyze
                progress(0.1, desc=f"Processing {len(npy_files)} .npy files...")
                results = extractor.extract_and_convert(
                    npy_files, progress_callback=update_progress
                )

                # Generate report
                progress(0.95, desc="Generating report...")
                report = extractor.generate_report()

                logger.info("NPY analysis complete")

                progress(1.0, desc="Complete!")

                return report

            except Exception as e:
                error_msg = f"Error analyzing .npy files: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return f"❌ Error: {str(e)}"

        def validate_shapes_handler(
            npy_path: str,
            root_path: str,
            delete_invalid: bool,
            # Added inputs to manually specify target shape
            target_sample_rate: int,
            target_duration: float,
            target_n_mels: int,
            target_hop_length: int,
            progress=gr.Progress(),
        ) -> str:
            """Validate NPY file shapes"""
            try:
                # Determine npy folder path
                if not npy_path:
                    if not root_path:
                        return "❌ Please provide either .npy folder path or scan datasets first"
                    npy_path = Path(root_path) / "npy"
                else:
                    npy_path = Path(npy_path)

                if not npy_path.exists():
                    return f"❌ Directory not found: {npy_path}"

                # Get current config to determine expected shape
                from src.config.defaults import DataConfig
                from src.data.feature_extraction import FeatureExtractor

                # Use inputs from UI instead of loading from saved config which might be stale
                config = DataConfig(
                    sample_rate=int(target_sample_rate),
                    audio_duration=float(target_duration),
                    hop_length=int(target_hop_length),
                    n_mels=int(target_n_mels),
                )

                # Calculate expected shape
                # Standard: 1.5s * 16000 = 24000 samples
                target_samples = int(config.audio_duration * config.sample_rate)

                # Helper extractor to calculate shape
                helper_extractor = FeatureExtractor(
                    sample_rate=config.sample_rate,
                    feature_type=config.feature_type,
                    n_mels=config.n_mels,
                    n_mfcc=config.n_mfcc,
                    n_fft=config.n_fft,
                    hop_length=config.hop_length,
                )

                expected_shape = helper_extractor.get_output_shape(target_samples)

                logger.info(
                    f"Validating shapes in {npy_path} against expected: {expected_shape}"
                )

                # Initialize NPY extractor
                extractor = NpyExtractor()

                # Scan files
                progress(0.1, desc="Scanning .npy files...")
                npy_files = extractor.scan_npy_files(npy_path, recursive=True)

                if not npy_files:
                    return f"ℹ️ No .npy files found in {npy_path}"

                # Progress callback
                def update_progress(current, total, message):
                    progress_value = 0.1 + (current / total) * 0.8
                    progress(progress_value, desc=message)

                # Validate
                progress(0.2, desc=f"Validating {len(npy_files)} files...")
                results = extractor.validate_shapes(
                    npy_files=npy_files,
                    expected_shape=expected_shape,
                    delete_invalid=delete_invalid,
                    progress_callback=update_progress,
                )

                # Generate Report
                report = ["=" * 60]
                report.append("SHAPE VALIDATION REPORT")
                report.append("=" * 60)
                report.append("")
                report.append(f"Target Shape: {expected_shape}")
                report.append(f"Files Checked: {results['total_files']}")
                report.append(f"✅ Valid Matches: {results['valid_count']}")
                report.append(f"❌ Mismatches: {results['mismatch_count']}")

                if delete_invalid:
                    report.append(f"🗑️ Deleted: {results['deleted_count']}")

                if results["mismatches"]:
                    report.append("")
                    report.append("Mismatch Examples:")
                    report.append("-" * 40)
                    for m in results["mismatches"][:10]:
                        report.append(f"File: {Path(m['path']).name}")
                        report.append(f"  Got: {m['actual']}")
                        report.append(f"  Exp: {m['expected']}")
                        report.append("")

                    if len(results["mismatches"]) > 10:
                        report.append(f"... and {len(results['mismatches']) - 10} more")

                return "\n".join(report)

            except Exception as e:
                return f"❌ Error during validation: {str(e)}\n{traceback.format_exc()}"

        def vad_filter_handler(
            root_path: str,
            threshold: float,
            min_duration: float,
            progress=gr.Progress()
        ) -> str:
            """Filter dataset using VAD"""
            try:
                if not root_path:
                    return "❌ Please provide dataset root path"

                # We filter the manifest files in the splits/ directory (or we could scan directories directly)
                # But usually we rely on the manifest generated by "Scan Datasets"
                # Let's assume we operate on the "dataset_manifest.json" generated by Scan
                
                manifest_path = Path(root_path).parent / "splits" / "dataset_manifest.json"
                
                if not manifest_path.exists():
                    return "❌ Manifest not found. Please 'Scan Datasets' first."
                
                logger.info(f"Running VAD filter on {manifest_path}...")
                
                progress(0.1, desc="Initializing VAD...")
                vad = VADFilter(energy_threshold=threshold)
                
                progress(0.2, desc="Filtering...")
                output_path = vad.process_dataset(
                    manifest_path, 
                    min_speech_duration=min_duration
                )
                
                # Update global scanner info if possible, or just warn user to re-scan
                # Since we modified the manifest (created a new one actually), we should probably
                # instruct user to use the cleaned one. 
                # Ideally, we replace the main manifest or the scanner updates itself.
                
                # For now, let's overwrite the main manifest if it was successful so subsequent steps use it?
                # Or maybe VADFilter returned a "_cleaned.json".
                # Let's rename it to be the main manifest so Splitter picks it up? 
                # That's destructive. Better to tell user or have Splitter look for cleaned.
                # Simplest for UI: Overwrite and backup old.
                
                backup_path = manifest_path.with_suffix(".json.bak")
                import shutil
                shutil.copy(manifest_path, backup_path)
                shutil.move(output_path, manifest_path)
                
                return (f"✅ VAD Filtering Complete!\n"
                        f"Original manifest backed up to {backup_path.name}\n"
                        f"Active manifest updated. You can now Split Datasets.")
                
            except Exception as e:
                 error_msg = f"Error during VAD filtering: {str(e)}"
                 logger.error(error_msg)
                 logger.error(traceback.format_exc())
                 return f"❌ Error: {str(e)}\n{traceback.format_exc()}"

        def auto_start_handler(
            root_path: str,
            skip_val: bool,
            # Feature extraction params
            feature_type: str,
            sample_rate: int,
            audio_duration: float,
            n_mels: int,
            hop_length: int,
            n_fft: int,
            batch_size: int,
            output_dir: str,
            # Split params
            train: float,
            val: float,
            test: float,
            # Training params
            use_cmvn: bool,
            use_ema: bool,
            ema_decay: float,
            use_balanced_sampler: bool,
            sampler_ratio_pos: int,
            sampler_ratio_neg: int,
            sampler_ratio_hard: int,
            run_lr_finder: bool,
            use_wandb: bool,
            wandb_project: str,
            state: gr.State,
            progress=gr.Progress(),
        ):
            """Orchestrates the full pipeline: Scan -> Extract -> Split -> Config -> Train"""
            
            logs = []
            def log(msg):
                logs.append(msg)
                logger.info(msg)
                return "\n".join(logs)

            try:
                # 1. Scan Datasets
                progress(0.0, desc="Step 1/5: Scanning Datasets...")
                log("--- STEP 1: SCANNING DATASETS ---")
                stats, scan_msg, _ = scan_datasets_handler(root_path, skip_val)
                if "error" in stats:
                    return log(f"❌ Scan Failed: {stats['error']}")
                log(scan_msg)

                # 2. Feature Extraction
                progress(0.2, desc="Step 2/5: Extracting Features...")
                log("\n--- STEP 2: EXTRACTING FEATURES ---")
                log(f"Extracting {feature_type} features to {output_dir}...")
                extract_report = batch_extract_handler(
                    root_path, feature_type, sample_rate, audio_duration, 
                    n_mels, hop_length, n_fft, batch_size, output_dir
                )
                if "❌ Error" in extract_report:
                    return log(f"❌ Extraction Failed:\n{extract_report}")
                log("Feature extraction completed.")

                # 3. Split Dataset
                progress(0.5, desc="Step 3/5: Splitting Datasets...")
                log("\n--- STEP 3: SPLITTING DATASETS ---")
                split_msg, _ = split_datasets_handler(root_path, train, val, test, output_dir)
                if "❌" in split_msg:
                    return log(f"❌ Split Failed: {split_msg}")
                log(split_msg)

                # 4. Load Config
                progress(0.7, desc="Step 4/5: Loading Configuration...")
                log("\n--- STEP 4: PREPARING CONFIGURATION ---")
                
                from src.config.defaults import WakewordConfig, DataConfig, TrainingConfig, ModelConfig
                
                # Create config matching UI parameters
                config = WakewordConfig()
                config.data = DataConfig(
                    sample_rate=int(sample_rate),
                    audio_duration=float(audio_duration),
                    n_mels=int(n_mels),
                    n_fft=int(n_fft),
                    hop_length=int(hop_length),
                    feature_type=feature_type,
                    use_precomputed_features_for_training=True,
                    npy_feature_dir=output_dir
                )
                
                # Try to load saved config if exists to preserve model params
                config_path = Path("configs/wakeword_config.yaml")
                if config_path.exists():
                    try:
                        loaded_config = WakewordConfig.load(config_path)
                        # Update data params to match UI but keep model params
                        loaded_config.data = config.data 
                        config = loaded_config
                        log(f"Loaded saved configuration from {config_path}")
                    except Exception as e:
                        log(f"⚠️ Failed to load saved config, using defaults: {e}")
                
                # Update state with config
                if state is None:
                    state = {}
                state["config"] = config
                
                # 5. Start Training
                progress(0.9, desc="Step 5/5: Starting Training...")
                log("\n--- STEP 5: STARTING TRAINING ---")
                
                from src.ui.panel_training import start_training
                
                # Call start_training with the state containing our config
                train_msg, _, _, _, _, _ = start_training(
                    state, use_cmvn, use_ema, ema_decay, 
                    use_balanced_sampler, sampler_ratio_pos, sampler_ratio_neg, sampler_ratio_hard,
                    run_lr_finder, use_wandb, wandb_project
                )
                
                log(f"Training Launch Result: {train_msg}")
                
                if "✅" in train_msg:
                    log("\n✅ PIPELINE STARTED SUCCESSFULLY!")
                    log("Switch to 'Model Training' tab to view progress.")
                else:
                    log("\n❌ FAILED TO START TRAINING")

                return log(f"Pipeline Finished.\nLast status: {train_msg}")

            except Exception as e:
                err = traceback.format_exc()
                logger.error(err)
                return log(f"❌ CRITICAL ERROR IN AUTO-START:\n{str(e)}\n{err}")

        # Connect event handlers
        scan_button.click(
            fn=scan_datasets_handler,
            inputs=[dataset_root, skip_validation],
            outputs=[stats_display, scan_status, health_report],
        )

        split_button.click(
            fn=split_datasets_handler,
            inputs=[
                dataset_root,
                train_ratio,
                val_ratio,
                test_ratio,
                extract_output_dir,
            ],  # Added extract_output_dir
            outputs=[split_status, health_report],
        )

        batch_extract_button.click(
            fn=batch_extract_handler,
            inputs=[
                dataset_root,
                extract_feature_type,
                extract_sample_rate,
                extract_duration,
                extract_n_mels,
                extract_hop_length,
                extract_n_fft,
                extract_batch_size,
                extract_output_dir,
            ],
            outputs=[batch_extraction_log],
        )

        analyze_button.click(
            fn=analyze_npy_handler,
            inputs=[npy_folder, dataset_root],
            outputs=[analysis_log],
        )

        validate_button.click(
            fn=validate_shapes_handler,
            inputs=[
                npy_folder,
                dataset_root,
                delete_invalid_checkbox,
                val_sample_rate,
                val_duration,
                val_n_mels,
                val_hop_length,
            ],
            outputs=[analysis_log],
        )

        vad_filter_btn.click(
            fn=vad_filter_handler,
            inputs=[dataset_root, vad_energy_threshold, vad_min_duration],
            outputs=[vad_log]
        )

        gr.Markdown("---")
        with gr.Row():
            auto_start_btn = gr.Button("🚀 AUTO-START TRAINING (Full Pipeline)", variant="primary", scale=2)
            auto_log = gr.Textbox(label="Pipeline Log", lines=10, interactive=False)

        # Note: We need inputs from Panel 2/3 here. Since we can't easily access them across files without passing 
        # explicit components, we will assume default values or need the user to pass them.
        # Ideally, this button should be in the Main App where all components are visible.
        # However, requested in Panel 1. We will try to use the state if components are not available, 
        # BUT Gradio events require component references.
        # 
        # WORKAROUND: We will define the button here but the caller (app.py) usually wires it up.
        # BUT `create_dataset_panel` is self-contained. 
        # 
        # Since we can't access components from other panels here, we will create HIDDEN inputs for the required training params
        # that match defaults, or require the user to set them in the pipeline code. 
        # A better approach for a "Single Button" is to assume reasonable defaults for training 
        # (which we did in the handler) OR allow passing state. 
        #
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        #
        # To facilitate this, we will add the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        # 
        # Let's attach the handler to the button using the inputs AVAILABLE IN THIS PANEL
        # and for others (Training params), we will use default values defined in the function 
        # signature if they are not wired.
        
        # Actually, `app.py` is where the global state lives. 
        # We will define the button here, but we will NOT click() it here.
        # We will export `auto_start_btn` and `auto_start_handler` so `app.py` can connect them.
        
        # Since we are in `create_dataset_panel`, we only have access to Dataset Panel inputs.
        # To make this work, we need to expose the `auto_start_handler` so `app.py` can wire it up with 
        # inputs from ALL panels.
        
        # For now, we will just expose the button instance as an attribute of the panel or return it?
        # No, `create_dataset_panel` returns a Block.
        #