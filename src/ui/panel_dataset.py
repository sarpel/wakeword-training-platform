"""
Panel 1: Dataset Management
- Dataset discovery and validation
- Train/test/validation splitting
- .npy file extraction
- Dataset health briefing
"""
import gradio as gr
from pathlib import Path
from typing import Tuple
import sys
import traceback
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.splitter import DatasetScanner, DatasetSplitter
from src.data.health_checker import DatasetHealthChecker
from src.data.npy_extractor import NpyExtractor
from src.exceptions import WakewordException
import structlog

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
                    value=str(Path(data_root) / "raw")  # Default value
                )

                gr.Markdown("**Expected Structure:**")
                gr.Markdown("""
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
                """)

                skip_validation = gr.Checkbox(
                    label="Fast Scan (Skip Validation)",
                    value=False,
                    info="Only count files without validation (much faster for large datasets)"
                )

                scan_button = gr.Button("🔍 Scan Datasets", variant="primary")
                scan_status = gr.Textbox(
                    label="Scan Status",
                    value="Ready to scan",
                    lines=1,
                    interactive=False
                )

            with gr.Column():
                gr.Markdown("### Dataset Statistics")
                stats_display = gr.JSON(
                    label="Dataset Summary",
                    value={"status": "No datasets scanned yet"}
                )

        gr.Markdown("---")

        # MOVED: NPY Feature Management to be BEFORE Splitting
        with gr.Row():
            gr.Markdown("### ⚡ NPY Feature Management (Optional but Recommended)")

        with gr.Tabs():
            # Tab 1: Batch Feature Extraction
            with gr.TabItem("⚡ Batch Feature Extraction"):
                gr.Markdown("**Performance**: Extract features NOW to speed up training by 40-60%.")

                with gr.Row():
                    with gr.Column():
                        extract_feature_type = gr.Dropdown(
                            label="Feature Type",
                            choices=["mel", "mfcc"],
                            value="mel",
                            info="Type of features to extract"
                        )
                        extract_batch_size = gr.Slider(
                            minimum=16, maximum=128, value=32, step=16,
                            label="Batch Size (GPU)",
                            info="Higher = faster but uses more memory"
                        )
                        extract_output_dir = gr.Textbox(
                            label="Output Directory (Source for Splitter)",
                            value=str(Path(data_root) / "raw" / "npy"),
                            info="Temporary location. Splitter will move these to data/npy/train|val|test"
                        )
                        batch_extract_button = gr.Button(
                            "⚡ Extract All Features to NPY",
                            variant="primary"
                        )

                    with gr.Column():
                        batch_extraction_log = gr.Textbox(
                            label="Batch Extraction Log",
                            lines=12,
                            value="Ready to extract features...\n\nRECOMMENDED WORKFLOW:\n1. 🔍 Scan Datasets (Top)\n2. ⚡ Extract Features (Here)\n3. ✂️ Split Datasets (Below)\n\nThis ensures NPY files are correctly included in your data splits.",
                            interactive=False
                        )

            # Tab 2: Analyze Existing NPY Files
            with gr.TabItem("📦 Analyze Existing NPY"):
                with gr.Row():
                    with gr.Column():
                        npy_folder = gr.Textbox(
                            label=".npy Files Directory",
                            placeholder="Path to .npy files (or leave empty to scan dataset_root/npy)",
                            lines=1
                        )
                        analyze_button = gr.Button("🔍 Analyze .npy Files", variant="secondary")
                        
                        gr.Markdown("### 📐 Shape Validation")
                        gr.Markdown("Check if NPY files match the expected input shape for training.")
                        
                        validate_shape_checkbox = gr.Checkbox(
                            label="Validate Shapes",
                            value=True,
                            info="Check dimensions against current config"
                        )
                        delete_invalid_checkbox = gr.Checkbox(
                            label="Delete Invalid Files",
                            value=False,
                            info="⚠️ Permanently delete files with mismatching shapes"
                        )
                        validate_button = gr.Button("📏 Validate & Clean Shapes", variant="secondary")

                    with gr.Column():
                        analysis_log = gr.Textbox(
                            label="Analysis Report",
                            lines=12,
                            value="Analyze existing .npy files...",
                            interactive=False
                        )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### ✂️ Train/Test/Validation Split")

                gr.Markdown("**Industry Standard Ratios:** Train: 70%, Val: 15%, Test: 15%")

                train_ratio = gr.Slider(
                    minimum=0.5, maximum=0.9, value=0.7, step=0.05,
                    label="Train Ratio",
                    info="Training set ratio (70% recommended)"
                )
                val_ratio = gr.Slider(
                    minimum=0.05, maximum=0.3, value=0.15, step=0.05,
                    label="Validation Ratio",
                    info="Validation set ratio (15% recommended)"
                )
                test_ratio = gr.Slider(
                    minimum=0.05, maximum=0.3, value=0.15, step=0.05,
                    label="Test Ratio",
                    info="Test set ratio (15% recommended)"
                )

                split_button = gr.Button("✂️ Split Datasets", variant="primary")
                split_status = gr.Textbox(
                    label="Split Status",
                    value="Step 1: Scan Datasets\nStep 2: (Optional) Extract NPY\nStep 3: Split Datasets",
                    lines=3,
                    interactive=False
                )

            with gr.Column():
                gr.Markdown("### Health Report")
                health_report = gr.Textbox(
                    label="Dataset Health Analysis",
                    lines=15,
                    value="Run dataset scan to see health report...",
                    interactive=False
                )

        gr.Markdown("---")

        # Event handlers with full implementation
        def scan_datasets_handler(root_path: str, skip_val: bool, progress=gr.Progress()) -> Tuple[dict, str, str]:
            """Scan datasets and return statistics and health report"""
            global _current_scanner, _current_dataset_info

            try:
                if not root_path:
                    return (
                        {"error": "Please provide dataset root path"},
                        "❌ No path provided",
                        "Run dataset scan to see health report..."
                    )

                root_path = Path(root_path)
                if not root_path.exists():
                    return (
                        {"error": f"Path does not exist: {root_path}"},
                        f"❌ Directory not found: {root_path}",
                        "Run dataset scan to see health report..."
                    )

                logger.info(f"Scanning datasets in: {root_path} (skip_validation={skip_val})")

                # Progress callback for Gradio
                def update_progress(progress_value, message):
                    progress(progress_value, desc=f"Scanning: {message}")

                # Create scanner with caching and parallel processing
                use_cache = not skip_val  # Only use cache when validating
                scanner = DatasetScanner(root_path, use_cache=use_cache)

                # Scan datasets with progress
                progress(0, desc="Initializing scan...")
                dataset_info = scanner.scan_datasets(
                    progress_callback=update_progress,
                    skip_validation=skip_val
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
                manifest_path = Path(root_path).parent / "splits" / "dataset_manifest.json"
                scanner.save_manifest(manifest_path)

                logger.info("Dataset scan complete")

                # Add cache info to status message
                cache_msg = ""
                if dataset_info.get('cached_files', 0) > 0:
                    cache_msg = f" ({dataset_info['cached_files']} from cache)"
                
                moved_msg = ""
                if moved_count > 0:
                    moved_msg = f"\n\nMoved {moved_count} low-quality files to 'unqualified_datasets' folder."

                mode_msg = " (fast scan)" if skip_val else ""

                progress(1.0, desc="Complete!")

                return (
                    stats,
                    f"✅ Scan complete! Found {stats['total_files']} audio files{cache_msg}{mode_msg}{moved_msg}",
                    health_report_text
                )

            except WakewordException as e:
                error_msg = f"❌ Data Error: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return (
                    {"error": error_msg},
                    f"❌ Error: {str(e)}",
                    f"Actionable suggestion: Please check your dataset for the following error: {e}"
                )
            except Exception as e:
                error_msg = f"Error scanning datasets: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return (
                    {"error": error_msg},
                    f"❌ Error: {str(e)}",
                    f"❌ Error during scan:\n{str(e)}"
                )

        def split_datasets_handler(
            root_path: str,
            train: float,
            val: float,
            test: float,
            npy_source: str,  # Added npy_source input
            progress=gr.Progress()
        ) -> Tuple[str, str]:
            """Split datasets into train/val/test"""
            global _current_dataset_info

            try:
                # Validate ratios
                total = train + val + test
                if abs(total - 1.0) > 0.01:
                    return (
                        f"❌ Ratios must sum to 1.0 (current: {total:.2f})",
                        "No split performed - fix ratios"
                    )

                if _current_dataset_info is None:
                    return (
                        "❌ Please scan datasets first",
                        "No split performed - scan required"
                    )

                logger.info(f"Splitting datasets: {train}/{val}/{test}")

                # Create splitter and split
                progress(0.1, desc="Initializing splitter...")
                splitter = DatasetSplitter(_current_dataset_info)

                # Determine npy source directory
                npy_source_path = Path(npy_source) if npy_source else Path(root_path) / "npy"
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
                    npy_source_dir=npy_source_path # Explicitly pass the source path
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

                return (
                    f"✅ Split complete! Saved to {output_dir}",
                    report_text
                )

            except WakewordException as e:
                error_msg = f"❌ Data Error: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return (
                    f"❌ Error: {str(e)}",
                    f"Actionable suggestion: Please check your dataset for the following error: {e}"
                )
            except Exception as e:
                error_msg = f"Error splitting datasets: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return (
                    f"❌ Error: {str(e)}",
                    f"❌ Error during split:\n{str(e)}"
                )

        def batch_extract_handler(
            root_path: str,
            feature_type: str,
            batch_size: int,
            output_dir: str,
            progress=gr.Progress()
        ) -> str:
            """Batch extract features to NPY files"""
            global _current_dataset_info

            try:
                # Validate dataset scanned
                if _current_dataset_info is None:
                    return "❌ Please scan datasets first (Step 1)"

                logger.info(f"Batch extracting {feature_type} features with batch_size={batch_size}")

                # Import batch extractor
                from src.data.batch_feature_extractor import BatchFeatureExtractor
                from src.config.defaults import DataConfig

                # Create config for feature extraction
                progress(0.05, desc="Initializing batch extractor...")
                config = DataConfig(
                    feature_type=feature_type,
                    sample_rate=16000,
                    audio_duration=1.5,
                    n_mels=64,
                    n_mfcc=40,
                    n_fft=400,
                    hop_length=160
                )

                # Initialize extractor
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                extractor = BatchFeatureExtractor(config=config, device=device)

                # Collect all audio files from scanned dataset
                progress(0.1, desc="Collecting audio files...")
                all_files = []
                for category, category_data in _current_dataset_info['categories'].items():
                    if category in ['positive', 'negative', 'hard_negative']:  # Skip background and rirs
                        for file_info in category_data['files']:
                            all_files.append(Path(file_info['path']))

                if not all_files:
                    return "❌ No audio files found in scanned dataset"

                logger.info(f"Found {len(all_files)} audio files to process")

                # Progress callback
                def update_progress(current, total, message):
                    progress_value = 0.1 + (current / total) * 0.8
                    progress(progress_value, desc=f"{message}")

                # Extract features
                progress(0.15, desc=f"Extracting {len(all_files)} files (batch={batch_size})...")
                output_path = Path(output_dir)
                results = extractor.extract_dataset(
                    audio_files=all_files,
                    output_dir=output_path,
                    batch_size=batch_size,
                    preserve_structure=True,
                    progress_callback=update_progress
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

                if results['failed_count'] > 0:
                    report.append("Failed Files:")
                    for failed in results['failed_files'][:10]:  # Show first 10
                        report.append(f"  - {Path(failed['path']).name}: {failed['error']}")
                    if results['failed_count'] > 10:
                        report.append(f"  ... and {results['failed_count'] - 10} more")
                    report.append("")

                report.append("=" * 60)
                report.append("Next Steps:")
                report.append("1. Split your dataset using 'Train/Test/Validation Split' section")
                report.append("   → This will automatically organize NPY files into data/npy/train|val|test")
                report.append("2. Go to Panel 2 (Configuration)")
                report.append("3. Verify 'NPY Feature Directory' is set to: data/npy")
                report.append(f"4. Verify 'NPY Feature Type' matches: {feature_type}")
                report.append("5. Start training (40-60% faster with precomputed features!)")
                report.append("=" * 60)

                report_text = "\n".join(report)

                logger.info(f"Batch extraction complete: {results['success_count']}/{results['total_files']}")

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

        def analyze_npy_handler(npy_path: str, root_path: str, progress=gr.Progress()) -> str:
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
                results = extractor.extract_and_convert(npy_files, progress_callback=update_progress)

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
            progress=gr.Progress()
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
                
                # Initialize standard config
                config = DataConfig()
                
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
                    hop_length=config.hop_length
                )
                
                expected_shape = helper_extractor.get_output_shape(target_samples)
                
                logger.info(f"Validating shapes in {npy_path} against expected: {expected_shape}")
                
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
                    progress_callback=update_progress
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
                
                if results['mismatches']:
                    report.append("")
                    report.append("Mismatch Examples:")
                    report.append("-" * 40)
                    for m in results['mismatches'][:10]:
                        report.append(f"File: {Path(m['path']).name}")
                        report.append(f"  Got: {m['actual']}")
                        report.append(f"  Exp: {m['expected']}")
                        report.append("")
                    
                    if len(results['mismatches']) > 10:
                        report.append(f"... and {len(results['mismatches']) - 10} more")

                return "\n".join(report)
                
            except Exception as e:
                return f"❌ Error during validation: {str(e)}\n{traceback.format_exc()}"

        # Connect event handlers
        scan_button.click(
            fn=scan_datasets_handler,
            inputs=[dataset_root, skip_validation],
            outputs=[stats_display, scan_status, health_report]
        )

        split_button.click(
            fn=split_datasets_handler,
            inputs=[dataset_root, train_ratio, val_ratio, test_ratio, extract_output_dir], # Added extract_output_dir
            outputs=[split_status, health_report]
        )

        batch_extract_button.click(
            fn=batch_extract_handler,
            inputs=[dataset_root, extract_feature_type, extract_batch_size, extract_output_dir],
            outputs=[batch_extraction_log]
        )

        analyze_button.click(
            fn=analyze_npy_handler,
            inputs=[npy_folder, dataset_root],
            outputs=[analysis_log]
        )

        validate_button.click(
            fn=validate_shapes_handler,
            inputs=[npy_folder, dataset_root, delete_invalid_checkbox],
            outputs=[analysis_log]
        )

    return panel


if __name__ == "__main__":
    # Test the panel
    demo = create_dataset_panel()
    demo.launch()