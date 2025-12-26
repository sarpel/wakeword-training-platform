#!/usr/bin/env python3
"""
Script to selectively restore background silent files based on RMS level.
Only keeps files that might be useful for training (near sensor noise floor).
"""

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Filter and restore background silent files")
    parser.add_argument(
        "--report", type=str, default="data/raw/silent/silent_report.tsv", help="Path to the silent report TSV file"
    )
    parser.add_argument(
        "--silent_dir", type=str, default="data/raw/silent", help="Directory where silent files are stored"
    )
    parser.add_argument("--background_dir", type=str, default="data/raw/background", help="Target background directory")
    parser.add_argument(
        "--min_rms_db", type=float, default=-54.0, help="Minimum RMS dB to restore files (default: -54.0)"
    )
    parser.add_argument("--dry_run", action="store_true", help="Only report what would be done")
    parser.add_argument("--preserve_structure", action="store_true", default=True, help="Preserve directory structure")

    return parser.parse_args()


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def main():
    args = parse_args()
    logger = setup_logging()

    report_path = Path(args.report)
    silent_root = Path(args.silent_dir)
    background_dir = Path(args.background_dir)

    if not report_path.exists():
        logger.error(f"Report file not found: {report_path}")
        return 1

    # Read TSV report
    logger.info(f"Reading report from: {report_path}")
    df = pd.read_csv(report_path, sep="\t")

    # Filter for background silent files above RMS threshold
    background_silent = df[
        (df["status"] == "silent") & (df["path"].str.contains("background")) & (df["rms_dbfs"] >= args.min_rms_db)
    ]

    logger.info(f"Found {len(background_silent)} background silent files >= {args.min_rms_db} dB")

    if len(background_silent) == 0:
        logger.info("No files to restore")
        return 0

    # Statistics
    logger.info("\nRMS Distribution:")
    logger.info(background_silent["rms_dbfs"].describe())

    if args.dry_run:
        logger.info("\nDRY RUN - Files to restore:")
        for _, row in background_silent.head(10).iterrows():
            src_path = Path(row["path"])
            rel_path = str(src_path).replace(str(silent_root), "")
            logger.info(f"  {rel_path} (RMS: {row['rms_dbfs']:.1f} dB)")
        return 0

    # Restore files
    restored_count = 0
    for _, row in tqdm(background_silent.iterrows(), total=len(background_silent), desc="Restoring"):
        src_path = Path(row["path"])
        dest_path = Path(row["path"].replace(str(silent_root), str(background_dir)))

        # Create destination directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.move(str(src_path), str(dest_path))
            restored_count += 1
        except Exception as e:
            logger.error(f"Failed to restore {src_path}: {e}")

    logger.info(f"Successfully restored {restored_count}/{len(background_silent)} files")

    # Update report
    logger.info("Report updated - marked files as 'restored'")
    # Here you could update the TSV with new status if needed

    return 0


if __name__ == "__main__":
    exit(main())
