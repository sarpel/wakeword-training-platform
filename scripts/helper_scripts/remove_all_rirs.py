#!/usr/bin/env python3
"""
Script to safely remove all RIR files from silent directory.
RIR files are typically not useful for ESP32S3 TinyConv training.
"""

import argparse
import logging
from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Remove all RIR files from silent directory")
    parser.add_argument("--silent_dir", type=str,
                       default="data/raw/silent",
                       help="Directory where silent files are stored")
    parser.add_argument("--report", type=str,
                       default="data/raw/silent/silent_report.tsv",
                       help="Path to the silent report TSV file")
    parser.add_argument("--dry_run", action="store_true",
                       help="Only report what would be done")
    parser.add_argument("--backup", action="store_true",
                       help="Create backup before deletion")
    parser.add_argument("--backup_dir", type=str,
                       default="data/raw/rirs_backup",
                       help="Backup directory for RIR files")
    
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    args = parse_args()
    logger = setup_logging()
    
    report_path = Path(args.report)
    backup_dir = Path(args.backup_dir)
    
    if not report_path.exists():
        logger.error(f"Report file not found: {report_path}")
        return 1
    
    # Read TSV report and find RIR files
    logger.info(f"Reading report from: {report_path}")
    df = pd.read_csv(report_path, sep='\t')
    
    # Find all silent RIR files
    rir_silent = df[
        (df['status'] == 'silent') & 
        (df['path'].str.contains('RIRs', case=False, na=False))
    ]
    
    logger.info(f"Found {len(rir_silent)} RIR silent files to process")
    
    if len(rir_silent) == 0:
        logger.info("No RIR files found")
        return 0
    
    # Calculate total size
    total_size_mb = 0
    for _, row in rir_silent.iterrows():
        file_path = Path(row['path'])
        if file_path.exists():
            total_size_mb += file_path.stat().st_size / (1024 * 1024)
    
    logger.info(f"Total size to remove: {total_size_mb:.1f} MB")
    
    if args.dry_run:
        logger.info("\nDRY RUN - RIR files to remove:")
        for _, row in rir_silent.head(10).iterrows():
            logger.info(f"  {Path(row['path']).name} (RMS: {row['rms_dbfs']:.1f} dB)")
        return 0
    
    # Create backup if requested
    if args.backup:
        logger.info(f"Creating backup in: {backup_dir}")
        backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove/back up files
    removed_count = 0
    backup_count = 0
    
    for _, row in tqdm(rir_silent.iterrows(), total=len(rir_silent), desc="Processing RIRs"):
        src_path = Path(row['path'])
        
        if not src_path.exists():
            logger.warning(f"File not found: {src_path}")
            continue
        
        if args.backup:
            # Copy to backup
            backup_path = backup_dir / src_path.name
            try:
                shutil.copy2(src_path, backup_path)
                backup_count += 1
            except Exception as e:
                logger.error(f"Failed to backup {src_path}: {e}")
                continue
        
        # Remove original
        try:
            src_path.unlink()
            removed_count += 1
        except Exception as e:
            logger.error(f"Failed to remove {src_path}: {e}")
    
    logger.info(f"Successfully processed {removed_count} RIR files")
    if args.backup:
        logger.info(f"Created {backup_count} backup files in {backup_dir}")
    
    # Update report (optional - you'd need to modify the TSV)
    logger.info("Note: You may want to update the report TSV to remove these entries")
    
    return 0

if __name__ == "__main__":
    exit(main())