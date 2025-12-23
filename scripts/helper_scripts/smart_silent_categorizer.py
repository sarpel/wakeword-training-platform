import pandas as pd
from pathlib import Path
import shutil
import logging
import argparse
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Smart silent audio categorizer - keep useful files, move others to be deleted")
    parser.add_argument("--report", default="C:/Users/Sarpel/Desktop/project_1/data/raw/silent/silent_report.tsv", 
                       help="Path to the silent report TSV file")
    parser.add_argument("--run", action="store_true", 
                       help="Actually move files (without this flag, shows preview only)")
    parser.add_argument("--to_deleted_dir", default="C:/Users/Sarpel/Desktop/project_1/to_be_deleted",
                       help="Directory to move files to be deleted")
    
    args = parser.parse_args()
    
    report_path = Path(args.report)
    to_deleted_dir = Path(args.to_deleted_dir)
    
    # Read report with proper tab separation
    try:
        df = pd.read_csv(report_path, sep="\t")
        logging.info(f"Loaded {len(df)} files from report")
        print("Column names:", df.columns.tolist())  # Debug column names
        print("First few rows:")
        print(df.head())
    except Exception as e:
        logger.error(f"Error reading report file: {e}")
        return
    
    if args.run:
        logger.info("🚀 RUNNING MODE - Files will be moved")
    else:
        logger.info("🔍 PREVIEW MODE - No files will be moved")
    
    # Categorize silent files
    keep_background = []
    delete_background = []
    delete_rirs = []
    delete_negative = []
    keep_others = []
    
    logger.info("📊 Analyzing files...")
    
    # Process each row
    for idx, row in df.iterrows():
        status = str(row.iloc[0]).strip()  # First column is status
        path = str(row.iloc[1]).strip()    # Second column is path
        
        if pd.isna(path) or path == "":
            continue
            
        file_path = Path(path)
        
        if status == "silent":
            # Get RMS value from rms_dbfs column (column index 3 based on our inspection)
            rms_value = float(row.iloc[2]) if len(row) > 2 and not pd.isna(row.iloc[2]) else -240.0
            duration = float(row.iloc[3]) if len(row) > 3 and not pd.isna(row.iloc[3]) else 1.0
            
            # Skip damaged audio files
            if duration < 0.25 or duration > 10.0:
                logger.warning(f"Skipping unusual duration file: {file_path.name} (duration: {duration}s)")
                continue
            
            if "background" in path.lower():
                if rms_value >= -54.0:
                    keep_background.append((str(file_path), rms_value, duration))
                else:
                    delete_background.append((str(file_path), rms_value, duration))
            elif "rirs" in path.lower() or "air_rir" in path.lower() or "sim_rir" in path.lower():
                delete_rirs.append((str(file_path), rms_value, duration))
            elif "negative" in path.lower():
                if rms_value >= -54.0:
                    keep_others.append((str(file_path), rms_value, duration))
                else:
                    delete_negative.append((str(file_path), rms_value, duration))
            elif "positive" in path.lower():
                # Keep almost all positives, only delete truly silent ones
                if rms_value < -54.0:
                    keep_others.append((str(file_path), rms_value, duration))
    
    # Print summary
    logger.info("📈 RESULTS SUMMARY:")
    logger.info(f"  Keep Background (useful noise): {len(keep_background):,} files")
    logger.info(f"  Delete Background (too silent): {len(delete_background):,} files")
    logger.info(f"  Delete All RIRs: {len(delete_rirs):,} files")
    logger.info(f"  Delete Negative: {len(delete_negative):,} files")
    logger.info(f"  Keep Other positives: {len(keep_others):,} files")
    
    logger.info(f"\\n💾 SPACE SAVINGS:")
    total_delete = len(delete_background) + len(delete_rirs) + len(delete_negative)
    logger.info(f"  Files to delete: {total_delete:,}")
    logger.info(f"  Files to keep: {len(keep_background) + len(keep_others):,}")
    
    if args.run and total_delete > 0:
        logger.info(f"\\n🚚 Moving files...")
        
        # Create target directories
        bg_silent_dir = to_deleted_dir / "background_silent"
        rirs_dir = to_deleted_dir / "rirs_all"
        neg_silent_dir = to_deleted_dir / "negative_silent"
        
        bg_silent_dir.mkdir(parents=True, exist_ok=True)
        rirs_dir.mkdir(parents=True, exist_ok=True)
        neg_silent_dir.mkdir(parents=True, exist_ok=True)
        
        moved_count = 0
        
        # Move background silent files
        for file_path, rms, duration in delete_background:
            if Path(file_path).exists():
                try:
                    target = bg_silent_dir / Path(file_path).name
                    shutil.move(file_path, target)
                    moved_count += 1
                except Exception as e:
                    logger.error(f"Error moving {file_path}: {e}")
        
        # Move all RIR files
        for file_path, rms, duration in delete_rirs:
            if Path(file_path).exists():
                try:
                    target = rirs_dir / Path(file_path).name
                    shutil.move(file_path, target)
                    moved_count += 1
                except Exception as e:
                    logger.error(f"Error moving {file_path}: {e}")
        
        # Move silent negative files
        for file_path, rms, duration in delete_negative:
            if Path(file_path).exists():
                try:
                    target = neg_silent_dir / Path(file_path).name
                    shutil.move(file_path, target)
                    moved_count += 1
                except Exception as e:
                    logger.error(f"Error moving {file_path}: {e}")
        
        logger.info(f"✅ Successfully moved {moved_count:,} files to {to_deleted_dir}")
        
        # Save CSV with kept files marked as "keep"
        logger.info("\\n📝 Updating report file...")
        try:
            df_updated = df.copy()
            for file_path, _, _ in delete_background + delete_rirs + delete_negative: 
                mask = df_updated.iloc[:, 1].astype(str).str.strip() == str(file_path)        
                if mask.any():
                    df_updated.loc[mask, df_updated.columns[0]] = "moved_to_delete"   
            
            # Mark kept background as "keep_useful"
            for file_path, _, _ in keep_background:
                mask = df_updated.iloc[:, 1].astype(str).str.strip() == str(file_path)
                if mask.any():
                    df_updated.loc[mask, df_updated.columns[0]] = "keep_useful"       
            
            # Save with proper tab separator
            df_updated.to_csv(report_path, sep="\t", index=False)
            logger.info(f"✅ Updated report file: {report_path}")
        except Exception as e:
            logger.error(f"Error updating report: {e}")
    
    else:
        logger.info("\\n💡 To actually move files, run with --run flag")
        logger.info("Example: python smart_silent_categorizer.py --run")

if __name__ == "__main__":
    main()
