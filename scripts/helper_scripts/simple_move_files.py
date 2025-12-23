import pandas as pd
from pathlib import Path
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    report_path = Path('C:/Users/Sarpel/Desktop/project_1/data/raw/silent/silent_report.tsv')
    to_deleted_dir = Path('C:/Users/Sarpel/Desktop/project_1/to_be_deleted')
    
    # Create target directories
    bg_silent_dir = to_deleted_dir / 'background_silent'
    rirs_dir = to_deleted_dir / 'rirs_all'
    neg_silent_dir = to_deleted_dir / 'negative_silent'
    
    bg_silent_dir.mkdir(parents=True, exist_ok=True)
    rirs_dir.mkdir(parents=True, exist_ok=True)
    neg_silent_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info('🚀 MOVING FILES NOW...')
    
    # Read report with proper tab separation
    df = pd.read_csv(report_path, sep='\t')
    silent_df = df[df['status'] == 'silent'].copy()
    logger.info(f'Found {len(silent_df)} silent files to analyze')
    
    # Lists to hold categorized files
    delete_background = []
    delete_rirs = []
    delete_negative = []
    
    # Process each silent file
    for idx, row in silent_df.iterrows():
        path = str(row['path']).strip()
        rms_value = float(row['rms_dbfs']) if pd.notna(row['rms_dbfs']) else -240.0
        duration = float(row['duration_sec']) if pd.notna(row['duration_sec']) else 1.0
        
        if not path:
            continue
            
        # Skip very large files
        if duration > 10.0:
            logger.warning(f'Skipping large file: {Path(path).name} (duration: {duration}s)')
            continue
            
        file_path = Path(path)
        
        # Categorize by type
        if 'background' in path.lower():
            if rms_value < -54.0:  # Only move very silent background
                delete_background.append(str(file_path))
        elif any(rir in path.lower() for rir in ['rirs', 'air_rir', 'sim_rir']):
            # ALL RIR files should be deleted
            delete_rirs.append(str(file_path))
        elif 'negative' in path.lower():
            if rms_value < -54.0:  # Only move very silent negatives 
                delete_negative.append(str(file_path))
    
    logger.info(f'Moving {len(delete_background)} background, {len(delete_rirs)} RIR, and {len(delete_negative)} silent negative files')
    
    moved_count = 0
    
    # Move files
    for file_path in delete_background:
        if Path(file_path).exists():
            try:
                target = bg_silent_dir / Path(file_path).name
                shutil.move(file_path, target)
                moved_count += 1
                if moved_count % 5000 == 0:
                    logger.info(f'Moved {moved_count} files...')
            except Exception as e:
                logger.error(f'Error moving {file_path}: {e}')
    
    for file_path in delete_rirs:
        if Path(file_path).exists():
            try:
                target = rirs_dir / Path(file_path).name
                shutil.move(file_path, target)
                moved_count += 1
                if moved_count % 5000 == 0:
                    logger.info(f'Moved {moved_count} files...')
            except Exception as e:
                logger.error(f'Error moving {file_path}: {e}')
    
    for file_path in delete_negative:
        if Path(file_path).exists():
            try:
                target = neg_silent_dir / Path(file_path).name
                shutil.move(file_path, target)
                moved_count += 1
            except Exception as e:
                logger.error(f'Error moving {file_path}: {e}')
    
    logger.info(f'✅ Successfully moved {moved_count:,} files to {to_deleted_dir}')
    
    return moved_count

if __name__ == '__main__':
    main()