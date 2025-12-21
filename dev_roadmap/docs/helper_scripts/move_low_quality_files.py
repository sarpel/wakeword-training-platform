import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.splitter import DatasetScanner

def main():
    # Default dataset path
    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / "data" / "raw"
    
    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])
        
    print(f"Scanning dataset at: {dataset_path}")
    
    scanner = DatasetScanner(dataset_path)
    # Run scan (must include validation to get quality scores)
    scanner.scan_datasets(skip_validation=False)
    
    print("\nMoving low quality files...")
    # Move files with score < 100 (i.e., any warnings)
    moved_count = scanner.move_low_quality_files(threshold=100.0)
    
    print(f"\nOperation complete. Moved {moved_count} files to 'lowquality' folders.")

if __name__ == "__main__":
    main()
