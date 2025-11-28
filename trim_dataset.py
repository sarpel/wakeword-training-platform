import sys
import argparse
from pathlib import Path
from src.data.splitter import DatasetScanner

def main():
    parser = argparse.ArgumentParser(description="Trim silence from audio dataset")
    parser.add_argument("--dataset", type=str, default="data/raw", help="Path to dataset root")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: data/trimmed_dataset)")
    parser.add_argument("--top_db", type=float, default=20.0, help="Silence threshold in dB (default: 20.0)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite original files (DANGEROUS)")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    print(f"Scanning dataset at: {dataset_path}")
    
    scanner = DatasetScanner(dataset_path)
    # Run scan (skip validation for speed, we just need file paths)
    scanner.scan_datasets(skip_validation=True)
    
    print(f"\nTrimming silence (threshold={args.top_db}dB)...")
    if args.overwrite:
        print("WARNING: Overwriting original files!")
        
    count = scanner.trim_dataset_silence(
        output_dir=args.output,
        top_db=args.top_db,
        overwrite=args.overwrite
    )
    
    print(f"\nOperation complete. Processed {count} files.")

if __name__ == "__main__":
    main()
