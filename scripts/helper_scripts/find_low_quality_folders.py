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
    # Run scan (skip validation for speed if just checking paths,
    # but we need validation to get quality scores, so we can't skip it)
    scanner.scan_datasets(skip_validation=False)

    # Get low quality folders for negative category
    folders = scanner.get_low_quality_folders(category="negative")

    print(f"\nFound {len(folders)} folders containing low quality negative files:")
    for folder in folders:
        print(folder)

    # Also check RIRs to verify the fix (optional, but good for verification)
    rirs = scanner.dataset_info["categories"].get("rirs", {})
    rir_warnings = len(rirs.get("quality_warnings", []))
    print(f"\nRIR Quality Warnings: {rir_warnings}")
    if rir_warnings == 0:
        print("SUCCESS: RIRs no longer have quality warnings!")
    else:
        print(f"WARNING: RIRs still have {rir_warnings} warnings.")


if __name__ == "__main__":
    main()
