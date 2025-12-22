import json
import logging
from pathlib import Path
from typing import List, Dict

def extract_false_positives(log_path: str, threshold: float = 0.5) -> List[Dict]:
    """
    Extract False Positives from evaluation logs (JSONL).
    
    Args:
        log_path: Path to log file
        threshold: Confidence threshold to consider as high confidence
        
    Returns:
        List of FP entries
    """
    fps = []
    path_obj = Path(log_path)
    
    if not path_obj.exists():
        print(f"File not found: {log_path}")
        return []

    with open(path_obj, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Check if FP: label=0 (negative) AND score > threshold
                # OR label=0 AND prediction=1 (if threshold not provided, but we use threshold)
                
                # Normalize label/prediction keys if needed (assuming 'label', 'score')
                label = entry.get("label")
                score = entry.get("score")
                
                if label is not None and score is not None:
                    # Ensure label is int (0 or 1)
                    if int(label) == 0:
                        if float(score) >= threshold:
                            fps.append(entry)
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error parsing line: {e}")
                continue
        
    return fps

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract False Positives from evaluation logs")
    parser.add_argument("log_path", help="Path to evaluation log (JSONL)")
    parser.add_argument("--threshold", type=float, default=0.9, help="Confidence threshold")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()
    
    fps = extract_false_positives(args.log_path, args.threshold)
    print(f"Found {len(fps)} False Positives with score >= {args.threshold}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(fps, f, indent=2)
        print(f"Saved to {args.output}")
    else:
        for fp in fps:
            path_info = fp.get('path', fp.get('file', 'unknown'))
            print(f"{path_info}: {fp.get('score', 0.0):.4f}")
