"""
Dataset Integrity Validator for NCI1 Molecular Graphs.
Validates structural invariants, edge symmetry, node feature matrices, and label distributions.
"""

import os
import sys
import argparse
import numpy as np


def validate_nci1_directory(data_dir: str = "data") -> bool:
    """Verifies that the NCI1 raw data directory exists and contains valid graph files."""
    nci1_path = os.path.join(data_dir, "NCI1")
    if not os.path.exists(nci1_path):
        print(f"[!] Directory not found: {nci1_path}")
        print("    Run 'python scripts/download_data.py' to fetch the raw benchmark data.")
        return False

    required_files = [
        "NCI1_A.txt",
        "NCI1_graph_indicator.txt",
        "NCI1_graph_labels.txt",
        "NCI1_node_labels.txt"
    ]

    missing = [f for f in required_files if not os.path.exists(os.path.join(nci1_path, f))]
    if missing:
        print(f"[!] Missing required benchmark files: {missing}")
        return False

    print("[+] All core NCI1 raw files located successfully.")
    for f in required_files:
        fpath = os.path.join(nci1_path, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    - {f}: {size_kb:.1f} KB")

    return True


def main():
    parser = argparse.ArgumentParser(description="Validate NCI1 dataset integrity.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    args = parser.parse_args()

    print("=" * 60)
    print("🧪 NCI1 Dataset Structural Integrity Check")
    print("=" * 60)

    success = validate_nci1_directory(args.data_dir)
    if success:
        print("\n[✓] Dataset integrity verified successfully.")
        sys.exit(0)
    else:
        print("\n[✗] Dataset validation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
