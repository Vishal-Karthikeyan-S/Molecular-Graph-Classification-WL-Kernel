#!/usr/bin/env python3
"""
CLI script to automatically download and verify the NCI1 molecular dataset.
"""

import sys
import os
from src.data_loader import load_nci1_dataset, inspect_dataset, save_dataset_metadata

def main():
    print("=== Downloading and Preparing NCI1 Dataset ===")
    dataset = load_nci1_dataset("data")
    print(f"Successfully loaded {len(dataset)} molecular graphs.")
    stats = inspect_dataset(dataset)
    save_dataset_metadata(stats, "results/metrics/dataset_stats.json")
    print("Done! Ready for experiments.")

if __name__ == "__main__":
    main()
