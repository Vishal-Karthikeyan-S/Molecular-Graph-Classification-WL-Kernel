#!/usr/bin/env python3
"""
CLI script to execute the full benchmark experiments.
"""

import argparse
from src.experiments import run_benchmark

def main():
    parser = argparse.ArgumentParser(description="Run Molecular Graph Classification Benchmark on NCI1")
    parser.add_argument("--subset", type=int, default=1200, help="Number of graphs to sample (default: 1200, set 0 for all 4110)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split")
    args = parser.parse_args()

    subset_arg = None if args.subset <= 0 else args.subset
    run_benchmark(subset_size=subset_arg, random_seed=args.seed)

if __name__ == "__main__":
    main()
