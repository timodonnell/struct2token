#!/usr/bin/env python3
"""Download and prepare benchmark datasets (CASP14, CASP15).

Creates index parquets at data/benchmark/{dataset}_index.parquet.

Usage:
    uv run scripts/prepare_benchmark_data.py [--mmcif_dir PATH] [--datasets casp14 casp15]
"""

from __future__ import annotations

import argparse

from struct2token.benchmark.datasets import DATASET_TARGETS, prepare_benchmark_index


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets")
    parser.add_argument(
        "--mmcif_dir",
        type=str,
        default=None,
        help="Path to local mmCIF mirror (structures not found here will be downloaded)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["casp14", "casp15"],
        choices=list(DATASET_TARGETS.keys()),
        help="Which datasets to prepare",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/benchmark",
        help="Output directory for index parquets",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="Directory to store downloaded structures (default: data/benchmark/structures)",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        output_path = f"{args.output_dir}/{dataset}_index.parquet"
        prepare_benchmark_index(
            dataset_name=dataset,
            mmcif_dir=args.mmcif_dir,
            output_path=output_path,
            download_dir=args.download_dir,
        )
        print()

    print("Done! Index parquets are ready for benchmarking.")


if __name__ == "__main__":
    main()
