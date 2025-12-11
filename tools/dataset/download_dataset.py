#!/usr/bin/env python3
"""
Download the FRAMES benchmark dataset (test and validation splits).

Usage:
    uv run tools/dataset/download_dataset.py

Output:
    data/dataset/test.json (100 items)
    data/dataset/val.json (100 items)
"""

import json
import os

import fire
from datasets import load_dataset


WIKI_LINK_FIELDS = [f"wikipedia_link_{i}" for i in range(1, 12)] + [
    "wikipedia_link_11+"
]


def remove_wikipedia_links(example):
    """Remove wikipedia_link_1 ... wikipedia_link_11+ keys if present."""
    return {k: v for k, v in example.items() if k not in WIKI_LINK_FIELDS}


def main(
    output_dir: str = None,
    test_range: str = "0:100",
    val_range: str = "100:200",
):
    """
    Download FRAMES benchmark dataset.

    Args:
        output_dir: Output directory (default: data/dataset/)
        test_range: Index range for test set (default: "0:100")
        val_range: Index range for validation set (default: "100:200")
    """
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    if output_dir is None:
        output_dir = os.path.join(project_root, "data", "dataset")

    os.makedirs(output_dir, exist_ok=True)

    # Parse ranges
    test_start, test_end = map(int, test_range.split(":"))
    val_start, val_end = map(int, val_range.split(":"))

    print("Loading FRAMES benchmark dataset...")
    dataset = load_dataset("google/frames-benchmark", split="test")

    # Select subsets
    test_subset = dataset.select(range(test_start, test_end))
    val_subset = dataset.select(range(val_start, val_end))

    # Remove wikipedia_link fields and convert to list
    test_data = [remove_wikipedia_links(x) for x in test_subset.to_list()]
    val_data = [remove_wikipedia_links(x) for x in val_subset.to_list()]

    # Save test set
    test_path = os.path.join(output_dir, "test.json")
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    # Save validation set
    val_path = os.path.join(output_dir, "val.json")
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(test_data)} items to {test_path}")
    print(f"Saved {len(val_data)} items to {val_path}")


if __name__ == "__main__":
    fire.Fire(main)
