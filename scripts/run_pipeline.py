#!/usr/bin/env python3
"""
Standalone script to run JunRAG pipelines.

Usage:
    uv run scripts/run_pipeline.py naive --query "Question" --metadata_path metadata.json
    uv run scripts/run_pipeline.py parallel --query "Complex question" --metadata_path metadata.json
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from junrag.cli.pipeline import main

if __name__ == "__main__":
    main()
