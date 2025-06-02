"""
Dataset loading and handling utilities.

This module provides functions for loading datasets from various file formats,
particularly CSV files, with proper error handling and logging.
"""

from pathlib import Path

import pandas as pd

from src.globals import logger


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
