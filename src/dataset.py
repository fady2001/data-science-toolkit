from pathlib import Path

from globals import logger
import pandas as pd


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
