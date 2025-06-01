import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

# from dataset.dataset import Dataset
# from dataset.preprocessor import Preprocessor
from src.globals import logger


class Saver:
    @staticmethod
    def save_dataset_csv(dataset: pd.DataFrame, file_path: Path, file_name:str) -> None:
        """
        Save the dataset to the specified directory.
        """
        if os.path.exists(file_path) is False:
            os.makedirs(file_path)
        file_path = Path(file_path) / f"{file_name}.csv"
        dataset.to_csv(file_path, sep=",")
        logger.success(f"Dataset saved to {file_path}.")
        
    def save_dataset_parquet(dataset: pd.DataFrame, file_path: Path) -> None:
        """
        Save the dataset to the specified directory in Parquet format.
        """
        if os.path.exists(dir) is False:
            os.makedirs(dir)
        dataset.to_parquet(file_path,engine='pyarrow')
        logger.success(f"Dataset saved to {file_path}.")
        
    def save_dataset_npy(dataset: np.ndarray, file_path: Path, file_name:str) -> None:
        """
        Save the dataset to the specified directory in NPY format.
        """
        if os.path.exists(file_path) is False:
            os.makedirs(file_path)
        file_path = Path(file_path) / f"{file_name}.npy"
        np.save(file_path, dataset)
        logger.success(f"Dataset saved to {file_path}.")

    @staticmethod
    def save_model(model, model_path: Path, model_name:str) -> None:
        """
        Save the model to the specified directory.
        """
        if os.path.exists(model_path) is False:
            os.makedirs(model_path)
        model_path = Path(model_path) / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.success(f"Model saved to {model_path}.")
