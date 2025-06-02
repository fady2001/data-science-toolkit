"""
Data and model saving utilities.

This module provides a centralized Saver class with static methods for saving
datasets in various formats (CSV, Parquet, NPY) and machine learning models
using pickle serialization, with proper directory creation and logging.
"""

import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.globals import logger


class Saver:
    """
    A utility class for saving datasets and models to disk.
    
    Provides static methods for saving data in different formats with
    automatic directory creation and comprehensive logging.
    """
    
    @staticmethod
    def save_dataset_csv(dataset: pd.DataFrame, file_path: Path, file_name: str) -> None:
        """
        Save a pandas DataFrame to CSV format.
        
        Args:
            dataset (pd.DataFrame): The dataset to save
            file_path (Path): Directory path where the file will be saved
            file_name (str): Name of the CSV file (without extension)
            
        Returns:
            None
        """
        if os.path.exists(file_path) is False:
            os.makedirs(file_path)
        file_path = Path(file_path) / f"{file_name}"
        dataset.to_csv(file_path, sep=",")
        logger.success(f"Dataset saved to {file_path}.")
        
    @staticmethod
    def save_dataset_parquet(dataset: pd.DataFrame, file_path: Path) -> None:
        """
        Save a pandas DataFrame to Parquet format.
        
        Args:
            dataset (pd.DataFrame): The dataset to save
            file_path (Path): Full file path including filename
            
        Returns:
            None
        """
        if os.path.exists(dir) is False:
            os.makedirs(dir)
        dataset.to_parquet(file_path, engine='pyarrow')
        logger.success(f"Dataset saved to {file_path}.")
        
    @staticmethod
    def save_dataset_npy(dataset: np.ndarray, file_path: Path, file_name: str) -> None:
        """
        Save a numpy array to NPY format.
        
        Args:
            dataset (np.ndarray): The numpy array to save
            file_path (Path): Directory path where the file will be saved
            file_name (str): Name of the NPY file (without extension)
            
        Returns:
            None
        """
        if os.path.exists(file_path) is False:
            os.makedirs(file_path)
        file_path = Path(file_path) / f"{file_name}.npy"
        np.save(file_path, dataset)
        logger.success(f"Dataset saved to {file_path}.")

    @staticmethod
    def save_model(model, model_path: Path, model_name: str) -> None:
        """
        Save a machine learning model using pickle serialization.
        
        Args:
            model: The trained model object to save
            model_path (Path): Directory path where the model will be saved
            model_name (str): Name of the model file (without extension)
            
        Returns:
            None
        """
        if os.path.exists(model_path) is False:
            os.makedirs(model_path)
        model_path = Path(model_path) / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.success(f"Model saved to {model_path}.")
