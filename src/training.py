"""
Model training utilities and hyperparameter tuning.

This module provides functions for training machine learning models and performing
hyperparameter optimization using RandomizedSearchCV. It includes comprehensive
logging and error handling for the training process.
"""

from typing import Dict, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

from src.globals import logger


def train(model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    Train a machine learning model with the provided training data.

    Args:
        model (BaseEstimator): Scikit-learn compatible model to train
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target values

    Returns:
        None

    Raises:
        ValueError: If model is None
    """
    if model is None:
        logger.error("Model is None.")
        raise ValueError("Model is None.")
    model.fit(X_train, y_train)
    logger.success("Model trained.")


def train_RandomizedSearchCV(
    model: BaseEstimator,
    cfg: DictConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[BaseEstimator, Dict]:
    """
    Perform hyperparameter tuning using RandomizedSearchCV.

    Uses the configuration to define the parameter search space and performs
    randomized search with cross-validation to find the best hyperparameters.

    Args:
        model (BaseEstimator): Scikit-learn compatible model to tune
        cfg (DictConfig): Configuration containing tuning parameters
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target values

    Returns:
        Tuple[BaseEstimator, Dict]: Best estimator and best parameters found

    Raises:
        ValueError: If model is None
    """
    if model is None:
        logger.error("Model is None.")
        raise ValueError("Model is None.")

    params = OmegaConf.to_container(cfg["tuning"]["random_forest"], resolve=True)
    n_iter = cfg["tuning"]["n_iter"]
    cv = cfg["tuning"]["cv"]
    search = RandomizedSearchCV(model, params, n_iter=n_iter, cv=cv, verbose=2)
    search.fit(X_train, y_train)

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best score: {search.best_score_}")

    logger.success("Randomized Search CV completed.")
    return search.best_estimator_, search.best_params_
