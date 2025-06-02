"""
Model evaluation utilities and metrics calculation.

This module provides functions for evaluating trained machine learning models,
generating performance reports, and saving evaluation results to JSON files.
"""

import json
import os

import numpy as np
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from skore import EstimatorReport

from src.globals import logger


def evaluate(cfg: DictConfig, final_model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluate a trained model and generate a comprehensive evaluation report.

    Creates an evaluation report with various metrics including accuracy, precision,
    recall, and timing information. Saves the report as a JSON file in the reports directory.

    Args:
        cfg (DictConfig): Hydra configuration object containing paths and model settings
        final_model (BaseEstimator): Trained scikit-learn model to evaluate
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test target values

    Returns:
        None
    """
    final_report = EstimatorReport(final_model, X_test=X_test, y_test=y_test)
    logger.info("creating evaluation report")
    evaluation_report = {
        "model_name": cfg["names"]["model_name"],
        "estimator_name": final_report.estimator_name_,
        "fitting_time": final_report.fit_time_,
        "accuracy": final_report.metrics.accuracy(),
        "precision": final_report.metrics.precision(),
        "recall": final_report.metrics.recall(),
        "prediction_time": final_report.metrics.timings(),
    }
    logger.info("saving evaluation report")
    if not os.path.exists(
        os.path.join(cfg["paths"]["reports_parent_dir"], cfg["names"]["model_name"])
    ):
        os.makedirs(os.path.join(cfg["paths"]["reports_parent_dir"], cfg["names"]["model_name"]))
    with open(
        os.path.join(
            cfg["paths"]["reports_parent_dir"],
            cfg["names"]["model_name"],
            "evaluation_report.json",
        ),
        "w",
    ) as js:
        json.dump(evaluation_report, js, indent=4)