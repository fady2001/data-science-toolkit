"""
Data preprocessing utilities and pipeline construction.

This module provides a flexible preprocessor class that can handle various
preprocessing steps including encoding, scaling, and imputation based on
configuration. It uses scikit-learn's ColumnTransformer for efficient
column-wise transformations.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)


class Preprocessor:
    """
    A flexible data preprocessor that applies various transformations based on configuration.

    This class creates a scikit-learn pipeline with encoding, scaling, and imputation
    steps applied to different columns based on the provided configuration.
    """

    def __init__(
        self,
        pipeline_config: Dict[str, str] = None,
    ):
        """
        Initialize the preprocessor with configuration.

        Args:
            pipeline_config (Dict[str, str], optional): Configuration dictionary
                specifying preprocessing steps for different columns
        """
        self.pipeline_config = pipeline_config or {}

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        """
        Fit the preprocessing pipeline to the training data.

        Args:
            X (pd.DataFrame): Training features

        Returns:
            Preprocessor: Returns self for method chaining
        """
        transformers = []

        transformers.append(("drop_columns", "drop", self.pipeline_config.get("drop", [])))
        transformers.extend(self.__create_encode_steps())
        transformers.extend(self.__create_scaling_steps())
        transformers.extend(self.__create_imputing_steps())

        preprocessor = ColumnTransformer(transformers=transformers)

        self.pipeline = preprocessor.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the input data using the fitted pipeline.

        Args:
            X (pd.DataFrame): Input features to transform

        Returns:
            np.ndarray: Transformed features

        Raises:
            ValueError: If pipeline is not fitted
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() before transform().")
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit the pipeline and transform the data in one step.

        Args:
            X (pd.DataFrame): Input features

        Returns:
            np.ndarray: Transformed features
        """
        self.fit(X)
        return self.pipeline.fit_transform(X)

    def get_pipeline(self) -> ColumnTransformer:
        """
        Get the fitted scikit-learn pipeline.

        Returns:
            ColumnTransformer: The fitted preprocessing pipeline

        Raises:
            ValueError: If pipeline is not fitted
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() before get_pipeline().")
        return self.pipeline

    def __create_encode_steps(self) -> List[Tuple[str, BaseEstimator, List[str]]]:
        """
        Create encoding transformation steps based on configuration.

        Returns:
            List[Tuple[str, BaseEstimator, List[str]]]: List of encoding transformers
        """
        encode_steps = []
        if self.pipeline_config.get("encoding"):
            for strategy in self.pipeline_config["encoding"].keys():
                if strategy == "onehot":
                    cols = self.pipeline_config["encoding"]["onehot"]
                    encode_steps.append(
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                            ),
                            cols,
                        )
                    )
                elif strategy == "ordinal":
                    cols = self.pipeline_config["encoding"]["ordinal"]
                    encode_steps.append(
                        (
                            "ordinal",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                            cols,
                        )
                    )
                elif strategy == "target":
                    cols = self.pipeline_config["encoding"]["target"]
                    encode_steps.append(("target", TargetEncoder(), cols))
        return encode_steps

    def __create_scaling_steps(self) -> List[Tuple[str, BaseEstimator, List[str]]]:
        """
        Create scaling transformation steps based on configuration.

        Returns:
            List[Tuple[str, BaseEstimator, List[str]]]: List of scaling transformers
        """
        scaling_steps = []
        if self.pipeline_config.get("scaling"):
            for strategy in self.pipeline_config["scaling"].keys():
                if strategy == "standard":
                    cols = self.pipeline_config["scaling"]["standard"]
                    scaling_steps.append(("standard", StandardScaler(), cols))
                elif strategy == "minmax":
                    cols = self.pipeline_config["scaling"]["minmax"]
                    scaling_steps.append(("minmax", MinMaxScaler(), cols))
                elif strategy == "robust":
                    cols = self.pipeline_config["scaling"]["robust"]
                    scaling_steps.append(("robust", RobustScaler(), cols))
        return scaling_steps

    def __create_imputing_steps(self) -> List[Tuple[str, BaseEstimator, List[str]]]:
        """
        Create imputation transformation steps based on configuration.

        Returns:
            List[Tuple[str, BaseEstimator, List[str]]]: List of imputation transformers
        """
        imputing_steps = []
        if self.pipeline_config.get("imputation"):
            for strategy in self.pipeline_config["imputation"].keys():
                if strategy == "mean":
                    cols = self.pipeline_config["imputation"]["mean"]
                    imputing_steps.append(
                        ("mean", SimpleImputer(strategy="mean", add_indicator=True), cols)
                    )
                elif strategy == "median":
                    cols = self.pipeline_config["imputation"]["median"]
                    imputing_steps.append(
                        ("median", SimpleImputer(strategy="median", add_indicator=True), cols)
                    )
                elif strategy == "most_frequent":
                    cols = self.pipeline_config["imputation"]["most_frequent"]
                    imputing_steps.append(
                        (
                            "most_frequent",
                            SimpleImputer(strategy="most_frequent", add_indicator=True),
                            cols,
                        )
                    )
                elif strategy == "constant":
                    cols_fill: Dict = self.pipeline_config["imputation"]["constant"]
                    for col, fill_value in cols_fill.items():
                        imputing_steps.append(
                            (
                                "constant",
                                SimpleImputer(
                                    strategy="constant", fill_value=fill_value, add_indicator=True
                                ),
                                col,
                            )
                        )
        return imputing_steps

    def get_feature_names_from_preprocessor(self) -> List[str]:
        """
        Extract feature names from a ColumnTransformer after encoding.

        Returns:
            List[str]: A list of feature names from the fitted pipeline
        """
        feature_names = []
        for name, transformer, columns in self.pipeline.transformers_:
            if transformer == "drop" or transformer is None:
                continue  # Skip dropped columns
            elif hasattr(transformer, "get_feature_names_out"):
                # For transformers like OneHotEncoder
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                # For other transformers, use the column names directly
                feature_names.extend(columns)
        return feature_names
