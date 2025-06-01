"""
Example usage of the transformer-based feature engineering pipeline.
This script demonstrates how to use the new sklearn-compatible transformers.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.features import (
    TitanicFeatureEngineer,
    create_feature_pipeline,
    create_modular_pipeline,
    extract_features,
)


def load_titanic_data():
    """Load and return Titanic dataset"""
    try:
        train_df = pd.read_csv("data/raw/train.csv")
        test_df = pd.read_csv("data/raw/test.csv")
        return train_df, test_df
    except FileNotFoundError:
        print("Dataset files not found. Please ensure train.csv and test.csv are in data/raw/")
        return None, None


def example_1_basic_usage():
    """Example 1: Basic usage with the complete transformer"""
    print("=" * 50)
    print("Example 1: Basic Usage")
    print("=" * 50)

    train_df, test_df = load_titanic_data()
    if train_df is None:
        print("Skipping example - no data found")
        return

    # Create and fit the feature engineer
    engineer = TitanicFeatureEngineer()

    # Fit on training data and transform both train and test
    train_transformed = engineer.fit_transform(train_df)
    test_transformed = engineer.transform(test_df)

    print(f"Original features: {train_df.shape[1]}")
    print(f"After transformation: {train_transformed.shape[1]}")
    print(f"New features added: {train_transformed.shape[1] - train_df.shape[1]}")

    # Show some of the new features
    new_features = [col for col in train_transformed.columns if col not in train_df.columns]
    print(f"\nNew features created: {new_features}")

    return train_transformed, test_transformed


def example_2_modular_pipeline():
    """Example 2: Using individual transformers in a modular pipeline"""
    print("\n" + "=" * 50)
    print("Example 2: Modular Pipeline")
    print("=" * 50)

    train_df, test_df = load_titanic_data()
    if train_df is None:
        print("Skipping example - no data found")
        return

    # Create a modular pipeline where you can customize individual steps
    pipeline = create_modular_pipeline()

    # You can also create a custom pipeline with only specific transformers
    from src.features import TitleExtractor, FamilySizeFeatures, DeckExtractor

    custom_pipeline = Pipeline(
        [
            ("title_extractor", TitleExtractor()),
            ("family_features", FamilySizeFeatures()),
            ("deck_extractor", DeckExtractor()),
        ]
    )

    # Transform the data
    train_transformed = custom_pipeline.fit_transform(train_df)
    test_transformed = custom_pipeline.transform(test_df)

    print(f"Custom pipeline features: {train_transformed.shape[1]}")

    return train_transformed, test_transformed


def example_3_full_ml_pipeline():
    """Example 3: Complete ML pipeline with feature engineering"""
    print("\n" + "=" * 50)
    print("Example 3: Complete ML Pipeline")
    print("=" * 50)

    train_df, test_df = load_titanic_data()
    if train_df is None:
        print("Skipping example - no data found")
        return

    # Separate features and target
    X = train_df.drop("Survived", axis=1)
    y = train_df["Survived"]

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a complete pipeline that includes feature engineering and modeling
    # Define categorical and numerical columns after feature engineering
    feature_engineer = TitanicFeatureEngineer()

    # First, let's see what columns we'll have after feature engineering
    sample_transformed = feature_engineer.fit_transform(X_train[:5])

    # Identify categorical and numerical columns
    categorical_features = ["Sex", "Embarked", "Title", "Deck", "Sex_Pclass"]
    numerical_features = [
        "Age",
        "Fare",
        "Pclass",
        "SibSp",
        "Parch",
        "FamilySize",
        "IsAlone",
        "TicketGroupSize",
        "FarePerPerson",
        "AgeBin",
        "FareBin",
        "Pclass*AgeBin",
        "CabinMissing",
        "AgeMissing",
    ]

    # Create preprocessing pipeline for categorical features
    categorical_transformer = Pipeline(
        [
            ("label_encoder", LabelEncoder())  # This is simplified; in practice use OneHotEncoder
        ]
    )

    # Create the complete pipeline
    complete_pipeline = Pipeline(
        [
            ("feature_engineer", TitanicFeatureEngineer()),
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(), numerical_features),
                        # Note: For categorical features, you'd typically use OneHotEncoder
                        # This is simplified for demonstration
                    ],
                    remainder="passthrough",
                ),
            ),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # Simplified version without ColumnTransformer for demonstration
    simple_pipeline = Pipeline(
        [
            ("feature_engineer", TitanicFeatureEngineer()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # Fit the pipeline
    simple_pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = simple_pipeline.predict(X_val)

    # Evaluate
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

    return simple_pipeline


def example_4_backward_compatibility():
    """Example 4: Using the legacy function for backward compatibility"""
    print("\n" + "=" * 50)
    print("Example 4: Backward Compatibility")
    print("=" * 50)

    train_df, test_df = load_titanic_data()
    if train_df is None:
        print("Skipping example - no data found")
        return

    # Use the legacy function - same interface as before
    train_transformed = extract_features(train_df)

    print("Legacy function still works!")
    print(f"Transformed shape: {train_transformed.shape}")

    return train_transformed


def run_all_examples():
    """Run all examples"""
    print("Transformer-based Feature Engineering Examples")
    print("=" * 60)

    # Example 1: Basic usage
    train_1, test_1 = example_1_basic_usage()

    # Example 2: Modular pipeline
    train_2, test_2 = example_2_modular_pipeline()

    # Example 3: Complete ML pipeline
    pipeline = example_3_full_ml_pipeline()

    # Example 4: Backward compatibility
    train_4 = example_4_backward_compatibility()

    print("\n" + "=" * 60)
    print("All examples completed!")


if __name__ == "__main__":
    run_all_examples()
