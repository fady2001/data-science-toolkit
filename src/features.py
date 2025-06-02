"""
Feature engineering transformers and pipelines for the Titanic dataset.

This module provides a comprehensive set of feature engineering transformers
that can be used individually or combined in pipelines to extract meaningful
features from the Titanic dataset for machine learning models.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class TitleExtractor(BaseEstimator, TransformerMixin):
    """Extract and clean title from Name column"""
    
    def __init__(self):
        """
        Initialize the TitleExtractor.
        
        Sets up mappings for rare titles and title variations.
        """
        self.rare_titles = [
            "Lady", "Countess", "Capt", "Col", "Don", "Dr", 
            "Major", "Rev", "Sir", "Jonkheer", "Dona"
        ]
        self.title_mapping = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for this transformer).
        
        Args:
            X (pd.DataFrame): Input features
            y (array-like, optional): Target values
            
        Returns:
            self: Returns the instance itself
        """
        return self
    
    def transform(self, X):
        """
        Extract titles from passenger names and standardize them.
        
        Args:
            X (pd.DataFrame): Input features containing 'Name' column
            
        Returns:
            pd.DataFrame: Features with added 'Title' column
        """
        X = X.copy()
        # Extract title from Name
        X["Title"] = X["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
        # Replace rare titles
        X["Title"] = X["Title"].replace(self.rare_titles, "Rare")
        # Map variations
        X["Title"] = X["Title"].replace(self.title_mapping)
        return X


class FamilySizeFeatures(BaseEstimator, TransformerMixin):
    """Create family size related features"""
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for this transformer).
        
        Args:
            X (pd.DataFrame): Input features
            y (array-like, optional): Target values
            
        Returns:
            self: Returns the instance itself
        """
        return self
    
    def transform(self, X):
        """
        Create family size and alone indicator features.
        
        Args:
            X (pd.DataFrame): Input features containing 'SibSp' and 'Parch' columns
            
        Returns:
            pd.DataFrame: Features with added 'FamilySize' and 'IsAlone' columns
        """
        X = X.copy()
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        return X


class DeckExtractor(BaseEstimator, TransformerMixin):
    """Extract deck information from Cabin"""
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for this transformer).
        
        Args:
            X (pd.DataFrame): Input features
            y (array-like, optional): Target values
            
        Returns:
            self: Returns the instance itself
        """
        return self
    
    def transform(self, X):
        """
        Extract deck letter from cabin information.
        
        Args:
            X (pd.DataFrame): Input features containing 'Cabin' column
            
        Returns:
            pd.DataFrame: Features with added 'Deck' column
        """
        X = X.copy()
        X["Deck"] = X["Cabin"].astype(str).str[0]
        X["Deck"] = X["Deck"].fillna("U")
        return X


class TicketFeatures(BaseEstimator, TransformerMixin):
    """Create ticket-related features"""
    
    def __init__(self):
        """Initialize the TicketFeatures transformer."""
        self.ticket_counts_ = None
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for this transformer).
        
        Args:
            X (pd.DataFrame): Input features
            y (array-like, optional): Target values
            
        Returns:
            self: Returns the instance itself
        """
        return self
    
    def transform(self, X):
        """
        Create ticket group size features.
        
        Args:
            X (pd.DataFrame): Input features containing 'Ticket' column
            
        Returns:
            pd.DataFrame: Features with added 'TicketGroupSize' column
        """
        X = X.copy()
        ticket_counts_ = X["Ticket"].value_counts().to_dict()
        # Use fitted ticket counts, default to 1 for unseen tickets
        X["TicketGroupSize"] = X["Ticket"].map(ticket_counts_).fillna(1)
        return X


class FareFeatures(BaseEstimator, TransformerMixin):
    """Create fare-related features"""
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for this transformer).
        
        Args:
            X (pd.DataFrame): Input features
            y (array-like, optional): Target values
            
        Returns:
            self: Returns the instance itself
        """
        return self
    
    def transform(self, X):
        """
        Create fare per person feature.
        
        Args:
            X (pd.DataFrame): Input features containing 'Fare' and 'FamilySize' columns
            
        Returns:
            pd.DataFrame: Features with added 'FarePerPerson' column
        """
        X = X.copy()
        # Fare per person (requires FamilySize from previous transformer)
        X["FarePerPerson"] = X["Fare"] / X["FamilySize"]
        X.replace({"FarePerPerson": {np.inf: np.nan, -np.inf: np.nan}}, inplace=True)
        
        return X


class BinningFeatures(BaseEstimator, TransformerMixin):
    """Create binned features for Age and Fare"""
    
    def __init__(self):
        """Initialize the BinningFeatures transformer."""
        self.fare_bins_ = None
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for this transformer).
        
        Args:
            X (pd.DataFrame): Input features
            y (array-like, optional): Target values
            
        Returns:
            self: Returns the instance itself
        """
        # Fit fare quantile bins on training data
        return self
    
    def transform(self, X):
        """
        Create binned age and fare features.
        
        Args:
            X (pd.DataFrame): Input features containing 'Age' and 'Fare' columns
            
        Returns:
            pd.DataFrame: Features with added 'AgeBin' and 'FareBin' columns
        """
        X = X.copy()
        try:
            fare_bins_ = pd.qcut(X["Fare"].dropna(), 4, retbins=True, duplicates='drop')[1]
        except ValueError:
            # If qcut fails due to duplicates, use regular cut
            fare_bins_ = [X["Fare"].min(), X["Fare"].quantile(0.25), 
                              X["Fare"].quantile(0.5), X["Fare"].quantile(0.75), 
                              X["Fare"].max()]
        # Age binning
        X["AgeBin"] = pd.cut(
            X["Age"], bins=[0, 12, 20, 40, 60, 80], labels=False, include_lowest=True
        )
        # Fare binning using fitted bins
        X["FareBin"] = pd.cut(X["Fare"], bins=fare_bins_, labels=False, include_lowest=True)
        return X


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Create interaction features"""
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for this transformer).
        
        Args:
            X (pd.DataFrame): Input features
            y (array-like, optional): Target values
            
        Returns:
            self: Returns the instance itself
        """
        return self
    
    def transform(self, X):
        """
        Create interaction features between different variables.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with added interaction columns
        """
        X = X.copy()
        X["Pclass*AgeBin"] = X["Pclass"] * X["AgeBin"].fillna(0).astype(int)
        X["Sex_Pclass"] = X["Sex"].astype(str) + "_" + X["Pclass"].astype(str)
        return X


class MissingValueIndicators(BaseEstimator, TransformerMixin):
    """Create missing value indicator features"""
    
    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for this transformer).
        
        Args:
            X (pd.DataFrame): Input features
            y (array-like, optional): Target values
            
        Returns:
            self: Returns the instance itself
        """
        return self
    
    def transform(self, X):
        """
        Create binary indicators for missing values.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with added missing value indicator columns
        """
        X = X.copy()
        X["CabinMissing"] = X["Cabin"].isnull().astype(int)
        X["AgeMissing"] = X["Age"].isnull().astype(int)
        return X


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Complete feature engineering pipeline for Titanic dataset"""
    
    def __init__(self):
        """
        Initialize the complete feature engineering pipeline.
        
        Creates a pipeline with all feature engineering transformers.
        """
        self.pipeline = Pipeline([
            ('title_extractor', TitleExtractor()),
            ('family_features', FamilySizeFeatures()),
            ('deck_extractor', DeckExtractor()),
            ('ticket_features', TicketFeatures()),
            ('fare_features', FareFeatures()),
            ('binning_features', BinningFeatures()),
            ('interaction_features', InteractionFeatures()),
            ('missing_indicators', MissingValueIndicators()),
        ])
    
    def fit(self, X, y=None):
        """
        Fit the complete feature engineering pipeline.
        
        Args:
            X (pd.DataFrame): Input features
            y (array-like, optional): Target values
            
        Returns:
            self: Returns the instance itself
        """
        self.pipeline.fit(X, y)
        return self
    
    def transform(self, X):
        """
        Transform features using the complete pipeline.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Transformed features
        """
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform features in one step.
        
        Args:
            X (pd.DataFrame): Input features
            y (array-like, optional): Target values
            
        Returns:
            pd.DataFrame: Transformed features
        """
        return self.fit(X, y).transform(X)


def create_modular_pipeline() -> Pipeline:
    """
    Create a modular pipeline where individual steps can be customized.
    
    Returns:
        Pipeline: Scikit-learn pipeline with all feature engineering steps
    """
    return Pipeline([
        ('title_extractor', TitleExtractor()),
        ('family_features', FamilySizeFeatures()),
        ('deck_extractor', DeckExtractor()),
        ('ticket_features', TicketFeatures()),
        ('fare_features', FareFeatures()),
        ('binning_features', BinningFeatures()),
        ('interaction_features', InteractionFeatures()),
        ('missing_indicators', MissingValueIndicators()),
    ])