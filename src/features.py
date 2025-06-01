import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Individual transformers for each feature engineering step

class TitleExtractor(BaseEstimator, TransformerMixin):
    """Extract and clean title from Name column"""
    
    def __init__(self):
        self.rare_titles = [
            "Lady", "Countess", "Capt", "Col", "Don", "Dr", 
            "Major", "Rev", "Sir", "Jonkheer", "Dona"
        ]
        self.title_mapping = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
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
        return self
    
    def transform(self, X):
        X = X.copy()
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        return X


class DeckExtractor(BaseEstimator, TransformerMixin):
    """Extract deck information from Cabin"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["Deck"] = X["Cabin"].astype(str).str[0]
        X["Deck"] = X["Deck"].fillna("U")
        return X


class TicketFeatures(BaseEstimator, TransformerMixin):
    """Create ticket-related features"""
    
    def __init__(self):
        self.ticket_counts_ = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        ticket_counts_ = X["Ticket"].value_counts().to_dict()
        # Use fitted ticket counts, default to 1 for unseen tickets
        X["TicketGroupSize"] = X["Ticket"].map(ticket_counts_).fillna(1)
        return X


class FareFeatures(BaseEstimator, TransformerMixin):
    """Create fare-related features"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Fare per person (requires FamilySize from previous transformer)
        X["FarePerPerson"] = X["Fare"] / X["FamilySize"]
        X.replace({"FarePerPerson": {np.inf: np.nan, -np.inf: np.nan}}, inplace=True)
        
        return X


class BinningFeatures(BaseEstimator, TransformerMixin):
    """Create binned features for Age and Fare"""
    
    def __init__(self):
        self.fare_bins_ = None
    
    def fit(self, X, y=None):
        # Fit fare quantile bins on training data
        return self
    
    def transform(self, X):
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
        return self
    
    def transform(self, X):
        X = X.copy()
        X["Pclass*AgeBin"] = X["Pclass"] * X["AgeBin"].fillna(0).astype(int)
        X["Sex_Pclass"] = X["Sex"].astype(str) + "_" + X["Pclass"].astype(str)
        return X


class MissingValueIndicators(BaseEstimator, TransformerMixin):
    """Create missing value indicator features"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["CabinMissing"] = X["Cabin"].isnull().astype(int)
        X["AgeMissing"] = X["Age"].isnull().astype(int)
        return X


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Complete feature engineering pipeline for Titanic dataset"""
    
    def __init__(self):
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
        self.pipeline.fit(X, y)
        return self
    
    def transform(self, X):
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def create_modular_pipeline() -> Pipeline:
    """Create a more modular pipeline where you can customize individual steps"""
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


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load data (example)
    # df = pd.read_csv("data/raw/train.csv")
    
    # Option 1: Use the complete transformer
    # engineer = TitanicFeatureEngineer()
    # transformed_df = engineer.fit_transform(df)
    
    # Option 2: Use individual transformers in a pipeline
    # pipeline = create_modular_pipeline()
    # transformed_df = pipeline.fit_transform(df)
    
    # Option 3: Use legacy function
    # transformed_df = extract_features(df)
    
    pass
