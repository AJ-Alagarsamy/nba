import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

class NBAModel:
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),  # Scale features
            ('logreg', LogisticRegression(
                C=0.5,  # Regularization - smaller = more regularization
                max_iter=1000,
                random_state=42,
                solver='lbfgs',
                class_weight='balanced'  # Handle any class imbalance
            ))
        ])
        self.features_to_use = None  # Will store which features to use
    
    def train(self, X, y):
        """Train the model and identify which features to use."""
        # Find and remove highly correlated features
        correlated_features = self._find_correlated_features(X)
        print(f"Removing {len(correlated_features)} highly correlated features")
        
        # Keep track of which features to use
        self.features_to_use = [col for col in X.columns if col not in correlated_features]
        X_reduced = X[self.features_to_use]
        
        # Train the model
        self.model.fit(X_reduced, y)
        
        # Optional: Print cross-validation scores
        try:
            scores = cross_val_score(self.model, X_reduced, y, cv=5)
            print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
        except:
            print("Could not perform cross-validation")
    
    def _find_correlated_features(self, X, threshold=0.85):
        """Find features with correlation above threshold."""
        correlated_features = set()
        correlation_matrix = X.corr()
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    colname = correlation_matrix.columns[i]
                    correlated_features.add(colname)
        
        return correlated_features
    
    def predict_probs(self, features):
        """Predict probability of home team winning."""
        if self.features_to_use is None:
            raise ValueError("Model must be trained before prediction")
        
        # Use only the features we kept during training
        features_reduced = features[self.features_to_use]
        
        # Predict probability of class 1 (home win)
        probabilities = self.model.predict_proba(features_reduced)
        return probabilities[:, 1]  # Return probability of home win