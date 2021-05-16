import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, features: pd.DataFrame, target: pd.Series = None):
        self.scaler.fit(features)
        return self

    def transform(self, features: pd.DataFrame, target: pd.Series = None):
        transformed = features.copy()
        self.scaler.transform(features)
        return transformed
