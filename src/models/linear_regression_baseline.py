"""Linear regression baseline model."""

import pandas as pd
from sklearn.linear_model import LinearRegression


class LinearRegressionBaseline:
    """A simple linear regression baseline for traffic forecasting."""

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=X.index)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        return self.model.score(X, y)
