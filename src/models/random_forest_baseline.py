"""Random forest baseline model."""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Optional, Union


class RandomForestBaseline:
    """
    A baseline model using Random Forest Regressor for regression tasks.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs: Optional[int] = 1,
    ):
        """
        Initialize the Random Forest baseline model.

        Args:
            n_estimators: Number of trees in the forest.
            random_state: Random state for reproducibility.
            n_jobs: Number of parallel jobs for tree training (-1 uses all cores).
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the model to the training data.

        Args:
            X: Feature DataFrame.
            y: Target Series.
        """
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Make predictions on the input data.

        Args:
            X: Feature DataFrame.

        Returns:
            Predicted values as a Series.
        """
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=X.index)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate the R-squared score on the given data.

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            R-squared score.
        """
        return self.model.score(X, y)


if __name__ == "__main__":
    # Example usage with dummy data
    import numpy as np

    # Create dummy data
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    y = pd.Series(X['feature1'] + X['feature2'] + np.random.randn(100) * 0.1)

    # Split into train/test
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Train model
    model = RandomForestBaseline()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate
    score = model.score(X_test, y_test)
    print(f"R-squared score: {score:.3f}")
    print(f"Sample predictions: {y_pred.head()}")