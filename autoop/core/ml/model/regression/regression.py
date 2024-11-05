# autoop/core/ml/model/regression.py
from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor


class MultipleLinearRegression(Model):
    """Multiple Linear Regression model."""
    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__(model_type="regression")
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the training data."""
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target variable for the input data."""
        return self.model.predict(X)


class LinearRegressionModel(Model):
    """Linear Regression model."""
    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__(model_type="regression")
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the training data."""
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target variable for the input data."""
        return self.model.predict(X)


class RidgeRegressionModel(Model):
    """Ridge Regression model."""
    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__(model_type="regression")
        self.model = Ridge()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the training data."""
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target variable for the input data."""
        return self.model.predict(X)


class DecisionTreeRegressorModel(Model):
    """Decision Tree Regressor model."""
    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__(model_type="regression")
        self.model = DecisionTreeRegressor()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the training data."""
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target variable for the input data."""
        return self.model.predict(X)
