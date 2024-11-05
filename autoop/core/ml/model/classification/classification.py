# autoop/core/ml/model/classification.py
from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class LogisticRegressionModel(Model):
    """Logistic Regression model."""
    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__(model_type="classification")
        self.model = LogisticRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the training data."""
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target variable for the input data."""
        return self.model.predict(X)


class DecisionTreeClassifierModel(Model):
    """Decision Tree Classifier model."""
    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__(model_type="classification")
        self.model = DecisionTreeClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the training data."""
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target variable for the input data."""
        return self.model.predict(X)


class RandomForestClassifierModel(Model):
    """Random Forest Classifier model."""
    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__(model_type="classification")
        self.model = RandomForestClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the training data."""
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target variable for the input data."""
        return self.model.predict(X)
