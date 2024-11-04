from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Dict, Any, Literal
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Base Model Class

class Model(ABC):
    def __init__(self, model_type: Literal["classification", "regression"]):
        self.model_type = model_type  # "classification" or "regression"
        self.parameters: Dict[str, Any] = {}  # Dictionary to hold model parameters

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model with input data X and target labels y."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output using the trained model on input data X."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return deepcopy(self.parameters)

    def set_params(self, **params: Any):
        """Set model parameters."""
        self.parameters.update(params)

    def to_artifact(self, name: str) -> Artifact:
        """Convert model and its parameters into an artifact."""
        data = {
            "model_type": self.model_type,
            "parameters": self.parameters,
        }
        return Artifact(name=name, data=data)

    def save(self, path: str):
        """Save the model parameters to a file."""
        with open(path, 'wb') as file:
            np.save(file, self.parameters)

    def load(self, path: str):
        """Load the model parameters from a file."""
        with open(path, 'rb') as file:
            self.parameters = np.load(file, allow_pickle=True).item()


# Classification Models

class LogisticRegressionModel(Model):
    def __init__(self):
        super().__init__(model_type="classification")
        self.model = LogisticRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class DecisionTreeClassifierModel(Model):
    def __init__(self):
        super().__init__(model_type="classification")
        self.model = DecisionTreeClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RandomForestClassifierModel(Model):
    def __init__(self):
        super().__init__(model_type="classification")
        self.model = RandomForestClassifier()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# Regression Models

class LinearRegressionModel(Model):
    def __init__(self):
        super().__init__(model_type="regression")
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class RidgeRegressionModel(Model):
    def __init__(self):
        super().__init__(model_type="regression")
        self.model = Ridge()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class DecisionTreeRegressorModel(Model):
    def __init__(self):
        super().__init__(model_type="regression")
        self.model = DecisionTreeRegressor()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
