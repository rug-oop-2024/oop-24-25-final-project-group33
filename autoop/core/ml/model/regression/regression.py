# autoop/core/ml/model/regression.py
from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

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
