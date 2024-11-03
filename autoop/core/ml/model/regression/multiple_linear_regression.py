import numpy as np
from pydantic import BaseModel, PrivateAttr
from sklearn.linear_model import LinearRegression
from typing import Optional

class MultipleLinearRegression(BaseModel):
    _model: Optional[LinearRegression] = PrivateAttr(default=None)
    _parameters: dict = PrivateAttr(default_factory=dict)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        self._model = LinearRegression()
        self._model.fit(observations, ground_truth)
        self._parameters["coefficients"] = self._model.coef_
        self._parameters["intercept"] = self._model.intercept_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("The model has not been fitted yet.")
        return self._model.predict(observations)

    @property
    def parameters(self):
        return self._parameters
