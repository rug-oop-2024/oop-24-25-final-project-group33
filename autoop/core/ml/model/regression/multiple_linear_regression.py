import numpy as np

from pydantic import BaseModel, Field


class MultipleLinearRegression(BaseModel):
    def __init__(self):
        super().__init__()
        self._parameters = {}

    @property
    def parameters(self):
        return self._parameters

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray):
        ones_column = np.ones((observations.shape[0], 1))
        X = np.hstack([observations, ones_column])

        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        XtY = X.T @ ground_truth
        w = XtX_inv @ XtY

        self._parameters["coefficients"] = w[:-1]
        self._parameters["intercept"] = w[-1]

    def predict(self, observations: np.ndarray) -> np.ndarray:
        ones_column = np.ones((observations.shape[0], 1))
        X = np.hstack([observations, ones_column])

        w = np.hstack(
            [self._parameters["coefficients"],
             self._parameters["intercept"]]
             )
        predictions = X @ w

        return predictions
